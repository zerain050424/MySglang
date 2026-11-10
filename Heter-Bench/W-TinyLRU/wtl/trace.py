"""trace 读取与到达调度（与 ``Heter-Bench/scripts/bench_hetercache.py`` 同语义）。

为什么要自己重写而不是 import
-----------------------------
``bench_hetercache.py`` 依赖 aiohttp / yaml，而离线回放只需 jsonl。
这里复刻它三个可直接对齐的行为，**随机数与排序公式完全一致**：

* ``load_sessions``：``{"<id>": {id, num_turns, turn_1..N}}`` 逐行读；
* ``generate_poisson_requests``：``Exp(lambda_rate)`` 累加得到到达时刻；
* 轮间等待：per-session ``random.Random(stable_seed_from_str(sid, seed+task_id*1000))``
  采样 ``lognormvariate(mu, sigma)``（与在线端同一条 trace 同一 seed 可复现）。

与真机的已知差异（写进报告 limitation）
---------------------------------------
1. 在线端下一轮要等上一轮**完成**才开始，离线端默认 ``service_s=0``
   （可用 ``--service-ms-per-1k`` 近似补上），因此离线复用间隔略短；
2. 离线不建模调度/批处理，只建模「到达顺序 + KV 池容量」。
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parents[1]          # Heter-Bench/W-TinyLRU
UNIBENCH_ROOT = Path(os.environ.get("UNIBENCH_ROOT", PKG_ROOT.parent))

# 负载注册表：name -> (相对 Heter-Bench 的 jsonl 路径, mu, sigma, 说明)
WORKLOADS: Dict[str, Dict[str, Any]] = {
    "alfred": {
        "path": "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl",
        "mu": 1.81, "sigma": 1.092, "task": "AgentBank_alfred",
        "note": "多轮 agentic：共享 system + tool schema（共享前缀最深）",
    },
    "apps": {
        "path": "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl",
        "mu": 0.0, "sigma": 0.0, "task": "AgentBank_apps",
        "note": "单轮 API 调用：短请求、低复用（干扰项）",
    },
    "codeparrot": {
        "path": "TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl",
        "mu": 1.81, "sigma": 1.092, "task": "Codeparrot",
        "note": "单轮代码：长 body，几乎不共享前缀",
    },
    "sharegpt": {
        "path": "TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl",
        "mu": 4.15, "sigma": 0.971, "task": "ShareGPT",
        "note": "多轮 chat：共享前缀是会话内历史（H5 护栏臂）",
    },
    "hybrid": {
        "path": "configs/hybrid_tasks_config.yaml",  # 走 --tasks-config
        "mu": 0.0, "sigma": 0.0, "task": "hybrid",
        "note": "四负载混合（1:1:1:1），与 Heter-Bench 在线实验同构",
    },
}


def workload_path(name: str) -> Path:
    cfg = WORKLOADS.get(name)
    if cfg is None:
        raise KeyError(f"未知 workload {name!r}；可用：{', '.join(WORKLOADS)}")
    return (UNIBENCH_ROOT / cfg["path"]).resolve()


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Request:
    """一次前向请求（一个 turn）。"""

    idx: int
    ts: float
    sid: str
    task: str
    turn: int
    prompt: List[int]
    output: List[int]

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    @property
    def total_len(self) -> int:
        return len(self.prompt) + len(self.output)

    @property
    def rid(self) -> str:
        # 与 bench_hetercache.py 完全一致，便于离线/在线逐请求对齐
        return f"{self.sid}_turn_{self.turn}"


@dataclasses.dataclass
class SessionArrival:
    ts: float
    counter: int
    task: str
    cfg: Dict[str, Any]
    session: Dict[str, Any]


# --------------------------------------------------------------------------
# 与 bench_hetercache.py 逐行对齐的两个小工具
# --------------------------------------------------------------------------
def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """字符串 -> 稳定 32-bit seed（sha256，与 bench_hetercache 相同）。"""
    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


def generate_poisson_requests(lambda_rate: float, n: int, seed: int) -> List[float]:
    """Poisson 过程：间隔 ~ Exp(lambda_rate)，累加得到到达时刻。"""
    rng = random.Random(seed)
    out, acc = [], 0.0
    for _ in range(n):
        acc += rng.expovariate(max(lambda_rate, 1e-9))
        out.append(acc)
    return out


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------
def load_sessions(
    path: str | Path,
    num_sessions: int = -1,
    seed: Optional[int] = None,
    shuffle: bool = True,
    read_limit: int = 0,
) -> List[Dict[str, Any]]:
    """读 session_data.jsonl；语义与 ``bench_hetercache.load_sessions`` 相同。

    ``read_limit``：只读文件前 N 行再做 shuffle/截取。大文件（2~3GB 的
    AgentBank/ShareGPT）全量读入要几十 GB Python 对象，离线矩阵没必要；
    设成 ``max(2000, num_sessions*10)`` 量级即可，代价是采样窗口偏向文件前段
    （会写进结果 meta，报告里要说明）。
    """
    sessions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(list(json.loads(line).values())[0])
            if read_limit and len(sessions) >= read_limit:
                break
            if (
                not shuffle
                and num_sessions
                and num_sessions > 0
                and len(sessions) >= num_sessions
            ):
                break
    if shuffle:
        random.Random(seed).shuffle(sessions)
    if num_sessions and num_sessions > 0:
        sessions = sessions[:num_sessions]
    return sessions


def load_tasks_config(path: str | Path) -> Dict[str, Any]:
    import yaml  # 延迟导入：只有混合模式需要

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_path(rel: str | Path, config_path: Path) -> Path:
    """解析配置里的 ``data_path``。

    在线端 ``bench_hetercache`` 以 **Heter-Bench 根目录** 为基准，而配置文件放在
    ``mixexp/configs/`` 下，所以不能简单按 ``config_path.parent`` 拼。这里依次尝试
    当前工作目录、配置目录、以及配置目录的各级祖先，取第一个存在的路径；
    都不存在时返回按配置目录拼接的结果（让下游报错时能看出找的是哪个位置）。
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    bases = [Path.cwd(), config_path.parent, *config_path.parents]
    for base in bases:
        cand = base / p
        if cand.exists():
            return cand
    return config_path.parent / p


# --------------------------------------------------------------------------
# 调度
# --------------------------------------------------------------------------
def _turns(session: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for t in range(1, int(session["num_turns"]) + 1):
        turn = session.get(f"turn_{t}")
        if turn and turn.get("input_ids"):
            yield turn


def _service_time(req_tokens: int, ms_per_1k: float) -> float:
    return (req_tokens / 1000.0) * (ms_per_1k / 1000.0)


def _session_turns(ev: SessionArrival, wait_scale: float, service_ms_per_1k: float,
                   service_ms: float = 0.0):
    """展开一条会话的轮次与相对时间偏移（不含绝对起点）。

    轮间间隔用 ``stable_seed_from_str(sid, seed+task_id*1000)`` 采样 lognormal，
    与在线端逐位一致；因此同一条会话无论被排到时间轴哪一段，轮间节奏都不变。
    """
    cfg = ev.cfg
    mu = float(cfg.get("mu", 0.0) or 0.0)
    sigma = float(cfg.get("sigma", 0.0) or 0.0)
    base_seed = int(cfg.get("seed", 42)) + int(cfg.get("task_id", 0)) * 1000
    rng = random.Random(stable_seed_from_str(ev.session["id"], base_seed=base_seed))

    turns = list(_turns(ev.session))
    offsets = [0.0]
    t = 0.0
    for ti, turn in enumerate(turns):
        if ti + 1 >= len(turns):
            break
        if wait_scale > 0:
            wait = 0.0 if (mu == 0.0 and sigma == 0.0) else rng.lognormvariate(mu, sigma)
            t += wait * wait_scale
        # 下一轮要等上一轮**完成**才开始：真机 bench 的 run_session 是
        # ``await request(); sleep(wait)``，所以轮间隔 = 服务时长 + 采样等待。
        t += _service_time(len(turn["input_ids"]), service_ms_per_1k) + service_ms / 1000.0
        offsets.append(t)
    return turns, offsets


def build_schedule(
    arrivals: Sequence[SessionArrival],
    *,
    wait_scale: float = 1.0,
    arrival_scale: float = 1.0,
    service_ms_per_1k: float = 0.0,
    service_ms: float = 0.0,
    concurrency: int = 0,
) -> List[Request]:
    """把 session 到达流展开成逐请求时间线（轮内串行、会话间并发）。

    ``arrival_scale``：整体缩放到达节奏（1=真实 Poisson，0=全部 t=0）。
    ``wait_scale``：轮间等待缩放（0=不等；1=按各负载 lognormal 采样）。
    ``concurrency``：并发会话上限（0=不限）。在线端 ``bench_hetercache`` 用
    ``asyncio.Semaphore(concurrency)`` 限制的正是**会话数**——一条会话从第 1 轮
    跑到最后一轮才释放名额。离线端若不限，500 条 40 轮的 alfred 会话会全部同时
    活着抢同一个池子，命中率会低一个数量级；这是离线/在线最大的差异来源。

    限流实现：按到达顺序排队，维护 ``concurrency`` 个 worker 的下次空闲时刻
    （最小堆）。会话开始 = ``max(到达时刻, 最早空闲时刻)``，占用 worker 直到
    最后一轮结束。等价于信号量 + FIFO 队列的离散事件形式。
    """
    out: List[Request] = []
    idx = 0
    free_at: List[float] = []          # worker 空闲时刻的最小堆
    limit = max(0, int(concurrency or 0))

    for ev in arrivals:
        turns, offsets = _session_turns(ev, wait_scale, service_ms_per_1k, service_ms)
        if not turns:
            continue
        arrive = ev.ts * arrival_scale
        if limit and len(free_at) >= limit:
            start = max(arrive, heapq.heappop(free_at))
        else:
            start = arrive
        if limit:
            # 会话占用信号量直到**最后一轮返回**：真机 run_session 在
            # ``await request()`` 之后才 exit，所以还要加一个服务时长。
            heapq.heappush(free_at, start + offsets[-1] + service_ms / 1000.0)

        sid, task = ev.session["id"], ev.task
        for ti, turn in enumerate(turns):
            out.append(
                Request(
                    idx=idx, ts=start + offsets[ti], sid=sid, task=task, turn=ti + 1,
                    prompt=list(turn["input_ids"]),
                    output=list(turn.get("response_ids") or []),
                )
            )
            idx += 1

    out.sort(key=lambda r: (r.ts, r.idx))
    for i, r in enumerate(out):
        r.idx = i
    return out


def schedule_single_workload(
    workload: str,
    *,
    num_sessions: int,
    seed: int,
    wait_scale: float,
    arrival_scale: float,
    service_ms_per_1k: float = 0.0,
    data_path: Optional[str] = None,
    lambda_rate: float = 1.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    task_name: Optional[str] = None,
    read_limit: int = 0,
    shuffle: bool = True,
    concurrency: int = 0,
    service_ms: float = 0.0,
) -> List[Request]:
    cfg = WORKLOADS.get(workload, {})
    path = Path(data_path).resolve() if data_path else workload_path(workload)
    # 与 bench_hetercache 单负载模式一致：用同一个 seed 打乱再截取（否则跨进程不可复现）
    sessions = load_sessions(
        path, num_sessions, seed=seed, shuffle=shuffle, read_limit=read_limit
    )
    task_cfg = dict(cfg)
    task_cfg["seed"] = seed
    task_cfg["task_id"] = 0
    if task_name:
        # 显式 task 标签（--data-path 自定义 trace 时必须，否则会沿用 name 的标签）
        task_cfg["name"] = task_name
    if mu is not None:
        task_cfg["mu"] = mu
    if sigma is not None:
        task_cfg["sigma"] = sigma
    elif workload not in WORKLOADS:
        # 未注册负载（合成 trace / 自定义 jsonl）：给一个温和的轮间间隔，
        # 否则 wait_scale 无效、会话内复用在时间上退化成 0 间隔。
        task_cfg.setdefault("mu", 1.8)
        task_cfg.setdefault("sigma", 1.0)
    times = generate_poisson_requests(lambda_rate, len(sessions), seed)
    label = task_name or task_cfg.get("name") or cfg.get("task") or workload
    arrivals = [
        SessionArrival(t, i, label, task_cfg, s)
        for i, (s, t) in enumerate(zip(sessions, times))
    ]
    arrivals.sort(key=lambda e: (e.ts, e.counter))
    return build_schedule(
        arrivals,
        wait_scale=wait_scale,
        arrival_scale=arrival_scale,
        service_ms_per_1k=service_ms_per_1k,
        service_ms=service_ms,
        concurrency=concurrency,
    )


def schedule_hybrid(
    *,
    tasks_config: str | Path,
    seed: int,
    wait_scale: float,
    arrival_scale: float,
    service_ms_per_1k: float = 0.0,
    num_sessions_override: int = -1,
    concurrency: int = 0,
    service_ms: float = 0.0,
) -> List[Request]:
    """多负载混合（对齐 bench_hetercache.build_hybrid_arrival_schedule）。"""
    config_path = Path(tasks_config).resolve()
    cfg = load_tasks_config(config_path)
    arrivals: List[SessionArrival] = []
    counter = 0
    for task in cfg.get("tasks", []):
        data_path = resolve_data_path(task["data_path"], config_path)
        max_record = int(task.get("max_record", -1))
        if num_sessions_override and num_sessions_override > 0:
            max_record = min(max_record, num_sessions_override) if max_record > 0 else num_sessions_override
        sessions = load_sessions(data_path, max_record, seed=None, shuffle=False)
        task_id = int(task.get("task_id", 0))
        rng = random.Random(int(task.get("seed", seed)) + task_id)
        if task.get("shuffle", True):
            rng.shuffle(sessions)
        times = generate_poisson_requests(
            float(task.get("lambda_rate", 1.0)), len(sessions),
            int(task.get("seed", seed)) + task_id,
        )
        task_cfg = dict(task)
        for sess, at in zip(sessions, times):
            arrivals.append(SessionArrival(at, counter, task["name"], task_cfg, sess))
            counter += 1
    arrivals.sort(key=lambda e: (e.ts, e.counter))
    return build_schedule(
        arrivals,
        wait_scale=wait_scale,
        arrival_scale=arrival_scale,
        service_ms_per_1k=service_ms_per_1k,
        service_ms=service_ms,
        concurrency=concurrency,
    )


def trace_fingerprint(requests: Sequence[Request]) -> str:
    """请求时间线指纹：用于离线/在线、真/假后端的一致性对照。"""
    h = hashlib.sha256()
    for r in requests:
        h.update(f"{r.ts:.6f}|{r.sid}|{r.turn}|{r.prompt_len}|{r.total_len}\n".encode())
    return h.hexdigest()[:16]
