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


def build_schedule(
    arrivals: Sequence[SessionArrival],
    *,
    wait_scale: float = 1.0,
    arrival_scale: float = 1.0,
    service_ms_per_1k: float = 0.0,
) -> List[Request]:
    """把 session 到达流展开成逐请求时间线（轮内串行）。

    ``arrival_scale``：整体缩放到达节奏（1=真实 Poisson，0=全部 t=0）。
    ``wait_scale``：轮间等待缩放（0=不等；1=按各负载 lognormal 采样）。
    """
    out: List[Request] = []
    idx = 0
    for ev in arrivals:
        cfg = ev.cfg
        sid = ev.session["id"]
        task = ev.task
        mu = float(cfg.get("mu", 0.0) or 0.0)
        sigma = float(cfg.get("sigma", 0.0) or 0.0)
        base_seed = int(cfg.get("seed", 42)) + int(cfg.get("task_id", 0)) * 1000
        rng = random.Random(stable_seed_from_str(sid, base_seed=base_seed))

        t = ev.ts * arrival_scale
        turns = list(_turns(ev.session))
        for ti, turn in enumerate(turns):
            prompt = list(turn["input_ids"])
            output = list(turn.get("response_ids") or [])
            out.append(
                Request(
                    idx=idx, ts=t, sid=sid, task=task, turn=ti + 1,
                    prompt=prompt, output=output,
                )
            )
            idx += 1
            if ti + 1 < len(turns):
                if wait_scale > 0:
                    wait = 0.0 if (mu == 0.0 and sigma == 0.0) else rng.lognormvariate(mu, sigma)
                    t += wait * wait_scale
                t += _service_time(len(prompt), service_ms_per_1k)
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
    )


def schedule_hybrid(
    *,
    tasks_config: str | Path,
    seed: int,
    wait_scale: float,
    arrival_scale: float,
    service_ms_per_1k: float = 0.0,
    num_sessions_override: int = -1,
) -> List[Request]:
    """多负载混合（对齐 bench_hetercache.build_hybrid_arrival_schedule）。"""
    config_path = Path(tasks_config).resolve()
    cfg = load_tasks_config(config_path)
    base_dir = config_path.parent
    arrivals: List[SessionArrival] = []
    counter = 0
    for task in cfg.get("tasks", []):
        data_path = Path(task["data_path"])
        if not data_path.is_absolute():
            data_path = base_dir / data_path
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
    )


def trace_fingerprint(requests: Sequence[Request]) -> str:
    """请求时间线指纹：用于离线/在线、真/假后端的一致性对照。"""
    h = hashlib.sha256()
    for r in requests:
        h.update(f"{r.ts:.6f}|{r.sid}|{r.turn}|{r.prompt_len}|{r.total_len}\n".encode())
    return h.hexdigest()[:16]
