#!/usr/bin/env python3
"""Heter-Bench：堆驱动的真实墙钟发射器
  全局最小堆只存 (due_ts, seq, 待发轮次)。主循环 peek → sleep 到真实墙钟 →
  create_task 发请求，不 await 它；每轮流式响应结束后按 lognormal 算 next_t 再压回堆。
  轮间空窗不占任何名额。
用法：
  python scripts/bench_heap.py \
      --tasks-config mixexp/TaskData/exp3/chat/tasks.yaml \
      --base-url http://127.0.0.1:30000 --wait-scale 1.0 \
      --output result/heap_chat.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import heapq
import itertools
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import aiohttp
import yaml

# configs 里的 data_path 相对 Heter-Bench 根目录解析；按「配置文件所在目录」
# 解析会得到 configs/TaskData/...，与实际布局差一层。
HETER_BENCH_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_sessions(
    path: str, num_sessions: int, seed: Optional[int] = None, shuffle: bool = True
) -> List[Dict[str, Any]]:
    """读取 session_data.jsonl（每行 {"<id>": {id, num_turns, turn_1..N}}）。

    shuffle=False 时保持文件原始顺序：用于 mashqa 这类"同一文档的多个问题
    在文件中相邻"的数据，避免打乱后同文档兄弟问题被拆散、共享长前缀失效。
    混合负载模式下 shuffle 由 build_hybrid_arrival_schedule 用 (seed+task_id)
    的 RNG 完成（对齐 HeterCache init_heap），故此处传 shuffle=False。
    """
    sessions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每行只有一个键，取该 session 的 value
            sessions.append(list(json.loads(line).values())[0])
            # 提前终止：仅当不需要「先打乱全集再截取」（shuffle=False）时，
            # 读满 num_sessions 即停，避免读入 GB 级大文件。
            # 对齐 HeterCache data_parser.read_simulator_data 的 max_records 提前 break。
            if (
                not shuffle
                and num_sessions
                and num_sessions > 0
                and len(sessions) >= num_sessions
            ):
                break
    # 固定种子打乱，保证可复现（shuffle=True 时需先读全集再打乱再截取）
    if shuffle:
        random.Random(seed).shuffle(sessions)
    if num_sessions and num_sessions > 0:
        sessions = sessions[:num_sessions]
    return sessions


# --------------------------------------------------------------------------
# 会话轮数口径校验
# --------------------------------------------------------------------------
def assert_turn_policy(
    task: Dict[str, Any], sessions: List[Dict[str, Any]], data_path: str
) -> None:
    """校验选中的会话都长过 TRANS 带:num_turns >= 配置里的 min_turns。

    依据：分类器（``SessionRegistry._classify``）对 ``turn_count < MULTI_TURN_MIN``
    的会话判 TRANS。多轮源里这类会话**终身**是 TRANS，会以「一次性流量」身份
    抢 one-shot 带，把自己正在复用的前缀一起清掉——这是多轮负载均匀命中率
    损失的主要来源。数据侧的剔除由 ``tools/drop_short_sessions.py`` 完成
    （``mixexp/prepare_data.sh`` 会调），这里只拦截，防的是换了 data_path、
    或者旧数据没重新生成时，静默跑出一组不可比较的数字。

    配置里没写 ``min_turns`` 的 task 不校验（单轮负载本来就是 1 轮）。
    """
    min_turns = task.get("min_turns")
    if not min_turns:
        return
    min_turns = int(min_turns)
    max_turns = task.get("max_turns")
    if max_turns and int(max_turns) < min_turns:
        raise ValueError(
            f"task {task.get('name')!r}: max_turns={max_turns} 会把手里的会话截到 "
            f"< min_turns={min_turns}，截完就全判 TRANS 了"
        )
    short = [s for s in sessions if int(s.get("num_turns") or 0) < min_turns]
    if short:
        turns = sorted(int(s.get("num_turns") or 0) for s in short)
        raise ValueError(
            f"task {task.get('name')!r}: {len(short)}/{len(sessions)} 条会话轮数 "
            f"< {min_turns}（{turns[0]}–{turns[-1]}），会终身判 TRANS。\n"
            f"  数据：{data_path}\n"
            f"  先生成过滤后数据：bash mixexp/prepare_data.sh，"
            f"data_path 指向 *.mt{min_turns}.jsonl"
        )


# --------------------------------------------------------------------------
# 到达流：各 task 独立 Poisson，合并进同一个堆
# --------------------------------------------------------------------------
def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """字符串 -> 稳定 32-bit seed（模仿 HeterCache simulator.py stable_seed_from_str）。"""
    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


def pick_arrival_params(task_cfg: Dict[str, Any], args) -> tuple:
    """按负载取 (mu, sigma)；显式指定 --mu/--sigma 时全局 override。

    对应 HeterCache simulator.py 的：
        task_para = task_paras[task_name]
        wait_time = rng.lognormal(mean=task_para["mu"], sigma=task_para["sigma"])
    """
    if args.mu is not None and args.sigma is not None:
        return args.mu, args.sigma
    return task_cfg.get("mu", 0.0), task_cfg.get("sigma", 0.0)


def generate_poisson_requests(
    lambda_rate: float, num_sessions: int, seed: int
) -> List[float]:
    """模仿 HeterCache simulator/utils.py::generate_poisson_requests。
    Poisson 过程：session 到达间隔服从 Exp(lambda_rate)（均值 1/lambda_rate），
    累加得到每个 session 的到达时刻。
    numpy 的 default_rng.exponential(scale=1/λ) 与 stdlib random.expovariate(λ)
    同分布（均值 1/λ），这里用 stdlib 以避免引入 numpy。
    """
    rng = random.Random(seed)
    arrival_times: List[float] = []
    acc = 0.0
    for _ in range(num_sessions):
        acc += rng.expovariate(lambda_rate)  # 指数分布到达间隔
        arrival_times.append(acc)
    return arrival_times


class SessionArrival(NamedTuple):
    """一个「会话首轮到达事件」：timestamp + 全局序号 + 负载标签 + 会话数据。"""

    arrival_offset: float
    counter: int
    task_name: str
    task_cfg: Dict[str, Any]
    session: Dict[str, Any]


def build_hybrid_arrival_schedule(
    tasks_config: Dict[str, Any], base_dir: Optional[str] = None
) -> List[SessionArrival]:
    """把多个异构负载按 lambda_rate「比例」混成一条全局到达流。

    逐行对齐 HeterCache simulator.py::init_heap：
      1. 对每个 task：读其 session 数据（max_record 限制数量）
      2. 用 rng(seed + task_id) 打乱该 task 自身 session 顺序
      3. 用 Poisson(lambda_rate) 生成该 task 的 session 到达时刻
      4. 所有 task 的 (arrival_time, counter, session) 合并进同一个列表
      5. 按 (arrival_time, counter) 升序排序 → 异构负载在时间轴上交错混合

    返回按到达时刻升序的 SessionArrival 列表。「比例」即各 task 的 lambda_rate
    之比（例如 task A lambda_rate=2、task B lambda_rate=1 → 2:1 混合）。
    """
    events: List[SessionArrival] = []
    counter = 0  # 全局计数器（模仿 itertools.count）：时间戳相同按入堆顺序稳定排序

    for task in tasks_config.get("tasks", []):
        name = task["name"]
        data_path = task["data_path"]
        # data_path 相对 Heter-Bench 根目录解析（而非 CWD），与 wtl 的 UNIBENCH_ROOT 一致
        if base_dir and not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)

        max_record = task.get("max_record", -1)
        lambda_rate = task.get("lambda_rate", 1)
        seed = task.get("seed", 42)
        task_id = task.get("task_id", 0)
        shuffle = task.get("shuffle", True)  # mashqa 等同文档问题相邻的数据可设 false

        # 读 session（保持文件原序），再由该 task 独立 RNG 打乱（对齐 HeterCache rng.shuffle）
        sessions = load_sessions(data_path, max_record, seed=None, shuffle=False)
        assert_turn_policy(task, sessions, data_path)
        rng = random.Random(seed + task_id)
        if shuffle:
            rng.shuffle(sessions)

        # arrival.mode=fixed：确定性到达（victim 需同相位，Poisson 会把 pause 打散）
        arrival = task.get("arrival") or {}
        if arrival.get("mode") == "fixed":
            base = float(arrival.get("offset_s", 0.0))
            step = float(arrival.get("step_s", 0.05))
            arrival_times = [base + step * i for i in range(len(sessions))]
        else:
            # Poisson 到达时刻（seed+task_id 保证各 task 到达流相互独立且可复现）
            arrival_times = generate_poisson_requests(
                lambda_rate, len(sessions), seed + task_id
            )

        for sess, at in zip(sessions, arrival_times):
            events.append(SessionArrival(at, counter, name, task, sess))
            counter += 1

    events.sort(key=lambda e: (e.arrival_offset, e.counter))
    return events


# --------------------------------------------------------------------------
# 指标累计
# --------------------------------------------------------------------------
class Stats:
    """统计窗内的累计量；预热窗（``--warmup-s``）内的请求只进 CSV，不进这里。

    为什么要有预热窗：VD（已兑现价值密度）的读数是一个时间常数为 W_q 的指数
    低通，W_q = max(horizon, k*T50_q)。chat 的 T50≈118s、k=4 → W_q≈472s，所以
    前 472s 的 VD 被系统性低估 1-e^{-t/W_q} 倍（t=100s 时 5.2 倍，而 struct /
    agentic 的 W=120s 只低估 1.8 倍）。这个差分偏差正好落在「chat 与 struct 谁
    先削」这条边界上。预热不足时整段统计都跑在冷启动的瞬态里。
    """

    def __init__(self) -> None:
        self.num_requests = 0
        self.num_sessions = 0
        # 预热窗：截止时刻（time.time() 绝对时刻；None = 不设窗，全量计入）
        self.warmup_until: Optional[float] = None
        self.warmup_s = 0.0
        self.warmup_requests = 0
        self.warmup_sessions = 0
        self.warmup_prompt_tokens = 0
        # 完成的请求总数（含预热），只用于进度条与吞吐分母之外的观测
        self.done_requests = 0
        # 统计窗内第一个请求的完成时刻，用来算实测窗口长度
        self.measured_since: Optional[float] = None
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.ttfts: List[float] = []  # 首 token 时延（秒）
        self.e2es: List[float] = []  # 端到端时延（秒）
        self.errors = 0
        # 混合负载：按 task 细分统计（异构负载各自的命中率）
        self.per_task: Dict[str, Dict[str, int]] = {}
        # 逐请求明细（供 CSV 输出：TTFT vs 重算量逐点分析）
        self.rows: List[Dict[str, Any]] = []

    def in_warmup(self, now_wall: float) -> bool:
        """该时刻是否仍在预热窗内（按请求**完成**时刻判定）。"""
        return self.warmup_until is not None and now_wall < self.warmup_until

    def record_warmup(self, prompt_tokens: int) -> None:
        """预热窗内完成的请求：只记计数，用于报告预热期的规模。"""
        self.warmup_requests += 1
        self.warmup_prompt_tokens += prompt_tokens

    def record_request(
        self,
        task_name: str,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
    ) -> None:
        """累计一次请求的全局与 per-task 计数。"""
        self.num_requests += 1
        self.prompt_tokens += prompt_tokens
        self.cached_tokens += cached_tokens
        self.completion_tokens += completion_tokens
        t = self.per_task.setdefault(
            task_name,
            {
                "num_sessions": 0,
                "num_requests": 0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
            },
        )
        t["num_requests"] += 1
        t["prompt_tokens"] += prompt_tokens
        t["cached_tokens"] += cached_tokens

    def record_session(self, task_name: str) -> None:
        """累计一个 session 的全局与 per-task 计数。"""
        self.num_sessions += 1
        t = self.per_task.setdefault(
            task_name,
            {
                "num_sessions": 0,
                "num_requests": 0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
            },
        )
        t["num_sessions"] += 1

    def summary(self) -> Dict[str, Any]:
        def pct(xs: List[float], p: float) -> Optional[float]:
            if not xs:
                return None
            xs = sorted(xs)
            # 线性插值分位数
            k = (len(xs) - 1) * p / 100.0
            lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
            return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

        hit_rate = (
            self.cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0.0
        )

        # 混合负载：输出每个 task 的命中率细分
        per_task = {}
        for name, t in self.per_task.items():
            per_task[name] = {
                "num_sessions": t["num_sessions"],
                "num_requests": t["num_requests"],
                "prompt_tokens": t["prompt_tokens"],
                "cached_tokens": t["cached_tokens"],
                "cache_hit_rate_pct": round(
                    t["cached_tokens"] / t["prompt_tokens"] * 100, 2
                )
                if t["prompt_tokens"]
                else 0.0,
            }

        return {
            "warmup_s": self.warmup_s,
            "warmup_requests": self.warmup_requests,
            "warmup_sessions": self.warmup_sessions,
            "warmup_prompt_tokens": self.warmup_prompt_tokens,
            "done_requests": self.done_requests,
            "num_sessions": self.num_sessions,
            "num_requests": self.num_requests,
            "num_errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_rate": round(hit_rate, 4),
            "cache_hit_rate_pct": round(hit_rate * 100, 2),
            "ttft_p50_ms": round(pct(self.ttfts, 50) * 1000, 2) if self.ttfts else None,
            "ttft_p90_ms": round(pct(self.ttfts, 90) * 1000, 2) if self.ttfts else None,
            "ttft_p99_ms": round(pct(self.ttfts, 99) * 1000, 2) if self.ttfts else None,
            "e2e_mean_ms": round(statistics.mean(self.e2es) * 1000, 2)
            if self.e2es
            else None,
            "e2e_p99_ms": round(pct(self.e2es, 99) * 1000, 2) if self.e2es else None,
            "per_task": per_task,
        }


# --------------------------------------------------------------------------
# 单次请求（流式：可测 TTFT）
# --------------------------------------------------------------------------
async def one_request(
    http: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    """发一次 /generate（流式），返回 ttft/e2e/prompt_tokens/cached_tokens。"""
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    last_meta: Dict[str, Any] = {}

    body = dict(payload)
    body["stream"] = True  # 流式：第一个 chunk 到达即 TTFT

    async with http.post(
        url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            meta = obj.get("meta_info")
            if meta:
                last_meta = meta

    e2e = time.perf_counter() - t0
    # TTFT 分解：forward_entry_time → prefill_finished_time 是纯 prefill，
    # 其余为 tokenizer 入队与调度排队。用于把尾部归因到重算而非排队。
    fe = last_meta.get("forward_entry_time")
    pf = last_meta.get("prefill_finished_time")
    return {
        "ttft": ttft,
        "e2e": e2e,
        "prompt_tokens": last_meta.get("prompt_tokens", 0),
        "cached_tokens": last_meta.get("cached_tokens", 0),
        "completion_tokens": last_meta.get("completion_tokens", 0),
        "queue_time_ms": round((last_meta.get("queue_time") or 0.0) * 1000, 3),
        "prefill_ms": round((pf - fe) * 1000, 3) if fe and pf else None,
    }


# --------------------------------------------------------------------------
# 堆与待发轮次
# --------------------------------------------------------------------------
class PendingTurn(NamedTuple):
    """一个待发送的（会话，轮次）。"""

    task_name: str
    task_cfg: Dict[str, Any]
    session: Dict[str, Any]
    turn_idx: int


class ArrivalHeap:
    """(due_ts, seq) 排序的最小堆。

    seq 是全局自增计数器，用途与 HeterCache `future_request_heap` 的 counter 相同：
    时刻相同时按压入顺序弹出，保证同一 seed 下到达流的顺序可复现。
    """

    def __init__(self) -> None:
        self._items: List[Tuple[float, int, PendingTurn]] = []
        self._seq = itertools.count()

    def push(self, *, due_ts: float, turn: PendingTurn) -> None:
        heapq.heappush(self._items, (due_ts, next(self._seq), turn))

    @property
    def next_due_ts(self) -> Optional[float]:
        return self._items[0][0] if self._items else None

    def pop_due(self, *, now: float) -> List[Tuple[float, PendingTurn]]:
        """弹出所有 due_ts <= now 的轮次，按 (due_ts, seq) 升序。"""
        due: List[Tuple[float, PendingTurn]] = []
        while self._items and self._items[0][0] <= now:
            due_ts, _, turn = heapq.heappop(self._items)
            due.append((due_ts, turn))
        return due

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------
# 完整历史 prompt
# --------------------------------------------------------------------------
def _resolve_prompt_ids(
    *, session: Dict[str, Any], turn_idx: int
) -> Tuple[List[int], bool]:
    """返回本轮要发的完整历史 token ids，以及客户端拼接是否与数据集口径一致。

    数据集里 `turn_i.input_ids` 本身就是完整历史（`tools/prepare_session_data.py`
    用 chat template 渲染 conversations[:2i] 后 tokenize）。这里额外验证
    「上一轮 prompt + 上一轮 response_ids + 本轮新增」是否等于它：

      Qwen 模板把 response 直接接在 assistant 头之后，response_ids 是字面后缀 → 一致；
      Llama 模板在 response 前插 `assistant<|end_header_id|>`，后缀并不对齐 → 不一致，
      此时只能以数据集为准（否则拼出来的历史与数据集的 token 序列不同，前缀共享失效）。

    返回的 ids 恒等于数据集的 `turn_i.input_ids`；第二个返回值只用于统计，
    让「客户端拼历史」这件事在结果里可验证，而不是靠注释声称。
    """
    full = session[f"turn_{turn_idx}"]["input_ids"]
    if turn_idx == 1:
        return full, True
    prev = session[f"turn_{turn_idx - 1}"]
    base = prev["input_ids"] + prev["response_ids"]
    return full, full[: len(base)] == base


# --------------------------------------------------------------------------
# 发射器
# --------------------------------------------------------------------------
class HeapEmitter:
    """最小堆 → 真实墙钟 → 异步 HTTP；每个请求一个协程消费流式响应。"""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        stats: Stats,
        heap: ArrivalHeap,
        http: aiohttp.ClientSession,
        url: str,
    ) -> None:
        self._args = args
        self._stats = stats
        self._heap = heap
        self._http = http
        self._url = url

        self._running: set[asyncio.Task] = set()
        self._wake = asyncio.Event()
        self._inflight = 0
        self._send_lags: List[float] = []
        self._session_rngs: Dict[str, random.Random] = {}
        self._prev_end: Dict[str, float] = {}
        self._concat_ok = 0
        self._concat_fallback = 0

    # ---- 主循环 ----------------------------------------------------------
    async def run(self) -> None:
        """到点就发；堆空或无槽位时等「有请求结束」这一事件。"""
        while self._heap or self._inflight:
            if not self._can_dispatch():
                await self._wake.wait()
                self._wake.clear()
                continue

            due_ts = self._heap.next_due_ts
            wait_s = due_ts - time.monotonic()
            if wait_s > 0:
                await self._sleep_until(wait_s)
                continue

            self._dispatch(self._heap.pop_due(now=time.monotonic()))

        # 收尾：正常路径下为空；留作把「任务里逃到 _run_turn 之外的异常」显式报出来，
        # 否则 asyncio 只会静默丢一个未取的 Task 结果
        if self._running:
            for exc in await asyncio.gather(*self._running, return_exceptions=True):
                if isinstance(exc, Exception):
                    self._stats.errors += 1
                    print(f"[ERR] 发射协程异常：{type(exc).__name__}: {exc}")

    def _can_dispatch(self) -> bool:
        if not self._heap:
            return False
        cap = self._args.max_inflight
        return not cap or self._inflight < cap

    async def _sleep_until(self, wait_s: float) -> None:
        """睡到下一个到点事件；期间若有请求结束并压回更早的时刻，提前醒来。"""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=wait_s)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    def _dispatch(self, due: List[Tuple[float, PendingTurn]]) -> None:
        for due_ts, turn in due:
            self._inflight += 1
            task = asyncio.create_task(self._fire(due_ts=due_ts, turn=turn))
            # 持有引用：asyncio 只保留弱引用，任务被 GC 会静默丢请求
            self._running.add(task)
            task.add_done_callback(self._running.discard)

    async def _fire(self, *, due_ts: float, turn: PendingTurn) -> None:
        try:
            await self._run_turn(due_ts=due_ts, turn=turn)
        finally:
            self._inflight -= 1
            self._wake.set()

    # ---- 单轮 ------------------------------------------------------------
    async def _run_turn(self, *, due_ts: float, turn: PendingTurn) -> None:
        sess = turn.session
        sid = sess["id"]
        idx = turn.turn_idx
        row = sess[f"turn_{idx}"]
        prompt_ids, concat_ok = _resolve_prompt_ids(session=sess, turn_idx=idx)
        if concat_ok:
            self._concat_ok += 1
        else:
            self._concat_fallback += 1

        prev_end = self._prev_end.get(sid)
        # 请求真正开始跑的时刻：滞后在任务启动这一层，与 POST 握手无关
        send_ts = time.monotonic()
        send_lag_s = send_ts - due_ts
        self._send_lags.append(send_lag_s)
        payload = {
            "input_ids": prompt_ids,
            "max_new_tokens": len(row["response_ids"]),
            "session_id": sid,
            "rid": f"{sid}_turn_{idx}",
            # 与旧 driver 同口径：本轮只暴露上一轮已结束的工具调用
            "tool_call": sess[f"turn_{idx - 1}"].get("tool_call") if idx > 1 else None,
        }

        try:
            r = await one_request(self._http, self._url, payload, self._args.timeout_s)
        except Exception as exc:  # noqa: BLE001  单轮失败不终止会话，与旧 driver 一致
            self._stats.errors += 1
            print(f"[ERR] {sid} turn {idx}: {type(exc).__name__}: {exc}")
            r = None

        now = time.monotonic()
        if r is not None:
            self._record(
                turn=turn, r=r, send_ts=send_ts, lag_s=send_lag_s, prev_end=prev_end
            )
        self._prev_end[sid] = now

        if idx < sess["num_turns"]:
            self._push_next_turn(turn=turn, finished_ts=now)
        elif self._stats.in_warmup(time.time()):
            # 整条会话在预热窗内跑完：计入 warmup_sessions，不进统计窗
            self._stats.warmup_sessions += 1
        else:
            self._stats.record_session(turn.task_name)
        self._maybe_report_progress()

    def _record(
        self,
        *,
        turn: PendingTurn,
        r: Dict[str, Any],
        send_ts: float,
        lag_s: float,
        prev_end: Optional[float],
    ) -> None:
        prompt_tokens = r["prompt_tokens"]
        cached_tokens = r["cached_tokens"]
        # 预热窗按完成时刻切：一行明细的 ts 与这里的判定必须同源，否则 CSV 与
        # JSON 汇总会切出两个不同的子集。
        now_wall = time.time()
        self._stats.done_requests += 1
        warm = self._stats.in_warmup(now_wall)
        if warm:
            self._stats.record_warmup(prompt_tokens)
        else:
            if self._stats.measured_since is None:
                self._stats.measured_since = now_wall
            self._stats.record_request(
                turn.task_name, prompt_tokens, cached_tokens, r["completion_tokens"]
            )
            if r["ttft"] is not None:
                self._stats.ttfts.append(r["ttft"])
            self._stats.e2es.append(r["e2e"])
        self._stats.rows.append(
            {
                "warmup": warm,
                "task": turn.task_name,
                "phase": turn.task_cfg.get("phase"),
                "session_id": turn.session["id"],
                "turn": turn.turn_idx,
                "prior_tool_call": bool(
                    turn.session[f"turn_{turn.turn_idx - 1}"].get("tool_call")
                    if turn.turn_idx > 1
                    else None
                ),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "recompute_tokens": prompt_tokens - cached_tokens,
                "ttft_ms": round(r["ttft"] * 1000, 2)
                if r["ttft"] is not None
                else None,
                "e2e_ms": round(r["e2e"] * 1000, 2),
                "gap_since_prev_end_s": (
                    round(send_ts - prev_end, 3) if prev_end is not None else None
                ),
                # 驱动侧迟到：实际发出时刻 - 计划到达时刻。旧 driver 里这项被信号量吞掉
                "send_lag_ms": round(lag_s * 1000, 3),
                "queue_time_ms": r["queue_time_ms"],
                "prefill_ms": r["prefill_ms"],
                "ts": round(now_wall, 3),
            }
        )

    # ---- 下一轮 ----------------------------------------------------------
    def _push_next_turn(self, *, turn: PendingTurn, finished_ts: float) -> None:
        next_idx = turn.turn_idx + 1
        wait_s = self._inter_turn_wait(task_cfg=turn.task_cfg, session=turn.session)
        self._heap.push(
            due_ts=finished_ts + wait_s,
            turn=turn._replace(turn_idx=next_idx),
        )
        # 新事件可能早于原堆顶（例如 0.5s 空窗 vs 原本 100s 后的事件），叫醒主循环
        self._wake.set()

    def _inter_turn_wait(
        self, *, task_cfg: Dict[str, Any], session: Dict[str, Any]
    ) -> float:
        mu, sigma = pick_arrival_params(task_cfg, self._args)
        if mu == 0.0 and sigma == 0.0:
            return 0.0
        rng = self._session_rng(task_cfg=task_cfg, session=session)
        return rng.lognormvariate(mu, sigma) * self._args.wait_scale

    def _session_rng(
        self, *, task_cfg: Dict[str, Any], session: Dict[str, Any]
    ) -> random.Random:
        """每 session 一条独立的空窗序列（seed 由 sid + task 决定，跨轮复用）。"""
        sid = session["id"]
        rng = self._session_rngs.get(sid)
        if rng is None:
            base_seed = (
                task_cfg.get("seed", self._args.seed)
                + task_cfg.get("task_id", 0) * 1000
            )
            rng = random.Random(stable_seed_from_str(sid, base_seed=base_seed))
            self._session_rngs[sid] = rng
        return rng

    # ---- 报告 ------------------------------------------------------------
    def _maybe_report_progress(self) -> None:
        every = self._args.progress_every
        if not every or self._stats.done_requests % every:
            return
        done, total = (
            self._stats.done_requests,
            self._stats.done_requests + len(self._heap),
        )
        hit = (
            self._stats.cached_tokens / self._stats.prompt_tokens
            if self._stats.prompt_tokens
            else 0.0
        )
        print(
            f"[heap] req={done} inflight={self._inflight} heap={len(self._heap)} "
            f"hit={hit:.1%} lag_p50={_pct(self._send_lags, 50):.1f}ms "
            f"({done}/{total})"
        )

    def summary_extra(self) -> Dict[str, Any]:
        return {
            "emitter": "heap",
            "max_inflight": self._args.max_inflight,
            "send_lag": {
                "p50_ms": round(_pct(self._send_lags, 50) * 1000, 3),
                "p99_ms": round(_pct(self._send_lags, 99) * 1000, 3),
                "max_ms": round(max(self._send_lags) * 1000, 3)
                if self._send_lags
                else None,
            },
            "history_concat": {
                "ok": self._concat_ok,
                "fallback_to_dataset": self._concat_fallback,
            },
        }


def _pct(xs: List[float], p: float) -> float:
    """线性插值分位数（与 Stats.summary 同口径）。"""
    if not xs:
        return 0.0
    ordered = sorted(xs)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


# --------------------------------------------------------------------------
# 组装与入口
# --------------------------------------------------------------------------
def build_heap(*, tasks_config: Dict[str, Any], arrival_scale: float) -> ArrivalHeap:
    """首轮到达流一次性压进堆：各 task 独立 Poisson，压入顺序即 (arrival, seq)。"""
    schedule = build_hybrid_arrival_schedule(tasks_config, str(HETER_BENCH_ROOT))
    start = time.monotonic()
    heap = ArrivalHeap()
    for ev in schedule:
        heap.push(
            due_ts=start + ev.arrival_offset * arrival_scale,
            turn=PendingTurn(
                task_name=ev.task_name,
                task_cfg=ev.task_cfg,
                session=ev.session,
                turn_idx=1,
            ),
        )
    return heap


async def main_async(args: argparse.Namespace) -> Tuple[Dict[str, Any], Stats]:
    with open(args.tasks_config, encoding="utf-8") as f:
        tasks_config = yaml.safe_load(f)

    heap = build_heap(tasks_config=tasks_config, arrival_scale=args.arrival_scale)
    print(
        f"[heap] tasks={len(tasks_config['tasks'])} sessions={len(heap)} "
        f"wait_scale={args.wait_scale} arrival_scale={args.arrival_scale} "
        f"max_inflight={args.max_inflight or '∞'}"
    )

    stats = Stats()
    stats.warmup_s = max(0.0, float(args.warmup_s))
    if stats.warmup_s > 0:
        # 用与明细行同一个时钟（time.time）定截止时刻，避免 perf_counter 与墙钟
        # 混用后 CSV 与汇总切出不同子集。
        stats.warmup_until = time.time() + stats.warmup_s
        print(f"[heap] warmup={stats.warmup_s:.0f}s（该窗口内的请求只进 CSV）")
    url = args.base_url.rstrip("/") + "/generate"
    started = time.perf_counter()
    async with aiohttp.ClientSession(
        # limit=0：不设连接上限。默认的每主机 100 会把请求堵在连接池里，
        # 那正是旧 driver 信号量位置的替代品，会让 send_lag 失去意义。
        connector=aiohttp.TCPConnector(limit=0),
        read_bufsize=10 * 1024**2,
    ) as http:
        emitter = HeapEmitter(args=args, stats=stats, heap=heap, http=http, url=url)
        await emitter.run()
        extra = emitter.summary_extra()

    wall_s = time.perf_counter() - started
    summary = stats.summary()
    summary.update(extra)
    summary["wall_time_s"] = round(wall_s, 2)
    # 统计窗 = 总时长扣掉预热；吞吐与命中率都只在统计窗内可比。
    measured_s = max(0.0, wall_s - stats.warmup_s)
    summary["measured_wall_s"] = round(measured_s, 2)
    summary["request_throughput"] = (
        round(stats.num_requests / measured_s, 2) if measured_s > 0 else None
    )
    summary["warmup_ratio"] = (
        round(stats.warmup_s / wall_s, 3) if wall_s > 0 else None
    )
    if stats.warmup_s > 0 and stats.num_requests == 0:
        # 预热吃掉了整条 trace：汇总里没有任何样本，会被误读成 0% 命中率。
        print(
            f"[heap] 警告：预热 {stats.warmup_s:.0f}s 覆盖了整条 trace"
            f"（总时长 {wall_s:.0f}s，统计窗内 0 个请求）。"
            "请缩短 --warmup-s 或放大数据规模（max_record）。"
        )
    summary["mode"] = "hybrid"
    summary["tasks_config"] = args.tasks_config
    summary["base_url"] = args.base_url
    summary["arrival_scale"] = args.arrival_scale
    summary["wait_scale"] = args.wait_scale
    return summary, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heter-Bench：堆驱动的真实墙钟发射器（UniCache 语义）"
    )
    p.add_argument(
        "--tasks-config",
        required=True,
        help="混合负载 tasks_config.yaml（含 data_path/lambda_rate/mu/sigma/seed）",
    )
    p.add_argument(
        "--base-url", default="http://127.0.0.1:30000", help="sglang server 地址"
    )
    p.add_argument(
        "--wait-scale",
        type=float,
        default=1.0,
        help="轮间空窗缩放：1=按 lognormal 真实等待；0=立即发下一轮",
    )
    p.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="首轮到达时刻缩放：1=真实 Poisson 节奏；0=全部立即到达",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="同时在飞的请求上限；0=不限（排队交给服务端，同 UniCache 口径）",
    )
    p.add_argument(
        "--mu", type=float, default=None, help="轮间空窗 lognormal mean 全局覆盖"
    )
    p.add_argument(
        "--sigma", type=float, default=None, help="轮间空窗 lognormal sigma 全局覆盖"
    )
    p.add_argument("--seed", type=int, default=42, help="空窗 RNG 的兜底种子")
    p.add_argument("--timeout-s", type=float, default=1800.0, help="单请求超时（秒）")
    p.add_argument(
        "--warmup-s",
        type=float,
        default=0.0,
        help=(
            "预热窗（秒）：完成时刻早于 t0+该值的请求只进 CSV，不计入汇总。"
            "VD 是时间常数为 W_q=max(horizon,k*T50) 的低通，t<W_q 时被系统性低估；"
            "取 2*max(W_q) 才算收敛（缺省参数下 chat 的 W_q≈472s → 建议 1000）。"
            "0=不设窗（旧口径，冷启动偏差留在统计里）。"
        ),
    )
    p.add_argument(
        "--progress-every", type=int, default=200, help="每 N 个请求打一行进度；0=关闭"
    )
    p.add_argument(
        "--output", default=None, help="结果 JSON 输出路径（同时写同名 CSV）"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary, stats = asyncio.run(main_async(args))
    print("\n================ Heter-Bench 结果（heap emitter）================")
    for key, value in summary.items():
        print(f"  {key:24s}: {value}")
    print("================================================================")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[heap] 结果已写入 {args.output}")
        if stats.rows:
            csv_path = str(args.output).rsplit(".json", 1)[0] + ".csv"
            import csv as _csv

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=list(stats.rows[0].keys()))
                w.writeheader()
                w.writerows(stats.rows)
            print(f"[heap] 逐请求明细已写入 {csv_path} ({len(stats.rows)} rows)")


if __name__ == "__main__":
    main()
