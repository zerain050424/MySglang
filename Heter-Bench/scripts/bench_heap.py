#!/usr/bin/env python3
"""Heter-Bench：堆驱动的真实墙钟 HTTP 压测发射器。

全局最小堆按 (due_ts, seq, PendingTurn) 调度各会话轮次；主循环按真实墙钟
并发发射请求，轮间空窗不占用并发槽位，每轮流式响应结束后按 lognormal
采样下一轮到达时刻并压回堆中。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import heapq
import itertools
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import yaml

if TYPE_CHECKING:
    import aiohttp

# data_path 相对 Heter-Bench 根目录解析，避免受运行工作目录影响。
HETER_BENCH_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# 配置与数据模型
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EmitterConfig:
    """发射器运行参数。"""

    wait_scale: float = 1.0
    max_inflight: int = 0
    mu: Optional[float] = None
    sigma: Optional[float] = None
    seed: int = 42
    timeout_s: float = 1800.0
    progress_every: int = 200

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EmitterConfig":
        return cls(
            wait_scale=args.wait_scale,
            max_inflight=args.max_inflight,
            mu=args.mu,
            sigma=args.sigma,
            seed=args.seed,
            timeout_s=args.timeout_s,
            progress_every=args.progress_every,
        )

    def resolve_wait_params(self, task_cfg: Dict[str, Any]) -> Tuple[float, float]:
        """解析轮间等待的 lognormal 参数 (mu, sigma)，全局显式参数优先。"""
        if self.mu is not None and self.sigma is not None:
            return self.mu, self.sigma
        return task_cfg.get("mu", 0.0), task_cfg.get("sigma", 0.0)


@dataclass(frozen=True)
class RequestMetrics:
    """单次流式请求的观测指标。"""

    ttft: Optional[float]
    e2e: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    queue_time_ms: float
    prefill_ms: Optional[float]


RequestFn = Callable[[Any, str, Dict[str, Any], float], Awaitable[RequestMetrics]]


class SessionArrival(NamedTuple):
    """会话首轮到达事件。"""

    arrival_offset: float
    counter: int
    task_name: str
    task_cfg: Dict[str, Any]
    session: Dict[str, Any]


class PendingTurn(NamedTuple):
    """堆中待发送的（会话，轮次）。"""

    task_name: str
    task_cfg: Dict[str, Any]
    session: Dict[str, Any]
    turn_idx: int


# --------------------------------------------------------------------------
# 工具与数据加载
# --------------------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """线性插值分位数；空序列返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def load_sessions(
    path: str, num_sessions: int, seed: Optional[int] = None, shuffle: bool = True
) -> List[Dict[str, Any]]:
    """读取 session_data.jsonl（每行格式 {"<id>": {id, num_turns, turn_1..N}}）。

    shuffle=False 时保持文件原始顺序，并在读满 num_sessions 后提前终止，
    避免打散同文档邻近问题并跳过大文件尾部扫描。
    """
    sessions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(list(json.loads(line).values())[0])
            if not shuffle and num_sessions and num_sessions > 0 and len(sessions) >= num_sessions:
                break
    if shuffle:
        random.Random(seed).shuffle(sessions)
    if num_sessions and num_sessions > 0:
        sessions = sessions[:num_sessions]
    return sessions


def assert_turn_policy(
    task: Dict[str, Any], sessions: List[Dict[str, Any]], data_path: str
) -> None:
    """校验选中会话均满足配置声明的 min_turns，防止短会话被误归为 TRANS。"""
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
# 到达流与最小堆
# --------------------------------------------------------------------------
def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """将字符串映射为跨进程确定的 32-bit 随机种子（与 wtl/trace.py 对齐）。"""
    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


def generate_poisson_requests(
    lambda_rate: float, num_sessions: int, seed: int
) -> List[float]:
    """按 Exp(lambda_rate) 生成确定性泊松到达时刻序列（与 wtl/trace.py 对齐）。"""
    rng = random.Random(seed)
    arrival_times: List[float] = []
    acc = 0.0
    for _ in range(num_sessions):
        acc += rng.expovariate(lambda_rate)
        arrival_times.append(acc)
    return arrival_times


def _build_task_arrivals(task: Dict[str, Any], sessions: List[Dict[str, Any]]) -> List[float]:
    """为单个 task 生成会话首轮到达偏移（支持 fixed 等步长与泊松两种模式）。"""
    arrival = task.get("arrival") or {}
    if arrival.get("mode") == "fixed":
        base = float(arrival.get("offset_s", 0.0))
        step = float(arrival.get("step_s", 0.05))
        return [base + step * i for i in range(len(sessions))]
    lambda_rate = task.get("lambda_rate", 1)
    seed = task.get("seed", 42)
    task_id = task.get("task_id", 0)
    return generate_poisson_requests(lambda_rate, len(sessions), seed + task_id)


def build_hybrid_arrival_schedule(
    tasks_config: Dict[str, Any], base_dir: Optional[str] = None
) -> List[SessionArrival]:
    """将多个异构负载按各自到达率交错合并为按 (arrival_offset, counter) 升序的到达流。"""
    events: List[SessionArrival] = []
    counter = 0

    for task in tasks_config.get("tasks", []):
        name = task["name"]
        data_path = task["data_path"]
        if base_dir and not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)

        max_record = task.get("max_record", -1)
        seed = task.get("seed", 42)
        task_id = task.get("task_id", 0)
        shuffle = task.get("shuffle", True)

        sessions = load_sessions(data_path, max_record, seed=None, shuffle=False)
        assert_turn_policy(task, sessions, data_path)
        if shuffle:
            random.Random(seed + task_id).shuffle(sessions)

        arrival_times = _build_task_arrivals(task, sessions)
        for sess, at in zip(sessions, arrival_times):
            events.append(SessionArrival(at, counter, name, task, sess))
            counter += 1

    events.sort(key=lambda e: (e.arrival_offset, e.counter))
    return events


class ArrivalHeap:
    """按 (due_ts, seq) 排序的最小堆，保证同刻事件按压入序确定性弹出。"""

    def __init__(self) -> None:
        self._items: List[Tuple[float, int, PendingTurn]] = []
        self._seq = itertools.count()

    def push(self, *, due_ts: float, turn: PendingTurn) -> None:
        heapq.heappush(self._items, (due_ts, next(self._seq), turn))

    @property
    def next_due_ts(self) -> Optional[float]:
        return self._items[0][0] if self._items else None

    def pop_due(
        self, *, now: float, limit: Optional[int] = None
    ) -> List[Tuple[float, PendingTurn]]:
        """弹出到期轮次（due_ts <= now），受 limit 上限约束，按 (due_ts, seq) 升序返回。"""
        due: List[Tuple[float, PendingTurn]] = []
        while self._items and self._items[0][0] <= now:
            if limit is not None and len(due) >= limit:
                break
            due_ts, _, turn = heapq.heappop(self._items)
            due.append((due_ts, turn))
        return due

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)


def build_heap(
    *,
    tasks_config: Dict[str, Any],
    arrival_scale: float,
    start_ts: Optional[float] = None,
) -> ArrivalHeap:
    """将混合到达流的首轮事件按到达时刻压入最小堆。"""
    schedule = build_hybrid_arrival_schedule(tasks_config, str(HETER_BENCH_ROOT))
    start = time.monotonic() if start_ts is None else start_ts
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


# --------------------------------------------------------------------------
# 统计与输出
# --------------------------------------------------------------------------
@dataclass
class TaskMetrics:
    """单个 task 的统计累计。"""

    num_sessions: int = 0
    num_requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        hit_pct = (
            round(self.cached_tokens / self.prompt_tokens * 100, 2)
            if self.prompt_tokens
            else 0.0
        )
        return {
            "num_sessions": self.num_sessions,
            "num_requests": self.num_requests,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate_pct": hit_pct,
        }


class Stats:
    """压测指标统计器：按完成时刻切分预热窗，分别累计汇总量与逐请求明细。"""

    def __init__(
        self, warmup_s: float = 0.0, warmup_until: Optional[float] = None
    ) -> None:
        self.num_requests = 0
        self.num_sessions = 0
        self.warmup_until: Optional[float] = warmup_until
        self.warmup_s = float(warmup_s)
        self.warmup_requests = 0
        self.warmup_sessions = 0
        self.warmup_prompt_tokens = 0
        self.done_requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.ttfts: List[float] = []
        self.e2es: List[float] = []
        self.errors = 0
        self.per_task: Dict[str, TaskMetrics] = {}
        self.rows: List[Dict[str, Any]] = []

    def configure_warmup(self, warmup_s: float, *, now_wall: float) -> None:
        """设置预热窗截止墙钟时刻；CSV 行时间戳与判定必须同源。"""
        self.warmup_s = max(0.0, float(warmup_s))
        self.warmup_until = now_wall + self.warmup_s if self.warmup_s > 0 else None

    def in_warmup(self, now_wall: float) -> bool:
        """判断给定墙钟完成时刻是否仍在预热窗内。"""
        return self.warmup_until is not None and now_wall < self.warmup_until

    def _task_bucket(self, task_name: str) -> TaskMetrics:
        bucket = self.per_task.get(task_name)
        if bucket is None:
            bucket = self.per_task[task_name] = TaskMetrics()
        return bucket

    def record_warmup(self, prompt_tokens: int) -> None:
        """预热期内完成的请求仅计入预热规模。"""
        self.warmup_requests += 1
        self.warmup_prompt_tokens += prompt_tokens

    def record_request(
        self,
        task_name: str,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
    ) -> None:
        """累计统计窗内的全局与分 task 请求量。"""
        self.num_requests += 1
        self.prompt_tokens += prompt_tokens
        self.cached_tokens += cached_tokens
        self.completion_tokens += completion_tokens
        bucket = self._task_bucket(task_name)
        bucket.num_requests += 1
        bucket.prompt_tokens += prompt_tokens
        bucket.cached_tokens += cached_tokens

    def record_session(self, task_name: str) -> None:
        """累计统计窗内完成的会话数。"""
        self.num_sessions += 1
        self._task_bucket(task_name).num_sessions += 1

    def record_error(self) -> None:
        self.errors += 1

    def record_turn(
        self,
        *,
        task_name: str,
        result: RequestMetrics,
        row: Dict[str, Any],
        now_wall: float,
    ) -> None:
        """记录单轮请求结果：按完成时刻切分预热窗并追加逐请求明细。"""
        self.done_requests += 1
        warm = self.in_warmup(now_wall)
        if warm:
            self.record_warmup(result.prompt_tokens)
        else:
            self.record_request(
                task_name,
                result.prompt_tokens,
                result.cached_tokens,
                result.completion_tokens,
            )
            if result.ttft is not None:
                self.ttfts.append(result.ttft)
            self.e2es.append(result.e2e)
        # 保持 CSV 16 列顺序：warmup 在首、ts 在末
        full_row = {"warmup": warm, **row, "ts": round(now_wall, 3)}
        self.rows.append(full_row)

    def record_session_finished(self, task_name: str, *, now_wall: float) -> None:
        """会话末轮结束时归属到预热会话或统计窗会话。"""
        if self.in_warmup(now_wall):
            self.warmup_sessions += 1
        else:
            self.record_session(task_name)

    def summary(self) -> Dict[str, Any]:
        hit_rate = (
            self.cached_tokens / self.prompt_tokens if self.prompt_tokens > 0 else 0.0
        )
        ttft_p50 = percentile(self.ttfts, 50)
        ttft_p90 = percentile(self.ttfts, 90)
        ttft_p99 = percentile(self.ttfts, 99)
        e2e_p99 = percentile(self.e2es, 99)

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
            "ttft_p50_ms": round(ttft_p50 * 1000, 2) if ttft_p50 is not None else None,
            "ttft_p90_ms": round(ttft_p90 * 1000, 2) if ttft_p90 is not None else None,
            "ttft_p99_ms": round(ttft_p99 * 1000, 2) if ttft_p99 is not None else None,
            "e2e_mean_ms": round(statistics.mean(self.e2es) * 1000, 2)
            if self.e2es
            else None,
            "e2e_p99_ms": round(e2e_p99 * 1000, 2) if e2e_p99 is not None else None,
            "per_task": {name: t.to_dict() for name, t in self.per_task.items()},
        }


def write_outputs(
    output_path: str, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]
) -> None:
    """将汇总 JSON 与逐请求 CSV 写入磁盘。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[heap] 结果已写入 {output_path}")
    if rows:
        csv_path = str(output_path).rsplit(".json", 1)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[heap] 逐请求明细已写入 {csv_path} ({len(rows)} rows)")


# --------------------------------------------------------------------------
# 流式 HTTP 客户端
# --------------------------------------------------------------------------
def _parse_sse_line(raw: bytes) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """解析单行 SSE 响应；返回 (是否 [DONE], JSON 对象或 None)。"""
    line = raw.strip()
    if not line or not line.startswith(b"data:"):
        return False, None
    data = line[len(b"data:") :].strip()
    if data == b"[DONE]":
        return True, None
    try:
        return False, json.loads(data)
    except json.JSONDecodeError:
        return False, None


async def one_request(
    http: "aiohttp.ClientSession",
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> RequestMetrics:
    """发送流式 /generate 请求并返回 TTFT、E2E 与服务端 meta_info 指标。"""
    import aiohttp

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    last_meta: Dict[str, Any] = {}

    body = dict(payload)
    body["stream"] = True

    async with http.post(
        url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            done, obj = _parse_sse_line(raw)
            if done:
                break
            if obj is None:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            meta = obj.get("meta_info")
            if meta:
                last_meta = meta

    e2e = time.perf_counter() - t0
    fe = last_meta.get("forward_entry_time")
    pf = last_meta.get("prefill_finished_time")
    return RequestMetrics(
        ttft=ttft,
        e2e=e2e,
        prompt_tokens=last_meta.get("prompt_tokens", 0),
        cached_tokens=last_meta.get("cached_tokens", 0),
        completion_tokens=last_meta.get("completion_tokens", 0),
        queue_time_ms=round((last_meta.get("queue_time") or 0.0) * 1000, 3),
        prefill_ms=round((pf - fe) * 1000, 3) if fe and pf else None,
    )


# --------------------------------------------------------------------------
# 异步发射器
# --------------------------------------------------------------------------
def _resolve_prompt_ids(
    *, session: Dict[str, Any], turn_idx: int
) -> Tuple[List[int], bool]:
    """返回本轮完整历史 token ids，以及「上轮输入+输出」是否为其字面前缀。

    由于不同模型 chat template 可能在回复前插入角色头，恒以数据集 `turn_i.input_ids`
    为准，避免前缀共享失配；第二个返回值用于核验拼接一致性。
    """
    full = session[f"turn_{turn_idx}"]["input_ids"]
    if turn_idx == 1:
        return full, True
    prev = session[f"turn_{turn_idx - 1}"]
    base = prev["input_ids"] + prev["response_ids"]
    return full, full[: len(base)] == base


class HeapEmitter:
    """事件驱动发射器：按堆序到点并发发射，在响应完成后调度下一轮。"""

    def __init__(
        self,
        *,
        config: EmitterConfig,
        stats: Stats,
        heap: ArrivalHeap,
        http: Any,
        url: str,
        request_fn: Optional[RequestFn] = None,
        mono_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._stats = stats
        self._heap = heap
        self._http = http
        self._url = url
        self._request_fn: RequestFn = request_fn or one_request
        self._mono = mono_clock
        self._wall = wall_clock

        self._running: set[asyncio.Task[None]] = set()
        self._wake = asyncio.Event()
        self._inflight = 0
        self._send_lags: List[float] = []
        self._session_rngs: Dict[str, random.Random] = {}
        self._prev_end: Dict[str, float] = {}
        self._concat_ok = 0
        self._concat_fallback = 0

    async def run(self) -> None:
        """主调度循环：无可用槽位或堆空时等待事件唤醒，到点则批量并发发射。"""
        while self._heap or self._inflight:
            if not self._can_dispatch():
                await self._wake.wait()
                self._wake.clear()
                continue

            due_ts = self._heap.next_due_ts
            wait_s = due_ts - self._mono()
            if wait_s > 0:
                await self._sleep_until(wait_s)
                continue

            self._dispatch(
                self._heap.pop_due(now=self._mono(), limit=self._available_slots())
            )

        if self._running:
            for exc in await asyncio.gather(*self._running, return_exceptions=True):
                if isinstance(exc, Exception):
                    self._stats.record_error()
                    print(f"[ERR] 发射协程异常：{type(exc).__name__}: {exc}")

    def _available_slots(self) -> Optional[int]:
        cap = self._config.max_inflight
        return None if not cap else max(0, cap - self._inflight)

    def _can_dispatch(self) -> bool:
        if not self._heap:
            return False
        slots = self._available_slots()
        return slots is None or slots > 0

    async def _sleep_until(self, wait_s: float) -> None:
        """等待到下一个到点时刻；期间若有更早事件入堆可由 _wake 提前唤醒。"""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=wait_s)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    def _dispatch(self, due: List[Tuple[float, PendingTurn]]) -> None:
        for due_ts, turn in due:
            self._inflight += 1
            task = asyncio.create_task(self._fire(due_ts=due_ts, turn=turn))
            # 持有强引用防止任务被 GC，同时在结束时及时释放
            self._running.add(task)
            task.add_done_callback(self._running.discard)

    async def _fire(self, *, due_ts: float, turn: PendingTurn) -> None:
        try:
            await self._run_turn(due_ts=due_ts, turn=turn)
        finally:
            self._inflight -= 1
            self._wake.set()

    def _build_payload(
        self, turn: PendingTurn, prompt_ids: List[int]
    ) -> Dict[str, Any]:
        sess = turn.session
        sid = sess["id"]
        idx = turn.turn_idx
        row = sess[f"turn_{idx}"]
        # 仅透传上一轮已结束的 tool_call，避免未来信息泄漏
        prior_tool_call = sess[f"turn_{idx - 1}"].get("tool_call") if idx > 1 else None
        return {
            "input_ids": prompt_ids,
            "max_new_tokens": len(row["response_ids"]),
            "session_id": sid,
            "rid": f"{sid}_turn_{idx}",
            "tool_call": prior_tool_call,
        }

    async def _execute_turn(
        self, turn: PendingTurn, payload: Dict[str, Any]
    ) -> Optional[RequestMetrics]:
        try:
            return await self._request_fn(
                self._http, self._url, payload, self._config.timeout_s
            )
        except Exception as exc:  # noqa: BLE001  单轮失败不终止会话后续轮次
            self._stats.record_error()
            print(
                f"[ERR] {turn.session['id']} turn {turn.turn_idx}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def _build_row_fields(
        self,
        *,
        turn: PendingTurn,
        result: RequestMetrics,
        lag_s: float,
        gap_s: Optional[float],
    ) -> Dict[str, Any]:
        idx = turn.turn_idx
        prior_tool_call = bool(
            turn.session[f"turn_{idx - 1}"].get("tool_call") if idx > 1 else None
        )
        return {
            "task": turn.task_name,
            "phase": turn.task_cfg.get("phase"),
            "session_id": turn.session["id"],
            "turn": idx,
            "prior_tool_call": prior_tool_call,
            "prompt_tokens": result.prompt_tokens,
            "cached_tokens": result.cached_tokens,
            "recompute_tokens": result.prompt_tokens - result.cached_tokens,
            "ttft_ms": round(result.ttft * 1000, 2)
            if result.ttft is not None
            else None,
            "e2e_ms": round(result.e2e * 1000, 2),
            "gap_since_prev_end_s": round(gap_s, 3) if gap_s is not None else None,
            "send_lag_ms": round(lag_s * 1000, 3),
            "queue_time_ms": result.queue_time_ms,
            "prefill_ms": result.prefill_ms,
        }

    async def _run_turn(self, *, due_ts: float, turn: PendingTurn) -> None:
        sess = turn.session
        sid = sess["id"]
        idx = turn.turn_idx
        prompt_ids, concat_ok = _resolve_prompt_ids(session=sess, turn_idx=idx)
        if concat_ok:
            self._concat_ok += 1
        else:
            self._concat_fallback += 1

        prev_end = self._prev_end.get(sid)
        send_ts = self._mono()
        lag_s = send_ts - due_ts
        self._send_lags.append(lag_s)

        payload = self._build_payload(turn, prompt_ids)
        result = await self._execute_turn(turn, payload)

        now = self._mono()
        if result is not None:
            gap_s = send_ts - prev_end if prev_end is not None else None
            row = self._build_row_fields(
                turn=turn, result=result, lag_s=lag_s, gap_s=gap_s
            )
            self._stats.record_turn(
                task_name=turn.task_name,
                result=result,
                row=row,
                now_wall=self._wall(),
            )
        self._prev_end[sid] = now

        if idx < sess["num_turns"]:
            self._push_next_turn(turn=turn, finished_ts=now)
        else:
            self._stats.record_session_finished(turn.task_name, now_wall=self._wall())
        self._maybe_report_progress()

    def _push_next_turn(self, *, turn: PendingTurn, finished_ts: float) -> None:
        wait_s = self._inter_turn_wait(task_cfg=turn.task_cfg, session=turn.session)
        self._heap.push(
            due_ts=finished_ts + wait_s,
            turn=turn._replace(turn_idx=turn.turn_idx + 1),
        )
        self._wake.set()

    def _inter_turn_wait(
        self, *, task_cfg: Dict[str, Any], session: Dict[str, Any]
    ) -> float:
        mu, sigma = self._config.resolve_wait_params(task_cfg)
        if mu == 0.0 and sigma == 0.0:
            return 0.0
        rng = self._session_rng(task_cfg=task_cfg, session=session)
        return rng.lognormvariate(mu, sigma) * self._config.wait_scale

    def _session_rng(
        self, *, task_cfg: Dict[str, Any], session: Dict[str, Any]
    ) -> random.Random:
        """为每个 session 维护独立随机流，跨多轮稳定复用。"""
        sid = session["id"]
        rng = self._session_rngs.get(sid)
        if rng is None:
            base_seed = (
                task_cfg.get("seed", self._config.seed)
                + task_cfg.get("task_id", 0) * 1000
            )
            rng = random.Random(stable_seed_from_str(sid, base_seed=base_seed))
            self._session_rngs[sid] = rng
        return rng

    def _maybe_report_progress(self) -> None:
        every = self._config.progress_every
        if not every or self._stats.done_requests % every:
            return
        done = self._stats.done_requests
        total = done + len(self._heap)
        hit = (
            self._stats.cached_tokens / self._stats.prompt_tokens
            if self._stats.prompt_tokens
            else 0.0
        )
        lag_p50_ms = (percentile(self._send_lags, 50) or 0.0) * 1000
        print(
            f"[heap] req={done} inflight={self._inflight} heap={len(self._heap)} "
            f"hit={hit:.1%} lag_p50={lag_p50_ms:.1f}ms ({done}/{total})"
        )

    def summary_extra(self) -> Dict[str, Any]:
        lag_p50 = percentile(self._send_lags, 50) or 0.0
        lag_p99 = percentile(self._send_lags, 99) or 0.0
        return {
            "emitter": "heap",
            "max_inflight": self._config.max_inflight,
            "send_lag": {
                "p50_ms": round(lag_p50 * 1000, 3),
                "p99_ms": round(lag_p99 * 1000, 3),
                "max_ms": round(max(self._send_lags) * 1000, 3)
                if self._send_lags
                else None,
            },
            "history_concat": {
                "ok": self._concat_ok,
                "fallback_to_dataset": self._concat_fallback,
            },
        }


# --------------------------------------------------------------------------
# CLI 与组装
# --------------------------------------------------------------------------
def _finalize_summary(
    summary: Dict[str, Any],
    extra: Dict[str, Any],
    stats: Stats,
    args: argparse.Namespace,
    wall_s: float,
) -> Dict[str, Any]:
    summary.update(extra)
    measured_s = max(0.0, wall_s - stats.warmup_s)
    summary["wall_time_s"] = round(wall_s, 2)
    summary["measured_wall_s"] = round(measured_s, 2)
    summary["request_throughput"] = (
        round(stats.num_requests / measured_s, 2) if measured_s > 0 else None
    )
    summary["warmup_ratio"] = (
        round(stats.warmup_s / wall_s, 3) if wall_s > 0 else None
    )
    if stats.warmup_s > 0 and stats.num_requests == 0:
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
    return summary


async def main_async(args: argparse.Namespace) -> Tuple[Dict[str, Any], Stats]:
    import aiohttp

    with open(args.tasks_config, encoding="utf-8") as f:
        tasks_config = yaml.safe_load(f)

    heap = build_heap(tasks_config=tasks_config, arrival_scale=args.arrival_scale)
    print(
        f"[heap] tasks={len(tasks_config['tasks'])} sessions={len(heap)} "
        f"wait_scale={args.wait_scale} arrival_scale={args.arrival_scale} "
        f"max_inflight={args.max_inflight or '∞'}"
    )

    stats = Stats()
    if args.warmup_s > 0:
        stats.configure_warmup(args.warmup_s, now_wall=time.time())
        print(f"[heap] warmup={stats.warmup_s:.0f}s（该窗口内的请求只进 CSV）")

    config = EmitterConfig.from_args(args)
    url = args.base_url.rstrip("/") + "/generate"
    started = time.perf_counter()
    # 不设客户端连接池上限（limit=0），避免排队在连接池内发生并掩盖真实并发压力
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        read_bufsize=10 * 1024**2,
    ) as http:
        emitter = HeapEmitter(
            config=config, stats=stats, heap=heap, http=http, url=url
        )
        await emitter.run()
        extra = emitter.summary_extra()

    wall_s = time.perf_counter() - started
    summary = _finalize_summary(stats.summary(), extra, stats, args, wall_s)
    return summary, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heter-Bench：堆驱动的真实墙钟 HTTP 压测发射器"
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
        help="同时在飞的请求上限；0=不限（排队交给服务端）",
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
        help="预热窗（秒）：完成时刻早于 t0+该值的请求只进 CSV，不计入汇总",
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
        write_outputs(args.output, summary, stats.rows)


if __name__ == "__main__":
    main()
