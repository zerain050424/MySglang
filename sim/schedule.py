"""把 ``bench_heap.py`` 的发射规则搬到虚拟时间（R2）。

emitter 的时间轴是闭环的：**本轮响应流读完的时刻**才决定下一轮的点到时刻::

    now = time.monotonic()          # 本轮结束
    due_ts = now + lognormal(mu, sigma) * wait_scale

虚拟时钟下没有「读流」，所以必须显式给一个**服务时长模型**，否则
``finished_ts == 开始时刻``，闭环退化成 0 延迟，仍然不等于真实负载。

对齐的三件事：

1. **首轮到达**：每 task 独立 Poisson（``expovariate(λ)`` 累加）或
   ``arrival.mode=fixed``（``offset + step*i``）；``arrival_scale`` 整体缩放。
2. **轮间空窗**：``lognormal(mu, sigma) * wait_scale``，每 session 一条独立 RNG，
   seed = ``stable_seed_from_str(sid, seed + task_id*1000)``（与 emitter 同式）。
3. **服务时长**：``prefill_base + prompt/prefill_rate + out/decode_rate``，
   ``prefill_*`` 可直接读 ``tools/calibrate.py`` 的产物
   （``prefill_rate_tok_per_s`` / ``ttft_intercept_ms``）。

并发窗口不在这里（R2 是单通道顺序队列）：``start = max(due, now)``，
同一时刻到点的请求会排队，等价 ``max_inflight=1``。R3 再放开。
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sim.metrics import SimResult
from sim.trace import Session, Turn, max_total_len


# ---------------------------------------------------------------------------
# 与 emitter 逐字相同的种子函数
# ---------------------------------------------------------------------------
def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """字符串 -> 稳定 32-bit seed（``bench_heap.stable_seed_from_str`` 同式）。"""

    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


# ---------------------------------------------------------------------------
# 每个 task 的发射参数
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """tasks_config 里一个 task 的发射相关字段（其余字段在加载期用掉）。"""

    name: str
    lambda_rate: float = 1.0
    mu: float = 0.0
    sigma: float = 0.0
    seed: int = 42
    task_id: int = 0
    arrival_mode: str = "poisson"  # poisson | fixed
    arrival_offset_s: float = 0.0
    arrival_step_s: float = 0.05

    @classmethod
    def from_config(cls, task: Mapping) -> "TaskSpec":
        arrival = task.get("arrival") or {}
        return cls(
            name=task["name"],
            lambda_rate=float(task.get("lambda_rate", 1)),
            mu=float(task.get("mu", 0.0)),
            sigma=float(task.get("sigma", 0.0)),
            seed=int(task.get("seed", 42)),
            task_id=int(task.get("task_id", 0)),
            arrival_mode=str(arrival.get("mode", "poisson")),
            arrival_offset_s=float(arrival.get("offset_s", 0.0)),
            arrival_step_s=float(arrival.get("step_s", 0.05)),
        )

    @property
    def rng_seed(self) -> int:
        """空窗 RNG 的基数：``seed + task_id*1000``（emitter 口径）。"""
        return self.seed + self.task_id * 1000


# ---------------------------------------------------------------------------
# 服务时长模型
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServiceModel:
    """服务时长（秒）= 固定开销 + prompt/prefill 速率 + out/decode 速率。

    ``prefill`` 部分可直接用 ``tools/calibrate.py`` 标定：
    ``prefill_base_s = ttft_intercept_ms / 1000``、
    ``prefill_tokens_per_s = prefill_rate_tok_per_s``。

    decode 速率标定脚本没有覆盖（它只测 TTFT），默认按 40 tok/s 给一个可调的锚点：
    它决定「响应什么时候读完」，而读完时刻正是下一轮的点到时刻。
    """

    prefill_base_s: float = 0.05
    prefill_tokens_per_s: float = 2000.0
    decode_tokens_per_s: float = 40.0

    def __post_init__(self) -> None:
        if self.prefill_tokens_per_s <= 0 or self.decode_tokens_per_s <= 0:
            raise ValueError("token 速率必须为正")

    def __call__(self, prompt_tokens: int, out_tokens: int = 0) -> float:
        return (
            self.prefill_base_s
            + max(0, prompt_tokens) / self.prefill_tokens_per_s
            + max(0, out_tokens) / self.decode_tokens_per_s
        )

    def prefill(self, tokens: int) -> float:
        """One prefill forward over a batch of ``tokens`` extend tokens."""

        return self.prefill_base_s + max(0, tokens) / self.prefill_tokens_per_s

    def decode_step(self) -> float:
        """One decode step; every running request emits one token in parallel.

        UniCache models TPOT as a constant, so the batch size does not scale the
        step time. Per-request latency still differs because requests finish on
        different steps.
        """

        return 1.0 / self.decode_tokens_per_s

    @classmethod
    def from_calibration(
        cls, path: str, *, decode_tokens_per_s: Optional[float] = None
    ) -> "ServiceModel":
        """读 ``calibrate.py`` 的 JSON 产物。"""

        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        rate = data.get("prefill_rate_tok_per_s")
        if not rate:
            raise ValueError(f"{path} 里没有 prefill_rate_tok_per_s，标定未完成？")
        return cls(
            prefill_base_s=float(data.get("ttft_intercept_ms", 0.0)) / 1000.0,
            prefill_tokens_per_s=float(rate),
            decode_tokens_per_s=(
                decode_tokens_per_s
                if decode_tokens_per_s is not None
                else cls.decode_tokens_per_s
            ),
        )


# ---------------------------------------------------------------------------
# 发射器
# ---------------------------------------------------------------------------
class Emitter:
    """按 emitter 的规则驱动 Simulator（虚拟时间，单通道）。

    用法::

        sessions = load_tasks(tasks_config["tasks"], base_dir=...)
        specs = [TaskSpec.from_config(t) for t in tasks_config["tasks"]]
        result = Emitter(sim, sessions, specs).run(arrival_scale=1.0, wait_scale=1.0)
    """

    def __init__(
        self,
        sim,  # Simulator（避免循环 import，鸭子类型）
        sessions: Sequence[Session],
        specs: Sequence[TaskSpec],
        service: Optional[ServiceModel] = None,
    ) -> None:
        self.sim = sim
        self.sessions = list(sessions)
        self.specs: Dict[str, TaskSpec] = {spec.name: spec for spec in specs}
        self.service = service or ServiceModel()
        self._by_sid: Dict[str, Session] = {}
        for session in self.sessions:
            if session.session_id in self._by_sid:
                raise ValueError(f"session_id 重复：{session.session_id}")
            if session.task_name not in self.specs:
                raise ValueError(
                    f"会话 {session.session_id} 的 task_name={session.task_name!r} "
                    f"没有对应 TaskSpec（有：{sorted(self.specs)}）"
                )
            self._by_sid[session.session_id] = session
        self._session_rngs: Dict[str, random.Random] = {}
        self.send_lags: List[float] = []

    # ---- 到达流 ----
    def _arrival_offsets(self, spec: TaskSpec, n: int) -> List[float]:
        if spec.arrival_mode == "fixed":
            return [
                spec.arrival_offset_s + spec.arrival_step_s * i for i in range(n)
            ]
        if spec.arrival_mode != "poisson":
            raise ValueError(f"未知 arrival 模式：{spec.arrival_mode!r}")
        rng = random.Random(spec.rng_seed)  # emitter: generate_poisson_requests(seed+task_id)
        out, acc = [], 0.0
        for _ in range(n):
            acc += rng.expovariate(spec.lambda_rate)
            out.append(acc)
        return out

    def _inter_turn_wait(self, spec: TaskSpec, sid: str, wait_scale: float) -> float:
        if spec.mu == 0.0 and spec.sigma == 0.0:
            return 0.0
        rng = self._session_rngs.get(sid)
        if rng is None:
            rng = random.Random(
                stable_seed_from_str(sid, base_seed=spec.rng_seed)
            )
            self._session_rngs[sid] = rng
        return rng.lognormvariate(spec.mu, spec.sigma) * wait_scale

    # ---- 主循环 ----
    def run(
        self,
        *,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
        max_inflight: int = 0,
    ) -> SimResult:
        """按 emitter 的规则跑完整条到达流。

        Args:
            arrival_scale: 首轮到达时刻整体缩放（1=真实 Poisson 节奏，0=全部立即）。
            wait_scale: 轮间空窗缩放（0=立即发下一轮）。
            max_inflight: 同时在飞上限；0 = 不限（emitter 的默认口径）。
                1 会退化成单通道顺序队列。

        并发语义：同一虚拟时刻到点的请求按 ``(due_ts, seq)`` 依次 ``begin``——
        各自 match / 挂锁 / 占槽，但**都不写树**；写树发生在各自完成时刻。
        因此同刻到点、前缀相同的两条请求会都 miss 然后各自插入（racing），
        与生产一致；顺序模型（``max_inflight=1``）下后到的那条会命中前一条，
        这是两种模型最容易观察到的分歧。
        """

        sim = self.sim
        if not self.sessions:
            return SimResult(records=[], summary={})
        if max_inflight < 0:
            raise ValueError(f"max_inflight 必须 >= 0，got {max_inflight}")

        # 首轮全部入堆：(due_ts, seq, sid, turn_idx)；seq 保证同刻按入堆顺序
        heap: List[Tuple[float, int, str, int]] = []
        seq = itertools.count()
        for spec in self.specs.values():
            group = [s for s in self.sessions if s.task_name == spec.name]
            for session, offset in zip(group, self._arrival_offsets(spec, len(group))):
                heapq.heappush(
                    heap, (offset * arrival_scale, next(seq), session.session_id, 1)
                )

        width = max_total_len(self.sessions)
        sim.prepare(width)
        sim.tasks = {s.session_id: s.task_name for s in self.sessions}

        # 在飞：按 (完成时刻, 完成序) 出堆，保证同刻完成的事件顺序稳定
        inflight: List[Tuple[float, int, object, str, int]] = []
        done_seq = itertools.count()
        eps = 1e-9
        now = 0.0
        inflight_peak = 0

        while heap or inflight:
            can_dispatch = max_inflight <= 0 or len(inflight) < max_inflight

            # 1) 先完成（同刻先腾地方/放锁，再发新的）
            if inflight and inflight[0][0] <= now + eps:
                finish, _dseq, pending, sid, idx = heapq.heappop(inflight)
                sim.clock.advance_to(finish)
                sim.end(pending)
                now = finish
                self._push_next_turn(heap, seq, self._by_sid[sid], idx, finish, wait_scale)
                continue

            # 2) 到点且名额够 -> 派发（begin：match + 挂锁 + 占槽，不写树）
            if heap and can_dispatch and heap[0][0] <= now + eps:
                due, _s, sid, idx = heapq.heappop(heap)
                self.send_lags.append(now - due)
                session = self._by_sid[sid]
                turn = replace(session.turns[idx - 1], t=now)
                pending = sim.begin(
                    turn,
                    sid,
                    task_name=session.task_name,
                    turn_idx=idx - 1,  # emitter 的 idx 是 1 基
                )
                finish = now + self.service(pending.prompt_len, pending.out_len)
                heapq.heappush(
                    inflight, (finish, next(done_seq), pending, sid, idx)
                )
                inflight_peak = max(inflight_peak, len(inflight))
                continue

            # 3) 推进时钟到下一个事件；名额被占时只看到点完成
            next_due = heap[0][0] if (heap and can_dispatch) else math.inf
            next_finish = inflight[0][0] if inflight else math.inf
            target = min(next_due, next_finish)
            if target == math.inf:
                raise RuntimeError("调度死锁：既不能派发也没有在飞请求")
            sim.clock.advance_to(max(target, now))
            now = sim.clock()

        result = sim.result()
        summary = result.summary or {}
        lags = sorted(self.send_lags)
        summary["emit"] = {
            "mode": "closed_loop_concurrent"
            if max_inflight != 1
            else "closed_loop_sequential",
            "max_inflight": max_inflight,
            "wall_s": round(now, 3),
            "inflight_peak": int(inflight_peak),
            "inflight_token_peak": int(sim.inflight_token_peak),
            "duplicate_insert_tokens": int(sim.duplicate_insert_tokens),
            "send_lag_p50_ms": round(_pct(lags, 50) * 1000, 3) if lags else 0.0,
            "send_lag_p99_ms": round(_pct(lags, 99) * 1000, 3) if lags else 0.0,
            "send_lag_max_ms": round(max(lags) * 1000, 3) if lags else 0.0,
            "arrival_scale": arrival_scale,
            "wait_scale": wait_scale,
            "service": {
                "prefill_base_s": self.service.prefill_base_s,
                "prefill_tokens_per_s": self.service.prefill_tokens_per_s,
                "decode_tokens_per_s": self.service.decode_tokens_per_s,
            },
        }
        result.summary = summary
        return result

    def _push_next_turn(
        self,
        heap: List[Tuple[float, int, str, int]],
        seq,
        session: Session,
        idx: int,
        finish: float,
        wait_scale: float,
    ) -> None:
        """本轮完成 -> 压入下一轮（``finish + lognormal * wait_scale``）。"""

        if idx >= session.n_turns:
            return
        spec = self.specs[session.task_name]
        wait = self._inter_turn_wait(spec, session.session_id, wait_scale)
        heapq.heappush(heap, (finish + wait, next(seq), session.session_id, idx + 1))


def _pct(values: Sequence[float], p: float) -> float:
    """线性插值分位数（与 emitter 的 ``_pct`` 同口径）。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


__all__ = [
    "Emitter",
    "ServiceModel",
    "TaskSpec",
    "stable_seed_from_str",
    "max_total_len",
]
