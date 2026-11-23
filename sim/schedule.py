"""闭环发射器；ArrivalPlan 与 ServiceModel 已迁至 sim.workload。"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from sim.metrics import SimResult
from sim.workload.arrival import (
    ArrivalPlan,
    TaskSpec,
    check_segments,
    first_turn_offsets,
    stable_seed_from_str,
)
from sim.workload.service import ServiceModel
from sim.workload.trace import Session, max_total_len


class Emitter:
    """按 bench_heap 规则驱动 Simulator（虚拟时间，闭环并发）。"""

    def __init__(
        self,
        sim,
        sessions: Sequence[Session],
        specs: Sequence[TaskSpec],
        service: ServiceModel | None = None,
    ) -> None:
        self.sim = sim
        self.plan = ArrivalPlan(sessions, specs)
        self.sessions = self.plan.sessions
        self.specs: Dict[str, TaskSpec] = self.plan.specs
        self.service = service or ServiceModel()
        self._by_sid = self.plan.by_sid
        self.send_lags: List[float] = []

    def _arrival_offsets(self, spec: TaskSpec, n: int) -> List[float]:
        return first_turn_offsets(spec, n)

    def _inter_turn_wait(self, spec: TaskSpec, sid: str, wait_scale: float) -> float:
        return self.plan.inter_turn_wait(spec, sid, wait_scale)

    def run(
        self,
        *,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
        max_inflight: int = 0,
    ) -> SimResult:
        """按闭环到达流推进仿真并收集汇总。"""
        sim = self.sim
        if not self.sessions:
            return SimResult(records=[], summary={})
        if max_inflight < 0:
            raise ValueError(f"max_inflight 必须 >= 0，got {max_inflight}")

        heap = self.plan.init_heap(arrival_scale=arrival_scale)
        width = max_total_len(self.sessions)
        sim.prepare(width)
        sim.tasks = {s.session_id: s.task_name for s in self.sessions}

        inflight: List[Tuple[float, int, object, str, int]] = []
        done_seq = itertools.count()
        eps = 1e-9
        now = 0.0
        inflight_peak = 0

        while heap or inflight:
            can_dispatch = max_inflight <= 0 or len(inflight) < max_inflight

            # 同一时刻优先完成老请求腾出槽位与锁，再派发新请求。
            if inflight and inflight[0][0] <= now + eps:
                finish, _dseq, pending, sid, idx = heapq.heappop(inflight)
                sim.clock.advance_to(finish)
                sim.end(pending)
                now = finish
                self.plan.push_next_turn(heap, sid, idx, finish, wait_scale)
                continue

            if heap and can_dispatch and heap[0][0] <= now + eps:
                due, _s, sid, idx = heapq.heappop(heap)
                self.send_lags.append(now - due)
                session = self.plan.by_sid[sid]
                turn = replace(session.turns[idx - 1], t=now)
                pending = sim.begin(
                    turn,
                    sid,
                    task_name=session.task_name,
                    turn_idx=idx - 1,
                )
                finish = now + self.service(pending.prompt_len, pending.out_len)
                heapq.heappush(
                    inflight, (finish, next(done_seq), pending, sid, idx)
                )
                inflight_peak = max(inflight_peak, len(inflight))
                continue

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


def _pct(values: Sequence[float], p: float) -> float:
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
    "check_segments",
    "stable_seed_from_str",
    "max_total_len",
]
