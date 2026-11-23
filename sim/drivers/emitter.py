"""闭环多轮发射器驱动"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from sim.core.lifecycle import RequestLifecycle
from sim.drivers.common import build_emit_stats
from sim.analysis.metrics import SimResult
from sim.workload.arrival import ArrivalPlan, TaskSpec
from sim.workload.service import ServiceModel
from sim.workload.trace import Session, max_total_len


class ClosedLoopEmitter:
    """在虚拟时钟下按闭环轮次发射请求，并支持 max_inflight 并发窗口。"""

    def __init__(
        self,
        lifecycle: RequestLifecycle,
        sessions: Sequence[Session],
        specs: Sequence[TaskSpec],
        service: Optional[ServiceModel] = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.plan = ArrivalPlan(sessions, specs)
        self.service = service or ServiceModel()
        self.send_lags: List[float] = []

    def _complete_due(
        self,
        now: float,
        eps: float,
        inflight: List[Tuple[float, int, object, str, int]],
        heap: List[Tuple[float, int, str, int]],
        wait_scale: float,
    ) -> Optional[float]:
        if not (inflight and inflight[0][0] <= now + eps):
            return None
        finish, _dseq, pending, sid, idx = heapq.heappop(inflight)
        self.lifecycle.clock.advance_to(finish)
        self.lifecycle.finish(pending)
        self.plan.push_next_turn(heap, sid, idx, finish, wait_scale)
        return finish

    def _dispatch_due(
        self,
        now: float,
        eps: float,
        can_dispatch: bool,
        heap: List[Tuple[float, int, str, int]],
        inflight: List[Tuple[float, int, object, str, int]],
        done_seq,
    ) -> bool:
        if not (heap and can_dispatch and heap[0][0] <= now + eps):
            return False
        due, _s, sid, idx = heapq.heappop(heap)
        self.send_lags.append(now - due)
        session = self.plan.by_sid[sid]
        turn = replace(session.turns[idx - 1], t=now)
        pending = self.lifecycle.start(
            turn,
            sid,
            task_name=session.task_name,
            turn_idx=idx - 1,
            arrival_t=due,
        )
        finish = now + self.service(pending.prompt_len, pending.out_len)
        heapq.heappush(inflight, (finish, next(done_seq), pending, sid, idx))
        return True

    def _advance_clock(
        self,
        now: float,
        can_dispatch: bool,
        heap: List[Tuple[float, int, str, int]],
        inflight: List[Tuple[float, int, object, str, int]],
    ) -> float:
        next_due = heap[0][0] if (heap and can_dispatch) else math.inf
        next_finish = inflight[0][0] if inflight else math.inf
        target = min(next_due, next_finish)
        if target == math.inf:
            raise RuntimeError("调度死锁：既不能派发也没有在飞请求")
        self.lifecycle.clock.advance_to(max(target, now))
        return self.lifecycle.clock()

    def run(
        self,
        *,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
        max_inflight: int = 0,
    ) -> SimResult:
        """运行闭环到达流并输出包含 emit 汇总的结果。"""
        if not self.plan.sessions:
            return SimResult(records=[], summary={})
        if max_inflight < 0:
            raise ValueError(f"max_inflight 必须 >= 0，got {max_inflight}")

        self.send_lags = []
        heap = self.plan.init_heap(arrival_scale=arrival_scale)
        width = max_total_len(self.plan.sessions)
        self.lifecycle.prepare(width)
        self.lifecycle.tasks = {
            s.session_id: s.task_name for s in self.plan.sessions
        }

        inflight: List[Tuple[float, int, object, str, int]] = []
        done_seq = itertools.count()
        eps = 1e-9
        now = 0.0
        inflight_peak = 0

        while heap or inflight:
            can_dispatch = max_inflight <= 0 or len(inflight) < max_inflight
            # 同刻优先完成老请求释放槽位和锁，再派发新请求。
            done_t = self._complete_due(now, eps, inflight, heap, wait_scale)
            if done_t is not None:
                now = done_t
                continue

            if self._dispatch_due(now, eps, can_dispatch, heap, inflight, done_seq):
                inflight_peak = max(inflight_peak, len(inflight))
                continue

            now = self._advance_clock(now, can_dispatch, heap, inflight)

        result = self.lifecycle.result()
        summary = result.summary or {}
        mode = (
            "closed_loop_concurrent" if max_inflight != 1 else "closed_loop_sequential"
        )
        summary["emit"] = build_emit_stats(
            mode=mode,
            wall_s=now,
            inflight_peak=inflight_peak,
            lifecycle=self.lifecycle,
            send_lags=self.send_lags,
            arrival_scale=arrival_scale,
            wait_scale=wait_scale,
            service=self.service,
            extra={"max_inflight": max_inflight},
        )
        result.summary = summary
        return result
