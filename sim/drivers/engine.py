"""SGLang 引擎步级调度驱动（批预算准入 + chunked prefill + decode 步进）。"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import List, Optional, Tuple

import msgspec

from sim.core.allocator import OutOfTokens
from sim.core.lifecycle import RequestLifecycle
from sim.drivers.admission import (
    ActiveReq,
    QueuedTurn,
    admit_waiting,
    trim_to_capacity,
)
from sim.drivers.budget import PrefillBudget, unallocated_reserved
from sim.drivers.common import build_emit_stats
from sim.drivers.policies import (
    CommitOnFinish,
    CommitPolicy,
    EagerReserve,
    LazyReserve,
    ProgressiveCommit,
    ReservePolicy,
)
from sim.analysis.metrics import SimResult
from sim.workload.arrival import ArrivalPlan, TaskSpec
from sim.workload.service import ServiceModel
from sim.workload.trace import Session, max_total_len


class EngineConfig(msgspec.Struct, frozen=True, kw_only=True):
    """引擎准入预算及分配/写树时机配置；chunked_prefill_size=-1 表示关闭分块。"""

    max_prefill_tokens: int = 16384
    chunked_prefill_size: int = -1
    max_running_requests: int | None = None
    max_prefill_bs: int | None = None
    schedule_policy: str = "fcfs"
    progressive_cache: bool = True
    lazy_decode_alloc: bool = True

    def __post_init__(self) -> None:
        if self.max_prefill_tokens <= 0:
            raise ValueError(
                f"max_prefill_tokens must be positive, got {self.max_prefill_tokens}"
            )
        if self.chunked_prefill_size != -1 and self.chunked_prefill_size <= 0:
            raise ValueError(
                "chunked_prefill_size must be -1 (disabled) or positive, "
                f"got {self.chunked_prefill_size}"
            )
        if self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ValueError(
                "max_running_requests must be positive or None, "
                f"got {self.max_running_requests}"
            )
        if self.max_prefill_bs is not None and self.max_prefill_bs <= 0:
            raise ValueError(
                f"max_prefill_bs must be positive or None, got {self.max_prefill_bs}"
            )
        if self.schedule_policy not in ("fcfs", "lpm"):
            raise ValueError(
                f"schedule_policy must be 'fcfs' or 'lpm', got {self.schedule_policy!r}"
            )


class EngineScheduler:
    """模拟 SGLang Scheduler 单步先批 Prefill、后全批 Decode 1 token 的执行序。"""

    def __init__(
        self,
        lifecycle: RequestLifecycle,
        sessions: Iterable[Session],
        specs: Iterable[TaskSpec],
        *,
        scheduler: Optional[EngineConfig] = None,
        service: Optional[ServiceModel] = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.scheduler = scheduler or EngineConfig()
        self.service = service or ServiceModel()
        self.plan = ArrivalPlan(list(sessions), list(specs))
        self.reserve_policy: ReservePolicy = (
            LazyReserve() if self.scheduler.lazy_decode_alloc else EagerReserve()
        )
        self.commit_policy: CommitPolicy = (
            ProgressiveCommit() if self.scheduler.progressive_cache else CommitOnFinish()
        )

        self._wait_scale = 1.0
        self._steps = 0
        self._prefill_steps = 0
        self._decode_steps = 0
        self._prefill_requests = 0
        self._chunked_requests = 0
        self._max_prefill_batch_tokens = 0
        self._max_prefill_batch_size = 0
        self._waiting_peak = 0
        self._running_peak = 0
        self._inflight_peak = 0
        self._send_lags: List[float] = []

    def _drain_arrived(
        self,
        heap: List[Tuple[float, int, str, int]],
        waiting: List[QueuedTurn],
        now: float,
        eps: float,
    ) -> None:
        while heap and heap[0][0] <= now + eps:
            due, _seq, sid, idx = heapq.heappop(heap)
            session = self.plan.by_sid[sid]
            waiting.append(
                QueuedTurn(
                    turn=session.turns[idx - 1],
                    session_id=sid,
                    task_name=session.task_name,
                    due=due,
                    idx=idx,
                )
            )

    def run(
        self,
        *,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
    ) -> SimResult:
        """按引擎步推进仿真直至所有会话全部轮次完成。"""
        if not self.plan.sessions:
            return SimResult(records=[], summary={})

        self._wait_scale = wait_scale
        width = max_total_len(self.plan.sessions)
        self.lifecycle.prepare(width)
        self.lifecycle.tasks = {
            s.session_id: s.task_name for s in self.plan.sessions
        }
        heap = self.plan.init_heap(arrival_scale=arrival_scale)

        waiting: List[QueuedTurn] = []
        running: List[ActiveReq] = []
        chunked: Optional[ActiveReq] = None
        now = 0.0
        eps = 1e-9

        while heap or waiting or running or chunked is not None:
            self._drain_arrived(heap, waiting, now, eps)

            if not waiting and not running and chunked is None:
                if not heap:
                    break
                now = max(now, heap[0][0])
                self.lifecycle.clock.advance_to(now)
                continue

            prefill = self._prefill_step(now, waiting, running, chunked, heap)
            if prefill is not None:
                now, chunked = prefill
                self._steps += 1
                self._prefill_steps += 1
                continue

            if running:
                now += self.service.decode_step()
                self.lifecycle.clock.advance_to(now)
                self._decode_step(running, now, heap)
                self._steps += 1
                self._decode_steps += 1
                continue

            raise OutOfTokens(
                "engine deadlock: requests are waiting but no prefill batch can be "
                f"admitted (waiting={len(waiting)} chunked={chunked is not None} "
                f"available={self.lifecycle.alloc.available_size()} "
                f"evictable={self.lifecycle.cache.evictable_size()})"
            )

        return self._result(now, arrival_scale, wait_scale)

    def _prefill_step(
        self,
        now: float,
        waiting: List[QueuedTurn],
        running: List[ActiveReq],
        chunked: Optional[ActiveReq],
        heap: List[Tuple[float, int, str, int]],
    ) -> Optional[Tuple[float, Optional[ActiveReq]]]:
        # 对懒分配策略仅在需要预测在途未来占槽时扣除未分配余量；eager 模式该值为 0。
        unallocated = unallocated_reserved(running, chunked)
        budget = PrefillBudget(
            self.lifecycle.alloc, self.lifecycle.cache, reserved_offset=unallocated
        )
        self._waiting_peak = max(self._waiting_peak, len(waiting))

        admitted, skipped_any, prefill_added, chunked_added = admit_waiting(
            now=now,
            cfg=self.scheduler,
            waiting=waiting,
            running=running,
            chunked=chunked,
            budget=budget,
            lifecycle=self.lifecycle,
            send_lags=self._send_lags,
        )
        self._prefill_requests += prefill_added
        self._chunked_requests += chunked_added

        dropped_p, dropped_c = trim_to_capacity(
            admitted=admitted,
            waiting=waiting,
            now=now,
            lifecycle=self.lifecycle,
            send_lags=self._send_lags,
            reserved_offset=unallocated,
        )
        self._prefill_requests -= dropped_p
        self._chunked_requests -= dropped_c

        try:
            self.reserve_policy.on_prefill_step(self.lifecycle, chunked, admitted)
        except BaseException:
            # 异常发生时尚未转交 Pending 的候选必须显式解锁，避免锁泄漏。
            for state in admitted:
                if state.pending is None:
                    self.lifecycle.cache.dec_lock_ref(state.matched.node)
            raise

        if not admitted and chunked is None:
            if skipped_any:
                return now, None
            return None
        return self._finish_prefill(now, admitted, chunked, running, heap)

    def _finish_prefill(
        self,
        now: float,
        admitted: List[ActiveReq],
        chunked: Optional[ActiveReq],
        running: List[ActiveReq],
        heap: List[Tuple[float, int, str, int]],
    ) -> Tuple[float, Optional[ActiveReq]]:
        tokens = sum(state.charge for state in admitted)
        if chunked is not None:
            tokens += chunked.charge
        self._max_prefill_batch_tokens = max(self._max_prefill_batch_tokens, tokens)
        self._max_prefill_batch_size = max(
            self._max_prefill_batch_size, len(admitted) + (1 if chunked else 0)
        )
        self._inflight_peak = max(
            self._inflight_peak,
            len(running) + len(admitted) + (1 if chunked is not None else 0),
        )

        now += self.service.prefill(tokens)
        self.lifecycle.clock.advance_to(now)

        next_chunked: Optional[ActiveReq] = None
        if chunked is not None:
            next_chunked = self._advance_prefill(
                chunked, running=running, now=now, heap=heap
            )
        for state in admitted:
            carried = self._advance_prefill(state, running=running, now=now, heap=heap)
            if carried is not None:
                if next_chunked is not None:
                    raise AssertionError(
                        "two chunked requests in one prefill step; SGLang keeps one"
                    )
                next_chunked = carried
        self._running_peak = max(self._running_peak, len(running))
        return now, next_chunked

    def _advance_prefill(
        self,
        state: ActiveReq,
        *,
        running: List[ActiveReq],
        now: float,
        heap: List[Tuple[float, int, str, int]],
    ) -> Optional[ActiveReq]:
        state.prefilled = min(state.prompt_len, state.prefilled + state.charge)
        if state.prefilled < state.prompt_len:
            self.commit_policy.on_chunk_done(self.lifecycle, state)
            return state
        if state.out_len == 0:
            self.lifecycle.finish(state.pending)
            self._push_next_turn(state, now, heap)
        else:
            self.commit_policy.on_prefill_done(self.lifecycle, state)
            running.append(state)
        return None

    def _decode_step(
        self,
        running: List[ActiveReq],
        now: float,
        heap: List[Tuple[float, int, str, int]],
    ) -> None:
        finished: List[ActiveReq] = []
        for state in running:
            state.generated += 1
            self.reserve_policy.on_decode_step(self.lifecycle, state)
            if state.generated >= state.out_len:
                finished.append(state)
        for state in finished:
            running.remove(state)
            self.lifecycle.finish(state.pending)
            self._push_next_turn(state, now, heap)
        self._running_peak = max(self._running_peak, len(running))
        self._inflight_peak = max(self._inflight_peak, len(running))

    def _push_next_turn(
        self,
        state: ActiveReq,
        finish: float,
        heap: List[Tuple[float, int, str, int]],
    ) -> None:
        queued = state.queued
        if queued is None:
            return
        self.plan.push_next_turn(
            heap, queued.session_id, queued.idx, finish, self._wait_scale
        )

    def _result(
        self, now: float, arrival_scale: float, wait_scale: float
    ) -> SimResult:
        result = self.lifecycle.result()
        summary = result.summary or {}
        summary["emit"] = build_emit_stats(
            mode="engine_step",
            wall_s=now,
            inflight_peak=self._inflight_peak,
            lifecycle=self.lifecycle,
            send_lags=self._send_lags,
            arrival_scale=arrival_scale,
            wait_scale=wait_scale,
            service=self.service,
            extra={
                "scheduler": {
                    "max_prefill_tokens": self.scheduler.max_prefill_tokens,
                    "chunked_prefill_size": self.scheduler.chunked_prefill_size,
                    "max_running_requests": self.scheduler.max_running_requests,
                    "max_prefill_bs": self.scheduler.max_prefill_bs,
                    "schedule_policy": self.scheduler.schedule_policy,
                    "progressive_cache": self.scheduler.progressive_cache,
                    "lazy_decode_alloc": self.scheduler.lazy_decode_alloc,
                },
                "steps": self._steps,
                "prefill_steps": self._prefill_steps,
                "decode_steps": self._decode_steps,
                "prefill_requests": self._prefill_requests,
                "chunked_requests": self._chunked_requests,
                "max_prefill_batch_tokens": self._max_prefill_batch_tokens,
                "max_prefill_batch_size": self._max_prefill_batch_size,
                "waiting_peak": self._waiting_peak,
                "running_peak": self._running_peak,
            },
        )
        result.summary = summary
        return result
