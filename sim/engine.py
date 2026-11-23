"""Engine-step driver that mirrors SGLang's prefill scheduling.

SGLang does not retire requests one by one: every ``engine.step`` first admits
as many waiting requests as the token/sequence budget allows, runs one prefill
forward for that batch, and only then advances the decode batch. Its default
``schedule_policy`` is ``fcfs`` and its admission loop lives in
``Scheduler._get_new_batch_prefill_raw`` + ``PrefillAdder.add_one_req``.

The production scheduler cannot be imported on this CPU-only path: importing
``sglang.srt.managers.schedule_policy`` pulls in ``schedule_batch`` -> model
config -> ``transformers``. This module therefore reproduces the admission
subset that changes cache behaviour, and reuses the production radix cache
through ``Simulator._match`` / ``Simulator.begin_matched`` / ``Simulator.end``:

* plan a whole prefill batch before committing any of it (so same-batch
  requests cannot hit each other's freshly written prefix);
* charge page-aligned extend tokens against ``max_prefill_tokens``;
* truncate to ``chunked_prefill_size`` and keep one chunked request across
  steps, like ``Scheduler.chunked_req``;
* keep ``max_running_requests`` / ``max_prefill_bs`` as sequence budgets;
* run one prefill step, then advance every running request by one decode token
  per step (constant TPOT, as in the UniCache simulator).

A prefill step does not also decode the existing running batch. SGLang mixes
them only when chunked prefill is active; modelling that mix would need a
second forward-time model and does not change prefix-cache admission.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence
from dataclasses import replace

import msgspec

from sim.allocator import OutOfTokens
from sim.metrics import SimResult
from sim.runner import Matched, Simulator
from sim.workload.arrival import ArrivalPlan, TaskSpec
from sim.workload.service import ServiceModel
from sim.workload.trace import Session, Turn, max_total_len


class EngineConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Admission budgets of one simulated engine.

    Names follow the SGLang scheduler fields. ``chunked_prefill_size = -1``
    disables chunked prefill, matching the SGLang CLI convention.
    """

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


class _Queued:
    """A turn that has arrived but has not been admitted yet."""

    __slots__ = ("due", "idx", "session_id", "task_name", "turn")

    def __init__(
        self,
        turn: Turn,
        session_id: str,
        task_name: str,
        due: float,
        idx: int,
    ) -> None:
        self.turn = turn
        self.session_id = session_id
        self.task_name = task_name
        self.due = due
        self.idx = idx


class _State:
    """A request with a cache reservation, progressing through prefill/decode."""

    __slots__ = ("charge", "generated", "matched", "pending", "prefilled", "queued")

    def __init__(
        self,
        queued: _Queued | None,
        matched: Matched,
        pending=None,
    ) -> None:
        self.queued = queued
        self.matched = matched
        self.pending = pending
        self.prefilled = matched.hit
        self.generated = 0
        self.charge = 0

    @property
    def prompt_len(self) -> int:
        return self.matched.prompt_len

    @property
    def out_len(self) -> int:
        return self.matched.out_len


class _PrefillBudget:
    """Batch admission budget over live allocator plus evictable tree tokens.

    The simulator reserves prompt + output KV at admission (``Matched.need``),
    whereas SGLang allocates output tokens as they are generated. The budget is
    therefore billed with the actual reservation: :meth:`fits` and :meth:`reserve`
    both take ``Matched.need``, so admission and allocation cannot disagree.

    Locking a matched prefix removes it from ``evictable_size()``; the next
    candidate sees that reduction, matching SGLang's ``_req_inc_lock_ref``
    effect during ``PrefillAdder.add_one_req``.
    """

    __slots__ = ("_offset", "alloc", "cache")

    def __init__(self, alloc, cache, reserved_offset: int = 0) -> None:
        self.alloc = alloc
        self.cache = cache
        self._offset = int(reserved_offset)

    def remaining_total(self) -> int:
        return self.alloc.available_size() + self.cache.evictable_size() - self._offset

    def fits(self, need: int) -> bool:
        return need <= self.remaining_total()

    def reserve(self, need: int) -> None:
        self._offset += need


def _unallocated_reserved(running: list[_State], chunked: _State | None) -> int:
    """已准入请求尚未物理占槽的未来输出（或后续分块）余量。"""
    states = list(running) + ([chunked] if chunked is not None else [])
    return sum(
        (s.matched.hit + s.matched.need) - s.pending.allocated_len
        for s in states
        if s.pending is not None and not s.pending.skipped
    )


class EngineSimulator:
    """Drive a :class:`Simulator` with SGLang-style engine steps.

    The arrival stream and inter-turn waits come from :class:`sim.schedule.Emitter`
    so this driver shares the seeds and service-time model with ``run_scheduled``.
    """

    def __init__(
        self,
        sim: Simulator,
        sessions: Iterable[Session],
        specs: Iterable[TaskSpec],
        *,
        scheduler: EngineConfig,
        service: ServiceModel | None = None,
    ) -> None:
        self.sim = sim
        self.scheduler = scheduler
        self.service = service or ServiceModel()
        self.plan = ArrivalPlan(list(sessions), list(specs))
        self.sessions = self.plan.sessions

        self._wait_scale = 1.0

        # Step-level counters exposed through summary["emit"].
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
        self._send_lags: list[float] = []

    # ------------------------------------------------------------------ run
    def run(
        self,
        *,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
    ) -> SimResult:
        if not self.sessions:
            return SimResult(records=[], summary={})

        sim = self.sim
        self._wait_scale = wait_scale
        width = max_total_len(self.sessions)
        sim.prepare(width)
        sim.tasks = {s.session_id: s.task_name for s in self.sessions}

        heap = self.plan.init_heap(arrival_scale=arrival_scale)

        waiting: list[_Queued] = []
        running: list[_State] = []
        chunked: _State | None = None
        now = 0.0
        eps = 1e-9

        while heap or waiting or running or chunked is not None:
            # Requests that have arrived are appended in FCFS order, exactly as
            # Scheduler._add_request_to_queue does before a step.
            while heap and heap[0][0] <= now + eps:
                due, _seq, sid, idx = heapq.heappop(heap)
                session = self.plan.by_sid[sid]
                waiting.append(
                    _Queued(
                        turn=session.turns[idx - 1],
                        session_id=sid,
                        task_name=session.task_name,
                        due=due,
                        idx=idx,
                    )
                )

            if not waiting and not running and chunked is None:
                if not heap:
                    break
                now = max(now, heap[0][0])
                sim.clock.advance_to(now)
                continue

            prefill = self._prefill_step(now, waiting, running, chunked, heap)
            if prefill is not None:
                now, chunked = prefill
                self._steps += 1
                self._prefill_steps += 1
                continue

            if running:
                now += self.service.decode_step()
                sim.clock.advance_to(now)
                self._decode_step(running, now, heap)
                self._steps += 1
                self._decode_steps += 1
                continue

            raise OutOfTokens(
                "engine deadlock: requests are waiting but no prefill batch can be "
                f"admitted (waiting={len(waiting)} chunked={chunked is not None} "
                f"available={sim.alloc.available_size()} "
                f"evictable={sim.cache.evictable_size()})"
            )

        return self._result(now, arrival_scale, wait_scale)

    # ------------------------------------------------------------------ prefill
    def _prefill_step(
        self,
        now: float,
        waiting: list[_Queued],
        running: list[_State],
        chunked: _State | None,
        heap: list[tuple[float, int, str, int]],
    ) -> tuple[float, _State | None] | None:
        unallocated = _unallocated_reserved(running, chunked)
        budget = _PrefillBudget(self.sim.alloc, self.sim.cache, reserved_offset=unallocated)
        self._waiting_peak = max(self._waiting_peak, len(waiting))

        admitted, skipped_any = self._admit_waiting(
            now, waiting, running, chunked, budget
        )
        self._trim_to_capacity(admitted, waiting, now, reserved_offset=unallocated)

        try:
            if self.scheduler.lazy_decode_alloc:
                if chunked is not None and chunked.pending is not None:
                    self.sim.reserve_upto(
                        chunked.pending,
                        min(chunked.prompt_len, chunked.prefilled + chunked.charge),
                    )
                for state in admitted:
                    pending = self.sim._make_pending(
                        state.matched, skipped=False, evicted=0
                    )
                    self.sim.reserve_upto(
                        pending,
                        min(state.prompt_len, state.prefilled + state.charge),
                    )
                    state.pending = pending
            else:
                for state in admitted:
                    state.pending = self.sim._reserve_locked(state.matched)
        except BaseException:
            for state in admitted:
                if state.pending is None:
                    self.sim.cache.dec_lock_ref(state.matched.node)
            raise

        if not admitted and chunked is None:
            if skipped_any:
                return now, None
            return None
        return self._finish_prefill(now, admitted, chunked, running, heap)

    def _admit_waiting(
        self,
        now: float,
        waiting: list[_Queued],
        running: list[_State],
        chunked: _State | None,
        budget: _PrefillBudget,
    ) -> tuple[list[_State], bool]:
        cfg = self.scheduler
        rem_input = cfg.max_prefill_tokens
        rem_chunk = None if cfg.chunked_prefill_size == -1 else cfg.chunked_prefill_size
        admitted: list[_State] = []
        skipped_any = False

        # SGLang calls add_chunked_req before scanning the waiting queue, so a
        # carried-over chunk gets the first slice of this step's budget.
        if chunked is not None:
            remaining = chunked.prompt_len - chunked.prefilled
            charge = remaining if rem_chunk is None else min(remaining, rem_chunk)
            chunked.charge = max(0, charge)
            rem_input -= chunked.charge
            if rem_chunk is not None:
                rem_chunk -= chunked.charge

        for queued in self._waiting_order(waiting, now):
            if rem_input <= 0:
                break
            if rem_chunk is not None and rem_chunk <= 0:
                break
            carried = 1 if chunked is not None else 0
            if (
                cfg.max_prefill_bs is not None
                and len(admitted) + carried >= cfg.max_prefill_bs
            ):
                break
            active = len(running) + len(admitted) + carried
            if (
                cfg.max_running_requests is not None
                and active >= cfg.max_running_requests
            ):
                break

            # Match now, but do not evict yet. SGLang locks every admitted
            # candidate during admission and only allocates after the whole
            # batch is chosen, so all candidates match the pre-batch tree.
            matched = self.sim._match(
                replace(queued.turn, t=now),
                queued.session_id,
                rid=None,
                task_name=queued.task_name,
                turn_idx=queued.idx - 1,  # emitter 的 idx 是 1 基
            )
            if matched.need > self.sim.alloc.size:
                pending = self.sim.begin_matched(matched)
                self.sim.end(pending)
                waiting.remove(queued)
                self._send_lags.append(now - queued.due)
                skipped_any = True
                continue

            extend = max(0, matched.prompt_len - matched.hit)
            is_chunked = rem_chunk is not None and extend > rem_chunk
            page = self.sim.cache.page_size
            paged_extend = -(-extend // page) * page
            charge = max(0, rem_chunk if is_chunked else paged_extend)
            if not budget.fits(matched.need) and (admitted or chunked is not None):
                break

            state = _State(queued=queued, matched=matched)
            state.charge = charge
            self.sim.lock_matched(matched)
            waiting.remove(queued)
            self._send_lags.append(now - queued.due)
            admitted.append(state)
            budget.reserve(matched.need)
            self._prefill_requests += 1
            if matched.hit + charge < state.prompt_len:
                self._chunked_requests += 1

            # SGLang's add_one_req commits the candidate that drains the last
            # token, then budget_state() stops the next one. Mirror that soft cap.
            rem_input -= charge
            if rem_chunk is not None:
                rem_chunk -= charge
            if rem_input <= 0 or (rem_chunk is not None and rem_chunk <= 0):
                break
        return admitted, skipped_any

    def _finish_prefill(
        self,
        now: float,
        admitted: list[_State],
        chunked: _State | None,
        running: list[_State],
        heap: list[tuple[float, int, str, int]],
    ) -> tuple[float, _State | None]:
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
        self.sim.clock.advance_to(now)

        next_chunked: _State | None = None
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

    def _trim_to_capacity(
        self,
        admitted: list[_State],
        waiting: list[_Queued],
        now: float,
        reserved_offset: int = 0,
    ) -> None:
        """Drop tail candidates until the locked batch fits the live pool.

        The admission budget is checked before locking, so a batch made almost
        entirely of its own evictable prefix could pass the check and still
        leave too little room once that prefix is pinned. Re-check after locking
        and return the excess to the waiting queue; the next step re-matches.
        """

        while admitted:
            free = (
                self.sim.alloc.available_size()
                + self.sim.cache.evictable_size()
                - reserved_offset
            )
            need = sum(state.matched.need for state in admitted)
            if need <= free:
                return
            state = admitted.pop()
            self.sim.cache.dec_lock_ref(state.matched.node)
            waiting.insert(0, state.queued)
            self._prefill_requests -= 1
            if state.matched.hit + state.charge < state.prompt_len:
                self._chunked_requests -= 1
            lag = now - state.queued.due
            if lag in self._send_lags:
                self._send_lags.remove(lag)

    def _waiting_order(self, waiting: list[_Queued], now: float) -> list[_Queued]:
        if self.scheduler.schedule_policy == "fcfs":
            return list(waiting)

        # SGLang's cache-aware LPM policy sorts by the longest matched prefix.
        # Its in-batch de-prioritisation pass builds a second radix tree; that
        # only affects ordering, so it is not reproduced here.
        pairs: list[tuple[_Queued, Matched]] = []
        for queued in waiting:
            matched = self.sim._match(
                replace(queued.turn, t=now),
                queued.session_id,
                rid=None,
                task_name=queued.task_name,
                turn_idx=queued.idx - 1,  # emitter 的 idx 是 1 基
            )
            pairs.append((queued, matched))
        pairs.sort(key=lambda pair: -pair[1].hit)
        return [queued for queued, _matched in pairs]

    def _advance_prefill(
        self,
        state: _State,
        *,
        running: list[_State],
        now: float,
        heap: list[tuple[float, int, str, int]],
    ) -> _State | None:
        state.prefilled = min(state.prompt_len, state.prefilled + state.charge)
        if state.prefilled < state.prompt_len:
            if self.scheduler.progressive_cache:
                self.sim.commit_upto(state.pending, state.prefilled, chunked=True)
            return state
        if state.out_len == 0:
            self.sim.end(state.pending)
            self._push_next_turn(state, now, heap)
        else:
            if self.scheduler.progressive_cache:
                self.sim.commit_upto(state.pending, state.prompt_len, chunked=False)
            running.append(state)
        return None

    # ------------------------------------------------------------------ decode
    def _decode_step(
        self,
        running: list[_State],
        now: float,
        heap: list[tuple[float, int, str, int]],
    ) -> None:
        finished: list[_State] = []
        for state in running:
            state.generated += 1
            if self.scheduler.lazy_decode_alloc:
                self.sim.reserve_upto(
                    state.pending, state.prompt_len + state.generated
                )
            if state.generated >= state.out_len:
                finished.append(state)
        for state in finished:
            running.remove(state)
            self.sim.end(state.pending)
            self._push_next_turn(state, now, heap)
        self._running_peak = max(self._running_peak, len(running))
        self._inflight_peak = max(self._inflight_peak, len(running))

    def _push_next_turn(
        self,
        state: _State,
        finish: float,
        heap: list[tuple[float, int, str, int]],
    ) -> None:
        queued = state.queued
        if queued is None:
            return
        self.plan.push_next_turn(
            heap, queued.session_id, queued.idx, finish, self._wait_scale
        )

    # ------------------------------------------------------------------ result
    def _result(
        self,
        now: float,
        arrival_scale: float,
        wait_scale: float,
    ) -> SimResult:
        result = self.sim.result()
        lags = sorted(self._send_lags)
        summary = result.summary or {}
        summary["emit"] = {
            "mode": "engine_step",
            "scheduler": {
                "max_prefill_tokens": self.scheduler.max_prefill_tokens,
                "chunked_prefill_size": self.scheduler.chunked_prefill_size,
                "max_running_requests": self.scheduler.max_running_requests,
                "max_prefill_bs": self.scheduler.max_prefill_bs,
                "schedule_policy": self.scheduler.schedule_policy,
                "progressive_cache": self.scheduler.progressive_cache,
                "lazy_decode_alloc": self.scheduler.lazy_decode_alloc,
            },
            "wall_s": round(now, 3),
            "steps": self._steps,
            "prefill_steps": self._prefill_steps,
            "decode_steps": self._decode_steps,
            "prefill_requests": self._prefill_requests,
            "chunked_requests": self._chunked_requests,
            "max_prefill_batch_tokens": self._max_prefill_batch_tokens,
            "max_prefill_batch_size": self._max_prefill_batch_size,
            "waiting_peak": self._waiting_peak,
            "running_peak": self._running_peak,
            "inflight_peak": self._inflight_peak,
            "inflight_token_peak": int(self.sim.inflight_token_peak),
            "duplicate_insert_tokens": int(self.sim.duplicate_insert_tokens),
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
    k = (len(values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


__all__ = ["EngineConfig", "EngineSimulator"]
