"""仿真请求驱动兼容门面；核心原语已迁至 sim.core.lifecycle。"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from sim.core.config import SimConfig
from sim.core.lifecycle import Matched, Pending, RequestLifecycle
from sim.metrics import RequestRecord, SimResult
from sim.workload.trace import Session, Turn, max_total_len


class Simulator:
    """按 trace 与调度器推进仿真的兼容门面，委托 RequestLifecycle 管理请求状态。"""

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.lifecycle = RequestLifecycle(config)
        self.config = self.lifecycle.config

    @property
    def sym(self):
        return self.lifecycle.sym

    @property
    def clock(self):
        return self.lifecycle.clock

    @property
    def alloc(self):
        return self.lifecycle.alloc

    @property
    def req_pool(self):
        return self.lifecycle.req_pool

    @property
    def cache(self):
        return self.lifecycle.cache

    @property
    def records(self):
        return self.lifecycle.records

    @property
    def tasks(self):
        return self.lifecycle.tasks

    @tasks.setter
    def tasks(self, value: Dict[str, str]) -> None:
        self.lifecycle.tasks = value

    @property
    def inflight_tokens(self) -> int:
        return self.lifecycle.inflight_tokens

    @property
    def inflight_token_peak(self) -> int:
        return self.lifecycle.inflight_token_peak

    @property
    def duplicate_insert_tokens(self) -> int:
        return self.lifecycle.duplicate_insert_tokens

    @property
    def skipped_requests(self) -> int:
        return self.lifecycle.skipped_requests

    @property
    def self_evictions(self) -> int:
        return self.lifecycle.self_evictions

    @property
    def unexpected_evictions(self) -> int:
        return self.lifecycle.unexpected_evictions

    @property
    def cost_model(self):
        return self.lifecycle.cost_model

    def prepare(self, width: int) -> None:
        self.lifecycle.prepare(width)

    def _match(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
        turn_idx: int = -1,
    ) -> Matched:
        return self.lifecycle.match(turn, session_id, rid, task_name, turn_idx=turn_idx)

    def _make_pending(
        self, matched: Matched, *, skipped: bool, evicted: int
    ) -> Pending:
        return self.lifecycle.new_pending(matched, skipped=skipped, evicted=evicted)

    def lock_matched(self, matched: Matched) -> None:
        self.lifecycle.lock(matched)

    def reserve_upto(self, pending: Pending, seq_len: int) -> int:
        return self.lifecycle.reserve_upto(pending, seq_len)

    def commit_upto(
        self, pending: Pending, seq_len: int, *, chunked: bool = False
    ) -> None:
        self.lifecycle.commit_upto(pending, seq_len, chunked=chunked)

    def _reserve_locked(self, matched: Matched) -> Pending:
        return self.lifecycle.admit_locked(matched)

    def begin_matched(self, matched: Matched) -> Pending:
        return self.lifecycle.admit(matched)

    def begin(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
        turn_idx: int = -1,
    ) -> Pending:
        return self.lifecycle.start(
            turn, session_id, rid, task_name, turn_idx=turn_idx
        )

    def end(self, pending: Pending) -> RequestRecord:
        return self.lifecycle.finish(pending)

    def request(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
        turn_idx: int = -1,
    ) -> RequestRecord:
        return self.lifecycle.serve(
            turn, session_id, rid, task_name, turn_idx=turn_idx
        )

    def _assert_balanced(self, where: str = "") -> None:
        self.lifecycle.assert_balanced(where=where)

    def run(self, sessions: Iterable[Session]) -> SimResult:
        sessions = list(sessions)
        seen = set()
        for session in sessions:
            if session.session_id in seen:
                raise ValueError(f"session_id 重复：{session.session_id}")
            seen.add(session.session_id)

        width = max_total_len(sessions)
        if width == 0:
            return SimResult(records=[], summary={})
        self.prepare(width)
        self.tasks = {session.session_id: session.task_name for session in sessions}

        events = sorted(
            (turn.t, session.session_id, idx)
            for session in sessions
            for idx, turn in enumerate(session.turns)
        )
        turns = {session.session_id: session.turns for session in sessions}
        for _t, sid, idx in events:
            self.request(
                turns[sid][idx], sid, task_name=self.tasks[sid], turn_idx=idx
            )

        return self.result()

    def result(self) -> SimResult:
        return self.lifecycle.result()

    def summary(self) -> Dict:
        return self.lifecycle.summary()

    def run_scheduled(
        self,
        sessions: Iterable[Session],
        specs: Iterable,
        *,
        service=None,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
        max_inflight: int = 0,
    ) -> SimResult:
        from sim.schedule import Emitter

        return Emitter(self, sessions, specs, service).run(
            arrival_scale=arrival_scale,
            wait_scale=wait_scale,
            max_inflight=max_inflight,
        )

    def run_engine(
        self,
        sessions: Iterable[Session],
        specs: Iterable,
        *,
        scheduler=None,
        service=None,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
    ) -> SimResult:
        from sim.engine import EngineConfig, EngineSimulator

        return EngineSimulator(
            self,
            sessions,
            specs,
            scheduler=scheduler or EngineConfig(),
            service=service,
        ).run(arrival_scale=arrival_scale, wait_scale=wait_scale)


def run_synthetic(config: Optional[SimConfig] = None, **trace_kwargs) -> SimResult:
    """构造四类混合合成负载并执行开环回放。"""
    from sim.workload.trace import synthetic_sessions

    return Simulator(config).run(synthetic_sessions(**trace_kwargs))
