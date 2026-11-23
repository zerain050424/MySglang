"""开环静态 trace 回放驱动。"""

from __future__ import annotations

from typing import Iterable

from sim.core.lifecycle import RequestLifecycle
from sim.metrics import SimResult
from sim.workload.trace import Session, max_total_len


class ReplayDriver:
    """按会话自带的 Turn.t 升序重放全部轮次，无并发冲突。"""

    def __init__(self, lifecycle: RequestLifecycle) -> None:
        self.lifecycle = lifecycle

    def run(self, sessions: Iterable[Session]) -> SimResult:
        """按全局事件时间戳重放全部会话。"""
        sessions = list(sessions)
        seen = set()
        for session in sessions:
            if session.session_id in seen:
                raise ValueError(f"session_id 重复：{session.session_id}")
            seen.add(session.session_id)

        width = max_total_len(sessions)
        if width == 0:
            return SimResult(records=[], summary={})
        self.lifecycle.prepare(width)
        self.lifecycle.tasks = {
            session.session_id: session.task_name for session in sessions
        }

        events = sorted(
            (turn.t, session.session_id, idx)
            for session in sessions
            for idx, turn in enumerate(session.turns)
        )
        turns = {session.session_id: session.turns for session in sessions}
        for t, sid, idx in events:
            self.lifecycle.serve(
                turns[sid][idx],
                sid,
                task_name=self.lifecycle.tasks[sid],
                turn_idx=idx,
                arrival_t=t,
            )

        return self.lifecycle.result()
