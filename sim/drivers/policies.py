"""引擎步级调度使用的物理占槽策略与前缀写树策略。"""

from __future__ import annotations

from typing import Protocol


class ReservePolicy(Protocol):
    """控制引擎步在 Prefill 与 Decode 阶段何时推进物理 KV 槽位。"""

    def on_prefill_step(self, lifecycle, chunked, admitted: list) -> None:
        ...

    def on_decode_step(self, lifecycle, state) -> None:
        ...


class EagerReserve:
    """准入时一次性预留全部 prompt + output 槽位，decode 期间不再额外分配。"""

    def on_prefill_step(self, lifecycle, chunked, admitted: list) -> None:
        del chunked
        for state in admitted:
            state.pending = lifecycle.admit_locked(state.matched)

    def on_decode_step(self, lifecycle, state) -> None:
        del lifecycle, state


class LazyReserve:
    """Prefill 仅占实际分块覆盖槽位，后续 output 随 decode 步进跨页边界时再分配。"""

    def on_prefill_step(self, lifecycle, chunked, admitted: list) -> None:
        if chunked is not None and chunked.pending is not None:
            lifecycle.reserve_upto(
                chunked.pending,
                min(chunked.prompt_len, chunked.prefilled + chunked.charge),
            )
        for state in admitted:
            pending = lifecycle.new_pending(state.matched, skipped=False, evicted=0)
            lifecycle.reserve_upto(
                pending, min(state.prompt_len, state.prefilled + state.charge)
            )
            state.pending = pending

    def on_decode_step(self, lifecycle, state) -> None:
        lifecycle.reserve_upto(state.pending, state.prompt_len + state.generated)


class CommitPolicy(Protocol):
    """控制引擎在分块边界与 Prefill 完成时刻是否提前提交前缀到前缀树。"""

    def on_chunk_done(self, lifecycle, state) -> None:
        ...

    def on_prefill_done(self, lifecycle, state) -> None:
        ...


class CommitOnFinish:
    """仅在请求完全结束时一次性写树（保守基准模式）。"""

    def on_chunk_done(self, lifecycle, state) -> None:
        del lifecycle, state

    def on_prefill_done(self, lifecycle, state) -> None:
        del lifecycle, state


class ProgressiveCommit:
    """分块完成及转入 decode 时即时提交满页前缀到树并移锁，消除并发可见延迟。"""

    def on_chunk_done(self, lifecycle, state) -> None:
        lifecycle.commit_upto(state.pending, state.prefilled, chunked=True)

    def on_prefill_done(self, lifecycle, state) -> None:
        lifecycle.commit_upto(state.pending, state.prompt_len, chunked=False)
