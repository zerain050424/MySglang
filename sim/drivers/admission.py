"""引擎步等待队列排序、批量准入规划与容量回退。"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from sim.core.lifecycle import Matched
from sim.drivers.budget import PrefillBudget
from sim.workload.trace import Turn


class QueuedTurn:
    """已到达但尚未被引擎准入的一轮请求。"""

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


class ActiveReq:
    """已获得 KV 预算、正在推进 Prefill 或 Decode 的在途请求状态。"""

    __slots__ = ("charge", "generated", "matched", "pending", "prefilled", "queued")

    def __init__(
        self,
        queued: Optional[QueuedTurn],
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


def order_waiting(
    waiting: List[QueuedTurn], now: float, schedule_policy: str, lifecycle
) -> List[QueuedTurn]:
    """按调度策略排列等待队列。"""
    if schedule_policy == "fcfs":
        return list(waiting)

    # 已知问题①：lpm 排序会对每条等待请求提前执行一次带副作用的 lifecycle.match()
    # （消耗 rid 序号、递增 reuse_count 与写入 _dispatched），暂原样保留以维持既有口径。
    pairs: List[Tuple[QueuedTurn, Matched]] = []
    for queued in waiting:
        matched = lifecycle.match(
            replace(queued.turn, t=now),
            queued.session_id,
            rid=None,
            task_name=queued.task_name,
            turn_idx=queued.idx - 1,
            arrival_t=queued.due,
        )
        pairs.append((queued, matched))
    pairs.sort(key=lambda pair: -pair[1].hit)
    return [queued for queued, _matched in pairs]


def _charge_chunked_head(
    chunked: Optional[ActiveReq],
    rem_input: int,
    rem_chunk: Optional[int],
) -> Tuple[int, Optional[int], int]:
    """把跨步延续的 chunked 请求优先计入本步 token 与分块预算。"""
    if chunked is None:
        return rem_input, rem_chunk, 0
    remaining = chunked.prompt_len - chunked.prefilled
    charge = remaining if rem_chunk is None else min(remaining, rem_chunk)
    chunked.charge = max(0, charge)
    rem_input -= chunked.charge
    if rem_chunk is not None:
        rem_chunk -= chunked.charge
    return rem_input, rem_chunk, 1


def _within_batch_limits(
    cfg,
    rem_input: int,
    rem_chunk: Optional[int],
    admitted_count: int,
    carried: int,
    running_count: int,
) -> bool:
    if rem_input <= 0:
        return False
    if rem_chunk is not None and rem_chunk <= 0:
        return False
    if cfg.max_prefill_bs is not None and admitted_count + carried >= cfg.max_prefill_bs:
        return False
    active = running_count + admitted_count + carried
    if cfg.max_running_requests is not None and active >= cfg.max_running_requests:
        return False
    return True


def admit_waiting(
    *,
    now: float,
    cfg,
    waiting: List[QueuedTurn],
    running: List[ActiveReq],
    chunked: Optional[ActiveReq],
    budget: PrefillBudget,
    lifecycle,
    send_lags: List[float],
) -> Tuple[List[ActiveReq], bool, int, int]:
    """扫描等待队列、统一匹配并上锁本步准入批次；返回 (admitted, skipped_any, prefill_added, chunked_added)。"""
    rem_input, rem_chunk, carried = _charge_chunked_head(
        chunked,
        cfg.max_prefill_tokens,
        None if cfg.chunked_prefill_size == -1 else cfg.chunked_prefill_size,
    )
    admitted: List[ActiveReq] = []
    skipped_any = False
    prefill_added = 0
    chunked_added = 0

    for queued in order_waiting(waiting, now, cfg.schedule_policy, lifecycle):
        if not _within_batch_limits(
            cfg, rem_input, rem_chunk, len(admitted), carried, len(running)
        ):
            break

        matched = lifecycle.match(
            replace(queued.turn, t=now),
            queued.session_id,
            rid=None,
            task_name=queued.task_name,
            turn_idx=queued.idx - 1,
            arrival_t=queued.due,
        )
        if matched.need > lifecycle.alloc.size:
            pending = lifecycle.admit(matched)
            lifecycle.finish(pending)
            waiting.remove(queued)
            send_lags.append(now - queued.due)
            skipped_any = True
            continue

        extend = max(0, matched.prompt_len - matched.hit)
        is_chunked = rem_chunk is not None and extend > rem_chunk
        page = lifecycle.cache.page_size
        paged_extend = -(-extend // page) * page
        charge = max(0, rem_chunk if is_chunked else paged_extend)
        if not budget.fits(matched.need) and (admitted or chunked is not None):
            break

        state = ActiveReq(queued=queued, matched=matched)
        state.charge = charge
        lifecycle.lock(matched)
        waiting.remove(queued)
        send_lags.append(now - queued.due)
        admitted.append(state)
        budget.reserve(matched.need)
        prefill_added += 1
        if matched.hit + charge < state.prompt_len:
            chunked_added += 1

        # 对齐 SGLang add_one_req 软上限：最后一条扣减后若预算见底才终止扫描。
        rem_input -= charge
        if rem_chunk is not None:
            rem_chunk -= charge
        if rem_input <= 0 or (rem_chunk is not None and rem_chunk <= 0):
            break

    return admitted, skipped_any, prefill_added, chunked_added


def trim_to_capacity(
    *,
    admitted: List[ActiveReq],
    waiting: List[QueuedTurn],
    now: float,
    lifecycle,
    send_lags: List[float],
    reserved_offset: int = 0,
) -> Tuple[int, int]:
    """批量上锁可能把候选自身原本可淘汰的前缀锁死，上锁后复核容量并将超额尾部退回等待队列。"""
    dropped_prefill = 0
    dropped_chunked = 0
    while admitted:
        free = (
            lifecycle.alloc.available_size()
            + lifecycle.cache.evictable_size()
            - reserved_offset
        )
        need = sum(state.matched.need for state in admitted)
        if need <= free:
            break
        state = admitted.pop()
        lifecycle.cache.dec_lock_ref(state.matched.node)
        waiting.insert(0, state.queued)
        dropped_prefill += 1
        if state.matched.hit + state.charge < state.prompt_len:
            dropped_chunked += 1
        # 已知问题②：按值相等从 send_lags 删除可能移除早先相同等待时长的样本，暂原样保留。
        lag = now - state.queued.due
        if lag in send_lags:
            send_lags.remove(lag)
    return dropped_prefill, dropped_chunked
