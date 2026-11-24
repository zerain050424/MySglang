"""队列占用账本：维护各可调度队列的 token 占用，支持增量更新与全树快照校准。"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from sglang.srt.mem_cache.heter_cache.utils import VALUE_QUEUES

# 承压判据：可用容量占池容量的比例低于它，就把校准窗口缩短到
# ``PRESSURE_WINDOW_S``。承压时水位变化最快，旧窗口会让仲裁按过期占用点名。
PRESSURE_AVAILABLE_RATIO = 0.05
PRESSURE_WINDOW_S = 3.0


class QueueOccupancyTracker:
    """三队列占用账本计数,只计可调度内容——one-shot 节点在写入方就已被排除"""

    __slots__ = ("_occ", "_capacity", "_last_reconcile", "_now", "_under_pressure")

    def __init__(self, now: Optional[Callable[[], float]] = None) -> None:
        self._occ: Dict[str, float] = {q: 0.0 for q in VALUE_QUEUES}
        self._capacity = 0.0
        self._now = now if now is not None else time.monotonic
        # -inf 而不是 0：首次淘汰必须校准（真时钟下 now-0 本来就巨大，
        # 而虚拟时钟从 0 起算时 now-0=0 会让第一次淘汰漏掉校准）。
        self._last_reconcile = float("-inf")
        self._under_pressure = False

    def use_clock(self, now: Callable[[], float]) -> None:
        """接管时间源（默认墙钟下等价于无操作）；不重置上一行校准时刻。"""
        self._now = now

    @property
    def occ(self) -> Dict[str, float]:
        """活引用，供热路径只读遍历；调用方不得修改。"""
        return self._occ

    @property
    def capacity(self) -> float:
        """最近一次校准得到的池容量（Token）。"""
        return self._capacity

    @property
    def under_pressure(self) -> bool:
        """当前池是否处于承压态（available / capacity < PRESSURE_AVAILABLE_RATIO）。"""
        return self._under_pressure

    def update_pressure(self, available: float, capacity: float) -> bool:
        """根据最新 allocator 水位刷新承压状态。"""
        self._under_pressure = capacity > 0 and (
            available / capacity < PRESSURE_AVAILABLE_RATIO
        )
        return self._under_pressure

    def snapshot(self) -> Dict[str, float]:
        """副本，供渲染或跨线程使用。"""
        return dict(self._occ)

    def insert(self, queue: str, tokens: int) -> None:
        """新节点挂载时增量计入。

        Args:
            queue: 队列名；不在 VALUE_QUEUES 内时忽略（random 不参与 VD 仲裁）。
            tokens: 该节点的 token 数。
        """
        if queue in self._occ:
            self._occ[queue] += tokens

    def evict(self, queue: str, tokens: int) -> None:
        """节点逐出时增量扣减，下界为 0。
        必须早于同一轮淘汰循环里的下一次_pick_victim：仲裁按这份占用选队列
        """
        if queue in self._occ:
            self._occ[queue] = max(0.0, self._occ[queue] - tokens)

    def wants_reconcile(
        self,
        available: float,
        capacity: float,
        *,
        normal_window_s: float,
        has_capacity: bool,
    ) -> bool:
        """校准时钟是否到期
        Args:
            available: 当前可用 token；has_capacity为 False 时不参考。
            capacity: 池容量；0 表示还没拿到。
            normal_window_s: 常规校准窗口（秒）。
            has_capacity: 调用方是否真的拿到了可用量。
        """
        now = self._now()
        if not has_capacity or capacity <= 0:
            return now - self._last_reconcile >= normal_window_s
        if available / capacity < PRESSURE_AVAILABLE_RATIO:
            return now - self._last_reconcile >= PRESSURE_WINDOW_S
        return now - self._last_reconcile >= normal_window_s

    def reconcile(self, snapshot: dict, capacity: float) -> None:
        """用全树快照校准，吸收增量维护跟不上的漂移。

        Args:
            snapshot: {queue: {"total": t, "one_shot": o}}，来自全树遍历。
            capacity: 本次拿到的池容量（Token）；拿不到时传 0，容量保持不变。
        """
        for q in VALUE_QUEUES:
            rec = snapshot.get(q) or {}
            total = float(rec.get("total", 0.0))
            one_shot = float(rec.get("one_shot", 0.0))
            self._occ[q] = max(0.0, total - one_shot)
        if capacity > 0:
            self._capacity = float(capacity)
        self._last_reconcile = self._now()

    def reset(self) -> None:
        """清空账本（缓存 flush / 跨 run 复用同一进程时调用）。"""
        for q in VALUE_QUEUES:
            self._occ[q] = 0.0
        self._last_reconcile = float("-inf")
        self._under_pressure = False
