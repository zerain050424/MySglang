"""引擎批预算检查与延迟分配未来余量计算。"""

from __future__ import annotations

from typing import Optional, Sequence


class PrefillBudget:
    """跨 waiting 扫描阶段的批量可准入 KV 预算追踪器。"""

    __slots__ = ("_offset", "alloc", "cache")

    def __init__(self, alloc, cache, reserved_offset: int = 0) -> None:
        self.alloc = alloc
        self.cache = cache
        # reserved_offset 包含当前批已承诺配额以及正在运行的懒分配请求尚未占槽的未来需求。
        self._offset = int(reserved_offset)

    def remaining_total(self) -> int:
        return self.alloc.available_size() + self.cache.evictable_size() - self._offset

    def fits(self, need: int) -> bool:
        return need <= self.remaining_total()

    def reserve(self, need: int) -> None:
        self._offset += need


def unallocated_reserved(running: Sequence, chunked: Optional[object]) -> int:
    """计算已在途请求尚未完成物理占槽的未来输出或后续分块余量，防止批准入超额导致死锁。"""
    states = list(running) + ([chunked] if chunked is not None else [])
    return sum(
        (s.matched.hit + s.matched.need) - s.pending.allocated_len
        for s in states
        if s.pending is not None and not s.pending.skipped
    )
