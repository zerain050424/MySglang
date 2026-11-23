"""按 token 计数的 KV 槽位分配器。"""

from __future__ import annotations


class OutOfTokens(RuntimeError):
    """槽位容量不足，表示驱动层未按先淘汰后分配的顺序执行。"""


class TokenAllocator:
    """仿真用 KV 槽位池，仅暴露缓存层所需的水位读取与释放接口。"""

    __slots__ = ("size", "device", "_used")

    def __init__(self, capacity: int, device: str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"容量必须为正，got {capacity}")
        self.size = int(capacity)
        self.device = device
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    def available_size(self) -> int:
        # 使用真实剩余容量而非 cache.evictable_size()，确保承压判据包含已锁节点与在飞请求。
        return self.size - self._used

    def snapshot(self) -> dict:
        """返回容量水位快照。"""
        return {"size": self.size, "used": self._used, "avail": self.available_size()}

    def alloc(self, tokens: int) -> int:
        """为节点或在飞请求预留槽位，超出容量直接报错以免命中率虚高。"""
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError(f"alloc 的 token 数必须非负，got {tokens}")
        if self._used + tokens > self.size:
            raise OutOfTokens(
                f"容量不足：used={self._used} + {tokens} > size={self.size}；"
                "先 evict 再 alloc"
            )
        self._used += tokens
        return self._used

    def release(self, tokens: int) -> None:
        """归还预留差额（例如并发竞态导致的重复段），防止池账高于树内实际用量。"""
        tokens = int(tokens)
        if tokens <= 0:
            return
        if tokens > self._used:
            raise RuntimeError(
                f"二次释放：release={tokens} 超过已分配 used={self._used}"
            )
        self._used -= tokens

    def free_segment(self, indices=None, start_pos: int = 0) -> None:
        """节点淘汰时的槽位归还入口，释放量恒为 len(indices)。"""
        # 生产中的 start_pos 仅作页对齐断言，不参与释放计数。
        del start_pos
        self.release(len(indices))

    def __repr__(self) -> str:
        return f"<TokenAllocator used={self._used}/{self.size}>"


def tree_tokens(cache) -> int:
    """返回当前驻留于前缀树上的 token 总数（含已锁前缀）。"""
    return cache.evictable_size() + cache.protected_size()
