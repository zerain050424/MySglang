"""按 token 计数的分配器：把「显存池」换成两个整数。

生产环境的 allocator 管的是 KV 槽位（分配、回收、页对齐、跨 rank 分片），
但前缀缓存只从它身上读四样东西：

    size               池容量（evict() 用来判承压、profile 用来算权重占比）
    device             折成 torch.device
    available_size()   当前可用 token（_observe_occupancy 用）
    free_segment()     节点被淘汰时归还

其余（页表、字节计量、多租户 class）对缓存决策没有影响，所以仿真只需要一个
计数器。**不要**用 ``cache.evictable_size()`` 冒充 ``available_size()``：
``evict()`` 里的承压判据要的是真实水位（含在飞请求与已锁节点），
用树内可淘汰量替代会让承压窗口永远不触发。

不变量（驱动层每一步都该成立）::

    allocator.used == cache.evictable_size() + cache.protected_size()

树上每个 token 都占一个槽位，反过来槽位里只有树内 token（顺序模型没有在飞请求）。
见 :func:`tree_tokens`；M3 的 runner 会逐请求校验它。
"""

from __future__ import annotations


class OutOfTokens(RuntimeError):
    """容量不足。出现即驱动层的顺序错了：必须先 evict 再 alloc。"""


class TokenAllocator:
    """最小 token 分配器（CPU 仿真用）。"""

    __slots__ = ("size", "device", "_used")

    def __init__(self, capacity: int, device: str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"容量必须为正，got {capacity}")
        self.size = int(capacity)
        self.device = device
        self._used = 0

    # ---- 读侧 ----
    @property
    def used(self) -> int:
        return self._used

    def available_size(self) -> int:
        return self.size - self._used

    def snapshot(self) -> dict:
        """供指标落盘的水位快照。"""
        return {"size": self.size, "used": self._used, "avail": self.available_size()}

    # ---- 写侧 ----
    def alloc(self, tokens: int) -> int:
        """为即将写入树的 token 预留槽位，返回新的 used。

        Raises:
            OutOfTokens: 预留后超过容量。缓存层不会替你腾地方——淘汰是驱动层
                的职责（生产里是 allocator 的 shortfall 回调），这里直接失败，
                避免仿真静默超配而让命中率虚高。
        """
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
        """直接归还 ``tokens`` 个槽位（不走 indices）。

        驱动层用它对账：插入前预留的是「页对齐后的全序列 - prompt 命中」，
        而实际树增长可能更少（response 段也可能是已缓存内容），差额应当立即归还，
        否则池账会比树高出一截。
        """
        tokens = int(tokens)
        if tokens <= 0:
            return
        if tokens > self._used:
            raise RuntimeError(
                f"二次释放：release={tokens} 超过已分配 used={self._used}"
            )
        self._used -= tokens

    def free_segment(self, indices=None, start_pos: int = 0) -> None:
        """归还一段槽位。签名与生产 allocator 一致，由 ``_release_node`` / ``_adopt`` 调用。

        ``start_pos`` 在生产里只是**页对齐断言**（见
        ``BaseTokenToKVPoolAllocator.free_segment``），不参与释放计数：释放量恒为
        ``len(indices)``。把它当偏移量会少还一截，池账会静默偏高。
        """
        self.release(len(indices))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<TokenAllocator used={self._used}/{self.size}>"


def tree_tokens(cache) -> int:
    """树上驻留的 token 总数（可淘汰 + 已锁），用于池账不变量校验。"""
    return cache.evictable_size() + cache.protected_size()
