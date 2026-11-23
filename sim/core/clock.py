"""事件驱动的单调虚拟时钟。"""

from __future__ import annotations


class VirtualClock:
    """对齐 time.monotonic 协议的单调虚拟秒表。"""

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        return self._t

    @property
    def now(self) -> float:
        return self._t

    def advance_to(self, t: float) -> float:
        """推进虚拟时钟至 t；禁止回退以防止节点驻留与老化时间出现负值。"""
        t = float(t)
        if t < self._t:
            raise ValueError(
                f"虚拟时钟不能回退：now={self._t:.6f}, 请求 {t:.6f}。"
            )
        self._t = t
        return self._t

    def step(self, dt: float) -> float:
        """向前步进 dt 秒。"""
        if dt < 0:
            raise ValueError(f"step 必须非负，got {dt}")
        return self.advance_to(self._t + dt)

    def __repr__(self) -> str:
        return f"<VirtualClock t={self._t:.3f}>"
