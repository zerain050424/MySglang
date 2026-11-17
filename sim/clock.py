"""虚拟时钟：把「秒」从墙钟换成事件时间轴。

缓存层的所有判据都是时间差——老化 TTL（120/180/300s）、one-shot 时间豁免
（``ONE_SHOT_IDLE_S=20``）、队内复用窗口（H=90s）、承压校准窗口（3s）、
profile 间隔（30s）。真时钟下这些只能靠 ``sleep`` 推进，回放一段 1 小时的会话
就得等 1 小时；注入本时钟后一步跳过。

契约（与 ``time.monotonic`` 对齐，缓存层不该看出区别）：

* 无参调用返回当前虚拟秒；
* **单调不减**：``advance_to`` 回退直接报错，而不是静默倒退——倒退会让
  ``now - node.last_access_time`` 变负，aging 判定无声失效；
* 不自动推进：时间只由驱动层（``advance_to`` / ``step``）决定，避免「跑得快就
  老化得快」这种隐式耦合。
"""

from __future__ import annotations


class VirtualClock:
    """可步进的单调秒表。"""

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    # ---- 时钟协议 ----
    def __call__(self) -> float:
        return self._t

    @property
    def now(self) -> float:
        return self._t

    # ---- 推进 ----
    def advance_to(self, t: float) -> float:
        """前进到绝对时刻 ``t``；同一时刻重复调用是幂等的。"""
        t = float(t)
        if t < self._t:
            raise ValueError(
                f"虚拟时钟不能回退：now={self._t:.6f}, 请求 {t:.6f}。"
                "回退会让 now - node.last_access_time 变负，aging 判定失效。"
            )
        self._t = t
        return self._t

    def step(self, dt: float) -> float:
        """前进 ``dt`` 秒（``dt >= 0``）。"""
        if dt < 0:
            raise ValueError(f"step 必须非负，got {dt}")
        return self.advance_to(self._t + dt)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<VirtualClock t={self._t:.3f}>"
