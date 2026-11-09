"""事件驱动的仿真时钟。

为什么需要它
------------
``HeterCacheStrategy`` 的复用分数（``_reuse_score``）用
``time.monotonic()``。离线回放若用真实时钟，
整条 trace 会在几秒内跑完，所有 ``elapsed`` 都≈0：

* ``reuse_prob = 1 - lognorm_cdf(elapsed)`` 全部≈1，队列间不再有区分度。

于是回放退化成「按 token 数量比大小」，与真机行为无关。
本模块把 sglang 这几个模块内的 ``time`` 换成一个由 trace 时间戳驱动的替身，
使回放既**可复现**又**保留时间局部性语义**。

只替换以下模块的模块级 ``time`` 名字，不碰全局 ``time`` 模块：

    sglang.srt.mem_cache.radix_cache
    sglang.srt.mem_cache.base_prefix_cache
    sglang.srt.mem_cache.evict_policy
    sglang.srt.mem_cache.heter_cache_radix_cache
    sglang.srt.mem_cache.victim_override
    sglang.srt.mem_cache.uni_utils

SLRU 的共享数账本不读时间（只读命中会话），故不在此列；
它的 `pick()` 读 `time.monotonic()`，因此本模块必须覆盖 `victim_override`。
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple

_PATCHED_MODULES = (
    "sglang.srt.mem_cache.radix_cache",
    "sglang.srt.mem_cache.base_prefix_cache",
    "sglang.srt.mem_cache.evict_policy",
    "sglang.srt.mem_cache.heter_cache_radix_cache",
    "sglang.srt.mem_cache.victim_override",
    "sglang.srt.mem_cache.uni_utils",
)

_EPOCH = 1_700_000_000.0  # 固定墙钟起点：只为了让 time() 看起来合理，不参与排序


class SimClock:
    """单调推进的仿真时间（秒）。``set`` 只允许向前，避免 trace 乱序导致时间倒流。"""

    __slots__ = ("_t",)

    def __init__(self, t0: float = 0.0) -> None:
        self._t = float(t0)

    # ---- 由回放驱动调用 ----
    def set(self, t: float) -> None:
        if t > self._t:
            self._t = float(t)

    def advance(self, dt: float) -> None:
        if dt > 0:
            self._t += float(dt)

    @property
    def now(self) -> float:
        return self._t

    # ---- 供被 patch 的模块调用 ----
    def monotonic(self) -> float:
        return self._t

    def perf_counter(self) -> float:
        return self._t

    def time(self) -> float:
        return _EPOCH + self._t

    def sleep(self, dt: float) -> None:  # 极少被调用；保持语义正确
        self.advance(dt)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SimClock({self._t:.3f})"


_installed: Optional[Tuple[SimClock, Dict[str, object]]] = None


def install(clock: SimClock) -> None:
    """把 clock 装进 sglang 相关模块。重复调用会先恢复再装。"""
    global _installed
    if _installed is not None:
        uninstall()
    saved: Dict[str, object] = {}
    for name in _PATCHED_MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:  # 模块不存在（不同版本）就跳过
            continue
        if hasattr(mod, "time"):
            saved[name] = mod.time
            mod.time = clock
    _installed = (clock, saved)


def uninstall() -> None:
    global _installed
    if _installed is None:
        return
    _clock, saved = _installed
    for name, mod_time in saved.items():
        try:
            importlib.import_module(name).time = mod_time
        except Exception:
            pass
    _installed = None
