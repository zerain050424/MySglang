"""已兑现价值密度（VD）：跨队列仲裁唯一的量。

``VD_q = H_q / max(occ_q, floor)``，``H_q`` 是以 ``tau_s`` 为时间常数、按可注入
时钟衰减的命中 token 质量。三个性质决定了它的用法：

* 已兑现：只统计过去一个 ``tau`` 内真实命中过的量，不预测未来（预测是 G 项的事）；
* 可跨队列比较：分母是驻留量，所以它是"每 token 换回多少重算节省"，与队列大小无关；
* 只用除法读数：``H`` 只在 reconcile 时物化（每个队列一次 ``exp``），淘汰热路径
  只做除法，不改变 ``record_evict`` 的常数开销。

不维护累计值：``HeterProfiler._total_hit_tokens`` 是累计量，只服务 profile 的
``hit`` 列；单调递增的累计值无法用于排序。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

from sglang.srt.mem_cache.heter_utils import (
    DEFAULT_VALUE_FLOOR_TOKENS,
    DEFAULT_VALUE_TAU_S,
    VALUE_QUEUES,
)

# exp 的参数下限：再小也只会得到 0，先截断避免无意义的浮点运算。
_MIN_EXP = -50.0


class QueueValueTracker:
    """三层状态：``_acc``（衰减中的命中质量）、``_stamp``（上次更新时刻）、
    ``_snapshot``（reconcile 时物化的只读快照）。

    ``random`` 不在 ``VALUE_QUEUES`` 里，命中直接丢弃：它整行 one-shot，
    不参与仲裁，记账只会稀释其他队列的密度。
    """

    __slots__ = ("_acc", "_stamp", "_snapshot", "_tau", "_floor", "_now")

    def __init__(
        self,
        *,
        tau_s: float = DEFAULT_VALUE_TAU_S,
        floor_tokens: float = DEFAULT_VALUE_FLOOR_TOKENS,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        self._tau = float(tau_s)
        self._floor = float(floor_tokens)
        self._now = now if now is not None else time.monotonic
        start = self._now()
        self._acc: Dict[str, float] = {q: 0.0 for q in VALUE_QUEUES}
        self._stamp: Dict[str, float] = {q: start for q in VALUE_QUEUES}
        self._snapshot: Dict[str, float] = {q: 0.0 for q in VALUE_QUEUES}

    def use_clock(self, now: Callable[[], float]) -> None:
        """接管时钟并清账：跨时钟的 ``elapsed`` 没有意义（虚拟时钟从 0 起算）。"""
        self._now = now
        self.reset()

    def reset(self) -> None:
        now = self._now()
        for q in VALUE_QUEUES:
            self._acc[q] = 0.0
            self._stamp[q] = now
            self._snapshot[q] = 0.0

    def on_hit(self, queue: str, tokens: int) -> None:
        """一次命中：先把旧质量衰减到当前时刻，再累加本次命中量。"""
        if queue not in self._acc:
            return
        now = self._now()
        self._acc[queue] = self._decayed(queue, now) + tokens
        self._stamp[queue] = now

    def materialize(self) -> None:
        """把 ``_acc`` 衰减到当前时刻并固化到 ``_snapshot``。

        必须同时回写 ``_acc`` 与 ``_stamp``：只更新快照会让下一次 ``on_hit``
        从"未衰减的质量 + 零间隔"起算，凭空多记一段历史。
        """
        now = self._now()
        for q in VALUE_QUEUES:
            value = self._decayed(q, now)
            self._acc[q] = value
            self._snapshot[q] = value
            self._stamp[q] = now

    def densities(self, occ: Dict[str, float]) -> Dict[str, float]:
        """``VD_q = H_q / max(occ_q, floor)``；占用为 0 时按 floor 计算。"""
        return {
            q: self._snapshot[q] / max(float(occ.get(q, 0.0)), self._floor)
            for q in VALUE_QUEUES
        }

    def snapshot(self) -> Dict[str, float]:
        """``H`` 的副本（供 profile 渲染；仲裁只读 ``densities``）。"""
        return dict(self._snapshot)

    def _decayed(self, queue: str, now: float) -> float:
        dt = now - self._stamp[queue]
        if dt <= 0.0:
            # 同刻多事件或时钟回退：不做衰减，保持累加语义。
            return self._acc[queue]
        exponent = -dt / self._tau
        if exponent < _MIN_EXP:
            return 0.0
        return self._acc[queue] * math.exp(exponent)
