"""已兑现价值密度（VD）：把"这段 KV 还值不值钱"折算成跨队列可比的密度。

两个时间尺度，职责必须分离：

* 视野（``horizon_s``，政策）：三个队列共享的决策视野。VD 的量纲由它定义——
  未来一个视野窗口内，每个驻留 token 预期换回的命中量。跨队列比较只在同一
  视野下成立，所以它不随队列变化。
* 估计窗（``W_q``，统计）：只用来平滑估计命中速率 ``r_q = H_q(W_q) / W_q``，
  取 ``W_q = max(horizon, k*T50_q)``。窗口远小于复用间隔时，窗口里常常一次命中
  都没有，读数在 0 与尖峰之间摆动（chat 的 T50 约 118s、与视野同量级，就是这种
  情形）。W_q 只影响方差，不改变比较尺度。

于是 ``VD_q = r_q * horizon / max(occ_q, floor)``，其中
``r_q * horizon = H_q(W_q) * (horizon / W_q)``。分母是驻留量，所以它是每 token
换回多少重算节省，与队列大小无关。

只用除法读数：H 只在 reconcile 时物化（每个队列一次 exp），淘汰热路径只做除法，
不改变 record_evict 的常数开销。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

from sglang.srt.mem_cache.heter_cache.utils import (
    DEFAULT_VALUE_FLOOR_TOKENS,
    DEFAULT_VALUE_HORIZON_S,
    DEFAULT_VALUE_SMOOTHING_K,
    VALUE_QUEUES,
    estimation_window_s,
)

# exp 的参数下限：再小也只会得到 0，先截断避免无意义的浮点运算。
_MIN_EXP = -50.0


class QueueValueTracker:
    """三层状态：_acc（衰减中的命中质量）、_stamp（上次更新时刻）、
    _snapshot（reconcile 时物化并投影到 horizon 的只读快照）。

    random 不在 VALUE_QUEUES 里，命中直接丢弃：它整行 one-shot，
    不参与仲裁，记账只会稀释其他队列的密度。
    """

    __slots__ = (
        "_acc",
        "_stamp",
        "_snapshot",
        "_window",
        "_projection",
        "_floor",
        "_now",
    )

    def __init__(
        self,
        *,
        horizon_s: float = DEFAULT_VALUE_HORIZON_S,
        smoothing_k: float = DEFAULT_VALUE_SMOOTHING_K,
        floor_tokens: float = DEFAULT_VALUE_FLOOR_TOKENS,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        horizon = float(horizon_s)
        # 估计窗与投影系数只依赖配置，建一次即可：热路径只查字典。
        self._window = {
            q: estimation_window_s(q, horizon, smoothing_k) for q in VALUE_QUEUES
        }
        self._projection = {q: horizon / self._window[q] for q in VALUE_QUEUES}
        self._floor = float(floor_tokens)
        self._now = now if now is not None else time.monotonic
        start = self._now()
        self._acc: Dict[str, float] = {q: 0.0 for q in VALUE_QUEUES}
        self._stamp: Dict[str, float] = {q: start for q in VALUE_QUEUES}
        self._snapshot: Dict[str, float] = {q: 0.0 for q in VALUE_QUEUES}

    def use_clock(self, now: Callable[[], float]) -> None:
        """接管时钟并清账"""
        self._now = now
        self.reset()

    def reset(self) -> None:
        now = self._now()
        for q in VALUE_QUEUES:
            self._acc[q] = 0.0
            self._stamp[q] = now
            self._snapshot[q] = 0.0

    def windows(self) -> Dict[str, float]:
        """各队列的估计窗 W_q（秒），只读副本；观测与单测用。"""
        return dict(self._window)

    def on_hit(self, queue: str, tokens: int) -> None:
        """一次命中：先把旧的命中质量衰减到当前时刻，再累加本次命中量。"""
        if queue not in self._acc:
            return
        now = self._now()
        self._acc[queue] = self._decayed(queue, now) + tokens
        self._stamp[queue] = now

    def materialize(self) -> None:
        """把命中质量衰减到当前时刻，投影到共同视野 horizon 后固化到快照。

        ``_acc`` 保留**未投影**的质量（下一段衰减的起点），``_snapshot`` 存
        ``H_q(W_q)*horizon/W_q``——两者不能混用，否则投影会被重复施加一次。
        """
        now = self._now()
        for q in VALUE_QUEUES:
            mass = self._decayed(q, now)
            self._acc[q] = mass
            self._stamp[q] = now
            self._snapshot[q] = mass * self._projection[q]

    def densities(self, occ: Dict[str, float]) -> Dict[str, float]:
        """VD_q = 未来 horizon 内的预期命中 / max(occ_q, floor)。

        floor=0（默认）时就是 H/occ；occ=0 的队列没有驻留物可评密度，给 0
        即可——它本来也不进 ``queue_order`` 的候选（候选过滤要求 occ>0），
        所以这里返回 0 不会把空队列排到最前。除零保护对 floor=0 是必需的，
        对 floor>0 是死分支。
        """
        out: Dict[str, float] = {}
        for q in VALUE_QUEUES:
            denom = max(float(occ.get(q, 0.0)), self._floor)
            out[q] = self._snapshot[q] / denom if denom > 0.0 else 0.0
        return out

    def snapshot(self) -> Dict[str, float]:
        return dict(self._snapshot)

    def _decayed(self, queue: str, now: float) -> float:
        dt = now - self._stamp[queue]
        if dt <= 0.0:
            # 同刻多事件或时钟回退：不做衰减
            return self._acc[queue]
        exponent = -dt / self._window[queue]
        if exponent < _MIN_EXP:
            return 0.0
        return self._acc[queue] * math.exp(exponent)
