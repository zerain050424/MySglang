"""HeterCacheStrategy 的遥测门面：把账本与事件流的多写、字段映射、开关判断收在一处。

业务层只表达意图（发生了一次命中 / 一次逐出 / 一次裁决），不关心要不要写
profiler、要不要判开关、参数怎么拼；上层的三个 ``record_*`` 因此各剩一行调用。

**属于实验脚手架**，不随 sglang 安装分发：由
``sglang/srt/mem_cache/heter_cache/bench_bridge.py`` 按仓库根定位后加载；Heter-Bench 不在场
时该桥直接报错，不留空实现。

开关判据是 ``capturing or enabled``，两者对应不同出口：

* ``capturing``：事件流落盘（``HETER_STRATEGY_EVENTS``），默认关；
* ``enabled``：文本 debug 日志（``STRAT_LOG_LEVEL=debug``），默认关。

只判 ``enabled`` 会让开了事件流但没开 debug 的跑法丢掉全部事件；只判
``capturing`` 又会让纯文本排查时看不到事件行。两个都不满足时直接返回，
避免白构造一份会被丢弃的事件字典。
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

from heter_logger import heterlog


class CacheTelemetry:
    """策略侧观测的唯一入口：把账本与事件流的多写、字段映射、开关判断收在一处。

    业务层只表达意图（发生了一次命中 / 一次逐出 / 一次裁决），不再关心
    "要不要写 profiler、要不要判开关、参数怎么拼"。上层的三个 ``record_*``
    因此各剩一行调用。

    放在本模块而不是策略模块：这里是 sglang 与 Heter-Bench 观测实现的唯一接缝，
    装配与降级都发生在这里，收敛点也应该在这里。

    开关判据是 ``capturing or enabled``，两者对应不同出口：

    * ``capturing``：事件流落盘（``HETER_STRATEGY_EVENTS``），默认关；
    * ``enabled``：文本 debug 日志（``STRAT_LOG_LEVEL=debug``），默认关。

    只判 ``enabled`` 会让开了事件流但没开 debug 的跑法丢掉全部事件；只判
    ``capturing`` 又会让纯文本排查时看不到事件行。两个都不满足时直接返回，
    避免白构造一份会被丢弃的事件字典。
    """

    # 顺带在这里定义 profile 的时钟与命中去重上限：它们都只服务观测。
    PROFILE_INTERVAL = 30.0
    MAX_SEEN_RIDS = 100_000

    __slots__ = ("profiler", "_last_profile", "_seen_rids", "_rid_order", "_now")

    def __init__(self, profiler, now: Optional[Callable[[], float]] = None):
        """绑定账本。

        Args:
            profiler: HeterProfiler 实例；策略侧同时保留引用（外部消费者直接读它）。
            now: 时间源；``None`` 表示 ``time.monotonic``。仿真器注入虚拟时钟后，
                profile 间隔按虚拟秒推进，而不是墙钟。
        """
        self.profiler = profiler
        self._now = now if now is not None else time.monotonic
        self._last_profile = self._now()
        # 每请求只归因一次命中的去重账本：同一 rid 的多次 chunked prefill 只算一次。
        self._seen_rids: set = set()
        self._rid_order: deque = deque()

    @staticmethod
    def _emitting() -> bool:
        """是否值得进入事件构造函数。"""
        return heterlog.capturing or heterlog.enabled

    def on_hit(self, tag, hit_tokens: int, req_id=None) -> bool:
        """一次前缀命中；同一 rid 只记一次（chunked prefill 会重复进入）。

        Args:
            tag: 节点标签。
            hit_tokens: 本次命中的 token 数。
            req_id: 请求 id；为空时直接忽略（与旧行为一致）。

        Returns:
            是否为该 rid 的首次命中。VD 账本据此去重，避免同一请求被记多次。
        """
        if req_id is None or req_id in self._seen_rids:
            return False
        self._seen_rids.add(req_id)
        self._rid_order.append(req_id)
        if len(self._rid_order) > self.MAX_SEEN_RIDS:
            self._seen_rids.discard(self._rid_order.popleft())

        self.profiler.record_hit(tag.queue_name, hit_tokens)
        if self._emitting():
            heterlog.hit(
                queue=tag.queue_name,
                tag=tag.value,
                hit_tokens=hit_tokens,
                req=req_id,
            )
        return True

    def on_evict(self, node_id: int, tag, tokens: int, one_shot: bool) -> None:
        """一次节点逐出（级联删除时逐点上报）。"""
        self.profiler.record_evict(tag.queue_name, tokens, one_shot=one_shot)
        if self._emitting():
            heterlog.evict(
                node_id=node_id,
                queue=tag.queue_name,
                tag=tag.value,
                tokens=tokens,
                one_shot=one_shot,
            )

    def on_victim_source(self, source: str, tokens: float) -> None:
        """一次淘汰裁决来源。"""
        self.profiler.record_victim_source(source, tokens)
        if self._emitting():
            heterlog.victim_source(source, tokens)

    def use_clock(self, now: Callable[[], float]) -> None:
        """接管时间源，并以当前时钟重新起算打印间隔。

        重建起点而不是保留旧值：旧值来自另一个时钟（墙钟或更早的虚拟起点），
        与新 ``now()`` 不同量级，留着会让第一行 profile 立刻到期或被无限推迟。
        """
        self._now = now
        self._last_profile = now()

    def wants_profile(self) -> bool:
        """profile 打印时钟是否到期。与账本校准时钟完全独立。"""
        return self._now() - self._last_profile >= self.PROFILE_INTERVAL

    def observe_profile(
        self,
        queue_occ: dict,
        session_summary,
        *,
        share_policy,
        reuse_policies=None,
        sched_occ: dict = None,
        values: dict = None,
    ) -> None:
        """渲染并输出一行 profile，同时推进打印时钟。

        只做渲染：可调度占用与已兑现价值密度由调用方传入，这里不做任何仲裁数学。
        """
        self._last_profile = self._now()
        rendered = self.profiler.summary(
            queue_occ,
            session_summary,
            share_policy=share_policy,
            reuse_policies=reuse_policies,
            sched_occ=sched_occ,
            values=values,
        )
        heterlog.info("Profile", f"[HeterCache profile]\n{rendered}")

    def reset(self) -> None:
        """清空跨 run 的累计量与去重账本（缓存 flush / reset 时调用）。"""
        self.profiler.reset()
        self._seen_rids.clear()
        self._rid_order.clear()
