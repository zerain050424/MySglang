"""HeterCacheStrategy 的遥测门面：把账本与事件流的多写、字段映射、开关判断收在一处。

业务层只表达意图（发生了一次命中 / 一次逐出 / 一次裁决），不关心要不要写
profiler、要不要判开关、参数怎么拼；上层的三个 ``record_*`` 因此各剩一行调用。

**属于实验脚手架**，不随 sglang 安装分发：由
``sglang/srt/mem_cache/heter_bench_bridge.py`` 按仓库根定位后加载；Heter-Bench 不在场
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

    __slots__ = ("profiler", "_last_profile", "_seen_rids", "_rid_order")

    def __init__(self, profiler):
        """绑定账本。

        Args:
            profiler: HeterProfiler 实例；策略侧同时保留引用（外部消费者直接读它）。
        """
        self.profiler = profiler
        self._last_profile = time.monotonic()
        # 每请求只归因一次命中的去重账本：同一 rid 的多次 chunked prefill 只算一次。
        self._seen_rids: set = set()
        self._rid_order: deque = deque()

    @staticmethod
    def _emitting() -> bool:
        """是否值得进入事件构造函数。"""
        return heterlog.capturing or heterlog.enabled

    def on_hit(self, tag, hit_tokens: int, req_id=None) -> None:
        """一次前缀命中；同一 rid 只记一次（chunked prefill 会重复进入）。

        Args:
            tag: 节点标签。
            hit_tokens: 本次命中的 token 数。
            req_id: 请求 id；为空时直接忽略（与旧行为一致）。
        """
        if req_id is None or req_id in self._seen_rids:
            return
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

    def wants_profile(self) -> bool:
        """profile 打印时钟是否到期。与配额校准时钟完全独立。"""
        return time.monotonic() - self._last_profile >= self.PROFILE_INTERVAL

    def observe_profile(
        self,
        queue_occ: dict,
        session_summary,
        *,
        slru_policy,
        mt_policy,
        quota_occ: dict,
        quota_targets: dict,
        quota_caps: dict,
    ) -> None:
        """渲染并输出一行 profile，同时推进打印时钟。

        只做渲染：目标配额与上限由调用方算好传进来，这里不碰配额数学。
        """
        self._last_profile = time.monotonic()
        rendered = self.profiler.summary(
            queue_occ,
            session_summary,
            slru_policy=slru_policy,
            mt_policy=mt_policy,
            quota_occ=quota_occ,
            quota_targets=quota_targets,
            quota_caps=quota_caps,
        )
        heterlog.info("Profile", f"[HeterCache profile]\n{rendered}")

    def reset(self) -> None:
        """清空跨 run 的累计量与去重账本（缓存 flush / reset 时调用）。"""
        self.profiler.reset()
        self._seen_rids.clear()
        self._rid_order.clear()
