"""HeterCache 后端 RadixCache 实现：分带清理 + 基于 VD 的跨队列单节点淘汰与后悔率观测。"""

import logging
import time
from typing import Optional

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy
from sglang.srt.mem_cache.heter_cache import bench_bridge, selector
from sglang.srt.mem_cache.heter_cache.bench_bridge import (
    content_key,
    heterlog,
    namespace_key,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


class HeterCacheRadixCache(RadixCache):
    """支持多队列分带、按已兑现价值密度淘汰的 RadixCache 实现。"""

    def __init__(self, params):
        super().__init__(params)
        # 后悔率账本：只在 HETER_REGRET=1 时装配；未装配时本类所有 regret
        # 钩子都以 ``is not None`` 短路，热路径零开销。
        self._regret = bench_bridge.regret_from_env(now=self._now)
        # 仲裁后悔弃权门（控制路径）：装配在策略上，这里只拿引用做接线。
        self._rgate = (
            self.eviction_strategy.regret_gate
            if isinstance(self.eviction_strategy, HeterCacheStrategy)
            else None
        )
        # 当前淘汰循环的裁决来源，由 evict() 在 _evict_one 之前写入：
        # record_victim_source 是整条链释放完才调的，届时节点已经没了。
        self._regret_source = "unknown"

    def reset(self):
        super().reset()
        # 基类 __init__ 末尾会先调一次 reset，那时账本还没装配。
        regret = getattr(self, "_regret", None)
        if regret is not None:
            regret.reset()
        rgate = getattr(self, "_rgate", None)
        if rgate is not None:
            rgate.reset()

    def _record_eviction_regret(self, victim, source: str) -> None:
        """在节点从树上摘除前登记后悔账本与弃权门 stamp（依赖完整父链）。"""
        self._regret_source = source
        regret = getattr(self, "_regret", None)
        if regret is not None:
            self._stamp_eviction(victim, victim, len(victim.value), 1)
        rgate = getattr(self, "_rgate", None)
        if rgate is not None:
            key = self._breakpoint_key(victim, rgate.params.context)
            rgate.on_evict(
                key=key,
                queue=self.eviction_strategy.node_tag(victim).queue_name,
                span=len(victim.value),
            )

    def evict(self, params: EvictParams) -> EvictResult:
        """批量淘汰节点，直至释放至少 params.num_tokens 个 token。"""
        if self.disable:
            return EvictResult()
        if not isinstance(self.eviction_strategy, HeterCacheStrategy):
            return super().evict(params)

        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        available = capacity = 0.0
        if self._needs_capacity:
            available = self.token_to_kv_pool_allocator.available_size()
            capacity = float(self.token_to_kv_pool_allocator.size)

        self._observe_occupancy(available, capacity)

        strategy = self.eviction_strategy
        num_evicted = 0
        while num_evicted < num_tokens:
            victim, source = self._pick_victim(self.evictable_leaves)
            if victim is None:
                break
            self._record_eviction_regret(victim, source)
            released = self._evict_one(victim)
            if released <= 0:
                logger.error(
                    "[struct evict] victim released %d tokens, abort to avoid spin",
                    released,
                )
                break
            num_evicted += released
            strategy.record_victim_source(source, released)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _observe_occupancy(self, available: float, capacity: float) -> None:
        """时钟到期时做一次全树快照，同一份快照先校准账本、再渲染 profile。

        编排留在驱动层：只有这里同时知道 allocator 水位、树与会话账本。
        两个时钟分属两个协作者——校准（tracker）服务账本正确性，打印
        （telemetry）服务观测；任一到期只遍历一次树。VD 的物化挂在校准分支里：
        它必须在 occ 校准之后、用同一时刻的占用做分母。

        Args:
            available: allocator 当前可用 token。
            capacity: 池容量；0 表示没拿到，此时校准走常规窗口。
        """
        tracker, telemetry = self.occupancy, self.telemetry
        strategy = self.eviction_strategy
        want_render = telemetry.wants_profile()
        want_reconcile = tracker.wants_reconcile(
            available,
            capacity,
            normal_window_s=telemetry.PROFILE_INTERVAL,
            has_capacity=capacity > 0,
        )
        if not (want_render or want_reconcile):
            return

        snapshot = self._per_queue_token_count()
        if want_reconcile:
            tracker.reconcile(snapshot, capacity)
            # 先校准 occ 再物化 VD：分子（命中质量）与分母（驻留量）必须同刻。
            strategy.realized.materialize()
        if not want_render:
            return
        telemetry.observe_profile(
            snapshot,
            self.session_registry.summary(),
            share_policy=strategy.node_share,
            reuse_policies=strategy.reuse_evicts,
            sched_occ=tracker.snapshot(),
            values=strategy.realized.densities(tracker.occ),
        )
        if self._regret is not None:
            # 与 profile 同拍输出：报告里的 join/tax 与占用读数同一时刻，
            # 对照实验里可以直接按时间对齐。
            heterlog.info("Regret", self._regret.emit())

    def _pick_victim(self, leaves):
        """转发至 selector 纯策略决策器。"""
        return selector.pick_victim(
            strategy=self.eviction_strategy, leaves=leaves
        )

    def _release_node(self, node) -> int:
        """释放一个节点的全部资源并从树上摘除，返回释放的 token 数。
        1. record_evict最先扣减队列占用
        2. 之后才是物理释放、对外事件与摘树。
        Args:
            node: 待释放的叶子节点。
        Returns:
            该节点释放的token数
        """
        released = len(node.value)
        self.eviction_strategy.record_evict(node)
        self.token_to_kv_pool_allocator.free_segment(node.value, start_pos=0)
        self.kv_events.record_remove(node)
        self._delete_leaf(node)
        return released

    def _evict_one(self, node) -> int:
        """单节点释放（不级联删除祖先），保证每步释放量与决策源一一对应。"""
        return self._release_node(node)

    # ------------------------------------------------------------------
    # 后悔率（regret）观测：树的读法在本模块，账本在 Heter-Bench。
    # ------------------------------------------------------------------
    def _stamp_eviction(self, victim, shallowest, span: int, chain: int) -> None:
        """把一次淘汰决策记进后悔率账本。

        Args:
            victim: 仲裁选出的牺牲节点（决策归属：队列 / 共享数 / 未访问时长）。
            shallowest: 本次级联释放的最浅节点（断点页所在，join 键取它的首页）。
            span: 本次决策腾出的 token 总数。
            chain: 本次级联释放的节点数。
        """
        regret = self._regret
        strategy = self.eviction_strategy
        if self.is_eagle or shallowest.key.is_bigram:
            # bigram 视图下 token 位置是二元组序号，窗口截取与请求侧对不上；
            # 记下来而不是静默丢样本。
            regret.note_skip("evict:bigram")
            return
        start = shallowest.abs_start
        regret.on_evict(
            key=self._breakpoint_key(shallowest, regret.context),
            span=span,
            source=self._regret_source,
            queue=strategy.node_tag(victim).queue_name,
            share=strategy.node_share.share_of(victim),
            chain=chain,
            depth=start // self.page_size,
            elapsed=max(0.0, self._now() - victim.last_access_time),
        )

    def _breakpoint_key(self, node, context: int) -> Optional[int]:
        """断点页内容键：观测账本（regret）与仲裁弃权门（RegretGate）同源。

        两侧的 join 全靠这个键对上；context 必须两侧一致——账本在场时以账本的
        为准（调用方传 ``regret.context``），本门单独启用时才用门自己的参数。

        Args:
            node: 断点页所在节点（必须尚未从树上摘除，父链完整）。
            context: 首页前随带的上下文 token 数。

        Returns:
            内容键；bigram/eagle 视角下返回 None（调用方各自记 skip）。
        """
        if self.is_eagle or node.key.is_bigram:
            return None
        window, base = self._regret_window(node, context)
        return content_key(
            window,
            node.abs_start,
            self.page_size,
            base=base,
            context=context,
            ns=namespace_key(node.key.extra_key, node.key.cache_salt),
        )

    def _regret_window(self, node, context: int):
        """取节点首页及其前 ``context`` 个 token 的连续窗口。

        Args:
            node: 断点页所在节点（必须尚未从树上摘除，父链完整）。
            context: 首页前随带的上下文 token 数（观测账本与弃权门同值）。

        Returns:
            ``(window, base)``：``window`` 是 token 列表，``window[0]`` 的绝对
            位置是 ``base``；链首不足 context 个 token 时从根截取。
        """
        pos = node.abs_start
        need = context
        segments = []
        cursor = node.parent
        while cursor is not None and need > 0 and len(cursor.key) > 0:
            ids = cursor.key.raw_token_ids()
            take = min(len(ids), need)
            segments.append(ids[len(ids) - take:])
            need -= take
            cursor = cursor.parent
        head = []
        for segment in reversed(segments):
            head.extend(segment)
        head.extend(node.key.raw_token_ids()[: self.page_size])
        # head = 祖先链尾部（页前上下文）+ 本页：base 对齐到上下文首 token，
        # 与请求侧 content_key(全量 ids, page_start, base=0) 取到同一窗口。
        return head, pos - (len(head) - self.page_size)

    def _record_match_regret(self, params, matched: int) -> None:
        """将 admission 匹配的断点页上报给后悔账本与弃权门。"""
        regret, gate = self._regret, self._rgate
        key = params.key
        if key.is_bigram or self.is_eagle:
            if regret is not None:
                regret.note_skip("request:bigram")
            if gate is not None:
                gate.note_skip("request:bigram")
            return
        key = key.page_aligned(self.page_size)
        prompt = len(key)
        recompute = prompt - matched
        if recompute <= 0:
            if regret is not None:
                regret.on_request(key=None, recompute=0, prompt=prompt)
            return
        context = regret.context if regret is not None else gate.params.context
        breakpoint_key = content_key(
            key.raw_token_ids(),
            matched,
            self.page_size,
            context=context,
            ns=namespace_key(key.extra_key, key.cache_salt),
        )
        if regret is not None:
            regret.on_request(key=breakpoint_key, recompute=recompute, prompt=prompt)
        if gate is not None:
            gate.on_request(key=breakpoint_key)

    def match_prefix(self, params):
        """admission 匹配后将断点页交给 regret 账本与弃权门（回扫不参与）。"""
        result = super().match_prefix(params)
        regret = getattr(self, "_regret", None)
        rgate = getattr(self, "_rgate", None)
        if (regret is not None or rgate is not None) and params.count_reuse:
            self._record_match_regret(params, int(result.device_indices.numel()))
        return result


def _heter_cache_factory(ctx):
    return HeterCacheRadixCache(ctx.params)


try:
    register_radix_cache_backend("heter_cache", _heter_cache_factory)
except ValueError:
    pass