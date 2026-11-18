"""HeterCache 后端 RadixCache 实现。

在原生 RadixCache 基础上叠加分带与按已兑现价值密度（VD）的跨队列淘汰：
- 优先清理无复用价值的 One-shot 临时节点；
- 可调度节点委托给 `selector.pick_victim`，按 VD 仲裁。
本模块专注执行层面的内存回收、Radix 树拓扑维护与级联修剪。

后悔率（regret）观测也挂在执行层：淘汰决策与 admission 匹配都只有这里
同时拿得到树拓扑与 token 序列。树的读法（窗口截取、页号、命名空间）留在
本模块；索引、归因、聚合与渲染在 Heter-Bench 的账本里，经由 ``bench_bridge``
装配（``HETER_REGRET=1`` 时才有账本，未装配时所有钩子短路）。
"""

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

    def evict(self, params: EvictParams) -> EvictResult:
        """批量淘汰节点，直至释放至少 params.num_tokens 个 Token。

        执行流程：
        1. 观测对齐：淘汰前触发快照采样，校准队列占用并驱动指标输出；
        2. 仲裁选点：调用 _pick_victim 决出牺牲节点及决策源；
        3. 物理释放：调用 _evict_one 执行显存释放与树结构修剪。
        """
        if self.disable:
            return EvictResult()
        # 非 HeterCache 策略回退至基类默认淘汰
        if not isinstance(self.eviction_strategy, HeterCacheStrategy):
            return super().evict(params)

        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        available = capacity = 0.0
        if self._needs_capacity:
            available = self.token_to_kv_pool_allocator.available_size()
            capacity = float(self.token_to_kv_pool_allocator.size)

        # 淘汰前同步全树占用，防止 VD 仲裁基于过期水位做出错误倾斜
        self._observe_occupancy(available, capacity)

        strategy = self.eviction_strategy
        num_evicted = 0
        while num_evicted < num_tokens:
            victim, source = self._pick_victim(self.evictable_leaves)
            if victim is None:
                break
            # stamp 要挂在裁决来源上，而 record_victim_source 在链释放完之后
            # 才调用；先存下来，_evict_one 内的树读法要用。
            self._regret_source = source
            # 观测账本与弃权门必须在 _evict_one 之前记 —— stamp 需要完整父链。
            # 单节点淘汰没有级联，victim 就是断点页所在的最浅节点，chain=1。
            if self._regret is not None:
                self._stamp_eviction(victim, victim, len(victim.value), 1)
            if self._rgate is not None:
                key = self._breakpoint_key(victim, self._rgate.params.context)
                self._rgate.on_evict(
                    key=key,
                    queue=strategy.node_tag(victim).queue_name,
                    span=len(victim.value),
                )
            released = self._evict_one(victim)
            if released <= 0:
                # 释放量为 0 表明节点状态异常，中止循环以防空转死锁（Spin）
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
        """释放**一个**叶子节点，不向上级联清理祖先链。

        历史：曾沿 One-shot 私有祖先链向上级联删除（遇根 / 上锁 / 分叉即止）。
        放弃级联的理由（exp3 scan cell 实测）：
        * 放大不可控：one_shot 淘汰 14.57M 里只有 2.67M 是被选中的节点本身，
          其余 11.9M 全是级联祖先——一次决策实际释放量是被选节点的 5.5 倍；
        * 越过价值判断：级联只问「祖先是否也 one-shot」，实测连带删掉 2.88M
          个 struct 标签 token，等于用 one-shot 的名义清掉了别的队列；
        * 决策与结果不可归因：victim_sources 记的是选中的那个节点，释放量却
          包含整条链，配额/密度账本没法按决策复盘。
        现在退回「一次淘汰一个节点」：``evict()`` 的 while 循环本来就按
        ``num_evicted < num_tokens`` 反复调用本方法，牺牲粒度回到节点级，
        每一步都可归因、可复现。

        单个节点的释放细节见 ``_release_node``。
        """
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

    def match_prefix(self, params):
        """admission 匹配后把断点页交给 regret 账本 join。

        断点页 = 首个未命中页；它若在坟场里，这次 miss 就有明确的责任决策。
        回扫（``count_reuse=False``）不是重读，不进税的分母也不参与 join。
        """
        result = super().match_prefix(params)
        regret = self._regret
        gate = self._rgate
        if (regret is None and gate is None) or not params.count_reuse:
            return result
        key = params.key
        if key.is_bigram or self.is_eagle:
            if regret is not None:
                regret.note_skip("request:bigram")
            if gate is not None:
                gate.note_skip("request:bigram")
            return result
        key = key.page_aligned(self.page_size)
        prompt = len(key)
        matched = int(result.device_indices.numel())
        recompute = prompt - matched
        if recompute <= 0:
            if regret is not None:
                regret.on_request(key=None, recompute=0, prompt=prompt)
            return result
        window = key.raw_token_ids()
        # 观测账本与弃权门同键：context 以账本为准（在场时），键才对得上。
        context = regret.context if regret is not None else gate.params.context
        breakpoint_key = content_key(
            window,
            matched,
            self.page_size,
            context=context,
            ns=namespace_key(key.extra_key, key.cache_salt),
        )
        if regret is not None:
            regret.on_request(
                key=breakpoint_key,
                recompute=recompute,
                prompt=prompt,
            )
        if gate is not None:
            gate.on_request(key=breakpoint_key)
        return result


def _heter_cache_factory(ctx):
    return HeterCacheRadixCache(ctx.params)


try:
    register_radix_cache_backend("heter_cache", _heter_cache_factory)
except ValueError:
    pass