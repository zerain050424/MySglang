"""HeterCache 后端 RadixCache 实现。

在原生 RadixCache 基础上叠加分带与按已兑现价值密度（VD）的跨队列淘汰：
- 优先清理无复用价值的 One-shot 临时节点；
- 可调度节点委托给 `queue_selector.pick_victim`，按 VD 仲裁。
本模块专注执行层面的内存回收、Radix 树拓扑维护与级联修剪。
"""

import logging
import time

from sglang.srt.mem_cache import queue_selector
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


class HeterCacheRadixCache(RadixCache):
    """支持多队列分带、按已兑现价值密度淘汰的 RadixCache 实现。"""

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
            slru_policy=strategy.slru_evict,
            multi_agentic_policy=strategy.multi_agentic_evict,
            sched_occ=tracker.snapshot(),
            values=strategy.realized.densities(tracker.occ),
        )

    def _pick_victim(self, leaves):
        """转发至 queue_selector 纯策略决策器。"""
        return queue_selector.pick_victim(
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
        """释放目标叶子节点，并对 One-shot 私有祖先链执行向上级联清理。

        优化说明：
        单轮会话的叶子被逐出后，其上游私有 Prompt 节点已失去复用价值。
        若不级联清理，这些孤立祖先会在下一轮重新变为叶子引发反复单步淘汰。
        沿祖先链向上连带删除，遇根节点、上锁节点或分叉点（含其它活跃子树）即止。

        单个节点的释放细节见 ``_release_node``；本方法只负责“往上走到哪”的判断。
        """
        total = 0
        cur = node
        strategy = self.eviction_strategy
        one_shot = strategy.is_one_shot(node)
        while True:
            parent = cur.parent
            total += self._release_node(cur)
            # 级联剪枝条件：整条祖先链均为未上锁且无其它分支的 One-shot 私有节点
            if (one_shot and parent is not None and parent is not self.root_node
                    and parent.lock_ref == 0 and not parent.children
                    and strategy.is_one_shot(parent)):
                cur = parent
                continue
            break
        return total


def _heter_cache_factory(ctx):
    return HeterCacheRadixCache(ctx.params)


try:
    register_radix_cache_backend("heter_cache", _heter_cache_factory)
except ValueError:
    pass