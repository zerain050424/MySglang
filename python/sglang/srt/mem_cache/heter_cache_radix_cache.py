"""HeterCache 后端 RadixCache 实现。

在原生 RadixCache 基础上叠加分带与配额淘汰机制：
- 优先清理无复用价值的 One-shot 临时节点；
- 可调度节点委托给 `quota_selector.pick_victim` 进行跨队列仲裁。
本模块专注执行层面的内存回收、Radix 树拓扑维护与级联修剪。
"""

import logging
import time

from sglang.srt.mem_cache import quota_selector
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


class HeterCacheRadixCache(RadixCache):
    """支持多队列分带配额淘汰的 RadixCache 实现。"""

    def evict(self, params: EvictParams) -> EvictResult:
        """批量淘汰节点，直至释放至少 params.num_tokens 个 Token。

        执行流程：
        1. 观测对齐：淘汰前触发快照采样，校准配额占用并驱动指标输出；
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

        # 淘汰前同步全树占用，防止配额仲裁基于过期水位做出错误倾斜
        if self._occupancy_observer is not None and self._occupancy_observer.wants_occupancy(
            available, capacity, has_capacity=self._needs_capacity
        ):
            self._occupancy_observer.observe_occupancy(
                self._per_queue_token_count(),
                self.session_registry.summary(),
                available=available,
                capacity=capacity,
            )

        strategy = self.eviction_strategy
        num_evicted = 0
        while num_evicted < num_tokens:
            victim, source = self._pick_victim(self.evictable_leaves, capacity)
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

    def _pick_victim(self, leaves, capacity: float):
        """转发至 quota_selector 纯策略决策器。"""
        return quota_selector.pick_victim(
            strategy=self.eviction_strategy,
            leaves=leaves,
            # 账本由驱动层持有，这里传只读投影给纯决策器。
            occ=self.occupancy.occ,
            capacity=capacity,
            page_size=self.page_size,
        )

    def _release_node(self, node) -> int:
        """释放一个节点的全部资源并从树上摘除，返回释放的 token 数。

        四个动作的顺序不可调换：

        1. ``record_evict`` 必须最先。它扣减配额账本，而 ``_pick_victim``
           在同一个淘汰循环里每轮都读这份账本——
           把扣减推后会让下一轮按错误的占用选受害者；
        2. 之后才是物理释放、对外事件与摘树。

        Args:
            node: 待释放的叶子节点。

        Returns:
            该节点释出的 token 数（先取值再释放，value 不会被释放动作改写）。
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