"""
  SimpleHYBRID 的 radix cache：RadixCache + 两带淘汰。
  one-shot 带（random 整个请求 / 单轮 response 段）永远先清空；
  可调度内容（各 workload 的 prompt 段 + 多轮 response 段）按复用分数
  全局竞争，分数最低者先走。

导入本模块即注册 "--radix-cache-backend=simple_hybrid" 后端。
"""

import logging
import time

from sglang.srt.mem_cache import quota_selector
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


class SimpleHybridRadixCache(RadixCache):
    """RadixCache + 多级队列淘汰"""

    def evict(self, params: EvictParams) -> EvictResult:
        # 禁用 radixcache 时无操作
        if self.disable:
            return EvictResult()
        if not isinstance(self.eviction_strategy, SimpleHybridStrategy):
            return super().evict(params)

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # 配额仲裁需要池容量与可用量（O(1)，不遍历树）；只在配额启用时读 allocator
        available = capacity = 0.0
        if self._needs_capacity:
            available = self.token_to_kv_pool_allocator.available_size()
            capacity = float(self.token_to_kv_pool_allocator.size)
        # profile 打印与配额账本校准共用一次全树遍历（两个时钟分开）
        if self._occupancy_observer is not None and self._occupancy_observer.wants_occupancy(
            available, capacity, has_capacity=self._needs_capacity
        ):
            self._occupancy_observer.observe_occupancy(
                self._per_workload_token_count(),
                self.session_registry.summary(),
                available=available,
                capacity=capacity,
            )

        strategy = self.eviction_strategy
        num_evicted = 0
        # 三级淘汰：
        #   阶段0 one-shot 带永远先清空（带内 LRU），不计配额
        #   阶段1 配额启用时：点名队列 -> 队内挑选（SLRU 分档 / RR / MT）-> 兜底
        #   否则保留原路径：全局 victim_key 竞争 + 改写器链
        while num_evicted < num_tokens:
            victim, source = self._pick_victim(self.evictable_leaves, capacity)
            if victim is None:
                break
            released = self._evict_one(victim)
            if released <= 0:
                # 淘汰必须单调推进；否则 evict 会陷入死循环
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
        """三级淘汰的候选选择；决策本身在 quota_selector.pick_victim。"""
        return quota_selector.pick_victim(
            strategy=self.eviction_strategy,
            leaves=leaves,
            capacity=capacity,
            page_size=self.page_size,
            overrides=self.victim_overrides,
            session_registry=self.session_registry,
        )

    def _evict_one(self, node) -> int:
        """
        淘汰一个叶子并释放其 KV，返回释放的 token 数。
        one-shot 节点向上追溯：把因本次删除而变得无其他子树、未上锁的
        祖先链一次删净 —— 单轮会话的response走后，其上方的prompt链
        不会再产生任何复用，逐叶淘汰只会让死链反复以新叶子的身份暴露，
        驱逐循环为此多跑若干轮。遇到根、分支点（还有其他子树）或上锁
        节点即停，它们仍可能是共享前缀。
        """
        total = 0
        cur = node
        strategy = self.eviction_strategy
        one_shot = strategy.is_one_shot(node)
        while True:
            parent = cur.parent
            self.eviction_strategy.record_evict(cur)
            self.token_to_kv_pool_allocator.free_segment(cur.value, start_pos=0)
            self.kv_events.record_remove(cur)
            total += len(cur.value)
            self._delete_leaf(cur)
            if (one_shot and parent is not None and parent is not self.root_node
                    and parent.lock_ref == 0 and not parent.children
                    and strategy.is_one_shot(parent)):
                cur = parent
                continue
            break
        return total


def _simple_hybrid_factory(ctx):
    """registry 工厂：由 TreeCacheBuildContext 构造 SimpleHybridRadixCache。"""
    return SimpleHybridRadixCache(ctx.params)


try:
    # 模块导入即注册后端，供 --radix-cache-backend=simple_hybrid 使用
    register_radix_cache_backend("simple_hybrid", _simple_hybrid_factory)
except ValueError:
    # 重复导入时 registry 拒绝重名，幂等处理
    pass
