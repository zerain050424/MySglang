"""``heter_cache`` 后端的 radix cache。

在 RadixCache 之上叠一层"分带淘汰"：one-shot 带（random 整轮请求、单轮 response
段）永远先清空，可调度内容（各 workload 的 prompt 段 + 多轮 response 段）再竞争。
本模块只负责**执行**淘汰（释放 KV、改树、记账），候选选择委托给
``quota_selector.pick_victim``，原因见该函数文档。

导入本模块即注册 ``--radix-cache-backend=heter_cache`` 后端。
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
    """RadixCache + 多级队列淘汰。

    HeterCache 的落地点：树结构与插入/匹配沿用 RadixCache，只有 ``evict`` 的
    候选选择换成按队列（struct / agentic / chat）仲裁。
    """

    def evict(self, params: EvictParams) -> EvictResult:
        """释放至少 ``params.num_tokens`` 个 token，必要时分批推进。

        Args:
            params: 淘汰请求；只用 ``num_tokens``。

        Returns:
            本次实际释放的 token 数（可能因树已空而少于请求量）。

        三步结构：
          1. 观测：按两个独立时钟跑一次全树遍历，分别用于配额账本校准与
             profile 打印（承压时校准窗口会缩短，打印仍固定 30s）；
          2. 选点：交 ``_pick_victim`` 决定受害者与来源标签；
          3. 执行：``_evict_one`` 释放并在 ``victim_src`` 记账。

        先做观测再做淘汰：``_occ`` 是配额仲裁的输入，过期一个窗口就会按错误
        的占用点名队列。
        """
        # 禁用 radix cache 时无操作
        if self.disable:
            return EvictResult()
        # 非 HeterCache 策略没有分带/配额语义，退回基类的 LRU 淘汰
        if not isinstance(self.eviction_strategy, HeterCacheStrategy):
            return super().evict(params)

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # 配额仲裁需要池容量与可用量（O(1)，不遍历树）；只在配额启用时读 allocator，
        # 这样未开配额的路径不依赖 allocator 提供 size/available_size。
        available = capacity = 0.0
        if self._needs_capacity:
            available = self.token_to_kv_pool_allocator.available_size()
            capacity = float(self.token_to_kv_pool_allocator.size)
        # 两个时钟分开：校准可以缩到秒级，打印固定 30s，否则日志会被刷屏。
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
        while num_evicted < num_tokens:
            victim, source = self._pick_victim(self.evictable_leaves, capacity)
            if victim is None:
                # 树里已无可淘汰叶子（全 locked 或已空）
                break
            released = self._evict_one(victim)
            if released <= 0:
                # 淘汰必须单调推进；释放量为 0 说明候选状态异常，
                # 继续循环会死转，直接中止并留下错误日志。
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
        """转发到 ``quota_selector.pick_victim``，补上本缓存持有的上下文。

        Args:
            leaves: 当前可淘汰叶子集合。
            capacity: 池容量（token）；未启用配额时为 0。

        Returns:
            ``(victim, source)``；``(None, None)`` 表示无可淘汰对象。
        """
        return quota_selector.pick_victim(
            strategy=self.eviction_strategy,
            leaves=leaves,
            capacity=capacity,
            page_size=self.page_size,
            overrides=self.victim_overrides,
            session_registry=self.session_registry,
        )

    def _evict_one(self, node) -> int:
        """淘汰一个叶子并释放其 KV，返回释放的 token 数。

        对 one-shot 节点会向上级联：本次删除后变成"空且未上锁"的祖先链一次删净。
        原因是单轮会话的 response 走后，其上方的 prompt 链不会再被任何请求复用，
        逐叶淘汰只会让这条死链反复以新叶子的身份重新进入候选，淘汰循环为此多跑
        若干轮。遇到根、分支点（还有其他子树）或上锁节点即停 —— 它们仍可能是
        别的请求的共享前缀。

        Args:
            node: 起始淘汰点，必须是当前可淘汰叶子。

        Returns:
            本次释放的 token 总数（含级联删除的祖先）。
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
            # 级联条件：整条链都必须是 one-shot，否则上方的共享前缀不能跟着删。
            if (one_shot and parent is not None and parent is not self.root_node
                    and parent.lock_ref == 0 and not parent.children
                    and strategy.is_one_shot(parent)):
                cur = parent
                continue
            break
        return total


def _heter_cache_factory(ctx):
    """registry 工厂：由 TreeCacheBuildContext 构造本缓存。"""
    return HeterCacheRadixCache(ctx.params)


try:
    # 模块导入即注册后端，供 --radix-cache-backend=heter_cache 使用
    register_radix_cache_backend("heter_cache", _heter_cache_factory)
except ValueError:
    # 重复导入（例如测试里按路径二次加载）时 registry 拒绝重名，幂等处理即可
    pass
