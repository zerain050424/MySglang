"""SimpleHYBRID 的 radix cache：RadixCache + 两带淘汰。
  one-shot 带（random 整个请求 / 单轮 response 段）永远先清空；
  可调度内容（各 workload 的 prompt 段 + 多轮 response 段）按复用分数
  全局竞争，分数最低者先走。

导入本模块即注册 "--radix-cache-backend=simple_hybrid" 后端。
"""

import time

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend


class SimpleHybridRadixCache(RadixCache):
    """RadixCache + 多级队列淘汰（面向文本对话 benchmark）。"""

    def evict(self, params: EvictParams) -> EvictResult:
        # 禁用 radix cache 时无操作
        if self.disable:
            return EvictResult()
        # 若策略不是 SimpleHybridStrategy，
        # 退回父类单堆淘汰，避免 AttributeError
        if not isinstance(self.eviction_strategy, SimpleHybridStrategy):
            return super().evict(params)

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # 每隔 update_interval 秒按命中效率刷新各类型 α 权重
        if self._alpha_updater is not None:
            # 传三份口径：alpha 用可调度容量，profile 用 workload 占用与会话缺口
            self._alpha_updater(
                *self._profile_token_counts(), self.session_registry.summary()
            )

        strategy = self.eviction_strategy
        num_evicted = 0

        # one-shot 永远先清空（one_shot_key 首项 -树上结束偏移：位置最深的块先走，
        # 对齐 UniCache evict_first 的 -num_hashed_tokens 口径）；
        # 清空后可调度内容按 victim_key 全局竞争（分数最低，同分 token 多者先走）。
        while num_evicted < num_tokens:
            leaves = self.evictable_leaves
            one_shot = [n for n in leaves if strategy.is_one_shot(n)]
            if one_shot:
                victim = min(one_shot, key=strategy.one_shot_key)
            elif leaves:
                victim = min(leaves, key=strategy.victim_key)
            else:
                break
            num_evicted += self._evict_one(victim)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_one(self, node) -> int:
        """
        淘汰一个叶子并释放其 KV，返回释放的 token 数。
        one-shot 节点向上追溯：把因本次删除而变得「无其他子树、未上锁」的
        祖先链一次删净 —— 单轮会话的 response 走后，其上方的 prompt 链
        不会再产生任何复用，逐叶淘汰只会让死链反复以新叶子的身份暴露，
        驱逐循环为此多跑若干轮。遇到根、分支点（还有其他子树）或上锁
        节点即停，它们仍可能是共享前缀。
        """
        total = 0
        cur = node
        one_shot = self.eviction_strategy.is_one_shot(node)
        while True:
            parent = cur.parent
            # 累计该队列的淘汰 token（供 profile）
            self.eviction_strategy.record_evict(cur)
            # 归还 KV slot（page 对齐的段）
            self.token_to_kv_pool_allocator.free_segment(cur.value, start_pos=0)
            # 记录 KV 事件（供指标/回放使用）
            self.kv_events.record_remove(cur)
            total += len(cur.value)
            # 从树中删除；_delete_leaf 会顺带维护 evictable_leaves 与父节点晋升
            self._delete_leaf(cur)
            if (one_shot and parent is not None and parent is not self.root_node
                    and parent.lock_ref == 0 and not parent.children):
                cur = parent  # 死链：随 one-shot 一并删除
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
