"""
  SimpleHYBRID 的 radix cache：RadixCache + 两带淘汰。
  one-shot 带（random 整个请求 / 单轮 response 段）永远先清空；
  可调度内容（各 workload 的 prompt 段 + 多轮 response 段）按复用分数
  全局竞争，分数最低者先走。

导入本模块即注册 "--radix-cache-backend=simple_hybrid" 后端。
"""

import logging
import time

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


class SimpleHybridRadixCache(RadixCache):
    """RadixCache + 多级队列淘汰"""

    def evict(self, params: EvictParams) -> EvictResult:
        # 禁用 radix cache 时无操作
        if self.disable:
            return EvictResult()
        if not isinstance(self.eviction_strategy, SimpleHybridStrategy):
            return super().evict(params)

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # 每 PROFILE_INTERVAL 秒按 workload 打印一次 profile
        # 门禁：仅当打印到期才做全树遍历（否则节流窗口内白发全线）
        if self._profile_printer is not None:
            _profile_due = getattr(self.eviction_strategy, "profile_due", None)
            if _profile_due is None or _profile_due():
                self._profile_printer(
                    self._per_workload_token_count(), self.session_registry.summary()
                )

        strategy = self.eviction_strategy
        num_evicted = 0
        # one-shot 永远先清空（淘汰策略为带内LRU）；
        # 清空后可调度内容按 victim_key 全局竞争（分数最低，同分 token 多者先走）。
        while num_evicted < num_tokens:
            # 下面两个分支必定会给victim赋值（one-shot 带 / victim_key基础候选）
            # 第三分支直接 break。因此 victim 绝不跨轮复用、也绝不为None
            # 复用上一轮已删除的 victim 会二次释放同一段 KV。
            leaves = self.evictable_leaves
            one_shot = [n for n in leaves if strategy.is_one_shot(n)]
            source = "global"
            if one_shot:
                victim = min(one_shot, key=strategy.one_shot_key)
                source = "one_shot"
            elif leaves:
                victim = min(leaves, key=strategy.victim_key)
                # 依次询问已注册的改写器（注册顺序即优先级，后者覆盖前者）。
                # 返回 None 表示不接管 -> 保留当前候选，绝不阻塞淘汰推进。
                for override in self.victim_overrides:
                    if override.applies_to(victim):
                        alt = override.pick(leaves, self.session_registry)
                        if alt is not None:
                            victim = alt
                            # queue用来profile本次淘汰由哪个队列的策略决定。
                            source = (
                                override.queue.value
                                if override.queue is not None
                                else override.name
                            )
            else:
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
        # 当祖先自己是一次性内容（未被跨会话共享）时级联删除，否则会
        # 连带删掉刚进 protected 的共享前缀。
        strategy = self.eviction_strategy
        one_shot = strategy.is_one_shot(node)
        while True:
            parent = cur.parent
            # 累计该队列的淘汰 token（供 profile）
            self.eviction_strategy.record_evict(cur)
            # 归还 KV slot
            self.token_to_kv_pool_allocator.free_segment(cur.value, start_pos=0)
            # 记录 KV 事件
            self.kv_events.record_remove(cur)
            total += len(cur.value)
            # 从树中删除；_delete_leaf 会顺带维护 evictable_leaves 与父节点晋升
            self._delete_leaf(cur)
            if (one_shot and parent is not None and parent is not self.root_node
                    and parent.lock_ref == 0 and not parent.children
                    # 同源闸：祖先必须也是 one-shot（共享数未达保护阈值）。
                    and strategy.is_one_shot(parent)):
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
