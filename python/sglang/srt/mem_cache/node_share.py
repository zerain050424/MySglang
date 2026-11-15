"""节点共享会话数账本：一段 KV 被多少个不同的会话命中过。

三处消费方共读一本账，口径必须单点定义：

* ``HeterCacheStrategy.is_aged`` / ``is_one_shot``：共享数达标的 struct 前缀
  不受老化与 one-shot 带影响；
* ``queue_selector.struct_tier``：struct 队内按共享数分档；
* ``HeterProfiler.summary``：只读 ``stats()`` 与 ``share_min`` 两项观测。

为什么不并进策略：与 ``QueueOccupancyTracker`` / ``QueueValueTracker`` 同构，
账本独立、策略只做判据与裁决。本类不参与选点，因此不继承 ``VictimOverride``
——那个基类的 ``pick`` 在这里是永远收不到调用的死代码。

侧表不内嵌到 ``TreeNode``：那会污染所有缓存实现（lru/lfu 用不到它）。
``WeakKeyDictionary`` 以节点为弱键，节点被 GC（驱逐 / 分裂清理 / reset）时条目
自动消失，生命周期与内嵌字段等价，不需要显式清理钩子。侧表是实例级的：模块级
会让同进程的多个 cache 实例互相污染共享数。
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.heter_utils import (
    DEFAULT_SHARE_MIN,
    SESSION_SHARE_MAX,
    session_bloom_bits,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode


class NodeShareLedger:
    """每节点一个饱和计数 + 一份 64-bit bloom 去重位图。

    ``share_of`` 的条目在节点首次被命中时创建，未命中过读作 0——共享保护只给
    "真的被别人复用过"的内容。

    Args:
        share_min: 保护阈值；省略时取 ``DEFAULT_SHARE_MIN``，下限 2。
    """

    __slots__ = (
        "_meta",
        "aged_evicted",
        "share_counted",
        "share_dedup",
        "share_min",
    )

    def __init__(self, share_min: int | None = None) -> None:
        self.share_min = max(
            2,
            int(DEFAULT_SHARE_MIN if share_min is None else share_min),
        )
        self.share_counted = 0   # 计入 share_sessions 的命中次数
        self.share_dedup = 0     # 同一会话重复命中的去重次数
        self.aged_evicted = 0    # 老化降级后被淘汰的 token（按事件累加）
        # node -> [share_sessions, session_bloom]
        self._meta: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def on_hit(self, node: TreeNode, session_id: str | None) -> None:
        """把 (node, session) 记进账本；同会话重复命中只算一次。"""
        if node is None or not session_id:
            return
        bits = session_bloom_bits(session_id)
        entry = self._meta.setdefault(node, [0, 0])
        if entry[1] & bits == bits:
            self.share_dedup += 1
            return
        entry[1] |= bits
        entry[0] = min(SESSION_SHARE_MAX, entry[0] + 1)
        self.share_counted += 1

    def on_inserted(self, node: TreeNode) -> None:
        """分裂回调：新父节点继承子节点的共享数条目。

        分裂不改变"哪些请求经过这段前缀"，两半都继承。缓存层只在分裂处回调
        （新叶子没有子节点，继承无意义），所以这里不需要再判空子节点。
        """
        share = 0
        bloom = 0
        for child in node.children.values():
            entry = self._meta.get(child)
            if entry:
                share = max(share, entry[0])
                bloom |= entry[1]
        if share or bloom:
            self._meta[node] = [share, bloom]

    def on_aged_evict(self, tokens: int) -> None:
        """老化降级后淘汰的 token（观测用，不参与判据）。"""
        self.aged_evicted += int(tokens)

    def share_of(self, node: TreeNode) -> int:
        """节点被多少个不同会话命中过；未命中过读作 0。"""
        entry = self._meta.get(node)
        return int(entry[0]) if entry else 0

    def stats(self) -> dict[str, float]:
        return {
            "share_counted": self.share_counted,
            "share_dedup": self.share_dedup,
            "aged_evicted": self.aged_evicted,
        }

    def reset(self) -> None:
        """清账并丢弃侧表（缓存 reset / 跨 run 复用同一进程）。"""
        self.share_counted = 0
        self.share_dedup = 0
        self.aged_evicted = 0
        # 弱引用条目会随节点回收消失，但被快照 / 观测句柄临时持住的旧节点会活过
        # reset；reset 的语义是"上一次 run 的状态清空"，所以显式丢弃。
        self._meta = weakref.WeakKeyDictionary()
