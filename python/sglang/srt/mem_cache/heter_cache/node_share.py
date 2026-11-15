"""节点共享会话数记账：一段 KV 被多少个不同的会话命中过。
* HeterCacheStrategy.is_aged/is_one_shot：共享数达标的 struct 前缀
  不受老化与 one-shot 带影响；
* selector.struct_tier：struct 队内按共享数分档；
* HeterProfiler.summary：只读stats()与share_min两项观测。
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.heter_cache.utils import (
    DEFAULT_SHARE_MIN,
    SESSION_SHARE_MAX,
    session_bloom_bits,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode


class NodeShareLedger:
    """每节点一个饱和计数/64-bit bloom 去重位图。
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
        """分裂回调：新父节点继承子节点的共享数条目"""
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
        """老化降级后淘汰的token。"""
        self.aged_evicted += int(tokens)

    def share_of(self, node: TreeNode) -> int:
        """节点被多少个不同会话命中过;未命中过读作0。"""
        entry = self._meta.get(node)
        return int(entry[0]) if entry else 0

    def stats(self) -> dict[str, float]:
        return {
            "share_counted": self.share_counted,
            "share_dedup": self.share_dedup,
            "aged_evicted": self.aged_evicted,
        }

    def reset(self) -> None:
        """清空并丢弃侧表"""
        self.share_counted = 0
        self.share_dedup = 0
        self.aged_evicted = 0
        self._meta = weakref.WeakKeyDictionary()
