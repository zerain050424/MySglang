"""标签版 TinyLRU：**分类即晋升**。

TinyLRU 分支的默认实现（`tiny_lru.policy = "tag_lru"`）。它把「谁该被保护」
从三区状态机改成一个纯规则：

    if 会话类型 == STRUCT（模板/共享前缀型）:
        share_sessions(node) >= struct_share_min(默认 2)  -> QSTRUCT   # 跨会话共享，LRU 淘汰
        share_sessions(node) <  struct_share_min:
            rids(node) >= struct_share_min                -> QCHAT     # 会话内复用（默认）
                                                             （或 QTRANS：local_hot="one_shot"）
            否则                                           -> QTRANS    # 真一次性，进 one-shot 带
    else:                                                  # chat / agentic / trans
        沿用会话类型映射（QCHAT / QAGENT / QTRANS）

三条关键点
----------
1. **晋升发生在分类/命中时，不需要「成员先变成可淘汰叶子」**。三区版最难的
   条件（M3 成员只在子链删光后才可淘汰）在这里消失。
2. **不做黏性**：标签是 `share_sessions` 的纯函数。会话共享数只增不减，因此
   一旦晋升就事实上保持；若将来做 aging，这个函数会自动跟着降级。
3. **不需要三区索引 / 准入 / aging / 预算**：`QSTRUCT` 内部按 LRU 排序由
   `SimpleHybridStrategy.victim_key` 的 class-rank 分支负责（见 evict_policy）。

与三区版共享的字段：`share_count`（rid 计数，诊断用）、`share_sessions`、
`rid_bloom`、`session_bloom`。两套策略可以随时切换，节点状态不用迁移。
"""

from __future__ import annotations

from typing import Dict, Optional

from sglang.srt.mem_cache.uni_utils import (
    SessionType,
    SimpleHybridQueueTag,
    session_bloom_bits,
)

_MAX_SHARE = 255


class TagLruPolicy:
    """无队列状态：只把「会话共享数 + 会话类型」映射成节点标签。"""

    name = "tag_lru"

    def __init__(
        self,
        struct_share_min: int = 2,
        share_metric: str = "session",
        local_hot: str = "chat",
    ):
        # 共享数达到该值才算「跨会话模板」；默认 2（>1）
        self.struct_share_min = max(2, int(struct_share_min))
        # "session"（默认，与规则一致）| "rid"（消融：按请求数判共享）
        self.share_metric = share_metric if share_metric in ("session", "rid") else "session"
        # 会话内热点（share_sessions==1 但被同一会话多次请求命中）怎么处置：
        #   "chat"（默认）：QCHAT -> rank 1，按复用概率曲线，不被一次性带清空
        #   "one_shot"（规则原样）：QTRANS -> 进 one-shot 硬优先带
        # 实测（ShareGPT/pool=0.3）：chat 78.10% vs one_shot 54.61%，
        # 合成多轮 struct 负载 82.46% vs 75.67%。默认必须是 chat。
        self.local_hot = local_hot if local_hot in ("one_shot", "chat") else "chat"

        # 同一会话连续命中整条路径：bloom bits 做一次局部缓存
        self._last_sid: Optional[str] = None
        self._last_sbits = 0

        self.stats = {
            "promote": 0,   # QTRANS/QCHAT/QAGENT -> QSTRUCT 的次数
            "demote": 0,    # QSTRUCT -> QTRANS 的次数（当前规则下不会发生）
            "dedup": 0,     # 同一会话重复命中同一个节点
            "counted": 0,   # 计入 share_sessions 的命中次数
            "born": 0,      # 新节点出生打标次数
        }

    # ------------------------------------------------------------------ 频数

    def _freq_of(self, node) -> int:
        if self.share_metric == "session":
            sessions = getattr(node, "share_sessions", None)
            if sessions is not None:
                return int(sessions)
        return int(getattr(node, "share_count", 0) or 0)

    def _sid_bits(self, session_id) -> int:
        if session_id != self._last_sid:
            self._last_sid = session_id
            self._last_sbits = session_bloom_bits(session_id)
        return self._last_sbits

    # ------------------------------------------------------------------ 标签

    def initial_tag(self, session_type) -> SimpleHybridQueueTag:
        """新节点出生标签。

        新节点一定是「本会话私有」的那一段（共享数为 0），所以 STRUCT 会话下
        先给 QTRANS：如果之后有第二个会话命中它，`observe_hit` 会把它晋升成
        QSTRUCT。其它会话类型保持原有映射。
        """
        self.stats["born"] += 1
        if session_type == SessionType.STRUCT:
            # 会话内热点口径：chat（默认）保留复用语义；one_shot 直接进一次性带
            return (
                SimpleHybridQueueTag.QCHAT
                if self.local_hot == "chat"
                else SimpleHybridQueueTag.QTRANS
            )
        return session_type_to_tag(session_type)

    def apply_tag(self, node, session_type) -> None:
        """按「会话类型 + 共享数」重贴标签（只在 STRUCT 会话上生效）。"""
        if node is None or session_type != SessionType.STRUCT:
            return
        rids = int(getattr(node, "share_count", 0) or 0)
        if self._freq_of(node) >= self.struct_share_min:
            want = SimpleHybridQueueTag.QSTRUCT
        elif self.local_hot == "chat" and rids >= self.struct_share_min:
            # 同一会话已经复用过：保留复用语义，不丢进 one-shot 硬优先带
            want = SimpleHybridQueueTag.QCHAT
        else:
            want = SimpleHybridQueueTag.QTRANS
        old = getattr(node, "queue_tag", None)
        if old == want:
            return
        node.queue_tag = want
        if want == SimpleHybridQueueTag.QSTRUCT:
            self.stats["promote"] += 1
        else:
            self.stats["demote"] += 1

    # ------------------------------------------------------------------ 钩子

    def observe_hit(self, node, session_id, session_type) -> None:
        """命中路径：先记「不同会话数」，再按规则贴标签。

        这一步就是「分类过程中的晋升」：第二个会话命中同一条前缀时，
        该前缀当场从 QTRANS 变成 QSTRUCT，不需要等它变成可淘汰叶子。
        """
        if node is None or session_id is None:
            return
        bits = self._sid_bits(session_id)
        if (getattr(node, "session_bloom", 0) & bits) == bits:
            self.stats["dedup"] += 1
            return
        node.session_bloom = getattr(node, "session_bloom", 0) | bits
        node.share_sessions = min(_MAX_SHARE, getattr(node, "share_sessions", 0) + 1)
        self.stats["counted"] += 1
        self.apply_tag(node, session_type)

    def on_split(self, prefix_half, suffix_half=None) -> None:
        """分裂：两半都继承频数（radix_cache 已复制字段），标签由命中路径再定。"""
        return

    # ------------------------------------------------------------------ 归属

    def owns(self, node) -> bool:
        """死链上溯闸 / 观测用：该节点是否属于「受保护的共享前缀」。"""
        return getattr(node, "queue_tag", None) == SimpleHybridQueueTag.QSTRUCT

    # ------------------------------------------------------------------ 观测

    def snapshot(self) -> Dict[str, object]:
        return {
            "policy": "tag_lru",
            "struct_share_min": self.struct_share_min,
            "share_metric": self.share_metric,
            "local_hot": self.local_hot,
            **self.stats,
        }

    def reset(self) -> None:
        self._last_sid = None
        self._last_sbits = 0
        for key in self.stats:
            self.stats[key] = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        s = self.snapshot()
        return (
            "[tag_lru] min={struct_share_min} promote={promote} "
            "dedup={dedup} counted={counted}".format(**s)
        )


def session_type_to_tag(session_type) -> SimpleHybridQueueTag:
    """SESSION 类型 -> 标签（STRUCT 由本类单独处理，不会走到这里）。"""
    return {
        SessionType.TRANS: SimpleHybridQueueTag.QTRANS,
        SessionType.AGENTIC: SimpleHybridQueueTag.QAGENT,
        SessionType.CHAT: SimpleHybridQueueTag.QCHAT,
        SessionType.STRUCT: SimpleHybridQueueTag.QTRANS,
    }.get(session_type, SimpleHybridQueueTag.QTRANS)
