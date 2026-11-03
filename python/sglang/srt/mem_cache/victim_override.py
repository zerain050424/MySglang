"""淘汰候选改写器（VictimOverride）：把「选谁」从 SimpleHybridStrategy 里拆成插件位。

契约（`RrEvictPolicy` / `StructTinyLruPolicy` 共用）：

    applies_to(node) -> bool        当前候选是否落在本改写器的管辖范围
    pick(leaves, registry) -> node  给出本改写器更愿意淘汰的节点；None = 不接管

硬性约束：
  * 改写器不释放 KV、不删节点，只返回候选；释放仍由 `_evict_one()` 统一执行。
  * 返回 None 时必须回退到原候选，不允许阻塞淘汰推进（否则 evict 会自旋）。
  * 改写器只能从 `leaves` 里挑节点，不允许跨队列造出新的可淘汰集合。

`SimpleHybridRadixCache.evict()` 按 `victim_overrides` 的注册顺序逐个询问，
后注册者覆盖前者（"越靠后越硬"）：RR 会话公平先改一次，struct 队列若有归属权再改一次。

`queue` 字段只做观测（profile 里打「本次淘汰由哪个队列的策略决定」），
不得参与任何仲裁判定 —— 跨队列仲裁仍由 victim_key 全局排序完成。

内置实现的分工：
  * `RrEvictPolicy`      —— 自带全部状态（会话轮询序号 / 配额违约计数）。
  * `StructTinyLruPolicy` —— 无状态适配器；队列状态在
    `struct_queue_policy.StructQueueState`，本类只暴露它的区头。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

from sglang.srt.mem_cache.uni_utils import (
    RrParams,
    SimpleHybridQueueTag,
    SimpleHybridTaskType,
    node_end_offset,
    task_type_of_tag,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.struct_queue_policy import StructQueueState
    from sglang.srt.mem_cache.uni_utils import SessionRegistry, SessionState


class VictimOverride(ABC):
    """插件位基类。子类只需实现 applies_to / pick。"""

    name: str = "override"

    # 观测字段：本改写器管辖的队列，仅用于 profile 记账。
    # 不得参与仲裁判定（否则退化为 proposer 形态，见设计讨论）。
    queue: Optional[SimpleHybridQueueTag] = None

    @abstractmethod
    def applies_to(self, node) -> bool:
        """当前候选是否归本改写器管。"""

    @abstractmethod
    def pick(self, leaves: Iterable["TreeNode"], registry) -> Optional["TreeNode"]:
        """返回本改写器更愿意淘汰的节点；None 表示不接管。"""

    def stats(self) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        """清空内部状态（缓存 reset / 进程复用）。"""

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"


class RrEvictPolicy(VictimOverride):
    """
    Agentic 会话轮询（round-robin）。
    语义：
      * 只处理 Agentic 队列的叶子，按会话聚合，每个会话取最深的那片叶子；
      * 本轮缺口 `d` 超过 `loss_cap(L)` 的会话直接跳过，避免把一条会话掏空；
      * 按照访问距离进行轮询删除。
    """

    name = "rr"
    queue = SimpleHybridQueueTag.QAGENT

    def __init__(self, params: RrParams):
        self.params = params
        self._rr_last: Dict[str, int] = {}
        self._rr_seq = 0
        self.quota_breach = 0

    # ---- 参数 ----

    def loss_cap(self, chain_len: int) -> float:
        """单会话本轮可以承受的 KV 缺口上限。"""
        keep = max(
            self.params.min_retain_tokens, self.params.min_retain_frac * chain_len
        )
        return max(
            0.0, min(self.params.kappa * chain_len, chain_len - keep)
        )

    # ---- 插件接口 ----

    def applies_to(self, node) -> bool: ##检测是否属于这个策略的作用域
        if not self.params.fair_rr or node is None:
            return False
        return (
            task_type_of_tag(getattr(node, "queue_tag", None))
            == SimpleHybridTaskType.MULTI_TURN_AGENTIC
        )

    def pick(self, leaves, registry) -> Optional["TreeNode"]:
        if not self.params.fair_rr:
            return None
        agentic = [node for node in leaves if self.applies_to(node)]
        if not agentic:
            return None
        victim = self._pick_round_robin(agentic, registry)
        if victim is None:
            self.quota_breach += 1
        return victim

    # ---- 内部 ----

    def _pick_round_robin(self, leaves, registry) -> Optional["TreeNode"]:
        """挑最久未被删除的 Agentic 会话的最深叶子。"""
        tails: Dict[str, Tuple["TreeNode", "SessionState"]] = {}
        for node in leaves:
            sid = getattr(node, "session_id", None)
            if sid is None:
                continue
            state = registry.get(sid)  # 会话 KV 记账
            if state is None or state.L <= 0:
                continue
            current = tails.get(sid)
            if current is None or node_end_offset(node) > node_end_offset(current[0]):
                # 会话内部只保留最深节点
                tails[sid] = (node, state)

        best = None
        for sid, (node, state) in tails.items():  # 跨会话轮询
            if state.d >= self.loss_cap(state.L):
                # 本会话缺口已经够大，本轮不再动它
                continue
            last = self._rr_last.get(sid, -1)  # -1 = 从未被选中
            if best is None or last < best[0]:
                best = (last, sid, node)
        if best is None:
            return None

        self._rr_last[best[1]] = self._rr_seq
        self._rr_seq += 1
        return best[2]

    # ---- 观测 ----

    def stats(self) -> Dict[str, float]:
        return {
            "rr_sessions": len(self._rr_last),
            "rr_rounds": self._rr_seq,
            "rr_quota_breach": self.quota_breach,
        }

    def reset(self) -> None:
        self._rr_last.clear()
        self._rr_seq = 0
        self.quota_breach = 0


class StructTinyLruPolicy(VictimOverride):
    """Q_struct 队内 W-TinyLRU 的**淘汰面**：把状态机的区头暴露成候选修正。

    拆分前 ``StructQueuePolicy`` 同时扮两种角色（缓存钩子宿主 + 淘汰插件）。
    现在：
      * 队列状态（成员判定 / 三区 / 准入 / aging）在
        ``struct_queue_policy.StructQueueState``，由缓存层持有；
      * 本类只做三件事：``applies_to = state.owns``、
        ``pick = state.eviction_candidate``、``stats/reset`` 转发。
    自身无状态，所以随时可以重建而不丢队列状态。
    ``forced``（只剩 protected 被迫出队）记在状态机里：那是区域级事实，
    不是本类的决策。
    """

    name = "struct_tinylru"
    queue = SimpleHybridQueueTag.QSTRUCT

    def __init__(self, state: "StructQueueState"):
        self.state = state

    def applies_to(self, node) -> bool:
        """O(1)：委托状态机做成员判定。"""
        return self.state.owns(node)

    def pick(self, leaves=None, registry=None):
        """队内区头；None = 不接管（上层保留原候选）。"""
        return self.state.eviction_candidate()

    # ---- 观测 / 生命周期（都转发）----

    def stats(self) -> Dict[str, float]:
        return dict(self.state.stats)

    def reset(self) -> None:
        self.state.reset()
