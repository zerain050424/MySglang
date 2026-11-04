"""淘汰候选改写器（VictimOverride）：把选择淘汰从 SimpleHybridStrategy 里拆成插件位。
    applies_to(node) -> bool        当前候选是否落在本改写器的管辖范围
    pick(leaves, registry) -> node  给出本改写器更愿意淘汰的节点；None = 不接管

硬性约束：
  * 改写器不释放 KV、不删节点，只返回候选；释放仍由 `_evict_one()` 统一执行。
  * 返回 None 时必须回退到原候选，不允许阻塞淘汰推进（否则 evict 会自旋）。
  * 改写器只能从 `leaves` 里挑节点，不允许跨队列造出新的可淘汰集合。

`SimpleHybridRadixCache.evict()` 按 `victim_overrides` 的注册顺序逐个询问，
后注册者覆盖前者（"越靠后越硬"）：RR 会话公平先改一次，struct 队列若有归属权再改一次。

`queue` 字段只做观测（profile 里打本次淘汰由哪个队列的策略决定），
不得参与任何仲裁判定 —— 跨队列仲裁仍由 victim_key 全局排序完成。

内置实现的分工：
  * `RrEvictPolicy`      —— 自带全部状态（会话轮询序号 / 配额违约计数）。
  * `RrEvictPolicy`      —— 自带全部状态（会话轮询序号 / 配额违约计数）。
  * `TlruEvictPolicy`    —— T-LRU 的 QAGENT 候选修正（自带高水位状态）。

  注：Q_struct 队内策略（标签版 TinyLRU）不通过插件实现 —— 它把「类间优先级 +
  类内 LRU」直接编码进 `SimpleHybridStrategy.victim_key`，因此这里不再有 struct
  适配器。
  * `TlruEvictPolicy`    —— T-LRU 的尾部优先修正；除候选选择外，还在
    插入/分裂时维护节点上的 `_tlru_*` 高水位（缓存层会调 `on_node_inserted`）。
  * `SlruEvictPolicy`    —— SLRU 的候选改写（probation → protected，段内 recency），
    同时独占「节点共享会话数」账本（`on_node_hit` 由缓存层命中路径调用）。
    `scope="non_one_shot"` 时与 `evict()` 的 one-shot 分支合起来就是
    「带 one-shot 带的 SLRU」：带内先清空，其余按两段排序。标签版 TinyLRU 仍把
    顺序编码进 `victim_key`，不走插件。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
from sglang.srt.mem_cache.uni_utils import (
    DEFAULT_STRUCT_SHARE_MIN,
    RrParams,
    SESSION_SHARE_MAX,
    SessionRegistry,
    SessionType,
    SimpleHybridQueueTag,
    SimpleHybridTaskType,
    TlruParams,
    node_end_offset,
    session_bloom_bits,
    task_type_of_tag,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.uni_utils import SessionState


class VictimOverride(ABC):
    """插件位基类。子类只需实现 applies_to / pick。"""

    name: str = "override"
    # 会话账本由构造方注入（见 SlruEvictPolicy.__init__）。基类不得在 import
    # 时抓死全局单例，否则 SessionRegistry.reset_global() / 测试里的
    # `cache.session_registry = SessionRegistry()` 都到不了插件。
    # 观测字段：本改写器管辖的队列，仅用于 profile 记账。
    # 不得参与仲裁判定（否则退化为 proposer 形态，见设计讨论）。
    queue: Optional[SimpleHybridQueueTag] = None

    @abstractmethod
    def applies_to(self, node) -> bool:
        """当前候选是否归本改写器管。"""

    @abstractmethod
    def pick(self, leaves: Iterable["TreeNode"], registry) -> Optional["TreeNode"]:
        """返回本改写器更愿意淘汰的节点；None表示不接管。"""

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


class TlruEvictPolicy(VictimOverride):

    name = "tlru"
    queue = SimpleHybridQueueTag.QAGENT

    def __init__(self, params: TlruParams):
        self.params = params
        # ---- 观测 ----
        self.tel_safe_picks = 0     # 选中的候选里有多少是 tel-safe（phase 1）
        self.lru_picks = 0          # tel-safe 用尽后按 recency 退化的次数（phase 2）
        self.fallbacks = 0          # 队内无 QAGENT 叶子，交回全局候选
        self.history_raises = 0     # 高水位上抬的次数（插入期开销的代理指标）

    # ------------------------------------------------------------ 插入期记账

    def on_node_inserted(self, node) -> None:
        """维护 T-LRU 的两个结构化字段；由缓存层在节点创建/分裂后回调。

        等价于 unified 的 `_set_tlru_lens_and_raise_history`，但用「自深 vs
        子树高水位取 max」统一覆盖 split：分裂出的新父节点只有 child 一个子树，
        max(自深, child.history) 恰好等于 child.history（分裂不增加分支深度）。
        """
        parent = getattr(node, "parent", None)
        base = getattr(parent, "_tlru_cached_prefix_len", 0) if parent is not None else 0
        node._tlru_cached_prefix_len = base + len(node.key or ())

        hi = node._tlru_cached_prefix_len
        for child in (node.children or {}).values():
            child_hi = getattr(child, "_tlru_history_len", 0)
            if child_hi > hi:
                hi = child_hi
        if getattr(node, "_tlru_history_len", 0) < hi:
            node._tlru_history_len = hi

        # 沿祖先链上抬（截断在已达标处，最常见情形是一次就停）。
        cur = parent
        while cur is not None and getattr(cur, "_tlru_history_len", 0) < hi:
            cur._tlru_history_len = hi
            self.history_raises += 1
            cur = cur.parent

    # ------------------------------------------------------------ 插件接口

    def applies_to(self, node) -> bool:
        if node is None:
            return False
        return (
            task_type_of_tag(getattr(node, "queue_tag", None))
            == SimpleHybridTaskType.MULTI_TURN_AGENTIC
        )

    def pick(self, leaves=None, registry=None) -> Optional["TreeNode"]:
        """tel-safe 的尾巴先走；同为 tel-safe 时取最久未访问（LRU 阶段）。"""
        if leaves is None:
            return None
        threshold = self.params.threshold
        next_prompt = self.params.next_prompt_estimate

        best = None
        for node in leaves:
            if not self.applies_to(node):
                continue
            history = getattr(node, "_tlru_history_len", 0)
            budget = max(history + next_prompt - threshold, 0)
            cached_without = getattr(node, "_tlru_cached_prefix_len", 0) - len(node.key or ())
            tel_safe = cached_without >= budget
            key = (0 if tel_safe else 1, node.last_access_time)
            if best is None or key < best[0]:
                best = (key, node, tel_safe)

        if best is None:
            self.fallbacks += 1
            return None
        if best[2]:
            self.tel_safe_picks += 1
        else:
            self.lru_picks += 1
        return best[1]

    # ------------------------------------------------------------ 观测

    def stats(self) -> Dict[str, float]:
        return {
            "tlru_tel_safe_picks": self.tel_safe_picks,
            "tlru_lru_picks": self.lru_picks,
            "tlru_fallbacks": self.fallbacks,
            "tlru_history_raises": self.history_raises,
        }

    def reset(self) -> None:
        """只清计数器；高水位存在节点上，随树一起丢弃。"""
        self.tel_safe_picks = 0
        self.lru_picks = 0
        self.fallbacks = 0
        self.history_raises = 0


class SlruEvictPolicy(VictimOverride):
    """S-LRU 的 QSTRUCT 候选修正，同时是「节点共享会话数」账本的唯一所有者。

    账本（本类独占）
    ----------------
    写 `node.share_sessions` / `node.session_bloom` 的唯一位置是 `on_node_hit()`；
    缓存层在命中路径上通过 `_hit_observers` 调用它，**与 `tiny_lru.policy` /
    `force_agentic` 无关**，所以关掉标签策略后账本照常维护。
    节点的初值（`TreeNode.__init__`）与分裂继承（`_split_node`）属于树结构维护，
    仍留在 radix_cache；本类只读它们。

    计数口径：不同**会话**命中同一节点才 +1；创建者自己的插入不计，
    同一会话重复命中由 `session_bloom` 去重，饱和在 `SESSION_SHARE_MAX`。

    候选修正
    --------
    `applies_to` 命中「STRUCT 会话 + 已被 >= struct_share_min 个外部会话
    命中过」的叶子，`pick` 在它们中间取最久未访问者（段内 LRU）。
    阈值与 TagLruPolicy 的晋升阈值是同一个旋钮
    （`TinyLRUParams.struct_share_min`），因此标签与淘汰不会各算一套。
    """

    # 观测字段：本改写器管辖的队列，仅用于 profile 记账。
    # 不得参与仲裁判定（否则退化为 proposer 形态，见设计讨论）。
    queue: Optional[SimpleHybridQueueTag] =  SimpleHybridQueueTag.QSTRUCT

    def __init__(self, session_registry=None, struct_share_min=None) -> None:
        # 与策略共用同一本会话账（SimpleHybridStrategy.__init__ 注入），
        # 保证 is_one_shot 与 applies_to 看到同一份 session_type。
        self.session_registry = session_registry or SessionRegistry.global_instance()
        # 共享数阈值：唯一旋钮是 TinyLRUParams.struct_share_min（由策略注入）。
        # 它同时决定「是否受 SLRU 保护」与「是否退出 one-shot 带」。
        self.struct_share_min = max(
            2,
            int(
                DEFAULT_STRUCT_SHARE_MIN
                if struct_share_min is None
                else struct_share_min
            ),
        )
        # 共享会话数账本的观测（原属 TagLruPolicy，已随记账一起搬过来）
        self.share_counted = 0   # 计入 share_sessions 的命中次数
        self.share_dedup = 0     # 同一会话重复命中的去重次数
        self.slru_picks = 0      # 本改写器实际给出的候选数

    # ------------------------------------------------------------ 共享会话数账本

    def on_node_hit(self, node, session_id=None) -> None:
        """命中路径的记账钩子：不同会话数 +1（同一会话去重）。

        由缓存层 `RadixCache._record_node_hit()` 在贴标签之前调用，
        因此 `tag_lru.apply_tag()` 读到的是本次命中后的新值。
        """
        if node is None or not session_id:
            return
        bits = session_bloom_bits(session_id)
        if getattr(node, "session_bloom", 0) & bits == bits:
            self.share_dedup += 1
            return
        node.session_bloom = getattr(node, "session_bloom", 0) | bits
        node.share_sessions = min(
            SESSION_SHARE_MAX, int(getattr(node, "share_sessions", 0) or 0) + 1
        )
        self.share_counted += 1

    @staticmethod
    def share_of(node) -> int:
        """节点已被多少个「其他会话」命中过（创建者不计）。"""
        return int(getattr(node, "share_sessions", 0) or 0)

    # ------------------------------------------------------------ 插件接口

    def applies_to(self, node) -> bool:
        """当前候选是否归本改写器管。"""
        if node is None:
            return False
        st = self.session_registry.get(getattr(node, "session_id", None))
        stype = st.session_type if st is not None else None
        return (
            stype == SessionType.STRUCT
            and self.share_of(node) >= self.struct_share_min
        )

    def pick(self, leaves: Iterable["TreeNode"], registry=None) -> Optional["TreeNode"]:
        """返回本改写器更愿意淘汰的节点；None表示不接管。"""
        struct = [node for node in leaves if self.applies_to(node)]
        if not struct:
            return None
        now = time.monotonic()
        nearest = None
        victim = None
        for node in struct:
            delta = now - node.last_access_time
            if nearest is None or delta > nearest:
                nearest = delta
                victim = node
        self.slru_picks += 1
        return victim

    def stats(self) -> Dict[str, float]:
        return {
            "slru_picks": self.slru_picks,
            "share_counted": self.share_counted,
            "share_dedup": self.share_dedup,
        }

    def reset(self) -> None:
        """清空内部状态（缓存 reset / 进程复用）。

        节点上的 `share_sessions` / `session_bloom` 随树一起丢弃，不用清。
        """
        self.share_counted = 0
        self.share_dedup = 0
        self.slru_picks = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"
