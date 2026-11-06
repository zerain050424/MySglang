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
  * `SlruEvictPolicy`    —— SLRU 的唯一实现：候选改写（probation → protected，
    段内 recency）
    同时独占「节点共享会话数」账本（`on_node_hit` 由缓存层命中路径调用）。
    `scope="non_one_shot"` 时与 `evict()` 的 one-shot 分支合起来就是
    「带 one-shot 带的 SLRU」：带内先清空，其余按两段排序。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
import weakref
from sglang.srt.mem_cache.uni_utils import (
    DEFAULT_SHARE_MIN,
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


# ---- HeterCache 插件侧表：替代 TreeNode 上的内嵌插件状态（方案 B 解耦）----
# 设计：策略插件私有状态不再内嵌 TreeNode 字段，改由本模块持侧表。
# WeakKeyDictionary 以节点为弱键：节点被 GC（分裂清理/驱逐/reset）时
# 条目自动消失，与旧内嵌字段生命周期等价，无需显式清理钩子。
# 键为 TreeNode，读侧缺省为 0，与旧节点字段初始值语义一致。
_TLRU_META = weakref.WeakKeyDictionary()  # node -> [cached_prefix_len, history_len]
_SLRU_META = weakref.WeakKeyDictionary()  # node -> [share_sessions, session_bloom]


def tlru_lens(node) -> Tuple[int, int]:
    """读取 T-LRU 侧表；节点无条目按 (0, 0)，与旧节点字段初始值等价。

    读方：TlruEvictPolicy.pick / TLRUStrategy.get_priority（独立策略）。
    写方：TlruEvictPolicy.on_node_inserted（插入/分裂期唯一写者）。
    """
    _e = _TLRU_META.get(node)
    return (_e[0], _e[1]) if _e else (0, 0)


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
        """维护 T-LRU 的两个结构化字段（存于 _TLRU_META 侧表）；由缓存层在节点创建/分裂后回调。

        等价于 unified 的 `_set_tlru_lens_and_raise_history`，但用「自深 vs
        子树高水位取 max」统一覆盖 split：分裂出的新父节点只有 child 一个子树，
        max(自深, child.history) 恰好等于 child.history（分裂不增加分支深度）。
        分裂时 child 的字段值不变（新 cached = 旧 base + 新前缀，数学上与旧
        公式一致），故只需回调新父节点。
        """
        parent = getattr(node, "parent", None)
        base = tlru_lens(parent)[0] if parent is not None else 0
        entry = _TLRU_META.setdefault(node, [0, 0])
        entry[0] = base + len(node.key or ())

        hi = entry[0]
        for child in (node.children or {}).values():
            child_hi = tlru_lens(child)[1]
            if child_hi > hi:
                hi = child_hi
        if entry[1] < hi:
            entry[1] = hi

        # 沿祖先链上抬（截断在已达标处，最常见情形是一次就停）。
        cur = parent
        while cur is not None:
            ce = _TLRU_META.setdefault(cur, [0, 0])
            if ce[1] >= hi:
                break
            ce[1] = hi
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
            cached, history = tlru_lens(node)
            budget = max(history + next_prompt - threshold, 0)
            cached_without = cached - len(node.key or ())
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
        """只清计数器；高水位在 _TLRU_META 侧表，随节点 GC 一起丢弃。"""
        self.tel_safe_picks = 0
        self.lru_picks = 0
        self.fallbacks = 0
        self.history_raises = 0


class SlruEvictPolicy(VictimOverride):
    """S-LRU 的 QSTRUCT 候选修正，同时是「节点共享会话数」账本的唯一所有者。

    账本（本类独占）
    ----------------
    共享会话账（share_sessions / session_bloom，存于 _SLRU_META 侧表）的唯一
    写入口是 `on_node_hit()`；分裂时的继承拷贝走 `on_node_inserted()`，分别由
    缓存层的 `_hit_observers` / `_insert_observers` 调度，**与 `force_agentic`
    等实验开关无关**。
    共享账本由本类独占持有（WeakKeyDictionary，节点被 GC 时条目自动消失），
    TreeNode 不再内嵌插件字段；树结构维护（节点创建、分裂）仍留在 radix_cache。

    计数口径：**每个不同会话 +1，创建者在插入那一刻就计入**，同一会话
    （含创建者后续轮次的重读）由 `session_bloom` 去重，饱和在 `SESSION_SHARE_MAX`。
    因此 ``share_min = 2`` 表示「共 2 个会话用过」＝创建者 + 1 次外部命中。

    为什么要记创建者：计数只对活节点有意义，节点被删就归零。若不计创建者，
    共享前缀要等第二次外部命中才达标，这段窗口里它仍是一次性内容，
    会被 one-shot 带连同死链一起删掉，重建后计数又从 0 开始——高压力下
    保护状态会永远建立不起来。

    候选修正
    --------
    `applies_to` 命中「STRUCT 会话 + 已被 >= share_min 个外部会话命中过」
    的叶子，`pick` 在它们中间取最久未访问者（段内 LRU）。
    阈值来自 `SlruParams.share_min`，与 `is_one_shot` 的分界同源。
    """

    # 观测字段：本改写器管辖的队列，仅用于 profile 记账。
    # 不得参与仲裁判定（否则退化为 proposer 形态，见设计讨论）。
    queue: Optional[SimpleHybridQueueTag] =  SimpleHybridQueueTag.QSTRUCT

    def __init__(self, session_registry=None, share_min=None) -> None:
        # 与策略共用同一本会话账（SimpleHybridStrategy.__init__ 注入），
        # 保证 is_one_shot 与 applies_to 看到同一份 session_type。
        self.session_registry = session_registry or SessionRegistry.global_instance()
        # 共享数阈值：唯一配置入口是 SlruParams.share_min（由策略注入）。
        # 它同时决定「是否受 SLRU 保护」与「是否退出 one-shot 带」。
        self.share_min = max(
            2,
            int(DEFAULT_SHARE_MIN if share_min is None else share_min),
        )
        # 共享会话数账本的观测（唯一写者就是本类）
        self.share_counted = 0   # 计入 share_sessions 的命中次数
        self.share_dedup = 0     # 同一会话重复命中的去重次数
        self.slru_picks = 0      # 本改写器实际给出的候选数

    # ------------------------------------------------------------ 共享会话数账本

    def _record_session(self, node, session_id) -> bool:
        """把 (node, session) 记进账本（_SLRU_META 侧表）；已记过则去重。返回是否真的新增。"""
        if node is None or not session_id:
            return False
        bits = session_bloom_bits(session_id)
        entry = _SLRU_META.setdefault(node, [0, 0])
        if entry[1] & bits == bits:
            self.share_dedup += 1
            return False
        entry[1] |= bits
        entry[0] = min(SESSION_SHARE_MAX, entry[0] + 1)
        self.share_counted += 1
        return True

    def on_node_inserted(self, node) -> None:
        """分裂回调：新父节点继承 child 的共享账本条目。

        分裂不改变「哪些请求经过这段前缀」，两半都继承（旧实现是
        radix_cache 分裂路径里的显式字段拷贝）；分裂出的新父节点只有一个
        子树。非分裂的普通插入不带子节点，自然落回 (0, 0) 缺省，不建条目。
        """
        if not node.children:
            return
        share = 0
        bloom = 0
        for child in node.children.values():
            ce = _SLRU_META.get(child)
            if ce:
                if ce[0] > share:
                    share = ce[0]
                bloom |= ce[1]
        if share or bloom:
            _SLRU_META[node] = [share, bloom]

    def on_node_hit(self, node, session_id=None) -> None:
        """命中路径的记账钩子：不同会话数 +1（同一会话去重）。

        由缓存层 `RadixCache._record_node_hit()` 在更新 `queue_tag` 之前调用，
        保证同一次命中里账本与标签看到的是同一份事实。
        """
        self._record_session(node, session_id)

    @staticmethod
    def share_of(node) -> int:
        """节点已被多少个不同会话命中过（创建者首次重读计 1）；读 _SLRU_META 侧表。"""
        entry = _SLRU_META.get(node)
        return int(entry[0]) if entry else 0

    # ------------------------------------------------------------ 插件接口

    def applies_to(self, node) -> bool:
        """当前候选是否归本改写器管。"""
        if node is None:
            return False
        st = self.session_registry.get(getattr(node, "session_id", None))
        stype = st.session_type if st is not None else None
        return (
            stype == SessionType.STRUCT
            and self.share_of(node) >= self.share_min
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

        共享账本条目在 _SLRU_META 侧表，随节点 GC 一起丢弃，不用清。
        """
        self.share_counted = 0
        self.share_dedup = 0
        self.slru_picks = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"
