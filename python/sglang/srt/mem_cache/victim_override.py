"""VictimOverride: 淘汰候选节点改写器插件。

【功能定位】
将多样化的淘汰裁决逻辑从SimpleHybridStrategy中解耦，提供可插拔的候选重选能力。
允许不同策略根据自身状态（如会话配额、分段水位等），对全局选出的默认淘汰候选进行重定向。

【核心接口协议】
    - applies_to(node) -> bool:
        判断当前候选节点是否落在该改写器的管辖范围内。
    - pick(leaves, registry) -> Optional[Node]:
        在合法的叶子节点池中挑出本策略优先淘汰的节点；返回 None 表示弃权/不干预。

【调度规则与硬性约束】
1. 链式覆盖原则（Last-in Precedence）：
   - 缓存层按注册顺序逐个询问改写器，后注册者若生效将覆盖前者的选择（如：会话级 RR 策略先修正一次，结构队列策略若命中管辖权可再次覆盖）。
2. 只挑不删（Separation of Concerns）：
   - 改写器仅负责挑选候选节点，绝对不执行释放 KV 或删除节点的操作，最终驱逐仍由 `_evict_one()` 统一落盘执行。
3. 安全回退（Fall-through Guarantee）：
   - 若返回 None 或无法决策，必须立即安全回退到原候选节点，严禁阻塞淘汰推进（防止 evict 循环自旋）。
4. 候选集封闭（Scope Boundary）：
   - 只能在传入的 `leaves`（可淘汰叶子集合）内部做重选，禁止跨范围构造或越界选取新节点。
5. 观测与仲裁分离：
   - 节点的 `queue` 属性仅用于性能分析与日志观测（记录该淘汰归属于哪个策略队列），绝对不得参与跨队列的仲裁判定（跨队列仲裁统一由全局 `victim_key` 排序决定）。

【内置策略实现清单】
- RrEvictPolicy:
    实现会话级轮询公平性（Round-Robin），内部维护独立的会话轮询序号及配额违约计数。
- SlruEvictPolicy:
    分段式 LRU（Probation → Protected，段内按 recency 排序）。同时独占维护节点共享会话数账本（通过 `on_node_hit` 触发）。
    配合 `scope="non_one_shot"` 可组装为带 One-shot 隔离带的 SLRU模式。

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
    MultiTurnParams,
    TASK_CONFIG,
    _cached_lognorm_cdf,
    node_end_offset,
    session_bloom_bits,
    task_type_of_tag,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.uni_utils import SessionState


# ---- HeterCache 插件侧表：替代 TreeNode 上的内嵌插件状态----
# 设计：策略插件私有状态不再内嵌 TreeNode 字段，改由本模块持侧表。
# WeakKeyDictionary 以节点为弱键：节点被 GC（分裂清理/驱逐/reset）时
# 条目自动消失，与旧内嵌字段生命周期等价，无需显式清理钩子。
# 键为 TreeNode，读侧缺省为 0。
_SLRU_META = weakref.WeakKeyDictionary()  # node -> [share_sessions, session_bloom]


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

    def pick(self, leaves, registry, *, max_deficit=None) -> Optional["TreeNode"]:
        if not self.params.fair_rr:
            return None
        agentic = [node for node in leaves if self.applies_to(node)]
        if not agentic:
            return None
        victim = self._pick_round_robin(agentic, registry, max_deficit=max_deficit)
        if victim is None:
            self.quota_breach += 1
        return victim

    # ---- 内部 ----

    def _pick_round_robin(self, leaves, registry, max_deficit=None) -> Optional["TreeNode"]:
        """挑最久未被删除的 Agentic 会话的最深叶子。

        `max_deficit` 是外部（跨队列配额）给的上限：本轮允许把单会话缺口再推高
        到多少 token。与 `loss_cap(L)` 取 min，即“配额缺口不得把单个会话削穿”。
        """
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
            limit = self.loss_cap(state.L)
            if max_deficit is not None:
                limit = min(limit, max_deficit)
            if state.d >= limit:
                # 本会话缺口已经够大（loss_cap 或配额缺口），本轮不再动它
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


class SlruEvictPolicy(VictimOverride):
    """S-LRU 针对 QSTRUCT 队列的淘汰候选改写器。

    【功能定位】
    本类承担双重职责：
    1. 独占维护全局的节点共享会话数（Node Share Sessions）账本。
    2. 针对 STRUCT 队列中脱离一次性带（One-shot）、进入共享保护段的节点，执行段内 LRU 候选改写。

    【账本维护机制（独占所有权）】
    - 独占写入点：
    共享会话账（`share_sessions` / `session_bloom`，存于本模块 `_SLRU_META`
    侧表）的唯一写入口为 `on_node_hit()`；分裂时的继承拷贝走
    `on_node_inserted()`。两者的触发分别由缓存层 `_hit_observers` 与
    `_insert_observers` 统一调度。
    - 职责边界划分：
    树基础拓扑维护（节点创建、分裂）仍留在 RadixCache 树核心层；共享账本
    由本类以侧表独占持有（节点被正常 GC 时条目随之消失），节点字段不再内嵌。

    【计数口径与设计权衡】
    - 去重与饱和计数：
    按独立会话为粒度计数，通过 `session_bloom` 过滤同一会话内的后续多轮重复命中，计数上限受限于 `SESSION_SHARE_MAX`。

    【候选改写裁决（Victim Selection）】
    - `applies_to(node) -> bool`:
    筛选落在管辖范围内的候选叶子：属于 STRUCT 会话，且跨会话使用计数已达标（`share_sessions >= share_min`）。
    - `pick(leaves, registry) -> Optional[Node]`:
    在受管辖的叶子节点集合中，挑选访问时间戳最早的节点返回（执行段内 LRU）。
    - 判定边界：
    阈值直接采用 `SlruParams.share_min`，确保候选拦截范围与系统的 `is_one_shot` 判定标准源头一致。
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


class MultiTurnEvictPolicy(VictimOverride):
    """多轮对话的带内改写器：复用 UniCache 的 lognormal 复用机制
    （带内归属按 SessionType ∈ {CHAT, AGENTIC} 判定）。

    【权重定义（1 - prob(e)）】
    带内排序权重 = 复用时间已过期的概率：

        w = 1 - reuse_prob = CDF(elapsed; μ, σ)

    其中 reuse_prob = 1 - CDF 是 UniCache calculate_priority 对多轮任务
    的原有口径（lognormal，μ/σ 来自 TASK_CONFIG）；**越大越先淘汰**。
    同权重时退化为段内 LRU（最久未访问先走）。

    【接管语义】
    带内归属按 **SessionType ∈ {CHAT, AGENTIC}** 判定（多轮两类负载），
    而非节点标签——分类器产出会话类型，NodeTag 不一定与之对齐。调用链
    的 applies_to(victim) 门禁只对落入本带的 victim 接管，绝不跨带改写。

    """

    name = "mt"
    queue = SimpleHybridQueueTag.QCHAT

    def __init__(self, params: MultiTurnParams, registry=None) -> None:
        self.params = params
        # 会话账本：applies_to 按 SessionType 判定（节点标签与会话类型可能不对齐，
        # 与 SlruEvictPolicy / is_one_shot 同用一本账，判定同源）。
        self.registry = registry or SessionRegistry.global_instance()
        # ---- 观测 ----
        self.mt_picks = 0       # 带内实际改写的次数
        self.mt_fallbacks = 0   # 管辖带内无可淘汰 多轮叶子

    # ------------------------------------------------------------ 插件接口

    def applies_to(self, node) -> bool:
        """按 SessionType 判定（CHAT / AGENTIC 两类多轮负载）。

        分类器产出的是**会话类型**，而节点标签（queue_tag）不一定与之对齐；
        因此带内归属只认 SessionRegistry——与 one-shot 判定同源，规避标签
        漂移。无会话证据（session_id 缺失，或账本已被 sweep）时不接管，
        保留原候选。
        """
        if node is None:
            return False
        st = self.registry.get(getattr(node, "session_id", None))
        return st is not None and st.session_type in (
            SessionType.CHAT,
            SessionType.AGENTIC,
        )

    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        mt_nodes = [node for node in (leaves or ()) if self.applies_to(node)]
        if not mt_nodes:
            self.mt_fallbacks += 1
            return None
        # CHAT 取 TASK_CONFIG[MULTI_TURN_CHAT]（μ=4.77 / σ=1.279）；AGENTIC 取
        # TASK_CONFIG[MULTI_TURN_AGENTIC]（μ=1.81 / σ=1.092）——与 _reuse_score 同一套参数。
        best = None
        for node in mt_nodes:
            st = self.registry.get(getattr(node, "session_id", None))
            stype = st.session_type if st is not None else SessionType.CHAT
            cfg_key = (
                SimpleHybridTaskType.MULTI_TURN_AGENTIC
                if stype == SessionType.AGENTIC
                else SimpleHybridTaskType.MULTI_TURN_CHAT
            )
            mu, sigma = TASK_CONFIG[cfg_key]
            elapsed = time.monotonic() - node.last_access_time
            weight = _cached_lognorm_cdf(int(elapsed), mu, sigma)  # 1-prob(e)
            # 权重大者先走；同分退化为 LRU（更久未访问 = 更小的 last_access）
            key = (weight, -node.last_access_time)
            if best is None or key > best[0]:
                best = (key, node)
        self.mt_picks += 1
        return best[1]

    # ------------------------------------------------------------ 观测

    def stats(self) -> Dict[str, float]:
        return {
            "mt_picks": self.mt_picks,
            "mt_fallbacks": self.mt_fallbacks,
        }

    def reset(self) -> None:
        """只清计数器；1-prob 权重无状态，无需清理。"""
        self.mt_picks = 0
        self.mt_fallbacks = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"
