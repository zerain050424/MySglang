"""HeterCache 的队内候选策略（VictimOverride）。

配额层先按队列把叶子筛成 ``pool``，再交给该队列的队内策略决出具体牺牲节点。
早期版本是"全局 ``victim_key`` 先选一个基础候选，再问改写器要不要改写"，
全局复用分删除后这条链已不存在，改写器变成了真正意义上的**队内选点者**：

* ``pick(leaves, registry) -> Optional[TreeNode]``：在给定池内选点，返回
  ``None`` 表示弃权（调用方跳下一个队列）。

四条硬约束（违反任何一条都会让淘汰循环出错）：

1. **只挑不删**：改写器不得释放 KV / 改树，副作用统一由缓存层 ``_evict_one``
   执行，否则同一节点会被二次释放。
2. **必可弃权**：返回 ``None`` 表示本队列无候选，调用方跳下一个队列；改写器
   也不许抛错阻塞淘汰推进（会导致 evict 自旋）。
3. **候选集封闭**：只能在传入的 ``leaves`` 内重选，不得构造新节点。
4. **观测与仲裁分离**：``queue`` 只用于 profile 归类，不参与仲裁；跨队列仲裁
   属于 ``quota_selector``。

内置实现：``RrEvictPolicy``（会话轮询）、``SlruEvictPolicy``（分段 LRU + 共享数
账本）、``MultiTurnEvictPolicy``（多轮带内 1-prob(e) 排序）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
import weakref
from sglang.srt.mem_cache.heter_utils import (
    DEFAULT_SHARE_MIN,
    RrParams,
    SESSION_SHARE_MAX,
    SessionRegistry,
    SessionType,
    HeterCacheNodeTag,
    MultiTurnParams,
    _cached_lognorm_cdf,
    node_end_offset,
    session_bloom_bits,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.heter_utils import SessionState


# ---- HeterCache 插件的侧表 ----
# 插件私有状态不内嵌到 TreeNode 上（那会污染所有缓存实现），改用本模块的侧表。
# WeakKeyDictionary 以节点为弱键：节点被 GC（驱逐/分裂清理/reset）时条目自动
# 消失，生命周期与内嵌字段等价，因此不需要任何显式清理钩子。
_SLRU_META = weakref.WeakKeyDictionary()  # node -> [share_sessions, session_bloom]


class VictimOverride(ABC):
    """队内策略的基类：选点者实现 ``pick``，纯观测者可以不实现。

    基类不再把 ``pick`` 定为抽象方法——SLRU 现在只是共享数账本 + 生命周期
    观测器，struct 队的选点由 ``quota_selector.struct_tier`` 直接完成，让它
    再实现一个永远收不到调用的 ``pick`` 就是死代码。
    """

    name: str = "override"
    # 本改写器管辖的队列；**仅**用于 profile 归类，不得参与仲裁判定。
    queue: Optional[HeterCacheNodeTag] = None

    def pick(self, leaves: Iterable["TreeNode"], registry) -> Optional["TreeNode"]:
        """在管辖范围内重选候选；返回 None 表示弃权。

        Raises:
            NotImplementedError: 纯观测型子类不参与选点。
        """
        raise NotImplementedError(
            f"{type(self).__name__} is an observer, not a victim picker"
        )

    def stats(self) -> Dict[str, float]:
        """返回本改写器的观测量；无观测的插件返回空字典。"""
        return {}

    def reset(self) -> None:
        """清空内部状态（缓存 reset / 跨 run 复用同一进程时调用）。"""

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"


class RrEvictPolicy(VictimOverride):
    """Agentic 队列的会话级轮询（round-robin）。

    目的：agentic 队列里常有几十条并行会话，按全局分数淘汰会让同一条会话
    连续挨刀，下一轮 prefill 时集中爆发。这里改成"按会话轮着削"：

    * 只处理 agentic 叶子，并归到所属会话（每会话只取最深的叶子——删它才能
      一次腾出最多空间）；
    * 本轮缺口 ``d`` 已达上限的会话直接跳过，避免把一条会话掏空；
    * 其余会话按"最久未被选中"排序，实现跨会话轮询。

    Args:
        params: ``RrParams``；``fair_rr`` 为假时整个改写器弃权。
    """

    name = "rr"
    queue = HeterCacheNodeTag.QAGENT

    def __init__(self, params: RrParams):
        self.params = params
        self._rr_last: Dict[str, int] = {}
        self._rr_seq = 0
        self.quota_breach = 0

    # ---- 参数 ----

    def loss_cap(self, chain_len: int) -> float:
        """单会话在本轮可以承受的 KV 缺口上限（token）。

        取 ``kappa * L`` 与 ``L - keep`` 的较小值：前者限制"一次最多削掉多少
        比例"，后者保证会话至少留下 ``min_retain_*`` 的底。

        Args:
            chain_len: 会话当前链长 L（token）。

        Returns:
            允许的缺口上限，``>= 0``。
        """
        keep = max(
            self.params.min_retain_tokens, self.params.min_retain_frac * chain_len
        )
        return max(
            0.0, min(self.params.kappa * chain_len, chain_len - keep)
        )

    # ---- 插件接口 ----


    def pick(self, leaves, registry, *, max_deficit=None) -> Optional["TreeNode"]:
        """按会话轮询选一个 agentic 叶子；没有可动会话时返回 None。

        Args:
            leaves: 当前可淘汰叶子集合。
            registry: 会话账本，提供每会话的 ``d`` / ``L``。
            max_deficit: 配额缺口上限（token）。跨队列配额点名 agentic 时传入，
                保证单次淘汰不会为了追配额而削穿一条会话；弹性区为 None，
                退回 ``loss_cap`` 语义。

        Returns:
            选中的叶子；无候选时返回 None 并累加违约计数。
        """
        if not self.params.fair_rr:
            return None
        agentic = [
            node
            for node in leaves
            if HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
            is HeterCacheNodeTag.QAGENT
        ]
        if not agentic:
            return None
        victim = self._pick_round_robin(agentic, registry, max_deficit=max_deficit)
        if victim is None:
            self.quota_breach += 1
        return victim

    # ---- 内部 ----

    def _pick_round_robin(self, leaves, registry, max_deficit=None) -> Optional["TreeNode"]:
        """挑"最久未被选中且缺口未达上限"的会话，返回它最深的叶子。

        Args:
            leaves: 已限定在 agentic 范围内的叶子。
            registry: 会话账本。
            max_deficit: 本轮缺口上限；与 ``loss_cap(L)`` 取 min，即"配额缺口
                不得把单个会话削穿"。

        Returns:
            选中的叶子；所有会话都达上限时返回 None（由调用方回退）。
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
                # 每会话只留最深叶子：删它才能一次腾出最多空间。
                tails[sid] = (node, state)

        best = None
        for sid, (node, state) in tails.items():
            limit = self.loss_cap(state.L)
            if max_deficit is not None:
                limit = min(limit, max_deficit)
            if state.d >= limit:
                # 缺口已达上限（loss_cap 或配额缺口），本轮不再动它。
                continue
            last = self._rr_last.get(sid, -1)  # -1 = 从未被选中，优先照顾
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
    """共享会话数账本 + 节点生命周期观测器。

职责：

1. **账本**：维护每个节点被多少个不同的会话命中过（``share``）与一份 bloom 位图
   （用于同会话去重）。这台账本被两处读取——``HeterCacheStrategy.is_one_shot``
   判 one-shot，和 ``quota_selector.struct_tier`` 判"是否受保护"，所以阈值只允许
   一个来源（``SlruParams.share_min``），否则两处口径会漂移。
2. **生命周期**：``on_node_hit`` / ``on_node_inserted`` 由缓存层调用。

**不参与选点**：struct 队的候选由 ``struct_tier`` 直接排，本类原来的 ``pick``
已删除——它永远收不到调用，``slru_picks`` 也恒为 0。

账本写入点只有两个，都由缓存层调度：``on_node_hit``（命中去重计数）与
``on_node_inserted``（分裂时新父节点继承）。树拓扑维护仍留在 RadixCache。

Args:
    session_registry: 会话账本；省略时取全局单例。
    share_min: 保护阈值；省略时取 ``DEFAULT_SHARE_MIN``，下限 2。
"""

    queue: Optional[HeterCacheNodeTag] =  HeterCacheNodeTag.QSTRUCT

    def __init__(self, session_registry=None, share_min=None) -> None:
        # 与策略共用同一本会话账（由 HeterCacheStrategy 注入）：is_one_shot 与
        # pick 必须看到同一份 session_type，否则"受保护"与"one-shot"会打架。
        self.session_registry = session_registry or SessionRegistry.global_instance()
        self.share_min = max(
            2,
            int(DEFAULT_SHARE_MIN if share_min is None else share_min),
        )
        # 共享会话数账本的观测（唯一写者就是本类）
        self.share_counted = 0   # 计入 share_sessions 的命中次数
        self.share_dedup = 0     # 同一会话重复命中的去重次数

    # ------------------------------------------------------------ 共享会话数账本

    def _record_session(self, node, session_id) -> bool:
        """把 (node, session) 记进共享数账本；同会话重复命中只算一次。

        去重用 64 bit bloom 而不是集合：会话数上万时集合会持续涨内存，而
        bloom 的误判方向是"当作已记过"（少计一次），对本策略足够安全。

        Args:
            node: 被命中的节点。
            session_id: 命中该节点的会话；缺失时不计数。

        Returns:
            True 表示该会话是首次命中，共享数确实 +1。
        """
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
        """分裂回调：新父节点继承子节点的共享账本条目。

        分裂不改变"哪些会话经过这段前缀"，所以两半都该继承。新父节点只有一个
        子树，取该子树的值即可；普通插入（不带子节点）直接返回，不建条目。

        Args:
            node: 新挂上树的节点（普通插入的叶子，或分裂出来的父节点）。
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
        """命中去重记账：新会话命中则共享数 +1。

        必须在缓存层更新 ``queue_tag`` 之前调用：同一次命中里账本与标签要看到
        同一份事实，否则 one-shot 判定会用到过期的共享数。

        Args:
            node: 被命中的节点。
            session_id: 命中该节点的会话。
        """
        self._record_session(node, session_id)

    @staticmethod
    def share_of(node) -> int:
        """返回节点被多少个不同会话命中过（创建者自己的首次重读计 1）。

        读侧表，无条目按 0：节点从未被命中过就不该享受共享保护。
        """
        entry = _SLRU_META.get(node)
        return int(entry[0]) if entry else 0

    # ------------------------------------------------------------ 插件接口


    def stats(self) -> Dict[str, float]:
        return {
            "share_counted": self.share_counted,
            "share_dedup": self.share_dedup,
        }

    def reset(self) -> None:
        """只清计数器。共享账本条目在侧表里随节点 GC 一起消失，不需要清。"""
        self.share_counted = 0
        self.share_dedup = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"


class MultiTurnEvictPolicy(VictimOverride):
    """CHAT 队列的带内改写器：用 lognormal 生存概率挑最冷的节点。

权重取"复用已经过期的概率"（UniCache 原机制）：

    w = 1 - reuse_prob = CDF(elapsed; mu, sigma)

权重取对数正态的 ``CDF(elapsed)``：``1 - CDF`` 才是"还没被复用"的概率，
所以这里**越大越先淘汰**。同权重时退化为段内 LRU。

带内归属按 **SessionType ∈ {CHAT, AGENTIC}** 判定，而不是节点标签：分类器产出
的是会话类型，``queue_tag`` 不一定与之对齐。本类只处理传入的池，绝不跨带取点。
"""

    name = "mt"
    queue = HeterCacheNodeTag.QCHAT

    def __init__(self, params: MultiTurnParams, registry=None) -> None:
        self.params = params
        # 与 SLRU / one-shot 同用一本会话账：带内归属只认 SessionType，
        # 不认 queue_tag（标签会因分叉、命中间隔等原因漂移）。
        self.registry = registry or SessionRegistry.global_instance()
        # ---- 观测 ----
        self.mt_picks = 0       # 带内实际改写的次数
        self.mt_fallbacks = 0   # 管辖带内无可淘汰 多轮叶子

    # ------------------------------------------------------------ 插件接口


    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """在多轮带内挑"过期概率"最高的节点；带内无候选时返回 None。

        Args:
            leaves: 当前可淘汰叶子集合。
            registry: 未使用；保留基类签名。

        Returns:
            选中的叶子；带内无候选时返回 None 并累加 fallback 计数。
        """
        mt_nodes = [
            node
            for node in (leaves or ())
            if (st := self.registry.get(getattr(node, "session_id", None))) is not None
            and st.session_type in (SessionType.CHAT, SessionType.AGENTIC)
        ]
        if not mt_nodes:
            self.mt_fallbacks += 1
            return None
        # 参数统一取自 HeterCacheNodeTag.reuse_config：多轮带的排序与其它
        # 复用间隔判断共用一个分布定义，避免两处各写一份 mu/sigma。
        best = None
        for node in mt_nodes:
            st = self.registry.get(getattr(node, "session_id", None))
            stype = st.session_type if st is not None else SessionType.CHAT
            cfg_key = (
                HeterCacheNodeTag.QAGENT
                if stype == SessionType.AGENTIC
                else HeterCacheNodeTag.QCHAT
            )
            mu, sigma = cfg_key.reuse_config
            elapsed = time.monotonic() - node.last_access_time
            weight = _cached_lognorm_cdf(int(elapsed), mu, sigma)  # 1-prob(e)
            # 权重大者先走；同分用 -last_access 让更久未访问的胜出（key 取 max）。
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
