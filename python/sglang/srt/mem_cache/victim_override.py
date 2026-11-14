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
账本）、``MultiTurnEvictPolicy``（agentic 带内 1-prob(e) 排序）、
``MultiAgenticEvict``（agentic 带内复用分排序）。
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
    HeterCacheNodeTag,
    MultiAgenticParams,
    MultiTurnParams,
    node_end_offset,
    node_num_tokens,
    reuse_window_probability,
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
    """CHAT 队列的会话级轮询（round-robin）。

    目的：多轮会话轮间间隔长且互不相同，按全局分数淘汰会让同一条会话
    连续挨刀，下一轮 prefill 时集中爆发。这里改成"按会话轮着削"：

    * 只处理调用方传入的 chat 叶子（配额层已按队列筛过池），每会话只取最
      深的叶子——删它才能一次腾出最多空间；
    * 本轮缺口d已达上限的会话直接跳过，避免把一条会话掏空；
    * 其余会话按"最久未被选中"排序，实现跨会话轮询。

    Args:
        params: ``RrParams``；``fair_rr`` 为假时整个改写器弃权。
    """

    name = "rr"
    queue = HeterCacheNodeTag.QCHAT

    def __init__(self, params: RrParams):
        self.params = params
        self._rr_last: Dict[str, int] = {}
        self._rr_seq = 0
        self.quota_breach = 0

    # ---- 参数 ----

    def loss_cap(self, chain_len: int) -> float:
        """单会话在本轮可以承受的 KV 缺口上限（token）。

        取kappa * L与L - keep的较小值：前者限制一次最多削掉多少
        比例，后者保证会话至少留下min_retain_*的底。

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
        """按会话轮询选一个 chat 叶子；没有可动会话时返回 None。

        Args:
            leaves: 配额层筛过的候选池（已限定在 chat 队列）。
            registry: 会话账本，提供每会话的 ``d`` / ``L``。
            max_deficit: 配额缺口上限（token）。跨队列配额点名 chat 时传入，
                保证单次淘汰不会为了追配额而削穿一条会话；弹性区为 None，
                退回 ``loss_cap`` 语义。

        Returns:
            选中的叶子；无候选时返回 None 并累加违约计数。
        """
        if not self.params.fair_rr:
            return None
        pool = list(leaves or ())
        if not pool:
            return None
        victim = self._pick_round_robin(pool, registry, max_deficit=max_deficit)
        if victim is None:
            self.quota_breach += 1
        return victim

    # ---- 内部 ----

    def _pick_round_robin(self, leaves, registry, max_deficit=None) -> Optional["TreeNode"]:
        """挑"最久未被选中且缺口未达上限"的会话，返回它最深的叶子。

        Args:
            leaves: 已限定在 chat 范围内的叶子。
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

    1. 维护每个节点被多少个不同的会话命中过与一份 bloom 位图
    2. 生命周期：on_node_hit/n_node_inserted由缓存层调用。
    Args:
        session_registry: 会话账本；省略时取全局单例。
        share_min: 保护阈值；省略时取 ``DEFAULT_SHARE_MIN``，下限 2。
    """

    queue: Optional[HeterCacheNodeTag] =  HeterCacheNodeTag.QSTRUCT

    def __init__(self, session_registry=None, share_min=None) -> None:
        # 与策略共用同一本会话账（由 HeterCacheStrategy 注入）：is_one_shot 与
        self.session_registry = session_registry or SessionRegistry.global_instance()
        self.share_min = max(
            2,
            int(DEFAULT_SHARE_MIN if share_min is None else share_min),
        )
        # 共享会话数账本的观测
        self.share_counted = 0   # 计入 share_sessions 的命中次数
        self.share_dedup = 0     # 同一会话重复命中的去重次数

    # ------------------------------------------------------------ 共享会话数账本

    def _record_session(self, node, session_id) -> bool:
        """把 (node, session) 记进共享数账本；同会话重复命中只算一次。
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
        Args:
            node: 被命中的节点。
            session_id: 命中该节点的会话。
        """
        self._record_session(node, session_id)

    @staticmethod
    def share_of(node) -> int:
        """返回节点被多少个不同会话命中过（创建者自己的首次重读计 1）。
        读侧表，无条目返回 0：节点从未被命中过就不该享受共享保护。
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
        self.share_counted = 0
        self.share_dedup = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"


class MultiTurnEvictPolicy(VictimOverride):
    """AGENTIC 队列的带内改写器：挑未来窗口内最不可能被复用的节点。

    分数取条件生存概率（H = ``reuse_window_s``）：

        p = P(reuse in (a, a+H] | not reused by a) = 1 - S(a+H)/S(a)

    a 是距上次访问的秒数，S = 1 - CDF（lognormal，参数来自
    ``HeterCacheNodeTag.reuse_config``）。p 越小越先淘汰；同分退化为段内 LRU。
    """

    name = "mt"
    queue = HeterCacheNodeTag.QAGENT

    def __init__(self, params: MultiTurnParams, registry=None, tag_of=None) -> None:
        self.params = params
        self.registry = registry or SessionRegistry.global_instance()
        # 生效标签而非原始 queue_tag：force_agentic 下全池都按 QAGENT 评分。
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        # ---- 观测 ----
        self.mt_picks = 0       # 带内实际改写的次数
        self.mt_fallbacks = 0   # 管辖带内无可淘汰叶子

    # ------------------------------------------------------------ 插件接口


    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """在 agentic 带内挑未来窗口内复用概率最低的节点；无候选时返回 None。

        Args:
            leaves: 配额层筛过的候选池（已限定在 agentic 队列）。
            registry: 未使用；保留基类签名。

        Returns:
            选中的叶子；带内无候选时返回 None 并累加 fallback 计数。
        """
        # 参数统一取自 HeterCacheNodeTag.reuse_config：带内排序与其它
        # 复用间隔判断共用一个分布定义，避免两处各写一份 mu/sigma。
        candidates = []
        for node in leaves or ():
            cfg = self._tag_of(node).reuse_config
            if cfg is not None:
                candidates.append((node, cfg))
        if not candidates:
            self.mt_fallbacks += 1
            return None
        # 一次取时钟：同一轮里所有节点必须用同一个 elapsed 基准，否则顺序会
        # 随遍历位置漂移，同 seed 的对照实验无法复现。
        now = time.monotonic()
        best = None
        for node, (mu, sigma) in candidates:
            score = reuse_window_probability(
                int(now - node.last_access_time),
                self.params.reuse_window_s,
                mu,
                sigma,
            )
            # 分数 = 未来 H 秒内被复用的概率：越小越先走；
            # 同分取更久未访问的（last_access 小者），保住段内 LRU 语义。
            key = (score, node.last_access_time)
            if best is None or key < best[0]:
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
        self.mt_picks = 0
        self.mt_fallbacks = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"


class MultiAgenticEvict(VictimOverride):
    """AGENTIC 队列的复用分改写器：按复用分升序淘汰。

    复用分只在本队内排序（不是跨队列的全局价值分）：

        reuse_score(node) = P(reuse in (a, a+H] | not reused by a)
                          = 1 - S(a+H)/S(a)

    a 是节点距上次访问的秒数，(mu, sigma) 取自
    HeterCacheNodeTag.reuse_config（agentic 节点即 QAGENT:(1.81, 1.092)）。
    分数越高说明未来 H 秒越可能被复用，越该保留；淘汰取最低分。

    与MT的分工：MT 用同一个分数，同分时退化为段内 LRU；本类同分时优先
    弃大片段（一次腾更多空间）。两者互斥且本类优先，开关是
    ``MultiAgenticParams.enabled``；关闭时策略侧不构造本类，agentic 队回落到
    MT / 兜底。
    """

    name = "multi_agentic"
    queue = HeterCacheNodeTag.QAGENT

    def __init__(self, params: MultiAgenticParams, tag_of=None) -> None:
        self.params = params
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        # ---- 观测 ----
        self.ma_picks = 0      # 实际给出的 agentic 候选数
        self.ma_fallbacks = 0  # 管辖带内无可淘汰叶子

    # ------------------------------------------------------------ 复用分

    def reuse_score(self, node, now: Optional[float] = None) -> float:
        """节点的复用分：未来 H 秒内被复用的条件概率，值域 [0, 1]。

        无时间维度的标签（如 QSTRUCT）返回 0，即最不保值；本类只处理 agentic
        池，那条分支仅作为口径兜底（仅非 ``force_agentic`` 的手工调用才能碰到）。
        """
        cfg = self._tag_of(node).reuse_config
        if cfg is None:
            return 0.0
        mu, sigma = cfg
        elapsed = int(
            (time.monotonic() if now is None else now) - node.last_access_time
        )
        return reuse_window_probability(
            elapsed, self.params.reuse_window_s, mu, sigma
        )

    def victim_key(self, node, now: Optional[float] = None) -> Tuple[float, int]:
        """队内排序键：复用分升序（低分先走），同分时大片段先走。"""
        return (self.reuse_score(node, now), -node_num_tokens(node))

    # ------------------------------------------------------------ 插件接口

    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """在 agentic 池内挑复用分最低的节点；带内无候选时返回 None。

        Args:
            leaves: 当前可淘汰叶子集合。
            registry: 未使用；保留基类签名。

        Returns:
            选中的叶子；无 agentic 叶子时返回 None 并累加 fallback 计数。
        """
        agentic = [node for node in (leaves or ()) if self._tag_of(node) is self.queue]
        if not agentic:
            self.ma_fallbacks += 1
            return None
        # 一次取时钟：同一轮里所有节点必须用同一个 elapsed 基准，否则顺序会
        # 随遍历位置漂移，同 seed 的对照实验无法复现。
        now = time.monotonic()
        self.ma_picks += 1
        return min(agentic, key=lambda node: self.victim_key(node, now))

    # ------------------------------------------------------------ 观测

    def stats(self) -> Dict[str, float]:
        return {
            "ma_picks": self.ma_picks,
            "ma_fallbacks": self.ma_fallbacks,
        }

    def reset(self) -> None:
        """只清计数器；复用分无状态，无需清理。"""
        self.ma_picks = 0
        self.ma_fallbacks = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"
