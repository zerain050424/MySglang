
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
from sglang.srt.mem_cache.heter_cache.utils import (
    RrParams,
    SessionRegistry,
    HeterCacheNodeTag,
    MultiAgenticParams,
    MultiTurnParams,
    node_end_offset,
    node_num_tokens,
    reuse_window_probability,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.heter_cache.utils import SessionState


class VictimOverride(ABC):
    """队内选点策略的基类：子类必须实现 ``pick``。

    账本不继承本类：共享数账本（``NodeShareLedger``）与占用账本一样不参与
    选点，挂在策略上由缓存层直接回调。
    """

    name: str = "override"
    # 本改写器管辖的队列；``pick`` 用它过滤候选池（见 MultiAgenticEvict）。
    queue: Optional[HeterCacheNodeTag] = None

    @abstractmethod
    def pick(self, leaves: Iterable["TreeNode"], registry) -> Optional["TreeNode"]:
        """在管辖范围内重选候选；返回 None 表示弃权。"""

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

    * 只处理调用方传入的 chat 叶子（权重层已按队列筛过池），每会话只取最
      深的叶子——删它才能一次腾出最多空间；
    * 本轮缺口d已达上限的会话直接跳过，避免把一条会话掏空；
    * 其余会话按"最久未被选中"排序，实现跨会话轮询。

    Args:
        params: RrParams；fair_rr为假时整个改写器弃权。
    """

    name = "rr"
    queue = HeterCacheNodeTag.QCHAT

    def __init__(self, params: RrParams):
        self.params = params
        self._rr_last: Dict[str, int] = {}
        self._rr_seq = 0
        self.deficit_breach = 0

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


    def pick(self, leaves, registry) -> Optional["TreeNode"]:
        """按会话轮询选一个 chat 叶子；没有可动会话时返回 None。

        Args:
            leaves: 权重层筛过的候选池（已限定在 chat 队列）。
            registry: 会话账本，提供每会话的 ``d`` / ``L``。

        Returns:
            选中的叶子；无候选时返回 None 并累加违约计数。
        """
        if not self.params.fair_rr:
            return None
        pool = list(leaves or ())
        if not pool:
            return None
        victim = self._pick_round_robin(pool, registry)
        if victim is None:
            self.deficit_breach += 1
        return victim

    # ---- 内部 ----

    def _pick_round_robin(self, leaves, registry) -> Optional["TreeNode"]:
        """挑最久未被选中且缺口未达上限的会话，返回它最深的叶子。

        Args:
            leaves: 已限定在 chat 范围内的叶子。
            registry: 会话账本。

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
            if state.d >= limit:
                # 缺口已达 loss_cap，本轮不再动它。
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
            "rr_deficit_breach": self.deficit_breach,
        }

    def reset(self) -> None:
        self._rr_last.clear()
        self._rr_seq = 0
        self.deficit_breach = 0


class MultiTurnEvictPolicy(VictimOverride):
    """AGENTIC 队列的带内改写器：挑未来窗口内最不可能被复用的节点。

    分数取条件生存概率（H =reuse_window_s）：

        p = P(reuse in (a, a+H] | not reused by a) = 1 - S(a+H)/S(a)

    a 是距上次访问的秒数，S = 1 - CDF p 越小越先淘汰；同分退化为段内 LRU。
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
            leaves: 权重层筛过的候选池（已限定在 agentic 队列）。
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
        reuse_score(node) = P(reuse in (a, a+H] | not reused by a)
                          = 1 - S(a+H)/S(a)
    """

    name = "multi_agentic"
    queue = HeterCacheNodeTag.QAGENT

    def __init__(self, params: MultiAgenticParams, tag_of=None, now=None) -> None:
        self.params = params
        self._now = now if now is not None else time.monotonic
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        # ---- 观测 ----
        self.ma_picks = 0      # 实际给出的 agentic 候选数
        self.ma_fallbacks = 0  # 管辖带内无可淘汰叶子

    def use_clock(self, now) -> None:
        """接管时间源（默认墙钟下等价于无操作）。

        复用分是 ``now - node.last_access_time`` 的差，节点时间戳由缓存盖；
        两边不同源时注入的假时钟会让分数整段失真。
        """
        self._now = now

    # ------------------------------------------------------------ 复用分

    def reuse_score(self, node, now: Optional[float] = None) -> float:
        """节点的复用分：未来 H 秒内被复用的条件概率，值域 [0, 1]"""
        cfg = self._tag_of(node).reuse_config
        if cfg is None:
            return 0.0
        mu, sigma = cfg
        elapsed = int(
            (self._now() if now is None else now) - node.last_access_time
        )
        return reuse_window_probability(
            elapsed, self.params.reuse_window_s, mu, sigma
        )

    def victim_key(self, node, now: Optional[float] = None) -> Tuple[float, int]:
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
        now = self._now()
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
