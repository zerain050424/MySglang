
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
from sglang.srt.mem_cache.heter_cache.utils import (
    ChainParams,
    ReuseParams,
    HeterCacheNodeTag,
    node_num_tokens,
    reuse_window_probability,
)

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.mem_cache.radix_cache import TreeNode


class VictimOverride(ABC):
    """队内选点策略的基类：子类必须实现 ``pick``。

    账本不继承本类：共享数账本（``NodeShareLedger``）与占用账本一样不参与
    选点，挂在策略上由缓存层直接回调。
    """

    name: str = "override"
    # 本改写器管辖的队列；``pick`` 用它过滤候选池（见 ReuseScoreEvict）。
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


class ReuseScoreEvict(VictimOverride):
    """队内复用分改写器（chat / agentic 共用一套实现）：按复用分升序淘汰。

        reuse_score(node) = P(reuse in (a, a+H] | not reused by a)
                          = 1 - S(a+H)/S(a)

    """

    name = "reuse"

    def __init__(
        self,
        params: ReuseParams,
        queue: HeterCacheNodeTag,
        tag_of=None,
        now=None,
    ) -> None:
        self.params = params
        self.queue = queue
        self._now = now if now is not None else time.monotonic
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        # ---- 观测 ----
        self.picks = 0     # 实际给出的本队候选数
        self.abstains = 0  # 管辖带内无可淘汰叶子

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
        """排序键：复用分升序；同分取更长段（一次淘汰腾出更多 token）。"""
        return (self.reuse_score(node, now), -node_num_tokens(node))

    # ------------------------------------------------------------ 插件接口

    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """在本队列池内挑复用分最低的节点；带内无候选时返回 None。

        Args:
            leaves: 当前可淘汰叶子集合。
            registry: 未使用；保留基类签名。

        Returns:
            选中的叶子；无本队叶子时返回 None 并累加 abstains 计数。
        """
        pool = [node for node in (leaves or ()) if self._tag_of(node) is self.queue]
        if not pool:
            self.abstains += 1
            return None
        # 一次取时钟：同一轮里所有节点必须用同一个 elapsed 基准，否则顺序会
        # 随遍历位置漂移，同 seed 的对照实验无法复现。
        now = self._now()
        self.picks += 1
        return min(pool, key=lambda node: self.victim_key(node, now))

    # ------------------------------------------------------------ 观测

    def stats(self) -> Dict[str, float]:
        return {"picks": self.picks, "abstains": self.abstains}

    def reset(self) -> None:
        """只清计数器；复用分无状态，无需清理。"""
        self.picks = 0
        self.abstains = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name} queue={self.queue.queue_name}>"


def _chain_walk(node) -> Tuple[list, int, int, int, set]:
    """自叶向根遍历一条链：返回 (nodes, remain, full, rerode_depth, session_ids)。

    remain = 淘汰该叶子后链上仍连续缓存的前缀（root→parent 的 token 数）；
    rerode_depth = 链上最深重读节点的 root→node 深度（0 = 无重读）。
    admission match 读连续前缀，重读集合就是链前缀，所以它等于该会话上次
    被命中的前缀长度——下一轮要读的位置。
    """
    leaf_len = len(node.key)
    nodes = [node]
    session_ids = {getattr(node, "session_id", None)}
    cursor = node.parent
    while cursor is not None and cursor.parent is not None:
        nodes.append(cursor)
        session_ids.add(getattr(cursor, "session_id", None))
        cursor = cursor.parent
    full = sum(len(n.key) for n in nodes)
    remain = full - leaf_len
    rerode_depth = 0
    depth = 0
    for item in reversed(nodes):
        depth += len(item.key)
        if getattr(item, "reuse_count", 0) > 0:
            rerode_depth = depth
    return nodes, remain, full, rerode_depth, session_ids


def chain_leaf_key(node, *, floor_tokens: int, weight: float,
                   tail_first: bool = True, registry=None, min_gaps: int = 1,
                   expired_of=None, weight_scale: float = 1.0,
                   session_registry=None) -> Tuple[float, float, int]:
    """链形定价键 ``(lat + penalty, -remain, id)``，``min`` 即淘汰。

    ChainEvict（队内）与 PricedFallback（全弃权兑底）共用同一套定价，口径
    单一来源：自适应锚 = max(floor 先验, 最深重读深度)；证据门 = 重读过 ∪
    会话回来过（``registry`` 查 gaps ≥ min_gaps）；会话终结释放锚。

    Args:
        weight_scale: 罚斜率乘数。PricedFallback 用 ``1 + β·R_队列`` 做
            后悔归因加权（历史亏得多的队列，锚罚更重）；队内改写器恒为 1。
        session_registry: 未使用；保留参数位避免调用方分支（与 pick 签名对齐）。
    """
    lat = float(node.last_access_time)
    nodes, remain, _full, rerode_depth, session_ids = _chain_walk(node)
    anchor = max(floor_tokens, rerode_depth)
    penalty = weight * weight_scale * max(0, anchor - remain)
    # 证据门：重读过（rerode_depth>0）∪ 会话回来过；无证据不护，
    # 否则一次性流量靠锚吃容量（四 cell 实测 gate=none 全线跌）。
    if penalty > 0 and not rerode_depth:
        returned = False
        if registry is not None:
            for sid in session_ids:
                if not sid:
                    continue
                state = registry.get(sid)
                if state is not None and len(state.gaps) >= min_gaps:
                    returned = True
                    break
        if not returned:
            penalty = 0.0
    # 锚出口：路径上所有会话终结才释放（混合路径只要有一段活着就保守保留）。
    if penalty > 0 and expired_of is not None:
        if all(expired_of(item) for item in nodes):
            penalty = 0.0
    tie = -float(remain) if tail_first else 0.0
    return (lat + penalty, tie, int(getattr(node, "id", 0)))


class PricedFallback(VictimOverride):
    """全弃权兑底的全局定价（档 B）：把「随机 fallback」换成「全树最便宜的叶子」。

    背景（chain_shape_report.md §七）：档 A 的弃权门打断了 VD 把削减确定性
    引向锚队的正反馈，但全弃权后的兑底是随机 fallback——warm 锚占叶子池大头
    照样被随机削到（warm 整链停在 ~48%）。本类给 heter 补上 chain_lru 的
    全局形状定价：每个候选叶子按 ``chain_leaf_key`` 定价，全树取最便宜；
    罚斜率乘上 ``1 + β·R_队列``（弃权门的后悔归因）做加权——历史削了亏的
    队列，锚罚更重，实测反馈与结构先验叠加。

    Stage 0 已在 pick_victim 前面拦下 one-shot 叶子，本类的候选池不含它们。
    """

    name = "priced_fallback"

    def __init__(self, params, gate, tag_of=None, expired_of=None) -> None:
        self.params = params          # RegretGateParams（floor/weight/beta）
        self._gate = gate             # RegretGate，提供 r(queue)
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        self._expired_of = expired_of
        self._registry = None
        # ---- 观测 ----
        self.picks = 0

    def bind_registry(self, registry) -> None:
        """接会话账本（与 ChainEvict.bind_registry 同约定）。"""
        self._registry = registry

    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """全树叶子按后悔归因加权的链形定价取最便宜；空池弃权。"""
        if registry is not None:
            self._registry = registry
        pool = [node for node in (leaves or ())]
        if not pool:
            return None
        self.picks += 1
        beta = self.params.beta
        weight = self.params.weight

        def price(node):
            scale = 1.0
            if beta > 0 and self._gate is not None:
                r_q = self._gate.r(self._tag_of(node).queue_name)
                if r_q is not None:
                    scale = 1.0 + beta * r_q
            return chain_leaf_key(
                node,
                floor_tokens=self.params.floor_tokens,
                weight=weight,
                tail_first=True,
                registry=self._registry,
                min_gaps=self.params.min_gaps,
                expired_of=self._expired_of,
                weight_scale=scale,
            )

        return min(pool, key=price)

    def stats(self) -> Dict[str, float]:
        return {"picks": self.picks}

    def reset(self) -> None:
        self.picks = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name}>"


class ChainEvict(VictimOverride):
    """队内链形保护改写器（chat / agentic）：浅段优先保护、砍链尾。

    排序键 = ``(last_access_time + penalty, -remain, node.id)``，``min`` 即淘汰，
    与独立策略 ``ChainLRUStrategy`` 同一条实测结论（详见 ``ChainParams`` 文档）。
    与独立策略的差异：

    * 锚自适应：penalty 的缺口以「链上最深重读节点的深度」与锚下限先验的
      较大者为准，不再是一个全局常数 floor——每条链护到**它自己**上次被读到的
      位置（admission match 读的是连续前缀，重读集合就是链前缀，所以
      最深重读节点深度 = 该会话上次被命中的前缀长度 = 下一轮要读的位置）。
    * 锚有出口：路径上所有会话都终结（``session_expired`` 口径，含「共享段
      不终结」的保护）时释放锚——死会话不许占着保护不放，否则长尾负载上
      死锚慢慢吃容量。

    跨队列仲裁不感知本改写器：``pick_victim`` 仍按 VD 定队列序，本类只在
    队内重选牺牲者，``record_evict``/``free_segment`` 口径不变。
    """

    name = "chain"

    def __init__(
        self,
        params: ChainParams,
        queue: HeterCacheNodeTag,
        tag_of=None,
        expired_of=None,
    ) -> None:
        self.params = params
        self.queue = queue
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        # 会话终结判据由策略提供（HeterCacheStrategy.session_expired）：
        # 共享段不终结、证据不足不终结，口径单一来源，不在这里重写一套。
        self._expired_of = expired_of
        self._registry = None  # 会话账本，pick/bind_registry 时推入
        # ---- 观测 ----
        self.picks = 0
        self.abstains = 0

    # ------------------------------------------------------------ 排序键

    def victim_key(self, node):
        """(lat + penalty, -remain, id)；penalty 缺口以自适应锚为准。

        定价走 :func:`chain_leaf_key`（与 PricedFallback 共用，口径单一来源）。
        reuse_count 只在 admission match 里自增（回扫不算），``hit_count``
        含写路径自增，不作证据。
        """
        return chain_leaf_key(
            node,
            floor_tokens=self.params.floor_tokens,
            weight=self.params.weight,
            tail_first=self.params.tail_first,
            registry=self._registry,
            min_gaps=self.params.min_gaps,
            # expire_anchors=False 时不出手，与旧实现一致
            expired_of=self._expired_of if self.params.expire_anchors else None,
        )

    # ------------------------------------------------------------ 插件接口

    def bind_registry(self, registry) -> None:
        """接会话账本（策略装配完后推入；与 ``use_clock`` 同一约定）。"""
        self._registry = registry

    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """在管辖队列池内挑「链形损失最小」的叶子；带内无候选时弃权。

        Args:
            leaves: 当前可淘汰叶子集合（``queue_victim`` 已按队列过滤，这里
                仍防御性再筛一次，与 ``ReuseScoreEvict`` 同约定）。
            registry: 会话账本；非 None 时绑定为后续查询源。
        """
        if registry is not None:
            self._registry = registry
        pool = [node for node in (leaves or ()) if self._tag_of(node) is self.queue]
        if not pool:
            self.abstains += 1
            return None
        self.picks += 1
        return min(pool, key=self.victim_key)

    # ------------------------------------------------------------ 观测

    def stats(self) -> Dict[str, float]:
        return {"picks": self.picks, "abstains": self.abstains}

    def reset(self) -> None:
        """只清计数器；排序键无状态，无需清理。"""
        self.picks = 0
        self.abstains = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name} queue={self.queue.queue_name}>"
