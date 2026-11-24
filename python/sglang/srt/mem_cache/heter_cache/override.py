
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
from sglang.srt.mem_cache.heter_cache.utils import (
    ChainParams,
    ReuseParams,
    HeterCacheNodeTag,
    VALUE_QUEUES,
    median,
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


def _node_len(node) -> int:
    key = getattr(node, "key", None)
    if key is not None and hasattr(key, "__len__"):
        return len(key)
    return node_num_tokens(node)


def _chain_walk(node, share_of=None, share_min: int = 2) -> Tuple[list, int, int, int, int, set]:
    """自叶向根遍历链条，返回 (nodes, remain, full, reread_depth, shared_depth, session_ids)。

    remain: 淘汰该叶子后链上仍连续缓存的前缀 token 数（root->parent）。
    reread_depth: 链上最深重读节点的 root->node 深度（0 表示无重读）。
    shared_depth: 链上最深跨会话共享节点的 root->node 深度（0 表示无共享）。
    """
    leaf_len = _node_len(node)
    nodes = [node]
    session_ids = {getattr(node, "session_id", None)}
    cursor = getattr(node, "parent", None)
    while cursor is not None and getattr(cursor, "parent", None) is not None:
        nodes.append(cursor)
        session_ids.add(getattr(cursor, "session_id", None))
        cursor = getattr(cursor, "parent", None)
    full = sum(_node_len(n) for n in nodes)
    remain = full - leaf_len
    reread_depth = 0
    shared_depth = 0
    depth = 0
    for item in reversed(nodes):
        depth += _node_len(item)
        if getattr(item, "reuse_count", 0) > 0:
            reread_depth = depth
        if getattr(item, "reuse_cross", 0) > 0 or (
            share_of is not None and share_of(item) >= share_min
        ):
            shared_depth = depth
    return nodes, remain, full, reread_depth, shared_depth, session_ids


def _has_return_evidence(session_ids: set, registry, min_gaps: int) -> bool:
    """链上任一已知会话是否已观测到至少 min_gaps 次轮间回访。"""
    if registry is None:
        return False
    for sid in session_ids:
        if not sid:
            continue
        state = registry.get(sid)
        if state is not None and len(state.gaps) >= min_gaps:
            return True
    return False


def _max_session_L(session_ids: set, registry) -> int:
    """链上已知会话最近一次 prompt 长度 L 的最大值（无有效记录时返回 0）。"""
    if registry is None:
        return 0
    max_l = 0
    for sid in session_ids:
        if not sid:
            continue
        state = registry.get(sid)
        if state is not None and state.L > max_l:
            max_l = int(state.L)
    return max_l


def chain_leaf_key(
    node,
    *,
    floor_tokens: int,
    weight: float,
    tail_first: bool = True,
    registry=None,
    min_gaps: int = 1,
    expired_of=None,
    weight_scale: float = 1.0,
    queue: Optional[str] = None,
    share_of=None,
    share_min: int = 2,
    cap_floor_by_session_L: bool = False,
) -> Tuple[float, float, int]:
    """链形定价键 (lat + penalty, -remain, id)，min 即淘汰。

    ChainEvict 与 PricedFallback 共用同一套定价：
    - struct 队列：以链上最深跨会话共享深度 shared_depth 作锚与证据门；
    - chat / agentic：锚深 = max(有效 floor, 最深重读深度)，支持按 SessionState.L 裁剪；
    - 证据门 = 链上存在重读节点 或 会话有回访记录；无证据不保护；
    - 锚出口 = 路径上所有会话均已终结时释放罚项。
    """
    lat = float(node.last_access_time)
    nodes, remain, _full, reread_depth, shared_depth, session_ids = _chain_walk(
        node, share_of=share_of, share_min=share_min
    )
    if queue == "struct":
        anchor = max(floor_tokens, shared_depth)
        has_evidence = shared_depth > 0
    else:
        eff_floor = floor_tokens
        if cap_floor_by_session_L and eff_floor > 0:
            sess_l = _max_session_L(session_ids, registry)
            if sess_l > 0:
                eff_floor = min(eff_floor, sess_l)
        anchor = max(eff_floor, reread_depth)
        has_evidence = bool(reread_depth) or _has_return_evidence(
            session_ids, registry, min_gaps
        )

    penalty = weight * weight_scale * max(0, anchor - remain)
    if penalty > 0 and not has_evidence:
        penalty = 0.0
    if penalty > 0 and expired_of is not None and all(expired_of(item) for item in nodes):
        penalty = 0.0
    tie = -float(remain) if tail_first else 0.0
    return (lat + penalty, tie, int(getattr(node, "id", 0)))


class PricedFallback(VictimOverride):
    """多队列软加权全树链形定价器：融合 VD_q / R_q 宏观乘数与队列级锚深 anchor_q。"""

    name = "priced_fallback"

    def __init__(
        self,
        params,
        gate,
        tag_of=None,
        expired_of=None,
        *,
        realized=None,
        occupancy=None,
        share_of=None,
        share_min: int = 2,
    ) -> None:
        self.params = params          # RegretGateParams（floor_by_queue/weight/alpha/beta/scale_min/max）
        self._gate = gate             # RegretGate，提供 r(queue)
        self._tag_of = tag_of or (
            lambda node: HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))
        )
        self._expired_of = expired_of
        self._realized = realized     # QueueValueTracker，提供 densities(occ)
        self._occupancy = occupancy   # QueueOccupancyTracker，提供 occ
        self._share_of = share_of     # NodeShareLedger.share_of
        self._share_min = int(share_min)
        self._registry = None
        # ---- 观测 ----
        self.picks = 0

    def bind_registry(self, registry) -> None:
        """接会话账本（与 ChainEvict.bind_registry 同约定）。"""
        self._registry = registry

    def queue_scales(self) -> Dict[str, float]:
        """计算各队列宏观保护乘数 scale_q = clip((VD_q / median_VD)^alpha * (1 + beta * R_q), s_min, s_max)。"""
        alpha = getattr(self.params, "alpha", 0.5)
        beta = self.params.beta
        s_min = getattr(self.params, "scale_min", 0.25)
        s_max = getattr(self.params, "scale_max", 4.0)

        vd_map: Dict[str, float] = {}
        med_vd = 0.0
        if self._realized is not None and self._occupancy is not None and alpha > 0:
            occ = self._occupancy.occ
            vd_map = self._realized.densities(occ)
            active_vds = [vd_map.get(q, 0.0) for q in VALUE_QUEUES if occ.get(q, 0.0) > 0.0]
            if active_vds:
                med_vd = median(active_vds)

        scales: Dict[str, float] = {}
        for q in VALUE_QUEUES:
            if med_vd > 0.0 and alpha > 0:
                vd_factor = (max(0.0, vd_map.get(q, 0.0)) / med_vd) ** alpha
            else:
                vd_factor = 1.0
            rg_factor = 1.0
            if beta > 0 and self._gate is not None:
                r_q = self._gate.r(q)
                if r_q is not None:
                    rg_factor = 1.0 + beta * r_q
            scales[q] = min(s_max, max(s_min, vd_factor * rg_factor))
        return scales

    def pick(self, leaves, registry=None) -> Optional["TreeNode"]:
        """全树叶子按队列软加权的链形定价取最便宜；空池弃权。"""
        if registry is not None:
            self._registry = registry
        pool = [node for node in (leaves or ())]
        if not pool:
            return None
        self.picks += 1
        weight = self.params.weight
        scales = self.queue_scales()
        floor_by_queue = getattr(self.params, "floor_by_queue", None) or {}
        default_floor = self.params.floor_tokens

        def price(node):
            q = self._tag_of(node).queue_name
            scale = scales.get(q, 1.0)
            floor_q = floor_by_queue.get(q, default_floor)
            return chain_leaf_key(
                node,
                floor_tokens=floor_q,
                weight=weight,
                tail_first=True,
                registry=self._registry,
                min_gaps=self.params.min_gaps,
                expired_of=self._expired_of,
                weight_scale=scale,
                queue=q,
                share_of=self._share_of,
                share_min=self._share_min,
                cap_floor_by_session_L=True,
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
