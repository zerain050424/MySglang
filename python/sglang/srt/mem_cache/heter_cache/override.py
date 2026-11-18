
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

        一次自叶向根的遍历同时算出：remain（root→parent 的 token 数）、
        重读边界深度、路径上的会话集合。reuse_count 只在 admission match 里
        自增（回扫不算），``hit_count`` 含写路径自增，不作证据。
        """
        lat = float(node.last_access_time)
        leaf_len = len(node.key)
        remain = 0
        full = leaf_len
        rerode_depth = 0  # 最深重读节点的 root→node 深度；0 = 链上无重读
        session_ids = {getattr(node, "session_id", None)}
        nodes = [node]
        cursor = node.parent
        while cursor is not None and cursor.parent is not None:
            nodes.append(cursor)
            full += len(cursor.key)
            session_ids.add(getattr(cursor, "session_id", None))
            cursor = cursor.parent
        remain = full - leaf_len
        # 自根向叶重放深度：depth(i) = root→nodes[i] 的 token 数。
        # 取最后一个重读节点（= 最深）的深度；admission match 读连续前缀，
        # 重读集合就是链前缀，所以它等于该会话上次被命中的前缀长度。
        depth = 0
        for item in reversed(nodes):
            depth += len(item.key)
            if getattr(item, "reuse_count", 0) > 0:
                rerode_depth = depth

        anchor = max(self.params.floor_tokens, rerode_depth)
        penalty = self.params.weight * max(0, anchor - remain)
        # 证据门：重读过（rerode_depth>0 即链上有重读节点）∪ 会话回来过。
        # 无证据的链不享受保护，否则一次性流量靠锚吃容量（四 cell 实测 gate=none 全跌）。
        if penalty > 0 and not rerode_depth:
            registry = self._registry
            returned = False
            if registry is not None:
                for sid in session_ids:
                    if not sid:
                        continue
                    state = registry.get(sid)
                    if state is not None and len(state.gaps) >= self.params.min_gaps:
                        returned = True
                        break
            if not returned:
                penalty = 0.0
        # 锚出口：路径上所有会话都终结才释放（混合路径只要有一段活着就保守保留，
        # 方向是多留、少剥，与目标同向）。共享段在 session_expired 里天然不终结。
        if penalty > 0 and self.params.expire_anchors and self._expired_of is not None:
            if all(self._expired_of(item) for item in nodes):
                penalty = 0.0
        tie = -float(remain) if self.params.tail_first else 0.0
        return (lat + penalty, tie, int(getattr(node, "id", 0)))

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
