
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple
import time
from sglang.srt.mem_cache.heter_cache.utils import (
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
