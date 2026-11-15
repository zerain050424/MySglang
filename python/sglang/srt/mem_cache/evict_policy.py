from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import random
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
from sglang.srt.mem_cache.queue_selector import queue_order as _queue_order
from sglang.srt.mem_cache.queue_selector import struct_tier
from sglang.srt.mem_cache.heter_utils import (
    node_num_tokens,
    ONE_SHOT_IDLE_S,
    RrParams,
    MultiAgenticParams,
    MultiTurnParams,
    SessionRegistry,
    SessionType,
    SlruParams,
    QueueValueParams,
    HeterCacheNodeTag,
)
from sglang.srt.mem_cache.queue_occupancy import QueueOccupancyTracker
from sglang.srt.mem_cache.queue_value import QueueValueTracker
from sglang.srt.mem_cache.node_share import NodeShareLedger
from sglang.srt.mem_cache.victim_override import (
    RrEvictPolicy,
    MultiAgenticEvict,
    MultiTurnEvictPolicy,
)
from sglang.srt.mem_cache.heter_bench_bridge import (
    CacheTelemetry,
    HeterProfiler,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)



# 节点口径统一在 heter_utils，避免策略/改写器/指标各算一套。
_node_num_tokens = node_num_tokens


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):

    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time ## LRU返回最近使用时间,较小的时间优先级越低


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.priority, node.last_access_time)


class TLRUStrategy(EvictionStrategy):
    def __init__(self, threshold: int = 0, next_prompt_estimate: int = 0):
        self.threshold = threshold
        self.next_prompt_estimate = next_prompt_estimate

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        budget = max(
            node._tlru_history_len + self.next_prompt_estimate - self.threshold, 0
        )
        cached_without_this_node = node._tlru_cached_prefix_len - len(node.key)
        tel_safe = cached_without_this_node >= budget
        return (-1 if tel_safe else 0, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class HeterCacheStrategy(EvictionStrategy):
    """HeterCache 多队列缓存淘汰策略。

    两层决策：
    1. 分带过滤：One-shot 节点优先清理，不计入 VD 账本；
       归属会话仍活跃的未匹配节点受时间豁免（ONE_SHOT_IDLE_S）保护，不进带。
    2. VD 仲裁：按 ``occ_q / (1 + VD_q)`` 排出候选队列，再由队内策略决出牺牲节点。
    """

    def __init__(
        self,
        *,
        force_agentic: bool = False,
        rr: Union[RrParams, dict, None] = None,
        slru: Union[SlruParams, dict, None] = None,
        mt: Union[MultiTurnParams, dict, None] = None,
        multi_agentic: Union[MultiAgenticParams, dict, None] = None,
        value: Union[QueueValueParams, dict, None] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        # 时间源可注入（单测用假时钟控制 one-shot 豁免窗口的推进）。
        self._now = now if now is not None else time.monotonic
        self.rr = RrParams.from_any(rr)
        self.slru = SlruParams.from_any(slru)
        self.mt = MultiTurnParams.from_any(mt)
        self.multi_agentic = MultiAgenticParams.from_any(multi_agentic)
        self.value = QueueValueParams.from_any(value)
        self.force_agentic = bool(force_agentic)

        self.occupancy = QueueOccupancyTracker()
        self.realized = QueueValueTracker(
            tau_s=self.value.tau_s,
            floor_tokens=self.value.floor_tokens,
            now=self._now,
        )

        self._fallback_rng = random.Random(self.value.seed)
        self.profiler = HeterProfiler()
        self.telemetry = CacheTelemetry(self.profiler)

        self.rr_evict = RrEvictPolicy(self.rr) if self.rr.fair_rr else None
        self.session_registry = SessionRegistry.global_instance()
        self.node_share = NodeShareLedger(self.slru.share_min)
        self.multi_agentic_evict = (
            MultiAgenticEvict(self.multi_agentic, self._node_tag)
            if self.multi_agentic.enabled
            else None
        )

    def use_clock(self, now: Callable[[], float]) -> None:
        """接管缓存层的时间源（默认墙钟下等价于无操作）。

        节点时间戳由 ``RadixCache`` 盖（``CacheInitParams.now``），所以策略与它
        持有的读时钟协作者（占用账本 / 遥测 / 会话账本 / 队内改写器）必须与缓存
        同源；否则 ``is_aged`` / one-shot 豁免 / 承压校准窗口会拿假 now 去减真
        时间戳，阈值全部失真。

        Args:
            now: 返回秒数的单调时钟。
        """
        self._now = now
        self.occupancy.use_clock(now)
        self.telemetry.use_clock(now)
        self.realized.use_clock(now)
        self.session_registry.use_clock(now)
        if self.multi_agentic_evict is not None:
            self.multi_agentic_evict.use_clock(now)

    def _node_tag(self, node) -> HeterCacheNodeTag:
        if self.force_agentic:
            return HeterCacheNodeTag.QAGENT
        return HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))

    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        raise NotImplementedError(
            "HeterCacheStrategy has no single-node ordering; "
            "use queue_weight arbitration (queue_selector.pick_victim) instead"
        )

    def is_aged(self, node) -> bool:
        """节点是否已老化：按生效标签取 TTL，0 表示该类型不老化。"""
        tag = self._node_tag(node)
        ttl = self.slru.ageing_ttl_s[tag.queue_name]
        if ttl <= 0:
            return False
        if tag is HeterCacheNodeTag.QSTRUCT and (
            self.node_share.share_of(node) < self.node_share.share_min
        ):
            return False
        return self._now() - node.last_access_time >= ttl

    def is_one_shot(self, node) -> bool:
        """判定节点是否属于 One-shot 带。
        标准：按类型 TTL 老化的节点，或
        属于 TRANS/STRUCT 且跨会话共享数未达到 share_min，
        且从未被任何请求匹配过
        且归属会话已搁置超过 ONE_SHOT_IDLE_S
        若节点无会话元数据，保守返回 False 防止误删热前缀。
        """
        # 老化必须先于 hit_count/share 的短路：共享前缀必然命中过且已达标。
        if self.is_aged(node):
            return True
        st = self.session_registry.get(getattr(node, "session_id", None))
        if st is None:
            return False
        if getattr(node, "hit_count", 0) != 0:
            # 内容已被再次用到：离开优先淘汰带
            return False
        # 时间豁免：会话还没凉，它的内容就还在可能的重访窗口内。
        anchor = max(st.last_active, getattr(node, "last_access_time", 0.0))
        if self._now() - anchor < ONE_SHOT_IDLE_S:
            return False
        if self.node_share.share_of(node) >= self.node_share.share_min:
            return False
        return st.session_type in (SessionType.TRANS, SessionType.STRUCT)

    def one_shot_key(self, node) -> tuple:
        """One-shot 带内淘汰排序：老化节点最优先，其余按访问时间升序"""
        return (0 if self.is_aged(node) else 1, node.last_access_time, node.id)

    # ---- 命中/淘汰记账 ----

    def record_hit(self, req, node, hit_tokens: int, by_queue: dict) -> None:
        """记录一次命中：telemetry 按 rid 去重，VD 只吃首次命中的分队列命中量。
        同一 rid 的 chunked prefill 会重复进入，去重由 telemetry 负责（它已按
        MAX_SEEN_RIDS 限长）；只有它判定首次见到时才更新 VD，否则一个 32k
        prompt 的 4 个 chunk 会被记 4 遍。

        Args:
            req: 请求对象（取 rid）。
            node: 最深命中节点（telemetry 的单事件归因口径，保持历史可比）。
            hit_tokens: 本次命中的总 token 数。
            by_queue: ``{queue: tokens}``，按命中节点的标签拆分，供 VD 归因。
        """
        first = self.telemetry.on_hit(
            tag=self._node_tag(node),
            hit_tokens=hit_tokens,
            req_id=getattr(req, "rid", None),
        )
        if not first:
            return
        for queue, tokens in by_queue.items():
            self.realized.on_hit(queue, tokens)

    def record_evict(self, node) -> None:
        """记录节点淘汰事件并扣减队列占用。

        晋升（hit_count 0→1）会把节点从 one-shot
        带挪进其标签队列，而该队列可能从未收到过对应的 insert 计入
        （one-shot 时 insert 跳过）——evict 会扣一个负项，直到下一次
        全树快照 reconcile 吸收。random 行仅作展示、不参与 VD 仲裁；
        struct 行的短暂低估方向是少淘汰、多保留，与目标同向。

        时间豁免引入反向漂移（轴一）：insert 时会话活跃被计入、evict 时
        已过搁置窗口进入带内被跳过 → 占用短暂虚高，方向是「多淘汰」，
        同样由 reconcile 吸收；承压时校准窗口为 QueueOccupancyTracker.PRESSURE_WINDOW_S
        （默认 3s），漂移窗口有界。
        """
        tokens = _node_num_tokens(node)
        tag = self._node_tag(node)
        queue = tag.queue_name
        one_shot = self.is_one_shot(node)
        # 仅维护非 one-shot 的三队列占用
        if not one_shot:
            self.occupancy.evict(queue, tokens)
        elif self.is_aged(node):
            # 老化淘汰单独计数：否则与真正的一次性流量混在 one_shot 里。
            self.node_share.on_aged_evict(tokens)
        self.telemetry.on_evict(
            node_id=id(node), tag=tag, tokens=tokens, one_shot=one_shot
        )

    def record_victim_source(self, source: str, tokens: float) -> None:
        """记录淘汰决策来源。"""
        self.telemetry.on_victim_source(source, tokens)

    def record_insert(self, node) -> None:
        """新节点挂载时增量计入队列占用（仅新建节点调用，分裂不触发）。"""
        if self.is_one_shot(node):
            return
        self.occupancy.insert(self._node_tag(node).queue_name, _node_num_tokens(node))

    # ---- 跨队列仲裁（VD） ----

    def queue_order(self) -> list:
        """按occ_q / (1 + VD_q)降序返回队列；越靠前越先削。"""
        occ = self.occupancy.occ
        return _queue_order(occ, self.realized.densities(occ))

    def queue_victim(self, queue: str, leaves):
        """在指定队列内挑一个淘汰节点；无合适候选返回 None"""
        target = HeterCacheNodeTag.from_queue(queue)
        if target is None:
            return None
        pool = [n for n in leaves if self._node_tag(n) is target]
        if not pool:
            return None
        if queue == "struct":
            # 模板分级淘汰：优先淘汰低共享度节点，同级按 LRU
            share_of = self.node_share.share_of
            share_min = self.node_share.share_min
            return min(pool, key=lambda n: struct_tier(n, share_of, share_min))
        if queue == "agentic" and self.multi_agentic_evict is not None:
            return self.multi_agentic_evict.pick(pool, self.session_registry)
        if queue == "chat" and self.rr_evict is not None:
            picked = self.rr_evict.pick(pool, self.session_registry)
            if picked is not None:
                return picked
        return None

    def fallback(self, leaves):
        """所有队列均弃权时的均匀随机淘汰"""
        pool = sorted(leaves, key=lambda n: n.id)
        if not pool:
            return None
        return self._fallback_rng.choice(pool)