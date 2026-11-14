"""HeterCache 的淘汰策略。

职责单一：多队列仲裁（``HeterCacheStrategy``）——把节点归到 struct / agentic /
chat 三个配额队列，按配额序列挑队，再把队内细节委托给 ``victim_override`` 的改写器。

**不再有全局复用分。** 旧版的 ``reuse_score``（多轮 lognormal 复用概率 + 单轮位置项）
与它的全局排序键 ``victim_key`` 已整体删除：配额现在必填，候选一律由队列仲裁产生，
没有任何路径需要“跨队列先算一个全局价值分”。

复用分本身还在，但只作为 **agentic 队内**的排序键（``MultiAgenticEvict``，
``multi_agentic.enabled`` 显式打开）：它不参与跨队列比较，也抢不了配额仲裁的
话语权——队列先由水填充点名，队内才按 ``1 - CDF(elapsed)`` 排序。

其余 ``get_priority`` 实现（LRU / LFU / TLRU / SLRU 等）依旧服务原生策略，
共同契约：**返回值越小越先被淘汰**。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import random
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
from sglang.srt.mem_cache.quota_selector import (
    effective_targets,
    queue_ranking,
    resolve_chunk,
    struct_tier,
)
from sglang.srt.mem_cache.heter_utils import (
    node_num_tokens,
    ONE_SHOT_IDLE_S,
    RrParams,
    MultiAgenticParams,
    MultiTurnParams,
    SessionRegistry,
    SessionType,
    SlruParams,
    QUOTA_QUEUES,
    QuotaParams,
    HeterCacheNodeTag,
)
from sglang.srt.mem_cache.queue_occupancy import QueueOccupancyTracker
from sglang.srt.mem_cache.victim_override import (
    RrEvictPolicy,
    SlruEvictPolicy,
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
        """返回节点的淘汰优先级；实现必须满足"越小越先淘汰"。"""
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
        # L 取节点的 high-water（历史最深前缀），故意不随尾部裁剪而回缩：
        # 若改用"当前驻留长度"当 L，被裁短的会话下一轮仍然超预算，T-LRU 会把它
        # 一路裁到空，而不是在 (threshold - Q_hat) 处停手。
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
    1. 分带过滤：One-shot 节点优先清理，不计入配额账本；
       归属会话仍活跃的未匹配节点受时间豁免（ONE_SHOT_IDLE_S）保护，不进带。
    2. 配额仲裁：先按配额承压情况排出候选队列，再由队内策略决出具体牺牲节点
    """

    def __init__(
        self,
        *,
        force_agentic: bool = False,
        rr: Union[RrParams, dict, None] = None,
        slru: Union[SlruParams, dict, None] = None,
        mt: Union[MultiTurnParams, dict, None] = None,
        multi_agentic: Union[MultiAgenticParams, dict, None] = None,
        quota: Union[QuotaParams, dict, None] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        # 时间源可注入（单测用假时钟控制 one-shot 豁免窗口的推进）。
        self._now = now if now is not None else time.monotonic
        self.rr = RrParams.from_any(rr)
        self.slru = SlruParams.from_any(slru)
        self.mt = MultiTurnParams.from_any(mt)
        self.multi_agentic = MultiAgenticParams.from_any(multi_agentic)
        self.quota = QuotaParams.from_any(quota)
        self.force_agentic = bool(force_agentic)

        self.occupancy = QueueOccupancyTracker()

        self._quota_rng = random.Random(self.quota.seed)
        self.profiler = HeterProfiler()
        self.telemetry = CacheTelemetry(self.profiler)

        self.rr_evict = RrEvictPolicy(self.rr) if self.rr.fair_rr else None
        self.session_registry = SessionRegistry.global_instance()
        self.slru_evict = SlruEvictPolicy(
            self.session_registry, self.slru.share_min
        )
        self.multi_agentic_evict = (
            MultiAgenticEvict(self.multi_agentic, self._node_tag)
            if self.multi_agentic.enabled
            else None
        )

        self.node_observers = (self.slru_evict,)

    def _node_tag(self, node) -> HeterCacheNodeTag:
        if self.force_agentic:
            return HeterCacheNodeTag.QAGENT
        return HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))

    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        raise NotImplementedError(
            "HeterCacheStrategy has no single-node ordering; "
            "use quota arbitration (quota_selector.pick_victim) instead"
        )

    def is_aged(self, node) -> bool:
        """节点是否已老化：按生效标签取 TTL，0 表示该类型不老化。

        QSTRUCT 额外要求 share >= share_min：未共享的 struct 节点由原有
        90s one-shot 规则处理，老化只负责"共享模板变冷"这一档。
        """
        tag = self._node_tag(node)
        ttl = self.slru.ageing_ttl_s[tag.queue_name]
        if ttl <= 0:
            return False
        if tag is HeterCacheNodeTag.QSTRUCT and (
            self.slru_evict.share_of(node) < self.slru_evict.share_min
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
        if self.slru_evict.share_of(node) >= self.slru_evict.share_min:
            return False
        return st.session_type in (SessionType.TRANS, SessionType.STRUCT)

    def one_shot_key(self, node) -> tuple:
        """One-shot 带内淘汰排序：老化节点最优先，其余按访问时间升序。"""
        return (0 if self.is_aged(node) else 1, node.last_access_time)

    # ---- 命中/淘汰记账 ----

    def record_hit(self, req, node, hit_tokens: int) -> None:
        """记录缓存命中指标。"""
        self.telemetry.on_hit(
            tag=self._node_tag(node),
            hit_tokens=hit_tokens,
            req_id=getattr(req, "rid", None),
        )

    def record_evict(self, node) -> None:
        """记录节点淘汰事件并扣减配额占用。

        晋升（hit_count 0→1）会把节点从 one-shot
        带挪进其标签队列，而该队列可能从未收到过对应的 insert 计入
        （one-shot 时 insert 跳过）——evict 会扣一个负项，直到下一次
        全树快照 reconcile 吸收。random 行仅作展示、不参与配额仲裁；
        struct 行的短暂低估方向是少淘汰、多保留，与目标同向。

        时间豁免引入反向漂移（轴一）：insert 时会话活跃被计入、evict 时
        已过搁置窗口进入带内被跳过 → 占用短暂虚高，方向是「多淘汰」，
        同样由 reconcile 吸收；承压时校准窗口为 quota.pressure_window_s
        （默认 3s），漂移窗口有界。
        """
        tokens = _node_num_tokens(node)
        tag = self._node_tag(node)
        queue = tag.queue_name
        one_shot = self.is_one_shot(node)
        # 仅维护启用了配额且非 one-shot 的三队列占用
        if not one_shot:
            self.occupancy.evict(queue, tokens)
        elif self.is_aged(node):
            # 老化淘汰单独计数：否则与真正的一次性流量混在 one_shot 里。
            self.slru_evict.on_aged_evict(tokens)
        self.telemetry.on_evict(
            node_id=id(node), tag=tag, tokens=tokens, one_shot=one_shot
        )

    def record_victim_source(self, source: str, tokens: float) -> None:
        """记录淘汰决策来源。"""
        self.telemetry.on_victim_source(source, tokens)

    def record_insert(self, node) -> None:
        """新节点挂载时增量计入配额（仅在节点创建时调用，节点分裂时不触发）。"""
        if self.is_one_shot(node):
            return
        self.occupancy.insert(self._node_tag(node).queue_name, _node_num_tokens(node))

    # ---- 跨队列配额 ----

    def quota_ranking(self, occ: dict, capacity: float, page_size: int) -> list:
        """当前淘汰候选队列的优先级序列。

        Args:
            occ: 配额口径的各队列占用（来自 QueueOccupancyTracker）。
            capacity: 池容量（Token）。
            page_size: 内存页大小，用于推算单次最小减持量。
        """
        targets = effective_targets(occ, capacity, self.quota)
        chunk = resolve_chunk(page_size, self.quota)
        return queue_ranking(
            occ, targets, self.quota, chunk=float(chunk), capacity=capacity
        )

    def quota_targets(self, occ: dict, capacity: float) -> dict:
        """各队列的目标 Token 配额（仅观测用途）。"""
        return effective_targets(occ, capacity, self.quota)

    def quota_victim(self, queue: str, leaves, occ: dict, capacity: float):
        """在指定队列内挑选淘汰节点。若无合适候选返回 None（触发上层跳队）。"""
        target = HeterCacheNodeTag.from_queue(queue)
        if target is None:
            return None
        pool = [n for n in leaves if self._node_tag(n) is target]
        if not pool:
            return None
        if queue == "struct":
            # 模板分级淘汰：优先淘汰低共享度节点，同级按 LRU
            share_of = self.slru_evict.share_of
            share_min = self.slru_evict.share_min
            return min(
                pool, key=lambda n: struct_tier(n, share_of, share_min)
            )
        if queue == "agentic":
            if self.multi_agentic_evict is not None:
                return self.multi_agentic_evict.pick(pool, self.session_registry)
        if queue == "chat" and self.rr_evict is not None:
            # RR 轮询调度，受配额超额缺口上限约束
            picked = self.rr_evict.pick(
                pool,
                self.session_registry,
                max_deficit=self._quota_gap("chat", occ, capacity),
            )
            if picked is not None:
                return picked
        return None

    def _quota_gap(self, queue: str, occ: dict, capacity: float):
        """计算队列超出有效配额的 Token 数；未超限时返回 None。"""
        if capacity <= 0:
            return None
        target = effective_targets(occ, capacity, self.quota)[queue]
        gap = occ[queue] - target
        return gap if gap > 0 else None

    def quota_fallback(self, leaves):
        """配额候选均耗尽时的均匀随机淘汰。"""
        pool = list(leaves)
        if not pool:
            return None
        return self._quota_rng.choice(pool)