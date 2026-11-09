"""HeterCache 的淘汰策略。

职责单一：多队列仲裁（``HeterCacheStrategy``）——把节点归到 struct / agentic /
chat 三个配额队列，按配额序列挑队，再把队内细节委托给 ``victim_override`` 的改写器。

**不再有全局复用分。** 旧版的 ``reuse_score``（多轮 lognormal 复用概率 + 单轮位置项）
与它的全局排序键 ``victim_key`` 已整体删除：配额现在必填，候选一律由队列仲裁产生，
没有任何路径需要“跨队列先算一个全局价值分”。

其余 ``get_priority`` 实现（LRU / LFU / TLRU / SLRU 等）依旧服务原生策略，
共同契约：**返回值越小越先被淘汰**。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import random
from typing import TYPE_CHECKING, Tuple, Union
from sglang.srt.mem_cache.quota_selector import (
    effective_targets,
    queue_capacities,
    queue_ranking,
    resolve_chunk,
    struct_tier,
)
from sglang.srt.mem_cache.heter_utils import (
    node_num_tokens,
    RrParams,
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
    MultiTurnEvictPolicy,
)
from sglang.srt.mem_cache.heter_logging import (
    CacheTelemetry,
    HeterProfiler,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)



# 节点口径统一在 heter_utils，避免策略/改写器/指标各算一套。
_node_num_tokens = node_num_tokens


class EvictionStrategy(ABC):
    """单节点淘汰排序的抽象基类。

    契约：``get_priority`` 返回可比较的键，**越小越先淘汰**。调用方只做
    ``min(candidates, key=strategy.get_priority)``，不关心内部是标量还是元组。

    返回元组时应把判据按优先级从高到低排列（如 ``(band, score, -tokens)``），
    靠元组字典序在单次比较中一并完成 tie-break，避免多次扫描。
    """

    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        """返回节点的淘汰优先级；实现必须满足"越小越先淘汰"。"""
        pass


class LRUStrategy(EvictionStrategy):
    """最近最少使用：直接以访问时间作优先级。"""

    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time ## LRU返回最近使用时间,较小的时间优先级越低


class LFUStrategy(EvictionStrategy):
    """最少使用频率：先比命中次数，再比访问时间。"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    """先进先出：以创建时间作优先级。"""

    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    """最近最常使用：取负把"越新越该走"倒过来。"""

    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    """后进先出：取负把"越新越该走"倒过来。"""

    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """优先级感知淘汰：小 priority 值先走，同优先级内退化 LRU。"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.priority, node.last_access_time)


class TLRUStrategy(EvictionStrategy):
    """Tail-Optimized LRU（Zhang et al., arXiv:2510.15152）。

    预算：一条历史长度为 L、预计下一轮还要追加 Q_hat 的会话，只需缓存
    ``L + Q_hat - threshold`` 个 token，就足以把下一轮 prefill 压在 TTFT 预算内。
    超过这条线的那部分尾部对本轮 TTFT 再无贡献，称为 TEL-safe，可以随便淘汰。

    实现方式按论文建议：把这类节点报成"无限老"（优先级 -1），淘汰驱动自然先
    把它们排干（论文的 phase 1），排干后回落到普通 recency 序（phase 2）。这样
    任何淘汰循环都不需要知道 T-LRU 的存在。

    threshold 与 next_prompt_estimate 都是 **token** 数；论文以 block 计，
    移植时需要乘 page_size。

    Args:
        threshold: 允许放弃的尾部预算（token），必须大于 ``next_prompt_estimate``，
            否则预算永远不小于 L、没有任何节点 TEL-safe，退化成纯 LRU。
        next_prompt_estimate: 下一轮 prompt 相对本轮的增量估计（token）。
    """

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
    """分段 LRU：按命中次数把节点分成 probation / protected 两段。

    先清 probation（``hit_count < protected_threshold``），再清 protected；
    段内按 recency 排序。用元组 ``(segment, last_access_time)`` 一次比较完成
    两段排序，不需要维护两个独立候选池。

    Args:
        protected_threshold: 进入 protected 段所需的命中次数。
    """

    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class HeterCacheStrategy:
    """HeterCache 多队列缓存淘汰策略。

    不继承 ``EvictionStrategy``：基类只规定 ``get_priority``（单节点排序键），
    而 HeterCache 的裁决来自配额仲裁（按队列与目标占用的水填充），根本不存在
    一个跨队列可比的节点分数。硬实现一个只为满足抽象方法的 ``get_priority``
    反而会误导调用方以为可以拿它当优先级堆用。

    两层决策：
    1. 分带过滤：One-shot 节点（无跨轮复用价值）优先清理，不计入配额账本。
    2. 配额仲裁：先按配额承压情况排出候选队列，再由队内策略（RR / SLRU / MT）
       决出具体牺牲节点。

    没有全局价值分：早期版本用 ``reuse_score`` 给每个节点打分、跨队列比较后再竞争，
    在配额为主的用法下已无意义，已整体删除。

    注：本策略只做裁决，KV 释放与 Radix 树拓扑修改由外层缓存引擎执行。
    """

    def __init__(
        self,
        *,
        force_agentic: bool = False,
        rr: Union[RrParams, dict, None] = None,
        slru: Union[SlruParams, dict, None] = None,
        mt: Union[MultiTurnParams, dict, None] = None,
        quota: Union[QuotaParams, dict, None] = None,
    ):
        self.rr = RrParams.from_any(rr)
        self.slru = SlruParams.from_any(slru)
        self.mt = MultiTurnParams.from_any(mt)
        self.quota = QuotaParams.from_any(quota)
        self.force_agentic = bool(force_agentic)

        # 配额账本：唯一写入口在缓存层（record_insert / record_evict），
        # 策略只读它的 occ 做仲裁。
        self.occupancy = QueueOccupancyTracker()
        # 兜底淘汰:采样独立 RNG，保证实验结果在相同 seed 下可复现
        self._quota_rng = random.Random(self.quota.seed)
        self.profiler = HeterProfiler()
        self.telemetry = CacheTelemetry(self.profiler)

        self.rr_evict = RrEvictPolicy(self.rr) if self.rr.fair_rr else None
        self.session_registry = SessionRegistry.global_instance()
        self.slru_evict = SlruEvictPolicy(
            self.session_registry, self.slru.share_min
        )
        self.mt_evict = MultiTurnEvictPolicy(self.mt, self.session_registry) if self.mt.enabled else None
        # 节点生命周期观测器：命中记账与分裂继承。与队内选点分开登记——RR / MT
        # 只在配额裁决里被调用，不参与节点生命周期；SLRU 两头都要（它的共享数
        # 账本既喂 is_one_shot，也喂 struct 分级）。
        self.node_observers = (self.slru_evict,)

    def reset_bookkeeping(self) -> None:
        """重置统计指标、去重状态与配额账本（缓存 flush / reset 时调用）。"""
        self.telemetry.reset()
        self.occupancy.reset()

    def _node_tag(self, node) -> HeterCacheNodeTag:
        """节点的生效标签（已收敛枚举）。force_agentic 开启时统一按 agentic 算。"""
        if self.force_agentic:
            return HeterCacheNodeTag.QAGENT
        return HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))


    def is_one_shot(self, node) -> bool:
        """判定节点是否属于 One-shot 带。

        标准：属于 TRANS/STRUCT 且跨会话共享数未达到 share_min。
        注：若节点无会话元数据，保守返回 False 防止误删热前缀。
        """
        st = self.session_registry.get(getattr(node, "session_id", None))
        if st is None:
            return False
        if self.slru_evict.share_of(node) >= self.slru_evict.share_min:
            return False
        return st.session_type in (SessionType.TRANS, SessionType.STRUCT)

    def one_shot_key(self, node) -> tuple:
        """One-shot 带内淘汰排序：按访问时间升序（带内 LRU）。"""
        return (node.last_access_time,)

    # ---- 命中/淘汰记账 ----

    def record_hit(self, req, node, hit_tokens: int) -> None:
        """记录缓存命中指标（去重由 telemetry 负责）。"""
        self.telemetry.on_hit(
            tag=self._node_tag(node),
            hit_tokens=hit_tokens,
            req_id=getattr(req, "rid", None),
        )

    def record_evict(self, node) -> None:
        """记录节点淘汰事件并扣减配额占用（支持级联删除逐点触发）。"""
        tokens = _node_num_tokens(node)
        tag = self._node_tag(node)
        queue = tag.queue_name
        one_shot = self.is_one_shot(node)
        # 仅维护启用了配额且非 one-shot 的三队列占用
        if not one_shot:
            self.occupancy.evict(queue, tokens)
        self.telemetry.on_evict(
            node_id=id(node), tag=tag, tokens=tokens, one_shot=one_shot
        )

    def record_victim_source(self, source: str, tokens: float) -> None:
        """记录淘汰决策来源。"""
        self.telemetry.on_victim_source(source, tokens)

    def wants_occupancy(self, available: float, capacity: float, *, has_capacity: bool) -> bool:
        """是否需要做一次全树快照：校准时钟或打印时钟任一到期。

        两个时钟分属两个协作者：校准（tracker）服务账本正确性，打印（telemetry）
        服务观测。调用方拿到 True 后只遍历一次，快照同时喂两边。
        """
        if self.telemetry.wants_profile():
            return True
        return self.occupancy.wants_reconcile(
            available,
            capacity,
            normal_window_s=self.telemetry.PROFILE_INTERVAL,
            pressure_window_s=self.quota.pressure_window_s,
            has_capacity=has_capacity,
        )


    def record_insert(self, node) -> None:
        """新节点挂载时增量计入配额（仅在节点创建时调用，节点分裂时不触发）。"""
        if self.is_one_shot(node):
            return
        self.occupancy.insert(self._node_tag(node).queue_name, _node_num_tokens(node))

    @property
    def needs_capacity(self) -> bool:
        """告诉缓存层：每次淘汰都必须注入 allocator 容量。

        这不是开关，是驱动层的探测信号：``RadixCache`` 用
        ``getattr(strategy, "needs_capacity", False)`` 决定要不要读
        ``available_size()/size``。没有这个属性，capacity 恒为 0，
        ``effective_targets`` 会把所有 target 算成 0，配额退化成
        “每个队列都超上限”。
        """
        return True

    def observe_occupancy(
        self,
        queue_occ: dict,
        session_summary=None,
        *,
        available: float,
        capacity: float,
    ) -> None:
        """全树快照的单一消费入口：同一份快照同时喂校准与渲染。

        调用方（``RadioCache.evict``）已按 ``wants_occupancy`` 判过时钟，
        这里再判一次是为了让该方法自身幂等——重复调用不会重复校准或刷屏。
        """
        if self.occupancy.wants_reconcile(
            available,
            capacity,
            normal_window_s=self.telemetry.PROFILE_INTERVAL,
            pressure_window_s=self.quota.pressure_window_s,
            has_capacity=capacity > 0,
        ):
            self.occupancy.reconcile(queue_occ, capacity)
        if not self.telemetry.wants_profile():
            return
        self.telemetry.observe_profile(
            queue_occ,
            session_summary,
            slru_policy=self.slru_evict,
            mt_policy=self.mt_evict,
            quota_occ=self.occupancy.occ,
            quota_targets=self.quota_targets(
                self.occupancy.occ, self.occupancy.capacity
            ),
            quota_caps=queue_capacities(self.quota, self.occupancy.capacity),
        )

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
        if queue == "agentic" and self.rr_evict is not None:
            # RR 轮询调度，受配额超额缺口上限约束
            picked = self.rr_evict.pick(
                pool,
                self.session_registry,
                max_deficit=self._quota_gap("agentic", occ, capacity),
            )
            if picked is not None:
                return picked
        if queue == "chat" and self.mt_evict is not None:
            picked = self.mt_evict.pick(pool, self.session_registry)
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
        """配额候选均耗尽时的均匀随机兜底（使用独立 RNG 确保可复现）。"""
        pool = list(leaves)
        if not pool:
            return None
        return self._quota_rng.choice(pool)