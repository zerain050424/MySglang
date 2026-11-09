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
import time
from collections import deque
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
from sglang.srt.mem_cache.victim_override import (
    RrEvictPolicy,
    SlruEvictPolicy,
    MultiTurnEvictPolicy,
)
from sglang.srt.mem_cache.heter_logging import (
    CacheTelemetry,
    heterlog,
    HeterProfiler,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)

# 承压判据（Q5=a）：可用容量占池容量的比例低于它就算承压，
# 此时配额账本的校准窗口从 30s 缩短到 pressure_window_s。
PRESSURE_AVAILABLE_RATIO = 0.05


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


class HeterCacheStrategy(EvictionStrategy):
    """HeterCache 多队列缓存淘汰策略。

    两层决策：
    1. 分带过滤：One-shot 节点（无跨轮复用价值）优先清理，不计入配额账本。
    2. 配额仲裁：先按配额承压情况排出候选队列，再由队内策略（RR / SLRU / MT）
       决出具体牺牲节点。

    没有全局价值分：早期版本用 ``reuse_score`` 给每个节点打分、跨队列比较后再竞争，
    在配额为主的用法下已无意义，已整体删除。

    注：本策略只做裁决，KV 释放与 Radix 树拓扑修改由外层缓存引擎执行。
    """
    PROFILE_INTERVAL = 30.0   # 监控输出节流周期（秒）
    MAX_SEEN_RIDS = 100_000   # 请求去重滑动窗口上限

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

        self._last_profile = time.monotonic()
        self._last_reconcile = 0.0
        self._capacity = 0.0
        # 各队列实时可调度占用（排除 one-shot，包含 locked）
        self._occ: dict = {q: 0.0 for q in QUOTA_QUEUES}
        # 兜底淘汰:采样独立 RNG，保证实验结果在相同 seed 下可复现
        self._quota_rng = random.Random(self.quota.seed)
        # Chunked prefill 请求去重队列，避免同一请求多次命中导致指标虚高
        self._seen_rids: set = set()
        self._rid_order: deque = deque()
        self.profiler = HeterProfiler()
        # 账本与事件流的多写收在门面里，下面三个 record_* 只管上报意图。
        # profiler 引用同时保留：外部消费者（Heter-Bench 的 struct_profile、测试）
        # 直接读 strategy.profiler，不经过门面。
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
        """重置统计指标与去重状态（缓存 flush / reset 时调用）。"""
        self.profiler.reset()
        self._seen_rids.clear()
        self._rid_order.clear()
        for q in QUOTA_QUEUES:
            self._occ[q] = 0.0

    def _node_tag(self, node) -> HeterCacheNodeTag:
        """节点的生效标签（已收敛枚举）。force_agentic 开启时统一按 agentic 算。"""
        if self.force_agentic:
            return HeterCacheNodeTag.QAGENT
        return HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))

    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        """HeterCache 没有单节点全局排序，调用方应当走配额裁决。

        旧实现先按 one-shot 分带、带内按 ``_reuse_score`` 排序。复用分删除后，
        任何“把 HeterCache 当成优先级堆”的用法都失去语义（配额是按队列与目标
        占用水填充的，不是按节点打分），所以这里直接失败而不是退回某个默认序。

        Raises:
            NotImplementedError: 总是抛出。HeterCache 必须搭配 HeterCacheRadixCache
                使用（它走 ``quota_selector.pick_victim``）；若被挂到统一树上，
                启动阶段应当被 registry 的守卫拦住。
        """
        raise NotImplementedError(
            "HeterCacheStrategy has no single-node ordering; "
            "use quota arbitration (quota_selector.pick_victim) instead"
        )

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
        """记录缓存命中指标。基于 rid 去重，规避 chunked prefill 重复记账。"""
        rid = getattr(req, "rid", None)
        if rid is None or rid in self._seen_rids:
            return
        self._seen_rids.add(rid)
        self._rid_order.append(rid)
        if len(self._rid_order) > self.MAX_SEEN_RIDS:
            self._seen_rids.discard(self._rid_order.popleft())

        tag = self._node_tag(node)
        self.telemetry.on_hit(tag=tag, hit_tokens=hit_tokens, req_id=rid)

    def record_evict(self, node) -> None:
        """记录节点淘汰事件并扣减配额占用（支持级联删除逐点触发）。"""
        tokens = _node_num_tokens(node)
        tag = self._node_tag(node)
        queue = tag.queue_name
        one_shot = self.is_one_shot(node)
        # 仅维护启用了配额且非 one-shot 的三队列占用
        if not one_shot and queue in QUOTA_QUEUES:
            self._occ[queue] = max(0.0, self._occ[queue] - tokens)
        self.telemetry.on_evict(
            node_id=id(node), tag=tag, tokens=tokens, one_shot=one_shot
        )

    def record_victim_source(self, source: str, tokens: float) -> None:
        """记录淘汰决策来源。"""
        self.telemetry.on_victim_source(source, tokens)

    def _reconcile_interval(self, available: float, capacity: float) -> float:
        """计算配额校准周期：内存承压时缩短至 pressure_window_s，平时为 30s。"""
        if capacity > 0 and available / capacity < PRESSURE_AVAILABLE_RATIO:
            return self.quota.pressure_window_s
        return self.PROFILE_INTERVAL

    def wants_occupancy(self, available: float, capacity: float, *, has_capacity: bool) -> bool:
        """判断是否需要执行全树扫描（兼顾指标打印与配额校准时钟）。"""
        now = time.monotonic()
        if now - self._last_profile >= self.PROFILE_INTERVAL:
            return True
        if not has_capacity:
            return now - self._last_reconcile >= self.PROFILE_INTERVAL
        return now - self._last_reconcile >= self._reconcile_interval(available, capacity)

    @property
    def needs_capacity(self) -> bool:
        """缓存层每次淘汰都要注入 allocator 容量：配额仲裁不能没有容量。"""
        return True

    def reconcile_occupancy(self, queue_occ: dict, capacity: float) -> None:
        """通过全树快照校准 _occ，吸收运行时标签变更、one-shot 晋升等增量漂移。"""
        for q in QUOTA_QUEUES:
            rec = queue_occ.get(q) or {}
            self._occ[q] = max(0.0, float(rec.get("total", 0.0)) - float(rec.get("one_shot", 0.0)))
        self._capacity = max(float(capacity), 0.0)
        self._last_reconcile = time.monotonic()

    def record_insert(self, node) -> None:
        """新节点挂载时增量计入配额（仅在节点创建时调用，节点分裂时不触发）。"""
        if self.is_one_shot(node):
            return
        queue = self._node_tag(node).queue_name
        if queue not in QUOTA_QUEUES:
            return
        self._occ[queue] += _node_num_tokens(node)

    def observe_occupancy(
        self,
        queue_occ: dict,
        session_summary=None,
        *,
        available: float,
        capacity: float,
    ) -> None:
        """全树快照聚合入口，统一触发配额校准与 profile 监控打印。"""
        now = time.monotonic()
        if now - self._last_reconcile >= self._reconcile_interval(available, capacity):
            self.reconcile_occupancy(queue_occ, capacity)
        if now - self._last_profile < self.PROFILE_INTERVAL:
            return
        self._last_profile = now
        _profile = self.profiler.summary(
            queue_occ,
            session_summary,
            slru_policy=self.slru_evict,
            mt_policy=self.mt_evict,
            quota_occ=self._occ,
            quota_targets=self.quota_targets(self._capacity),
            quota_caps=queue_capacities(self.quota, self._capacity),
        )
        heterlog.info("Profile", f"[HeterCache profile]\n{_profile}")

    # ---- 跨队列配额 ----

    def quota_ranking(self, capacity: float, page_size: int) -> list:
        """获取当前淘汰候选队列的优先级序列。"""
        targets = effective_targets(self._occ, capacity, self.quota)
        chunk = resolve_chunk(page_size, self.quota)
        return queue_ranking(
            self._occ, targets, self.quota, chunk=float(chunk), capacity=capacity
        )

    def quota_targets(self, capacity: float) -> dict:
        """获取各队列的目标 Token 配额（仅观测用途）。"""
        return effective_targets(self._occ, capacity, self.quota)

    def quota_victim(self, queue: str, leaves, capacity: float):
        """在指定队列内挑选淘汰节点。若无合适候选返回 None（触发上层跳队）。"""
        # 队列名来自配额层（字符串），先反查成枚举再与节点标签直接比较，
        # 中间不经过任何字符串投影。random 也反查得到 QTRANS，与旧判定等价；
        # 未知名字返回 None，直接跳队。
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
                pool, self.session_registry, max_deficit=self._quota_gap("agentic", capacity)
            )
            if picked is not None:
                return picked
        if queue == "chat" and self.mt_evict is not None:
            picked = self.mt_evict.pick(pool, self.session_registry)
            if picked is not None:
                return picked
        # 队内策略弃权（关闭或不接管）时本队列无候选，交由上层跳下一队；
        # 不再退回复用分排序。
        return None

    def _quota_gap(self, queue: str, capacity: float):
        """计算队列超出有效配额的 Token 数；未超限时返回 None。"""
        if capacity <= 0:
            return None
        target = effective_targets(self._occ, capacity, self.quota)[queue]
        gap = self._occ[queue] - target
        return gap if gap > 0 else None

    def quota_fallback(self, leaves):
        """配额候选均耗尽时的均匀随机兜底（使用独立 RNG 确保可复现）。"""
        pool = list(leaves)
        if not pool:
            return None
        return self._quota_rng.choice(pool)