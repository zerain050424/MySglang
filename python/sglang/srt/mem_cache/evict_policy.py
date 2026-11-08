from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Tuple, Union
from sglang.srt.mem_cache.uni_utils import (
    node_end_offset,
    node_num_tokens,
    RrParams,
    MultiTurnParams,
    SessionRegistry,
    SessionType,
    SlruParams,
    TASK_CONFIG,
    workload_label_of,
    SimpleHybridTaskType,
    _cached_lognorm_cdf,
    task_type_of_tag,
)
from sglang.srt.mem_cache.victim_override import (
    RrEvictPolicy,
    SlruEvictPolicy,
    MultiTurnEvictPolicy,
)
from sglang.srt.mem_cache.heter_logging import heterlog, HeterProfiler

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)


# 节点口径与标签映射统一在 uni_utils，避免策略/改写器/指标各算一套。
_node_num_tokens = node_num_tokens
_node_end_offset = node_end_offset


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
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class TLRUStrategy(EvictionStrategy):
    """Tail-Optimized LRU (Zhang et al., arXiv:2510.15152).

    A conversation with history length L whose next prompt is expected to add
    Q_hat tokens only has to keep L + Q_hat - threshold tokens cached to hold its
    next prefill under the TTFT budget; tokens past that budget cannot improve
    tail latency and are "TEL-safe", i.e. free to evict. Such nodes are reported
    as infinitely old, which is the implementation the paper suggests: the
    existing eviction driver then drains them before anything else (the paper's
    phase 1) and continues in plain recency order once they run out (phase 2),
    so neither eviction loop needs to know about T-LRU.

    threshold and next_prompt_estimate are token counts, whereas the paper states
    both in blocks; multiply the paper's values by page_size to convert.
    """

    def __init__(self, threshold: int = 0, next_prompt_estimate: int = 0):
        self.threshold = threshold
        self.next_prompt_estimate = next_prompt_estimate

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # node._tlru_history_len is the branch's high-water depth, i.e. the
        # paper's L, and deliberately does not shrink when the tail is trimmed.
        # Deriving L from what is still resident instead would leave the
        # shortened conversation over budget on the next pass too, and T-LRU
        # would walk it down to nothing rather than stopping after
        # (threshold - Q_hat) tokens.
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
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class SimpleHybridStrategy(EvictionStrategy):
    """
      多级别队列进行淘汰
    """
    # profile 打印的节流窗口（秒）
    PROFILE_INTERVAL = 30.0
    # Cap on the once-per-request dedup set (bounds memory in long runs).
    MAX_SEEN_RIDS = 100_000

    def __init__(
        self,
        max_offset: int = 2048,
        *,
        force_agentic: bool = False,
        rr: Union[RrParams, dict, None] = None,
        slru: Union[SlruParams, dict, None] = None,
        mt: Union[MultiTurnParams, dict, None] = None,
    ):
        """
        参数组由 RrParams / SlruParams / MultiTurnParams 承载（uni_utils）。
        配置只接受分组形式：
            {"rr": {...}, "slru": {...}, "mt": {...}}
        未知键在参数组构造处直接报错。

        max_offset 不属于任何一组：它是 struct 复用分数里的位置项上界。

        force_agentic 是实验开关：打开后所有节点都按 agentic 语义
        解释（评分、one-shot、插件作用域一致），并把 queue_tag 直接标成 QAGENT。
        用途是让 QAGENT-only 的插件拿到 100% 流量；它不修改 session_type，因此
        分类器自身的统计仍然可测。
        """
        self.rr = RrParams.from_any(rr)
        self.slru = SlruParams.from_any(slru)
        self.mt = MultiTurnParams.from_any(mt)
        self.force_agentic = bool(force_agentic)
        self.max_offset = max_offset or 2048

        # profile 打印的节流窗口
        self._last_profile = time.monotonic()
        # 每请求只归因一次命中的去重账本
        self._seen_rids: set = set()
        self._rid_order: deque = deque()
        #   观测账本（累计命中/淘汰/one_shot/淘汰来源）已外挂至
        #   Uni-Bench/profile 的 HeterProfiler（方案 B）：记账/渲染同口径，
        #   reset 随 reset_bookkeeping 委托清空；树采集仍在策略内。
        self.profiler = HeterProfiler()


        self.rr_evict = RrEvictPolicy(self.rr) if self.rr.fair_rr else None
        # 会话账本：SLRU 的共享数账本与 one-shot 判定共用它。
        self.session_registry = SessionRegistry.global_instance()
        # 共享数阈值只有一处配置：SlruParams.share_min。
        # 它同时喂给 SLRU 的准入判定与 one-shot 的分界。
        self.slru_evict = SlruEvictPolicy(
            self.session_registry, self.slru.share_min
        )
        # 多轮对话（QCHAT）带内改写器：1-prob(e) 排序（默认启用，见 MultiTurnParams）
        self.mt_evict = MultiTurnEvictPolicy(self.mt, self.session_registry) if self.mt.enabled else None
        self.victim_overrides = tuple(
            p for p in (self.rr_evict, self.slru_evict, self.mt_evict) if p is not None
        )

    def reset_bookkeeping(self) -> None:
        """清空窗口/累计/来源计数（缓存 flush 与 reset 同源）。

        不复位参数，也不动 `_last_profile`：它只影响打印节流。
        跨 run 复用 server 时，若不清理，`[SimpleHybrid profile]` 打印的是
        上一轮与这一轮的混合值。
        """
        self.profiler.reset()
        self._seen_rids.clear()
        self._rid_order.clear()


    def _task_type_of(self, node) -> SimpleHybridTaskType:
        # 队列标签由缓存状态机维护。映射表在 uni_utils，策略与改写器共用。
        # force_agentic 时短路：即使某个节点残留 QTRANS/QCHAT/QSTRUCT 标签，
        # 评分与队列归属也统一按 agentic 走（缓存层会同步把标签强制成 QAGENT）。
        # 注意：one-shot 判定不走这里 —— 它只看会话类型 + 跨会话共享数。
        if self.force_agentic:
            return SimpleHybridTaskType.MULTI_TURN_AGENTIC
        
        return task_type_of_tag(getattr(node, "queue_tag", None))

    def _reuse_score(self, node) -> float:
        """calculate_priority() from HeterCache, with time.monotonic() as now."""
        task_type = self._task_type_of(node)
        elapsed = time.monotonic() - node.last_access_time
        e = int(elapsed)
        cfg = TASK_CONFIG.get(task_type)
        if cfg is not None:  # 多轮任务
            mu, sigma = cfg
            return 1.0 - _cached_lognorm_cdf(e, mu, sigma)
        if task_type == SimpleHybridTaskType.STRUCT_SINGLE_TURN:
            # HeterCache 的 offset 是块在序列中的绝对结束位置，不是 radix 片段长度。
            offset_factor = 1.0 - _node_end_offset(node) / (self.max_offset or 1)
            return max(0.0, min(1.0, offset_factor))
        return 0.0


    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        """
        全序优先级：带 0 = 一次性内容（random / 单轮 response 段）先淘汰，
        带内按 last_access 升序（与 one_shot_key 同口径）；带 1 = 可调度内容
        按 reuse_score 竞争。
        """
        if self.is_one_shot(node):
            return (0, float(-_node_end_offset(node)), 0.0)
        return (1, self._reuse_score(node), float(-_node_num_tokens(node)))

    # 暴露给淘汰器的键语义

    def victim_key(self, node) -> tuple:
        """全局基础候选键：复用分数最低者先走，同分时 token 多者先走。

        队列内部顺序由插件接管（SLRU 对受保护的 struct 叶子改写成段内 LRU），
        因此这里不再编码「类间 rank」。
        """
        return (self._reuse_score(node), -_node_num_tokens(node))

    def is_one_shot(self, node) -> bool:
        """一次性内容（one-shot）：淘汰顺序永远先于可调度内容。

        判定只看会话类型与跨会话共享数

        one_shot = 会话类型 ∈ {TRANS, STRUCT} 且 share_sessions < share_min

        为什么不能看标签：`queue_tag` 是谁创建了这段的粗粒度标记，而
        共享数才是还会不会被复用的证据。典型反例：TRANS 会话插入的
        文档段后来被别的会话反复命中——标签仍挂着 QTRANS，但它是热前缀。

        `share_sessions >= share_min` = 已被 >=2 个不同会话命中过（创建者
        自己的首次重读计 1，首次外部命中即达 2），属于跨会话模板，
        不进一次性带——即使它的创建者当初被判成 TRANS。阈值取自
        `SlruParams.share_min`，与 `SlruEvictPolicy.applies_to` 是同一个值。
        """
        st = self.session_registry.get(getattr(node, "session_id", None))
        if st is None:
            # 无会话证据（客户端未带 session_id，或账本已被 sweep）：不判 one-shot。
            return False
        if self.slru_evict.share_of(node) >= self.slru_evict.share_min:
            # 已被足够多会话复用：即便创建者是 TRANS 也不再是一次性内容。
            return False
        return st.session_type in (SessionType.TRANS, SessionType.STRUCT)

    def workload_of(self, node) -> str:
        """节点所属的工作负载。"""
        return workload_label_of(self._task_type_of(node))

    def one_shot_key(self, node) -> tuple:
        """
        one-shot 带内排序：最久未访问者先走（LRU）。
        """
        return (node.last_access_time,)

    # ---- 命中/淘汰记账 ----

    def record_hit(self, req, node, hit_tokens: int) -> None:
        # Attribute a request's prefix-cache hit to its workload type, once
        # per request.
        rid = getattr(req, "rid", None)
        if rid is None or rid in self._seen_rids:
            return
        self._seen_rids.add(rid)
        self._rid_order.append(rid)
        if len(self._rid_order) > self.MAX_SEEN_RIDS:
            self._seen_rids.discard(self._rid_order.popleft())

        task_type = self._task_type_of(node)
        wkey = self.workload_of(node)
        # 累计命中（账本外挂：HeterProfiler）。
        self.profiler.record_hit(wkey, hit_tokens)
        # 策略事件流：高频路径，DEBUG 级别短路
        if heterlog.enabled:
            heterlog.hit(workload=wkey,
                         task_type=task_type,
                         hit_tokens=hit_tokens,
                         req=rid)

    def record_evict(self, node) -> None:
        """累计被淘汰的 token 数，按 workload 记账。
        one-shot 淘汰量单记一个标量；random 的淘汰量直接读 workload 行，
        单轮 response 段的淘汰量 = 标量 - evicted["random"]。"""
        tokens = _node_num_tokens(node)
        wkey = self.workload_of(node)
        self.profiler.record_evict(wkey, tokens, one_shot=self.is_one_shot(node))
        # 策略事件流：节点级驱逐事件（供 Uni-Bench 结果矩阵分析）
        heterlog.evict(node_id=id(node),
                       workload=wkey,
                       tokens=tokens,
                       one_shot=self.is_one_shot(node))

    def record_victim_source(self, source: str, tokens: float) -> None:
        """
        记录本次淘汰由谁决定，按来源累计 token 数。
        source 取值："one_shot"/ "global"（全局 victim_key 竞争）/
        插件的 queue 名（如 "qagent"、"qstruct"）。
        """
        self.profiler.record_victim_source(source, tokens)
        # 策略事件流：淘汰裁决来源
        heterlog.victim_source(source, tokens)


    def profile_due(self) -> bool:
        """调用点门禁：到 PROFILE_INTERVAL 才值得做一次全树口径计算。

        evict() 热路径先问这个，再做 _per_workload_token_count() 的全树遍历，
        避免在节流窗口内白白遍历整棵树。
        """
        return time.monotonic() - self._last_profile >= self.PROFILE_INTERVAL

    def maybe_print_profile(self, workload_occ, session_summary=None) -> None:
        now = time.monotonic()
        if now - self._last_profile < self.PROFILE_INTERVAL:
            return
        self._last_profile = now
        _profile = self.profiler.summary(
            workload_occ,
            session_summary,
            slru_policy=self.slru_evict,
            mt_policy=self.mt_evict,
        )
        heterlog.info("Profile", f"[SimpleHybrid profile]\n{_profile}")
