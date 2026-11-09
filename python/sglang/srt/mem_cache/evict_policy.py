"""HeterCache 的淘汰策略与评分。

两层职责：

1. 单节点排序（``EvictionStrategy.get_priority``）：LRU / LFU / TLRU / SLRU 等
   原生策略，以及 HeterCache 的 ``reuse_score``；
2. 多队列仲裁（``HeterCacheStrategy``）：把节点归到 struct / agentic / chat /
   random 四个队列，分带淘汰，并把队内细节委托给 ``victim_override`` 的改写器。

所有 ``get_priority`` 的共同契约：**返回值越小越先被淘汰**。这个约定必须由每个
子类自己保证（MRU / FILO 靠取负实现），调用方只做 ``min()``。
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
from sglang.srt.mem_cache.uni_utils import (
    node_end_offset,
    node_num_tokens,
    RrParams,
    MultiTurnParams,
    SessionRegistry,
    SessionType,
    SlruParams,
    QUOTA_QUEUES,
    QuotaParams,
    TASK_CONFIG,
    workload_label_of,
    HeterCacheTaskType,
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

# 承压判据（Q5=a）：可用容量占池容量的比例低于它就算承压，
# 此时配额账本的校准窗口从 30s 缩短到 pressure_window_s。
PRESSURE_AVAILABLE_RATIO = 0.05


# 节点口径与标签映射统一在 uni_utils，避免策略/改写器/指标各算一套。
_node_num_tokens = node_num_tokens
_node_end_offset = node_end_offset


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
    """HeterCache 的多队列淘汰策略（``--radix-eviction-policy heter_cache``）。

    把节点归入四个队列（struct / agentic / chat / random），并按三个层次做题：

    * 分带：one-shot 内容（random 整轮、单轮 response 段）永远排在带 0，
      先于任何可调度内容被清掉——清它没有机会成本。
    * 评分：可调度带内用 ``_reuse_score`` 给节点打分（多轮用复用概率，
      单轮用位置项），分数低者先走。
    * 仲裁：可选地开启跨队列配额（``QuotaParams``），先决定点名哪个队列，
      再交给队内改写器（RR / SLRU / MT）挑具体节点。

    本类只提供评分与候选选择接口；释放 KV、改树等副作用在缓存层。
    前缀缓存命中/淘汰的观测账本已外挂到 ``HeterProfiler``，本类只负责喂数据。
    """
    # profile 打印的节流窗口（秒）；与配额账本的校准时钟相互独立。
    PROFILE_INTERVAL = 30.0
    # 去重账本上限：一次 run 里请求数可能到百万级，不封顶会缓慢涨内存。
    MAX_SEEN_RIDS = 100_000

    def __init__(
        self,
        max_offset: int = 2048,
        *,
        force_agentic: bool = False,
        rr: Union[RrParams, dict, None] = None,
        slru: Union[SlruParams, dict, None] = None,
        mt: Union[MultiTurnParams, dict, None] = None,
        quota: Union[QuotaParams, dict, None] = None,
    ):
        """构造策略并完成插件接线。

        参数组全部由 ``uni_utils`` 的参数类承载，配置只接受分组形式：
        ``{"rr": {...}, "slru": {...}, "mt": {...}, "quota": {...}}``。
        分组设计的目的是让未知键在构造处立即报错，而不是静默忽略一项实验配置。

        Args:
            max_offset: struct 复用分数里位置项的上界（token）。不属于任何组：
                它是算法常量而非实验旋钮。
            force_agentic: 实验开关。打开后所有节点按 agentic 语义解释（评分、
                one-shot、插件作用域一致），并把 ``queue_tag`` 直接改成 QAGENT，
                让 QAGENT-only 的插件拿到 100% 流量；它不修改 session_type，
                所以分类器自身的统计仍然可测。
            rr: Agentic 会话级轮询参数；``fair_rr`` 为真时才建改写器。
            slru: SLRU 共享会话数阈值（同时决定 one-shot 分界与保护范围）。
            mt: 多轮对话（QCHAT）带内改写器参数。
            quota: 跨队列配额；``caps`` 为空时不启用，行为与引入配额前一致。

        Raises:
            ValueError: 任何参数组不合法（如 ``quota caps`` 之和大于 1、
                ``reservations`` 超过对应 ``caps``）。
        """
        self.rr = RrParams.from_any(rr)
        self.slru = SlruParams.from_any(slru)
        self.mt = MultiTurnParams.from_any(mt)
        self.quota = QuotaParams.from_any(quota)
        self.force_agentic = bool(force_agentic)
        self.max_offset = max_offset or 2048

        # 两个时钟：打印固定 30s，配额校准在承压时可缩到 pressure_window_s。
        self._last_profile = time.monotonic()
        self._last_reconcile = 0.0
        self._capacity = 0.0
        # 配额口径的队列占用（排除 one-shot、包含 locked）。
        # 增量只在“创建 / 消亡”两条路径上维护；改标签与 one-shot 迁移靠快照对齐，
        # 因为增量方案无法以 O(1) 跟踪这两类跳变（见 reconcile_occupancy）。
        self._occ: dict = {q: 0.0 for q in QUOTA_QUEUES}
        # 兜底抽样用策略私有 RNG：不碰全局 random，同 seed 可复现。
        self._quota_rng = random.Random(self.quota.seed)
        # 每请求只归因一次命中的去重账本（同一请求多次 chunked prefill 只算一次）。
        self._seen_rids: set = set()
        self._rid_order: deque = deque()
        # 观测账本已外挂到 Uni-Bench/profile 的 HeterProfiler：记账/渲染同口径，
        # reset 随 reset_bookkeeping 委托清空；树采集仍在本类内。
        self.profiler = HeterProfiler()

        # 插件只按开关实例化：关闭的插件不进覆盖链，避免每次淘汰多一层空调用。
        self.rr_evict = RrEvictPolicy(self.rr) if self.rr.fair_rr else None
        # 会话账本：SLRU 的共享数账本与 one-shot 判定共用它，因此必须先于两者建。
        self.session_registry = SessionRegistry.global_instance()
        self.slru_evict = SlruEvictPolicy(
            self.session_registry, self.slru.share_min
        )
        self.mt_evict = MultiTurnEvictPolicy(self.mt, self.session_registry) if self.mt.enabled else None
        self.victim_overrides = tuple(
            p for p in (self.rr_evict, self.slru_evict, self.mt_evict) if p is not None
        )

    def reset_bookkeeping(self) -> None:
        """清空跨 run 的累计量（缓存 flush / reset 时调用）。

        不清参数、也不动 ``_last_profile``：前者是配置，后者只影响打印节流。
        不清理就会在复用同一 server 跑多个 seed 时打印混合值，实验之间互相污染。
        """
        self.profiler.reset()
        self._seen_rids.clear()
        self._rid_order.clear()
        # 树已经空了，配额账本必须归零，否则下一轮从残留占用起步。
        for q in QUOTA_QUEUES:
            self._occ[q] = 0.0


    def _task_type_of(self, node) -> HeterCacheTaskType:
        """把节点的 ``queue_tag`` 映射回任务类型（评分与队列归属的唯一入口）。

        映射表在 ``uni_utils``，保证策略、改写器、指标不会各算一套。
        ``force_agentic`` 时短路：即使节点残留 QTRANS/QCHAT/QSTRUCT 标签，
        评分与队列归属也统一按 agentic 走（缓存层会同步把标签强制成 QAGENT）。

        注意 one-shot 判定不走这里：那个只看会话类型 + 跨会话共享数。
        """
        if self.force_agentic:
            return HeterCacheTaskType.MULTI_TURN_AGENTIC
        
        return task_type_of_tag(getattr(node, "queue_tag", None))

    def _reuse_score(self, node) -> float:
        """算节点的复用分数（越低越先被淘汰）。

        两类队列用两套模型，是 HeterCache 的核心假设：

        * 多轮（chat / agentic）：``1 - CDF(elapsed)``，即下一轮还会复用的概率。
          参数 μ/σ 取自 ``TASK_CONFIG``（lognormal，UniCache 原机制）。
        * 单轮（struct）：只剩位置项。单轮没有“下一轮”，时间在这里无意义；
          offset 越靠后越接近尾部，尾部复用的可能性越低。

        random 返回 0：它属于 one-shot 带，正常不会走到评分（带 0 已优先）。
        """
        task_type = self._task_type_of(node)
        elapsed = time.monotonic() - node.last_access_time
        e = int(elapsed)
        cfg = TASK_CONFIG.get(task_type)
        if cfg is not None:  # 多轮任务
            mu, sigma = cfg
            return 1.0 - _cached_lognorm_cdf(e, mu, sigma)
        if task_type == HeterCacheTaskType.STRUCT_SINGLE_TURN:
            # HeterCache 的 offset 是块在序列中的绝对结束位置，不是 radix 片段长度。
            offset_factor = 1.0 - _node_end_offset(node) / (self.max_offset or 1)
            return max(0.0, min(1.0, offset_factor))
        return 0.0


    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        """全序优先级（基类契约：越小越先淘汰）。

        带 0 = 一次性内容，按树内偏移倒序（越靠尾部越先）；带 1 = 可调度内容，
        按 ``_reuse_score`` 竞争，同分时 token 多者先走（一次腾更多空间）。
        """
        if self.is_one_shot(node):
            return (0, float(-_node_end_offset(node)), 0.0)
        return (1, self._reuse_score(node), float(-_node_num_tokens(node)))

    def victim_key(self, node) -> tuple:
        """全局基础候选键：复用分数最低者先走，同分时 token 多者先走。

        不编码“类间 rank”：队列内部顺序由改写器接管（SLRU 把受保护的 struct
        叶子改写成段内 LRU，RR 改成会话轮询），在这里再排一次会打架。
        """
        return (self._reuse_score(node), -_node_num_tokens(node))

    def is_one_shot(self, node) -> bool:
        """判节点是否属于 one-shot 带（永远先被清掉的那一带）。

        判据：会话类型 ∈ {TRANS, STRUCT} 且跨会话共享数 < ``share_min``。

        为什么不能只看标签：``queue_tag`` 只说明“谁创建了这段”，而共享数才是
        “还会不会被复用”的证据。典型反例：TRANS 会话插入的文档段后来被别的
        会话反复命中——标签仍是 QTRANS，但它已经是热前缀，不能再当一次性内容。

        无会话证据（未带 session_id，或账本已被 sweep）时返回 False：宁可多留
        一段垃圾，也不误删可能是跨会话共享的前缀。

        Args:
            node: 树节点。

        Returns:
            True 表示该节点属于 one-shot 带。
        """
        st = self.session_registry.get(getattr(node, "session_id", None))
        if st is None:
            return False
        if self.slru_evict.share_of(node) >= self.slru_evict.share_min:
            # 已被足够多会话复用：即使创建者是 TRANS 也不再是一次性内容。
            return False
        return st.session_type in (SessionType.TRANS, SessionType.STRUCT)

    def workload_of(self, node) -> str:
        """节点所属的工作负载标签（chat / agentic / struct / random）。"""
        return workload_label_of(self._task_type_of(node))

    def one_shot_key(self, node) -> tuple:
        """one-shot 带内排序：越久未访问越先走（带内 LRU）。

        不用 ``get_priority`` 的全序：带 0 内部按偏移排序是为了把同一轮请求的
        尾部先清，但带内整体顺序仍应服从 recency，否则会把老叶子钦在后面。
        """
        return (node.last_access_time,)

    # ---- 命中/淘汰记账 ----

    def record_hit(self, req, node, hit_tokens: int) -> None:
        """按 workload 记一次前缀缓存命中（账本在 ``HeterProfiler``）。

        同一请求只归因一次：chunked prefill 会让一个请求多次进入命中路径，
        重复计数会把命中率算虚高。去重账本用 rid 环形队列封顶，长跑不会涨内存。

        Args:
            req: 请求对象，只读 ``rid``。
            node: 命中的树节点，用于推导 workload 与任务类型。
            hit_tokens: 本次命中的 token 数。
        """
        rid = getattr(req, "rid", None)
        if rid is None or rid in self._seen_rids:
            return
        self._seen_rids.add(rid)
        self._rid_order.append(rid)
        if len(self._rid_order) > self.MAX_SEEN_RIDS:
            self._seen_rids.discard(self._rid_order.popleft())

        task_type = self._task_type_of(node)
        wkey = self.workload_of(node)
        self.profiler.record_hit(wkey, hit_tokens)
        # 高频路径：先判级别再组参数，关闭时零开销。
        if heterlog.enabled:
            heterlog.hit(workload=wkey,
                         task_type=task_type,
                         hit_tokens=hit_tokens,
                         req=rid)

    def record_evict(self, node) -> None:
        """按 workload 累计被淘汰的 token，并同步扣减配额账本。

        ``_evict_one`` 做级联删除时会逐节点调用本方法，所以祖先链上的节点
        也会各扣一次，不需要额外的级联钩子。

        Args:
            node: 即将被释放的节点（调用时 KV 仍然有效）。
        """
        tokens = _node_num_tokens(node)
        wkey = self.workload_of(node)
        one_shot = self.is_one_shot(node)
        # 配额只统计三个队列。注意 random 不参与：它的节点通常整行 one-shot，
        # 但缺席会话证据（未带 session_id）时 is_one_shot 会返回 False、wkey 为
        # "random"，必须用白名单挡掉，否则扣减会打到不存在的键上。
        if self.quota.enabled and not one_shot and wkey in QUOTA_QUEUES:
            self._occ[wkey] = max(0.0, self._occ[wkey] - tokens)
        self.profiler.record_evict(wkey, tokens, one_shot=one_shot)
        # 节点级驱逐事件，供 Uni-Bench 结果矩阵归因。
        heterlog.evict(node_id=id(node),
                       workload=wkey,
                       tokens=tokens,
                       one_shot=one_shot)

    def record_victim_source(self, source: str, tokens: float) -> None:
        """按裁决来源累计淘汰量，用于回答“到底是谁决定的”。

        Args:
            source: ``"one_shot"``（带 0 清理）/ ``"global"``（全局 victim_key）/
                ``"quota:<queue>"`` / ``"quota:fallback"`` /
                改写器的 queue 名（如 ``"qagent"``）。
            tokens: 本次释放的 token 数。
        """
        self.profiler.record_victim_source(source, tokens)
        heterlog.victim_source(source, tokens)


    def _reconcile_interval(self, available: float, capacity: float) -> float:
        """返回配额账本的校准窗口：承压时缩到 ``pressure_window_s``，否则 30s。

        校准本身是 O(树) 遍历，所以只在配额启用时才允许缩短窗口：没开配额时
        多跑一遍遍历没有任何收益（这段代码的前身就曾因此在热路径上白发全线）。

        Args:
            available: 当前可用 token（``allocator.available_size()``）。
            capacity: 池容量（token）；<=0 时视为拿不到信息，退回常规窗口。

        Returns:
            本次允许的最小校准间隔（秒）。
        """
        if capacity > 0 and available / capacity < PRESSURE_AVAILABLE_RATIO:
            return self.quota.pressure_window_s
        return self.PROFILE_INTERVAL

    def wants_occupancy(self, available: float, capacity: float, *, has_capacity: bool) -> bool:
        """门禁：打印时钟或校准时钟任一到期，才值得遍历一次树。

        两个时钟必须分开：校准可能要秒级（配额靠它抓突发），但 profile 打印
        固定 30s，否则日志会被 2~5s 一行浅没。

        Args:
            available: 当前可用 token。
            capacity: 池容量（token）。
            has_capacity: 调用方是否真拿到了可用量；False 表示配额未启用或后端
                不是 heter_cache，此时分不清承压，只能用常规窗口。

        Returns:
            True 表示调用方应该马上做一次 ``observe_occupancy``。
        """
        now = time.monotonic()
        if now - self._last_profile >= self.PROFILE_INTERVAL:
            return True
        if not self.quota.enabled:
            return False
        if not has_capacity:
            return now - self._last_reconcile >= self.PROFILE_INTERVAL
        return now - self._last_reconcile >= self._reconcile_interval(available, capacity)

    @property
    def needs_capacity(self) -> bool:
        """是否需要缓存层把 allocator 的可用量/容量传进来（仅配额启用时）。"""
        return self.quota.enabled

    def reconcile_occupancy(self, workload_occ: dict, capacity: float) -> None:
        """用全树快照把 ``_occ`` 拉回真值（Ground Truth）。

        增量记账故意只覆盖“创建 / 消亡”两条 O(1) 路径，剩下三类漂移靠这里吸收：

        * 运行期改标签（QTRANS → QAGENT/QCHAT/QSTRUCT，分叉点父节点→QSTRUCT）；
        * one-shot → 可调度的迁移（共享数涨过 ``share_min``）；
        * 任何漏掉的增量路径。

        这也是承压时把窗口压到秒级的理由：漂移只能被限制在两个快照之间。

        Args:
            workload_occ: ``_per_workload_token_count()`` 的输出，
                形如 ``{label: {"total": t, "one_shot": o}}``。
            capacity: 池容量（token），缓存下来供打印时算 target/cap。
        """
        for q in QUOTA_QUEUES:
            rec = workload_occ.get(q) or {}
            self._occ[q] = max(0.0, float(rec.get("total", 0.0)) - float(rec.get("one_shot", 0.0)))
        self._capacity = max(float(capacity), 0.0)
        self._last_reconcile = time.monotonic()

    def record_insert(self, node) -> None:
        """新节点入账（配额口径：排除 one-shot）。

        只由缓存层的**节点创建**路径调用。不能挂到 ``_insert_observers``：
        分裂也会触发它，而分裂时 ``child.key`` 同步变短、队列归属不变，
        占用净变化为 0，再记一次就是双计。

        Args:
            node: 刚挂上树的新节点（带 value）。
        """
        if not self.quota.enabled or self.is_one_shot(node):
            return
        wkey = self.workload_of(node)
        if wkey not in QUOTA_QUEUES:
            return  # random 不参与配额
        self._occ[wkey] += _node_num_tokens(node)

    def observe_occupancy(
        self,
        workload_occ: dict,
        session_summary=None,
        *,
        available: float,
        capacity: float,
    ) -> None:
        """缓存层唯一的观测入口：按两个时钟分别做账本校准与 profile 打印。

        一次全树遍历同时喂两个消费者，避免各自遍历一遍。

        Args:
            workload_occ: ``_per_workload_token_count()`` 的输出。
            session_summary: 会话账本的缺口摘要，只用于打印。
            available: 当前可用 token。
            capacity: 池容量（token）。
        """
        now = time.monotonic()
        if (
            self.quota.enabled
            and now - self._last_reconcile >= self._reconcile_interval(available, capacity)
        ):
            self.reconcile_occupancy(workload_occ, capacity)
        if now - self._last_profile < self.PROFILE_INTERVAL:
            return
        self._last_profile = now
        quota_on = self.quota.enabled
        _profile = self.profiler.summary(
            workload_occ,
            session_summary,
            slru_policy=self.slru_evict,
            mt_policy=self.mt_evict,
            quota_occ=self._occ if quota_on else None,
            quota_targets=(
                self.quota_targets(self._capacity) if quota_on else None
            ),
            quota_caps=(
                queue_capacities(self.quota, self._capacity) if quota_on else None
            ),
        )
        heterlog.info("Profile", f"[HeterCache profile]\n{_profile}")

    # ---- 跨队列配额（阶段 1a/1b）：缓存层只调下面这几个入口 ----

    def quota_ranking(self, capacity: float, page_size: int) -> list:
        """返回本轮点名的队列优先级（空列表 = 配额没有意见）。

        Args:
            capacity: 池容量（token）。
            page_size: 页大小，用于推导 chunk。

        Returns:
            队列名列表，调用方按序尝试，第一个有候选的即为 Q*。
        """
        if not self.quota.enabled:
            return []
        targets = effective_targets(self._occ, capacity, self.quota)
        chunk = resolve_chunk(page_size, self.quota)
        return queue_ranking(
            self._occ, targets, self.quota, chunk=float(chunk), capacity=capacity
        )

    def quota_targets(self, capacity: float) -> dict:
        """各队列的有效目标（token），仅用于观测；配额未启用时返回空字典。"""
        if not self.quota.enabled:
            return {}
        return effective_targets(self._occ, capacity, self.quota)

    def quota_victim(self, queue: str, leaves, capacity: float):
        """在指定队列内挑一个受害者；该队列没有候选时返回 None。

        返回 None 是给调用方的“跳队”信号，而不是错误：点名队列可能全是
        locked 叶子、全是 one-shot、或标签已全集中到别的队列。

        Args:
            queue: 队列名（struct / agentic / chat）。
            leaves: 当前可淘汰叶子集合。
            capacity: 池容量（token），用于把配额缺口传给 RR。

        Returns:
            选中的节点，或 None。
        """
        pool = [n for n in leaves if self.workload_of(n) == queue]
        if not pool:
            return None
        if queue == "struct":
            # 两遍降级：先牺牲“刚好够保护”的模板，再动高复用模板，同档内 LRU。
            share_of = self.slru_evict.share_of
            share_min = self.slru_evict.share_min
            return min(
                pool, key=lambda n: struct_tier(n, share_of, share_min)
            )
        if queue == "agentic" and self.rr_evict is not None:
            # 配额缺口作为 RR loss_cap 的上界：超配额时单次不得把一条会话削穿。
            picked = self.rr_evict.pick(
                pool, self.session_registry, max_deficit=self._quota_gap("agentic", capacity)
            )
            if picked is not None:
                return picked
        if queue == "chat" and self.mt_evict is not None:
            picked = self.mt_evict.pick(pool, self.session_registry)
            if picked is not None:
                return picked
        # 队内策略弃权时退化到全局分数，保证队列内总能推进。
        return min(pool, key=self.victim_key)

    def _quota_gap(self, queue: str, capacity: float):
        """该队列超出有效目标多少 token；未超出时返回 None。

        None 的语义是“配额不管这件事”：弹性区内不该限制队内策略的自由度，
        否则一个刚好压线的队列会被无谓地束缚。
        """
        if not self.quota.enabled or capacity <= 0:
            return None
        target = effective_targets(self._occ, capacity, self.quota)[queue]
        gap = self._occ[queue] - target
        return gap if gap > 0 else None

    def quota_fallback(self, leaves):
        """兜底：用策略私有 RNG 均匀抽一个可淘汰叶子。

        仅在配额点名的队列都拿不出候选时用。用 ``random.Random(seed)`` 而不是
        模块级 ``random``：同 trace + 同 seed 可复现（实验要可归因），也不会
        污染进程内其他地方的随机序列。

        Args:
            leaves: 当前可淘汰叶子集合。

        Returns:
            抽中的节点；集合为空时返回 None。
        """
        pool = list(leaves)
        if not pool:
            return None
        return self._quota_rng.choice(pool)
