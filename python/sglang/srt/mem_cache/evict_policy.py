from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from sglang.srt.mem_cache.struct_queue_policy import StructQueueState
from sglang.srt.mem_cache.uni_utils import (
    AlphaParams,
    node_end_offset,
    node_num_tokens,
    RrParams,
    TinyLRUParams,
    TlruParams,
    TASK_CONFIG,
    WORKLOAD_LABELS,
    workload_label_of,
    alpha_key_of,
    SimpleHybridTaskType,
    _cached_lognorm_cdf,
    task_type_of_tag,
)
from sglang.srt.mem_cache.tag_lru import TagLruPolicy
from sglang.srt.mem_cache.victim_override import (
    RrEvictPolicy,
    StructTinyLruPolicy,
    TlruEvictPolicy,
)

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
    # 衰减
    HIT_DECAY = 0.95
    # Cap on the once-per-request dedup set (bounds memory in long runs).
    MAX_SEEN_RIDS = 100_000

    def __init__(
        self,
        max_offset: int = 2048,
        *,
        force_agentic: bool = False,
        alpha: Union[AlphaParams, dict, None] = None,
        rr: Union[RrParams, dict, None] = None,
        tiny_lru: Union[TinyLRUParams, dict, None] = None,
        tlru: Union[TlruParams, dict, None] = None,
    ):
        """
        三组参数分别由 AlphaParams / RrParams / TinyLRUParams 承载（uni_utils）。
        配置只接受分组形式：
            {"alpha": {...}, "rr": {...}, "tiny_lru": {...}}
        未知键在参数组构造处直接报错（拼错不再静默失效）。

        max_offset 不属于任何一组：它是 Q_struct 复用分数里的位置项上界
        （HeterCache 的 offset 语义），与 TinyLRU 的频数准入无关。

        force_agentic 是实验开关（不是调参）：打开后所有节点都按 agentic 语义
        解释（评分、one-shot、插件作用域一致），并把 queue_tag 直接标成 QAGENT。
        用途是让 QAGENT-only 的插件拿到 100% 流量；它不修改 session_type，因此
        分类器自身的统计仍然可测。
        """
        self.alpha = AlphaParams.from_any(alpha)
        self.rr = RrParams.from_any(rr)
        self.tiny_lru = TinyLRUParams.from_any(tiny_lru)
        self.tlru = TlruParams.from_any(tlru)
        self.force_agentic = bool(force_agentic)
        self.max_offset = max_offset or 2048

        # 在线更新α状态
        self._last_update = time.monotonic()
        self._hit_tokens: Dict[str, float] = {}
        self._seen_rids: set = set()
        self._rid_order: deque = deque()
        # 累计计数器（不清空，供 profile 使用）
        #   - self._hit_tokens : 策略用的窗口命中量，键是alpha_key
        #   - self._total_hit_tokens / _total_evicted_tokens : profile用的累计量，键是 workload 标签（"random" 单列），与alpha无关。
        #   - self._total_evicted_one_shot : 一次性内容（random + 单轮 response 段）
        #     的累计淘汰量；单轮 response 段口径 = 该标量 − evicted["random"]。
        self._total_hit_tokens: Dict[str, float] = {}
        self._total_evicted_tokens: Dict[str, float] = {}
        self._total_evicted_one_shot = 0.0
        # 淘汰来源记账：one_shot 带 / 全局 victim_key / 各插件（queue 名），
        # 供 profile 打印本次淘汰由哪个队列的策略决定。
        self._victim_sources: Dict[str, float] = {}


        #   self.structState    = 状态机（缓存层持有：8 个钩子 + owns 死链闸）
        #   self.TinyLru_evict  = 淘汰面适配器（只进 victim_overrides）
        # 若让 self.structState 直接指向适配器，radix_cache 的 8 处钩子调用会静默
        # 变成 no-op（getattr 判空不报错），struct 策略会整体失效而日志无异常。
        self.rr_evict = RrEvictPolicy(self.rr) if self.rr.fair_rr else None
        self.structState = (
            StructQueueState(
                enabled=True,
                k=self.tiny_lru.k,
                window_entries=self.tiny_lru.window_entries,
                epoch_events=self.tiny_lru.epoch_events,
                admission_tie=self.tiny_lru.admission_tie,
                admission_metric=self.tiny_lru.admission_metric,
                inherit_share=self.tiny_lru.inherit_share,
                share_metric=self.tiny_lru.share_metric,
                protected_budget_frac=self.tiny_lru.protected_budget_frac,
                stale_grace_s=self.tiny_lru.stale_grace_s,
            )
            if self.tiny_lru.policy == "wtinylru"
            else None
        )
        self.TinyLru_evict = (
            StructTinyLruPolicy(self.structState)
            if self.structState is not None
            else None
        )
        # 标签版（TinyLRU 分支默认）：不建三区，只按「会话类型 + 会话共享数」贴标签。
        # 与三区版二选一；两者都关时是彻底的零开销路径。
        self.tagState = (
            TagLruPolicy(
                struct_share_min=self.tiny_lru.struct_share_min,
                share_metric=(
                    self.tiny_lru.share_metric
                    if self.tiny_lru.share_metric in ("session", "rid")
                    else "session"
                ),
                local_hot=self.tiny_lru.local_hot,
            )
            if self.tiny_lru.policy == "tag_lru"
            else None
        )
        # 按标签记账：标签版没有插件，`victim_src` 里不会出现 qstruct，
        # 这里单独统计「各标签被淘汰了多少 token」，profile 与实验套件都用它。
        self._evicted_by_tag: Dict[str, float] = {}
        # T-LRU 尾巴优先：只修 QAGENT 的候选，并在插入期维护 _tlru_* 高水位。
        # 与 struct 分属不同队列可共存；与 fair_rr 互斥（两者都修 QAGENT，
        # 链上后注册者会把前者完全遮蔽），由 _validate_scope() 拦住。
        self.tlru_evict = TlruEvictPolicy(self.tlru) if self.tlru.enabled else None
        self._validate_scope()
        self.victim_overrides = tuple(
            p
            for p in (self.rr_evict, self.TinyLru_evict, self.tlru_evict)
            if p is not None
        )

    def reset_bookkeeping(self) -> None:
        """清空窗口/累计/来源计数（缓存 flush 与 reset 同源）。

        不复位参数，也不动 `_last_update`：它只影响计时节流。
        跨 run 复用 server 时，若不清理，`[SimpleHybrid profile]` 打印的是
        上一轮与这一轮的混合值。
        """
        self._hit_tokens.clear()
        self._total_hit_tokens.clear()
        self._total_evicted_tokens.clear()
        self._total_evicted_one_shot = 0.0
        self._victim_sources.clear()
        self._seen_rids.clear()
        self._rid_order.clear()

    def owns_struct(self, node) -> bool:
        """该节点是否受「结构复用」保护（死链上溯闸用，两种模式统一入口）。"""
        if self.structState is not None:
            return bool(self.structState.owns(node))
        if self.tagState is not None:
            return bool(self.tagState.owns(node))
        return False

    def tag_snapshot(self) -> Optional[dict]:
        return self.tagState.snapshot() if self.tagState is not None else None

    def _validate_scope(self) -> None:
        """拦住会静默失效的插件组合。

        `victim_overrides` 是覆盖链（后者覆盖前者），而 RR 只要挑得出会话就会
        返回非 None，所以 fair_rr + tlru 同时开会把 TLRU 完全遮蔽。
        """
        if self.tagState is not None and self.tlru_evict is not None:
            # 标签版把 QSTRUCT 的 LRU 顺序放进全局 victim_key，而 T-LRU 改写的是
            # QAGENT 候选；两者同时开会互相遮蔽，直接拦住而不是静默失效。
            raise ValueError(
                "tiny_lru.policy='tag_lru' and tlru.enabled are mutually exclusive: "
                "tag_lru orders QSTRUCT via victim_key, T-LRU rewrites QAGENT "
                "candidates, and the latter silently wins."
            )
        if self.rr_evict is not None and self.tlru_evict is not None:
            raise ValueError(
                "rr.fair_rr and tlru.enabled are mutually exclusive: both rewrite "
                "the QAGENT candidate and the later override wins, so T-LRU would "
                "never be consulted. Enable only one of them."
            )

    def _task_type_of(self, node) -> SimpleHybridTaskType:
        # 队列标签由缓存状态机维护。映射表在 uni_utils，策略与改写器共用。
        # force_agentic 时短路：即使某个节点残留 QTRANS/QCHAT/QSTRUCT 标签，
        # 评分与 one-shot 判定也统一按 agentic 走（标签由缓存层同步强制为 QAGENT）。
        if self.force_agentic:
            return SimpleHybridTaskType.MULTI_TURN_AGENTIC
        return task_type_of_tag(getattr(node, "queue_tag", None))

    def _reuse_score(self, node) -> float:
        """calculate_priority() from HeterCache, with time.monotonic() as now."""
        task_type = self._task_type_of(node)
        num_tokens = len(node.value) if node.value is not None else 0
        elapsed = time.monotonic() - node.last_access_time
        e = int(elapsed)
        cfg = TASK_CONFIG.get(task_type)
        if cfg is not None: ## 多轮任务
            mu, sigma, alpha_key = cfg
            alpha = self.alpha.balance.get(alpha_key, 1.0) # 在线更新α
            reuse_prob = 1.0 - _cached_lognorm_cdf(e, mu, sigma)
            return alpha * reuse_prob
        if task_type == SimpleHybridTaskType.STRUCT_SINGLE_TURN:
            alpha = self.alpha.balance.get("struct", 1.0)
            # HeterCache 的 offset 是块在序列中的绝对结束位置，不是 radix 片段长度。
            offset_factor = 1.0 - _node_end_offset(node) / (self.max_offset or 1)
            offset_factor = max(0.0, min(1.0, offset_factor))
            #  单轮分数 = alpha * 位置项。
            return alpha * offset_factor
        return 0.0


    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        """
        全序优先级：带 0 = 一次性内容（random / 单轮 response 段）先淘汰，
        带内按树上结束偏移降序（与 one_shot_key 同口径）；带 1 = 可调度内容
        按 reuse_score 竞争。
        """
        if self.is_one_shot(node):
            return (0, float(-_node_end_offset(node)), 0.0)
        return (1, self._reuse_score(node), float(-_node_num_tokens(node)))

    # 暴露给淘汰器的键语义

    def victim_key(self, node) -> tuple:
        """统一淘汰键。

        三区版：``(reuse_score, -tokens)``，队内顺序由 struct 插件改写。
        标签版（tag_lru）：把「类间优先级 + 类内顺序」直接编码进键——

        * rank 0 = QTRANS / 一次性内容：最先走（区内 LRU）
        * rank 1 = QCHAT / QAGENT：仍按复用概率曲线（HeterCache 的 reuse_prob）
        * rank 2 = **QSTRUCT（跨会话共享前缀）：最受保护，类内 LRU**

        注意 rank 2 必须显式给 LRU，不能用 `_reuse_score` —— 后者对 STRUCT 用的是
        `1 - end_offset/max_offset` 的位置打分，长度 ≥ max_offset 的前缀会被判 0 分、
        反而最先淘汰。
        """
        tokens = _node_num_tokens(node)
        if self.tagState is not None:
            task = self._task_type_of(node)
            if task == SimpleHybridTaskType.STRUCT_SINGLE_TURN:
                return (2, node.last_access_time, -tokens)
            if task == SimpleHybridTaskType.RANDOM_SINGLE_TURN:
                return (0, node.last_access_time, -tokens)
            return (1, self._reuse_score(node), -tokens)
        return (self._reuse_score(node), -tokens)

    def is_one_shot(self, node) -> bool:
        """
        一次性内容（one-shot）：淘汰顺序永远先于可调度内容。
          - RANDOM_SINGLE_TURN：随机单轮，跨会话几乎不复用；
          - STRUCT_SINGLE_TURN 的生成段：会话发完即弃，response 不会再被任何请求当作前缀。
        多轮负载的 response 段不在此列：下一轮会复用，与其 prompt 段同队列参与调度。
        """
        task_type = self._task_type_of(node)
        if task_type == SimpleHybridTaskType.RANDOM_SINGLE_TURN:
            return True
        return node.is_decode and (
            task_type == SimpleHybridTaskType.STRUCT_SINGLE_TURN
        )

    def workload_of(self, node) -> str:
        """节点所属的工作负载。"""
        return workload_label_of(self._task_type_of(node))

    def one_shot_key(self, node) -> tuple:
        """one-shot 带内排序：树上结束偏移最深的块先淘汰。

        口径对齐 HeterCache 的 evict_first 堆键（-num_hashed_tokens，
        即块的绝对结束位置）：序列里越靠后的生成段越先走。"""
        return (-_node_end_offset(node), node.last_access_time)

    # ---- online alpha balancing ----

    def record_hit(self, req, node, hit_tokens: int) -> None:
        # Attribute a request's prefix-cache hit to its workload type, once
        # per request; decay other types to keep the window recent.
        rid = getattr(req, "rid", None)
        if rid is None or rid in self._seen_rids:
            return
        self._seen_rids.add(rid)
        self._rid_order.append(rid)
        if len(self._rid_order) > self.MAX_SEEN_RIDS:
            self._seen_rids.discard(self._rid_order.popleft())

        task_type = self._task_type_of(node)
        key = alpha_key_of(task_type)
        for q in self._hit_tokens:
            if q != key:
                self._hit_tokens[q] *= self.HIT_DECAY ## 命中期间其他队列的窗口衰减
        self._hit_tokens[key] = self._hit_tokens.get(key, 0.0) + hit_tokens
        # 累计命中。
        wkey = workload_label_of(task_type)
        self._total_hit_tokens[wkey] = self._total_hit_tokens.get(wkey, 0.0) + hit_tokens

    def record_evict(self, node) -> None:
        """累计被淘汰的 token 数，按 workload 记账（与 hit 同口径，可比较）。
        one-shot 淘汰量单记一个标量；random 的淘汰量直接读 workload 行，
        单轮 response 段的淘汰量 = 标量 − evicted["random"]。"""
        tokens = _node_num_tokens(node)
        wkey = self.workload_of(node)
        self._total_evicted_tokens[wkey] = (
            self._total_evicted_tokens.get(wkey, 0.0) + tokens
        )
        if self.is_one_shot(node):
            self._total_evicted_one_shot += tokens
        if self.tagState is not None:
            tag = getattr(node, "queue_tag", None)
            key = f"tag:{getattr(tag, 'value', tag)}"
            self._evicted_by_tag[key] = self._evicted_by_tag.get(key, 0.0) + tokens

    def record_victim_source(self, source: str, tokens: float) -> None:
        """
        记录本次淘汰由谁决定，按来源累计 token 数。
        source 取值："one_shot"（硬优先带）/ "global"（全局 victim_key 竞争）/
        插件的 queue 名（如 "qagent"、"qstruct"）。只做观测，不参与仲裁。
        """
        self._victim_sources[source] = self._victim_sources.get(source, 0.0) + tokens

    def profile_summary(self, workload_occ, session_summary=None) -> str:
        """返回各 workload 的 profile 摘要，供日志打印。

        字段含义：
          hit / evicted : 累计命中 / 淘汰 token，按 workload（与 occ 同口径，可比较）
          occ           : 该 workload 的全部驻留 token（含 response 段）；
                          `occ(-N)` 后缀 = 其中 one-shot 部分（random 整行、
                          struct 的 response 段），alpha 调度不了
          alpha         : 在线权重（random 不参与 alpha，恒为 "-"）

        Σocc == schedulable + Σone_shot（schedulable 即 alpha 分母）。
        多轮负载的 one_shot 恒为 0，不打印后缀；random 整行 one-shot。
        """
        lines = ["  workload           hit     evicted             occ   alpha"]
        pool = 0
        schedulable = 0
        for q in WORKLOAD_LABELS:
            c = workload_occ.get(q)
            if not c:
                continue
            pool += c["total"]
            schedulable += c["total"] - c["one_shot"]
            hit = self._total_hit_tokens.get(q, 0.0)
            evicted = self._total_evicted_tokens.get(q, 0.0)
            alpha = self.alpha.balance.get(q)
            alpha_s = f"{alpha:>6.3f}" if alpha is not None else "     -"
            occ_s = f"{c['total']:>10.0f}"
            if c["one_shot"]:
                occ_s += f"(-{c['one_shot']:.0f})"
            lines.append(
                f"  {q:8s} {hit:>12.0f} {evicted:>12.0f} {occ_s:>16s} {alpha_s}"
            )
        lines.append(
            f"  pool={pool:.0f} schedulable={schedulable:.0f} "
            f"one_shot_evicted={self._total_evicted_one_shot:.0f}"
        )
        if self._evicted_by_tag:
            by_tag = " ".join(
                f"{k}={v:.0f}" for k, v in sorted(self._evicted_by_tag.items())
            )
            lines.append(f"  evicted_by_tag: {by_tag}")
        if self.tagState is not None:
            s = self.tagState.snapshot()
            lines.append(
                f"  tag_lru: min_share={s['struct_share_min']} "
                f"promote={s['promote']} dedup={s['dedup']} counted={s['counted']}"
            )
        if self._victim_sources:
            src = " ".join(
                f"{k}={v:.0f}" for k, v in self._victim_sources.items()
            )
            lines.append(f"  victim_src: {src}")
        if self.tlru_evict is not None:
            # phase1/phase2 的混合比：tel_safe 为 0 说明尾部预算没起作用。
            s = self.tlru_evict.stats()
            lines.append(
                f"  tlru: tel_safe={s['tlru_tel_safe_picks']:.0f} "
                f"lru={s['tlru_lru_picks']:.0f} "
                f"fallback={s['tlru_fallbacks']:.0f} "
                f"hist_raises={s['tlru_history_raises']:.0f}"
            )
        if session_summary:
            # 会话级缺口：d_sum 与淘汰总量同量纲，可互相校验
            lines.append(
                f"  sessions={int(session_summary['sessions'])} "
                f"d_sum={session_summary['d_sum']:.0f} "
                f"d_p50={session_summary['d_p50']:.0f} "
                f"d_max={session_summary['d_max']:.0f} "
                f"d_nonzero={int(session_summary['d_nonzero'])}"
            )
        return "\n".join(lines)

    def maybe_update_alpha(self, sched, workload_occ, session_summary=None) -> None:
        # Time-guarded trigger for alpha rebalancing (called from evict()).
        now = time.monotonic()
        if now - self._last_update < self.alpha.update_interval:
            return ## 每经过一个时间间隔interval才进行更新
        self._last_update = now
        self.update_alpha_balance(sched)
        # 每个 update_interval 秒打印一次各 workload profile 摘要
        logger.info(
            "[SimpleHybrid profile]\n%s",
            self.profile_summary(workload_occ, session_summary),
        )

    def update_alpha_balance(
        self,
        sched,
        smooth: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """按「单位容量的命中产出」重算各队列 alpha（带稳定性约束）。
        效率定义
        --------
            eff(q) = 命中占比(q) / max(容量占比(q), min_cap_share)
                   = (hits_q/Σhits) / max(cap_q/Σcap, min_cap_share)

        无量纲、量级 O(1)：>1 表示这一级用给定容量换到了超额命中。

        为什么需要三个稳定性约束（实测 62 分钟的教训）
        -------------------------------------------------
        1. 容量份额下限：分母趋零时 eff 爆炸。struct 被压到 786/100000
           (0.8%) 时，一个窗口里少量命中就让 eff 冲高 -> r**5 顶到上界 10；
           下一窗口它没有容量自然没命中 -> EMA 逐窗 ×0.1 衰减回 0.001。
           实测形成 ~2.5 分钟周期的锯齿，alpha 跨度 10000×。
        2. 单步限速：即使信号极端，单次更新也不超过 max_alpha_step 倍，
           避免 Bang-Bang 式跳变。
        3. 升温 + 加重平滑：temperature 0.2 -> 0.5（5 次幂改平方），
           smooth 0.1 -> 0.7（90% 新值改为 70% 保留旧值）。

        参与规则
        --------
        只让「当前占着容量」的队列参与配额分配；但**不按命中过滤** ——
        占着容量却零命中的队列必须参与（eff=0），否则它永远拿不到惩罚。
        """
        if smooth is None:
            smooth = self.alpha.smooth
        if temperature is None:
            temperature = self.alpha.temperature

        hits = dict(self._hit_tokens)
        self._hit_tokens.clear()
        if not hits:
            return

        caps = {q: c for q, c in sched.items() if c > 0}
        if not caps:
            return

        total_hits = sum(hits.get(q, 0.0) for q in caps)
        if total_hits <= 0:
            return  # 本窗口没有任何命中，无反馈信号，保持现状
        total_cap = sum(caps.values())
        if total_cap <= 0:
            return

        # 容量份额加下限：使 eff 在 cap_share→0 时有界（约束 1）
        cap_share = {
            q: max(c / total_cap, self.alpha.min_cap_share) for q, c in caps.items()
        }
        eff = {q: (hits.get(q, 0.0) / total_hits) / cap_share[q] for q in caps}

        keys = list(eff.keys())
        vals = [eff[k] for k in keys]
        mean_eff = sum(vals) / len(vals)
        if mean_eff <= 0:
            return

        # 温度缩放 + 2σ 截断（抑制离群）
        scaled = [(v / mean_eff) ** (1.0 / temperature) for v in vals]
        mean_s = sum(scaled) / len(scaled)
        std_s = (sum((x - mean_s) ** 2 for x in scaled) / len(scaled)) ** 0.5
        lower = max(0.001, mean_s - 2 * std_s)
        # upper 必须 >= lower，否则 max(lower, min(upper, x)) 会退化成常数
        upper = max(lower, min(10.0, mean_s + 2 * std_s))

        for q, val in zip(keys, scaled):
            prev = self.alpha.balance.get(q, 1.0)
            updated = smooth * prev + (1 - smooth) * val          # EMA
            # 约束 2：单步限速，避免从下界一步跳到上界
            lo, hi = prev / self.alpha.max_alpha_step, prev * self.alpha.max_alpha_step
            updated = max(lo, min(hi, updated))
            self.alpha.balance[q] = max(lower, min(upper, updated))
    