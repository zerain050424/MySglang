from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from sglang.srt.mem_cache.uni_utils import (
    TASK_CONFIG,
    WORKLOAD_LABELS,
    workload_label_of,
    SimpleHybridTaskType,
    _cached_lognorm_cdf,
    alpha_key_of,
    classify_simple_hybrid_task_type,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)


def _node_num_tokens(node) -> int:
    """节点片段的 token 数（radix 片段的长度）。"""
    return len(node.value) if node.value is not None else 0


def _node_end_offset(node) -> int:
    """片段的绝对结束偏移"""
    num_tokens = _node_num_tokens(node)
    abs_start = getattr(node, "abs_start", None)
    if abs_start is None:
        return num_tokens
    return abs_start + num_tokens


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
        alpha_balance: Optional[Dict[str, float]] = None,
        update_interval: float = 30.0,
        min_cap_share: float = 0.02,
        max_alpha_step: float = 3.0,
        alpha_smooth: float = 0.7,
        alpha_temperature: float = 0.5,
    ):
        self.max_offset = max_offset or 2048
        # ---- alpha 控制器的稳定性参数----
        # 加下限使 eff 有界。
        self.min_cap_share = min_cap_share
        # 单次更新允许的最大倍数变化（上/下）。防止一个噪声窗口把权重从下界直接顶到上界。
        self.max_alpha_step = max_alpha_step
        # EMA 中旧值的保留权重。0.7 表示每次只吸收 30% 的新观测；
        self.alpha_smooth = alpha_smooth
        # 做 (1/temperature) 次幂放大。
        self.alpha_temperature = alpha_temperature
        self.alpha_balance = ( # 直接赋值是引用共享,外部改原来的字典,内部跟着变,传入dict进行构造,可以实现浅拷贝的内外隔离
            dict(alpha_balance)
            if alpha_balance
            else {
                "single": 1.0,
                "text": 1.0,
                "file": 1.0,
                "image": 1.0,
                "search": 1.0,
                "agentic": 1.0,
            }
        )
        # 在线更新α状态
        self.update_interval = update_interval
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
    @staticmethod
    def _task_type_of(node) -> SimpleHybridTaskType:
         return getattr(node, "task_type", None) or SimpleHybridTaskType.SINGLE_TURN

    def _reuse_score(self, node) -> float:
        """calculate_priority() from UniCache, with time.monotonic() as now."""
        task_type = self._task_type_of(node)
        num_tokens = len(node.value) if node.value is not None else 0
        elapsed = time.monotonic() - node.last_access_time
        e = int(elapsed)
        cfg = TASK_CONFIG.get(task_type)
        if cfg is not None: ## 多轮任务
            mu, sigma, alpha_key = cfg
            alpha = self.alpha_balance.get(alpha_key, 1.0) # 在线更新α
            reuse_prob = 1.0 - _cached_lognorm_cdf(e, mu, sigma)
            return alpha * reuse_prob
        if task_type == SimpleHybridTaskType.SINGLE_TURN:
            alpha = self.alpha_balance.get("single", 1.0)
            # UniCache 的 offset 是块在序列中的绝对结束位置，不是 radix 片段长度。
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
        """
            统一淘汰键：复用分数低者先走，同分时 token 多者先走。
        """
        return (self._reuse_score(node), -_node_num_tokens(node))

    def is_one_shot(self, node) -> bool:
        """
        一次性内容（one-shot）：淘汰顺序永远先于可调度内容。
          - SINGLE_TURN_RANDOM：随机单轮，跨会话几乎不复用；
          - 单轮负载（SINGLE_TURN）的生成段：会话发完即弃，response 不会再被任何请求当作前缀。
        多轮负载的 response 段不在此列：下一轮会复用，与其 prompt 段同队列参与调度。
        """
        task_type = self._task_type_of(node)
        if task_type == SimpleHybridTaskType.SINGLE_TURN_RANDOM:
            return True
        return bool(getattr(node, "is_decode", False)) and (
            task_type == SimpleHybridTaskType.SINGLE_TURN
        )

    def workload_of(self, node) -> str:
        """节点所属的工作负载。"""
        return workload_label_of(self._task_type_of(node))

    def one_shot_key(self, node) -> tuple:
        """one-shot 带内排序：树上结束偏移最深的块先淘汰。

        口径对齐 UniCache 的 evict_first 堆键（-num_hashed_tokens，
        即块的绝对结束位置）：序列里越靠后的生成段越先走。"""
        return (-_node_end_offset(node), node.last_access_time)

    # ---- online alpha balancing ----

    def record_hit(self, req, hit_tokens: int) -> None:
        # Attribute a request's prefix-cache hit to its workload type, once
        # per request; decay other types to keep the window recent.
        rid = getattr(req, "rid", None)
        if rid is None or rid in self._seen_rids:
            return
        self._seen_rids.add(rid)
        self._rid_order.append(rid)
        if len(self._rid_order) > self.MAX_SEEN_RIDS:
            self._seen_rids.discard(self._rid_order.popleft())

        task_type = classify_simple_hybrid_task_type(
            getattr(req, "session_id", None), getattr(req, "extra_key", None)
        )
        key = alpha_key_of(task_type)
        for q in self._hit_tokens:
            if q != key:
                self._hit_tokens[q] *= self.HIT_DECAY ## 命中期间其他队列的窗口衰减
        self._hit_tokens[key] = self._hit_tokens.get(key, 0.0) + hit_tokens
        # 累计命中。
        # 用 workload 标签记账（SINGLE_TURN_RANDOM 单列 "random"），
        # 与上面喂给 alpha 的 alpha_key 是两个口径。
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

    def profile_summary(self, workload_occ, session_summary=None) -> str:
        """返回各 workload 的 profile 摘要，供日志打印。

        字段含义：
          hit / evicted : 累计命中 / 淘汰 token，按 workload（与 occ 同口径，可比较）
          occ           : 该 workload 的全部驻留 token（含 response 段）；
                          `occ(-N)` 后缀 = 其中 one-shot 部分（random 整行、
                          single 的 response 段），alpha 调度不了
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
            alpha = self.alpha_balance.get(q)
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
        if now - self._last_update < self.update_interval:
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
        1. 容量份额下限：分母趋零时 eff 爆炸。single 被压到 786/100000
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
            smooth = self.alpha_smooth
        if temperature is None:
            temperature = self.alpha_temperature

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
            q: max(c / total_cap, self.min_cap_share) for q, c in caps.items()
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
            prev = self.alpha_balance.get(q, 1.0)
            updated = smooth * prev + (1 - smooth) * val          # EMA
            # 约束 2：单步限速，避免从下界一步跳到上界
            lo, hi = prev / self.max_alpha_step, prev * self.max_alpha_step
            updated = max(lo, min(hi, updated))
            self.alpha_balance[q] = max(lower, min(upper, updated))
    