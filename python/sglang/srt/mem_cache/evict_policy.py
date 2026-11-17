from __future__ import annotations

from abc import ABC, abstractmethod
import bisect
import dataclasses
import json
import logging
import math
import random
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union
from sglang.srt.mem_cache.heter_cache.selector import queue_order as _queue_order
from sglang.srt.mem_cache.heter_cache.selector import struct_tier
from sglang.srt.mem_cache.heter_cache.utils import (
    node_num_tokens,
    ONE_SHOT_IDLE_S,
    REUSE_QUEUES,
    ReuseParams,
    SessionRegistry,
    SessionType,
    ShareParams,
    QueueValueParams,
    HeterCacheNodeTag,
)
from sglang.srt.mem_cache.heter_cache.occupancy import QueueOccupancyTracker
from sglang.srt.mem_cache.heter_cache.value import QueueValueTracker
from sglang.srt.mem_cache.heter_cache.node_share import NodeShareLedger
from sglang.srt.mem_cache.heter_cache.override import ReuseScoreEvict
from sglang.srt.mem_cache.heter_cache.bench_bridge import (
    CacheTelemetry,
    HeterProfiler,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)



# 节点口径统一在 heter_cache.utils，避免策略/改写器/指标各算一套。
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


class ChainLRUStrategy(EvictionStrategy):
    """LRU + 链形修正：浅段优先保护、砍链尾（victim 按下游依赖体量选）。

    动机（tail-stress 诊断）：match 刷新的是整条匹配路径的 recency，一条私有
    单子链要么全新要么全旧。链一旦成为全局最旧，``evict`` 里「叶子出堆 → 父
    变新叶子再出堆」会一路剥穿——整链归零。归零后会话下轮 hit=0、整链重插、
    再整链变旧被剥，进入死循环（tail-stress 上 warm 整链丢失 52%，P90 是 OPT
    上界的 5 倍）。

    两个修正只读树形与上游已有计数，不引新账本、不改 evict 循环：

    * 浅段保护（``floor_tokens``）：淘汰叶子后，链上剩余前缀（root→parent 的
      token 数）若低于下限，且这条链有重读证据，就把 ``weight × 缺口`` 秒加进
      LRU 键——别的受害者必须比它还旧 ``weight × 缺口`` 秒才轮得到它。剥得越
      深罚得越重，链在重读锚附近自然刹住；重读证据只认 ``reuse_count``
      （admission match 计数），``hit_count`` 含写路径自增，会把刚写的段误判
      成「被重读过」。
    * 砍链尾（``tail_first``）：平票时先剥 ``remain`` 大的叶子——长链砍尾只损失
      尾巴，短链砍根损失整链。
    """

    _GATES = ("none", "rerode")

    def __init__(
        self,
        floor_tokens: int = 4096,
        weight: float = 0.01,
        gate: str = "rerode",
        tail_first: bool = True,
    ):
        if floor_tokens < 0:
            raise ValueError(f"floor_tokens must be >= 0, got {floor_tokens}")
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        if gate not in self._GATES:
            raise ValueError(f"gate must be one of {self._GATES}, got {gate!r}")
        self.floor_tokens = int(floor_tokens)
        self.weight = float(weight)
        self.gate = gate
        self.tail_first = bool(tail_first)

    def get_priority(self, node: TreeNode) -> Tuple[float, float, int]:
        lat = float(node.last_access_time)
        # remain = 淘汰该叶子后链上仍连续缓存的前缀（root→parent 的 token 数）。
        # 顺路收集重读证据：reuse_count 只在 admission match 里自增（回扫不算）。
        remain = 0
        rerode = getattr(node, "reuse_count", 0) > 0
        cursor = node.parent
        while cursor is not None and cursor.parent is not None:
            remain += len(cursor.key)
            rerode = rerode or getattr(cursor, "reuse_count", 0) > 0
            cursor = cursor.parent
        penalty = 0.0
        if self.gate == "rerode" and not rerode:
            penalty = 0.0
        elif self.floor_tokens > 0 and remain < self.floor_tokens:
            penalty = self.weight * (self.floor_tokens - remain)
        tie = -float(remain) if self.tail_first else 0.0
        return (lat + penalty, tie, int(getattr(node, "id", 0)))


class HeterCacheStrategy(EvictionStrategy):
    """HeterCache 多队列缓存淘汰策略。

    两层决策：
    1. 分带过滤：One-shot 节点优先清理，不计入 VD 账本；
       归属会话仍活跃的未匹配节点受时间豁免（ONE_SHOT_IDLE_S）保护，不进带。
    2. VD 仲裁：按 VD 升序排出候选队列（只读密度，不含占用），再由队内策略决出牺牲节点。
       chat / agentic 共用一套队内口径：按复用概率升序（``ReuseScoreEvict``），
       只有 mu/sigma 随标签不同。
    """

    def __init__(
        self,
        *,
        force_agentic: bool = False,
        share: Union[ShareParams, dict, None] = None,
        reuse: Union[ReuseParams, dict, None] = None,
        value: Union[QueueValueParams, dict, None] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        # 时间源可注入（单测用假时钟控制 one-shot 豁免窗口的推进）。
        self._now = now if now is not None else time.monotonic
        self.share = ShareParams.from_any(share)
        self.reuse = ReuseParams.from_any(reuse)
        self.value = QueueValueParams.from_any(value)
        self.force_agentic = bool(force_agentic)

        self.occupancy = QueueOccupancyTracker()
        self.realized = QueueValueTracker(
            horizon_s=self.value.horizon_s,
            smoothing_k=self.value.smoothing_k,
            floor_tokens=self.value.floor_tokens,
            now=self._now,
        )

        self._fallback_rng = random.Random(self.value.seed)
        self.profiler = HeterProfiler()
        self.telemetry = CacheTelemetry(self.profiler)

        self.session_registry = SessionRegistry.global_instance()
        self.node_share = NodeShareLedger(self.share.share_min)
        # chat / agentic 同一套复用分排序，各持一个实例只是为了分开计数与分队列筛池。
        self.reuse_evicts = (
            {
                q: ReuseScoreEvict(
                    self.reuse, HeterCacheNodeTag.from_queue(q), self._node_tag
                )
                for q in REUSE_QUEUES
            }
            if self.reuse.enabled
            else {}
        )

    def use_clock(self, now: Callable[[], float]) -> None:
        """接管缓存层的时间源
        Args:
            now: 返回秒数的单调时钟。
        """
        self._now = now
        self.occupancy.use_clock(now)
        self.telemetry.use_clock(now)
        self.realized.use_clock(now)
        self.session_registry.use_clock(now)
        for policy in self.reuse_evicts.values():
            policy.use_clock(now)

    def _node_tag(self, node) -> HeterCacheNodeTag:
        if self.force_agentic:
            return HeterCacheNodeTag.QAGENT
        return HeterCacheNodeTag.coerce(getattr(node, "queue_tag", None))

    def node_tag(self, node) -> HeterCacheNodeTag:
        """节点生效标签（含 force_agentic 归并）。观测层读这个口径，
        不要自己 coerce ``queue_tag``——那样 force_agentic 下回报的队列是假的。"""
        return self._node_tag(node)

    def get_priority(self, node: TreeNode) -> Tuple[int, float, float]:
        raise NotImplementedError(
            "HeterCacheStrategy has no single-node ordering; "
            "use queue_weight arbitration (heter_cache.selector.pick_victim) instead"
        )

    def is_aged(self, node) -> bool:
        """节点是否已老化。

        两种口径（``share.ageing_mode``）：

        * ``queue_ttl``：节点 ``TTL`` 秒未被访问即老化（历史口径）。
        * ``session_life``：只看**该会话自己的轮间隔分布**，超过 ``k × p90`` 才算
          会话终结；终结后的**私有**内容（跨会话共享数 < share_min）才可回收。
          这样"会话还活着、只是这轮间隔长"不会被误判成永不复用——drift3 里
          chat 的前缀有 59.5% 的 regret 就是被按队列的固定 TTL 误杀的。

        struct 两种口径下都走 TTL：它是模板族、单轮、``gaps`` 为空，没有会话级
        证据可用；私有 struct 本来归 one-shot 带管，不当模板老化。
        """
        tag = self._node_tag(node)
        queue = tag.queue_name
        if tag is HeterCacheNodeTag.QSTRUCT:
            ttl = self.share.ageing_ttl_s[queue]
            if ttl <= 0:
                return False
            if self.node_share.share_of(node) < self.node_share.share_min:
                return False
            return self._now() - node.last_access_time >= ttl
        if self.share.ageing_mode == "session_life":
            return self.session_expired(node)
        ttl = self.share.ageing_ttl_s[queue]
        if ttl <= 0:
            return False
        return self._now() - node.last_access_time >= ttl

    def session_life_s(self, state) -> Optional[float]:
        """该会话的存活上限（秒）：``k × 轮间隔 p90``，夹在 [min, max] 之间。

        证据不足（观测到的轮间隔少于 ``session_ttl_min_gaps``）时返回 None，
        调用方据此保守处理（不老化），避免用一条样本外推整条会话。
        """
        gaps = getattr(state, "gaps", None) or []
        if len(gaps) < self.share.session_ttl_min_gaps:
            return None
        ordered = sorted(gaps)
        idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        life = self.share.session_ttl_k * ordered[idx]
        return min(max(life, self.share.session_ttl_min_s), self.share.session_ttl_max_s)

    def session_expired(self, node) -> bool:
        """会话是否已终结（方案 A 的判据本体）。

        只回收私有内容：跨会话共享数达到 ``share_min`` 的节点即使归属会话终结，
        别的会话还会读它，不能按老化清掉。
        """
        if self.node_share.share_of(node) >= self.node_share.share_min:
            return False
        state = self.session_registry.get(getattr(node, "session_id", None))
        if state is None:
            return False
        life = self.session_life_s(state)
        if life is None:
            return False
        return self._now() - state.last_active > life

    def is_one_shot(self, node) -> bool:
        """判定节点是否属于 One-shot 带。
        标准：按类型 TTL 老化的节点，或
        属于 TRANS/STRUCT 且跨会话共享数未达到 share_min，
        且从未被任何请求重读过（`reuse_count == 0`）
        且归属会话已搁置超过 ONE_SHOT_IDLE_S
        若节点无会话元数据，保守返回 False 防止误删热前缀。
        """
        # 老化必须先于 reuse_count/share 的短路：共享前缀必然被重读过且已达标。
        if self.is_aged(node):
            return True
        st = self.session_registry.get(getattr(node, "session_id", None))
        if st is None:
            return False
        if getattr(node, "reuse_count", 0) > 0:
            # 内容已被再次读到：离开优先淘汰带
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
        """记录一次命中：telemetry 按 rid 去重,同一 rid 的 chunked prefill 会重复进入，
        去重由 telemetry 负责
        Args:
            req: 请求对象
            node: 最深命中节点
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

        晋升（reuse_count 从 0 变 1）会把节点从 one-shot
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
        """按 VD 升序返回队列；越靠前（密度越低）越先削。"""
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
        reuse = self.reuse_evicts.get(queue)
        if reuse is not None:
            # chat / agentic 同一条路径：复用概率升序，只有 mu/sigma 随标签不同。
            return reuse.pick(pool, self.session_registry)
        return None

    def fallback(self, leaves):
        """所有队列均弃权时的均匀随机淘汰"""
        pool = sorted(leaves, key=lambda n: n.id)
        if not pool:
            return None
        return self._fallback_rng.choice(pool)


# ---------------------------------------------------------------------------
# 离线最优（OPT / Belady）：策略上界，不用于线上选择
# ---------------------------------------------------------------------------
#
# 语义与 UniCache 的 OPT 一致：淘汰「下次被使用时刻最远」的块，永不复用者
# 视为 ``+∞``（最先出）。区别只在数据面：
#
# * 不维护全局 ``PREFILL_CNT`` 游标。策略接口只有 ``get_priority(node)``，
#   没有 tick 推进钩子；把 use map 存成「该前缀被使用的时刻（秒）」后，
#   每次查询用 ``self._now()`` 二分即可，策略变成**只读无状态**，不 pop、
#   不依赖调用顺序，重复调用结果相同。
# * 块标识直接复用缓存自己的 SHA256 累进 hash（``compute_node_hash_values``
#   写进 ``TreeNode.hash_value``），不另造一套 Python ``hash`` 体系。
#
# 因此这是「block 粒度贪心的近似上界」：淘汰单位仍是整个叶子节点，
# 做不到 Belady 的逐块精确最优。


@dataclasses.dataclass(frozen=True)
class OptParams:
    """OPT 上界的参数。

    Attributes:
        use_map_path: ``{block_hash: [使用时刻(秒), ...]}`` 的 JSON 路径。
            必填：缺失时若静默退化成「全部 +∞」，OPT 就变成了任意序，
            实验结果会看起来「跑通了」而实际没有上界语义。
        block_size: use map 的切块粒度（token），必须与缓存 ``page_size``
            一致；两者错配时 hash 分段不同 → 查询全部落空。
        miss_warn: use map 未命中的比例超过该值时告警一次，用于发现
            「粒度错配 / map 与负载不匹配」这类静默退化。
    """

    use_map_path: str
    block_size: int = 1
    miss_warn: float = 0.5
    # 代价加权上界（可选）：``{block_hash: cost}`` 的 JSON。给了之后 OPT 的排序键
    # 不再是「下次使用时刻」，而是按块重算代价缩放过的版本：
    #   cost_mode="protect_deep"：key = -next_use / cost  （贵的块更晚被淘汰）
    #   cost_mode="paper"       ：key = -next_use * cost  （论文 BeladyCompute 字面式）
    # 空字符串 = 退化成经典 hit-optimal Belady，行为与本参数引入前逐位一致。
    cost_map_path: str = ""
    cost_mode: str = "protect_deep"

    def __post_init__(self):
        if not isinstance(self.use_map_path, str) or not self.use_map_path:
            raise ValueError(
                f"opt.use_map_path must be a non-empty str, got {self.use_map_path!r}"
            )
        if self.block_size <= 0:
            raise ValueError(
                f"opt.block_size must be > 0, got {self.block_size}"
            )
        if not isinstance(self.miss_warn, (int, float)) or not (
            0.0 < self.miss_warn <= 1.0
        ):
            raise ValueError(
                f"opt.miss_warn must be in (0, 1], got {self.miss_warn!r}"
            )
        if self.cost_mode not in ("protect_deep", "paper"):
            raise ValueError(
                "opt.cost_mode must be 'protect_deep' or 'paper', "
                f"got {self.cost_mode!r}"
            )
        if not isinstance(self.cost_map_path, str):
            raise ValueError(
                f"opt.cost_map_path must be a str, got {self.cost_map_path!r}"
            )

    @classmethod
    def from_any(cls, value=None) -> "OptParams":
        if isinstance(value, cls):
            return value
        if value is None:
            raise ValueError(
                "opt 策略必须提供 use_map_path（离线 trace 生成的块使用表）；"
                "不做「缺表即全 +∞」的静默退化"
            )
        if not isinstance(value, dict):
            raise TypeError(
                f"opt params must be None/dict/{cls.__name__}, got {type(value).__name__}"
            )
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                f"unknown opt params {sorted(unknown)}; known: {sorted(known)}"
            )
        return cls(**value)


def load_use_map(path: str) -> Dict[str, Tuple[float, ...]]:
    """读 ``{hash: [t, ...]}``，排序去重后转成不可变元组。

    Args:
        path: JSON 文件路径。

    Returns:
        ``{hash: (t0 <= t1 <= ...)}``。

    Raises:
        ValueError: 文件为空、没有可用的时间序列，或格式不是 dict。
            空表必须显式失败：否则所有节点都拿 ``+∞``，淘汰顺序退化成
            「谁被先遍历谁先出」。
    """

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"use map must be a JSON object, got {type(raw).__name__}")

    table: Dict[str, Tuple[float, ...]] = {}
    for block_hash, times in raw.items():
        if not isinstance(times, (list, tuple)):
            raise ValueError(
                f"use map entry for {block_hash!r} must be a list, got {type(times).__name__}"
            )
        seq = tuple(sorted({float(t) for t in times}))
        if seq:
            table[str(block_hash)] = seq
    if not table:
        raise ValueError(f"use map at {path} carries no usable use times")
    return table


class OPTStrategy(EvictionStrategy):
    """离线 Belady/OPT 上界：优先级 = 下次使用时刻的相反数。

    ``RadixCache._evict`` 用 ``heapq`` 取**最小**优先级，所以返回
    ``-next_use``：下次使用越远（含 ``+∞``）越先被淘汰，与 OPT 语义一致。

    时间轴约定：use map 里的时刻与 ``self._now()`` 必须同源（生产是
    ``time.monotonic``，仿真里是 ``VirtualClock``）。单个节点可能覆盖多个
    page，取其所有 page 中**最早**的下次使用——保守侧，宁可少淘汰。
    """

    def __init__(
        self,
        *,
        use_map_path: str,
        block_size: int = 1,
        miss_warn: float = 0.5,
        cost_map_path: str = "",
        cost_mode: str = "protect_deep",
        now: Optional[Callable[[], float]] = None,
    ):
        self.params = OptParams.from_any(
            {
                "use_map_path": use_map_path,
                "block_size": block_size,
                "miss_warn": miss_warn,
                "cost_map_path": cost_map_path,
                "cost_mode": cost_mode,
            }
        )
        self._use_map = load_use_map(self.params.use_map_path)
        self._cost_map: Dict[str, int] = {}
        if self.params.cost_map_path:
            with open(self.params.cost_map_path, encoding="utf-8") as f:
                raw_costs = json.load(f)
            if not isinstance(raw_costs, dict):
                raise ValueError(
                    "cost map must be a JSON object {block_hash: cost}, got "
                    f"{type(raw_costs).__name__}"
                )
            self._cost_map = {str(k): int(v) for k, v in raw_costs.items()}
        self._page_size = self.params.block_size
        # 时间源可注入；缓存装配完会再推一次（见 RadixCache.__init__）。
        self._now = now if now is not None else time.monotonic

        # 退化可见性：命中率掉到 0 却没人发现是这类策略最贵的失败方式。
        self._lookups = 0
        self._misses = 0
        self._warned = False

    # ---- 协作者装配 -------------------------------------------------------
    def use_clock(self, now: Callable[[], float]) -> None:
        """接管缓存层的时间源（与 ``HeterCacheStrategy.use_clock`` 同一约定）。"""
        self._now = now

    def bind_page_size(self, page_size: int) -> None:
        """校验 use map 的切块粒度与缓存页大小一致。

        Raises:
            ValueError: 不一致。分段不同 → 每个 hash 都查不到 → 全 ``+∞``。
        """
        if page_size != self.params.block_size:
            raise ValueError(
                "OPT use map block_size="
                f"{self.params.block_size} does not match cache page_size={page_size}; "
                "regenerate the use map with the same page size"
            )
        self._page_size = page_size

    # ---- 查询 -------------------------------------------------------------
    def _ensure_hash(self, node) -> list:
        """补齐 ``node.hash_value``（含未计算的祖先）。

        ``compute_node_hash_values`` 只读 ``parent.hash_value[-1]``：父链没算时
        会静默退化成「没有上下文」的 hash，随后每个块都查不到。所以这里沿
        父链**自根向下**逐层补算，而不是只算当前节点。
        """
        if node.hash_value is not None:
            return node.hash_value

        # 延迟导入：utils 依赖本模块（工厂表），模块级导入会成环。
        from sglang.srt.mem_cache.utils import compute_node_hash_values

        pending = []
        cursor = node
        while cursor is not None and cursor.hash_value is None:
            pending.append(cursor)
            cursor = cursor.parent
        # 缓存层已经把 root 的 hash_value 置成 []，正常不会走到 root 之外。
        for item in reversed(pending):
            item.hash_value = compute_node_hash_values(item, self._page_size)
        return node.hash_value

    def _next_use(self, block_hash: str) -> float:
        """该块严格晚于当前时刻的下次使用；没有则 ``+∞``。

        计数在这里逐次累加：``_lookups`` 是**查询次数**，``_misses`` 是其中查不到
        的次数。二者都必须在同一条路径上各自独立计数，否则 `_misses/_lookups`
        恒为 1.0——那会让「粒度错配」的告警变成「miss 过 1000 次」的探测器，
        健康负载也照报 100%。
        """
        self._lookups += 1
        times = self._use_map.get(block_hash)
        if times is None:
            self._misses += 1
            self._maybe_warn()
            return math.inf
        # bisect_right：跳过所有 <= now 的已发生使用。
        index = bisect.bisect_right(times, self._now())
        return times[index] if index < len(times) else math.inf

    def _maybe_warn(self) -> None:
        """只做判据（读 ``_lookups``/``_misses``），不改计数。"""
        if self._warned or self._lookups < 1000:
            return
        ratio = self._misses / self._lookups
        if ratio >= self.params.miss_warn:
            self._warned = True
            logger.warning(
                "OPT use map miss ratio %.1f%% (%d/%d): block 粒度或负载与 map 不匹配，"
                "OPT 已退化为「全 +∞」的任意序",
                ratio * 100.0,
                self._misses,
                self._lookups,
            )

    def get_priority(self, node) -> Tuple[float, float, int]:
        """返回 ``(-next_use, last_access_time, id)``，``heapq`` 取最小即淘汰。

        装配了 ``cost_map_path`` 时（代价加权上界），第一项换成
        ``-next_use / cost``（``protect_deep``，贵的块更晚被淘汰）或
        ``-next_use * cost``（``paper``）。代价取该节点覆盖各块的平均值——
        节点是整段淘汰的，用它代表这段的价值密度。
        """
        hashes = self._ensure_hash(node)
        next_use = min((self._next_use(h) for h in hashes), default=math.inf)
        key = -next_use
        if self._cost_map:
            costs = [self._cost_map.get(h) for h in hashes]
            known = [c for c in costs if c is not None and c > 0]
            if known:
                mean_cost = sum(known) / len(known)
                if self.params.cost_mode == "protect_deep":
                    key = -next_use / mean_cost
                else:
                    key = -next_use * mean_cost
        return (
            key,
            float(getattr(node, "last_access_time", 0.0)),
            int(getattr(node, "id", 0)),
        )