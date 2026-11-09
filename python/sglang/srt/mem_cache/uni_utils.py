"""HeterCache 的共享数据模型：任务/会话分类、参数组、会话账本。

三个消费者都依赖这里，所以口径必须单点定义，避免"策略算一套、指标算一套"：

* ``evict_policy``：评分（lognormal 复用概率 / 位置项）与 one-shot 判定；
* ``victim_override``：改写器的作用域、共享数阈值、会话类型；
* ``quota_selector`` / profile：队列标签与占用口径。

命名约定：``SimpleHybridTaskType`` 是**节点级**任务类型（由 ``queue_tag`` 推得），
``SessionType`` 是**会话级**分类（由分类器推得）。两者经常不一致，判定归属时
必须明确用哪一个（见 ``is_one_shot`` 与 ``MultiTurnEvictPolicy.applies_to``）。

进程级状态只有一处：``SessionRegistry.global_instance()``。它被所有 radix cache
实例共享，因此 ``clear()`` 是全局语义（这正是 ``/flush_cache`` 需要的）。
"""

import math
import zlib
import dataclasses
import enum
import logging
import os
import math
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# SPDX-License-Identifier: Apache-2.0
from enum import Enum

class SimpleHybridTaskType(str, enum.Enum):
    """节点级任务类型，由 ``queue_tag`` 推导（见 ``task_type_of_tag``）。

    决定评分模型：多轮用复用概率，单轮用位置项——单轮没有"下一轮"，
    时间维度上无话可说，只保留位置项。
    """

    MULTI_TURN_CHAT = "multi_turn_chat"
    MULTI_TURN_AGENTIC = "multi_turn_agentic"
    STRUCT_SINGLE_TURN = "struct_single_turn"
    RANDOM_SINGLE_TURN = "random_single_turn"


class SessionType(str, enum.Enum):
    """会话级分类结果，由 ``SessionRegistry._classify`` 依据轮数/工具调用/间隔推出。

    与节点标签分属两层：一个 TRANS 会话插入的片段后来可能被多会话复用，
    那时标签仍是 QTRANS，但已不该算 one-shot。
    """

    TRANS = "trans"
    STRUCT = "struct"
    CHAT = "chat"
    AGENTIC = "agentic"


class SimpleHybridQueueTag(str, enum.Enum):
    """HeterCache 节点状态标签。

    标签由会话分类器与 radix 结构证据共同维护；非会话请求保留旧命中间隔规则。
    """

    QTRANS = "qtrans"
    QSTRUCT = "qstruct"
    QAGENT = "qagent"
    QCHAT = "qchat"


def session_bloom_bits(session_id) -> int:
    """把 session_id 哈希成 64 bit 布隆位图（4 个哈希函数），用于共享数去重。

    为什么是 bloom 而不是集合：共享数账本挂在每个节点上，会话数上万时集合
    会持续涨内存；bloom 固定 8 字节，误判方向是"当作已记过"（少计一次共享），
    对本策略是可接受的保守偏差。经验上 ~8 个会话后误判开始明显。

    Args:
        session_id: 会话标识；非 str 会先转字符串。

    Returns:
        64 bit 位图，位布局只与 session_id 有关，与任何标签策略无关。
    """
    text = "sid:" + (session_id if isinstance(session_id, str) else str(session_id))
    h = zlib.crc32(text.encode("utf-8", "replace")) & 0xFFFFFFFF
    h = (h * 0x9E3779B1) & 0xFFFFFFFF
    bits = 0
    for i in range(4):
        bits |= 1 << ((h >> (i * 6)) & 63)
    return bits


# 共享会话数的饱和上界：8 bit，够用且能原样塞进诊断快照。
SESSION_SHARE_MAX = 255


def session_type_to_queue_tag(session_type: SessionType) -> SimpleHybridQueueTag:
    """把会话分类结果映射成"新写入 KV 该归哪个队列"的标签。

    只在节点出生时使用：出生标签回答"谁创建了这段"，之后的复用事实由共享数
    账本记录，不回写标签。
    """
    return {
        SessionType.TRANS: SimpleHybridQueueTag.QTRANS,
        SessionType.STRUCT: SimpleHybridQueueTag.QSTRUCT,
        SessionType.AGENTIC: SimpleHybridQueueTag.QAGENT,
        SessionType.CHAT: SimpleHybridQueueTag.QCHAT,
    }[session_type]


# ---------------------------------------------------------------------------
# SimpleHybrid 的可调参数组
#
# 用 dataclass 而非裸 kwargs：让 eviction_policy_config 有层级、可校验、可整体
# 传递。配置只接受分组形式 {"rr":{...},"slru":{...},"mt":{...},"quota":{...}}，
# 拼错的键会在构造处报错——实验配置静默失效比启动失败更难查。
#
#   RrParams        : Agentic 会话级轮询配额（跨会话公平）
#   SlruParams      : SLRU 的共享会话数阈值（决定 one-shot 分界与保护范围）
#   MultiTurnParams : 多轮对话（QCHAT）带内权重
#   QuotaParams     : 跨队列配额（下限 + 上限，分母为池容量）
# 用 dataclass 而不是裸 kwargs，是为了让 `eviction_policy_config` 有层级、能校验、能整体
# 配置只接受 {"rr":{...},"slru":{...},"mt":{...},"quota":{...}} 分组形式。
# ---------------------------------------------------------------------------


def _coerce_params(cls, value, *, name: str):
    """把 None / dict / 参数组实例统一成实例。

    Args:
        cls: 参数组类。
        value: None（取默认值）、dict（按关键字展开）或已有实例。
        name: 错误消息里的配置分组名（如 ``"quota"``）。

    Returns:
        参数组实例。

    Raises:
        ValueError: dict 里出现未知键；消息会列出该组已知键名。
        TypeError: 传入既不是 None/dict 也不是该组的实例。
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        try:
            return cls(**value)
        except TypeError as exc:
            known = ", ".join(f.name for f in dataclasses.fields(cls))
            raise ValueError(
                f"Invalid {name} parameters {sorted(value)}: {exc}. Known keys: {known}."
            ) from None
    raise TypeError(
        f"{name} must be {cls.__name__}, dict or None, got {type(value).__name__}."
    )


@dataclasses.dataclass
class RrParams:
    """Agentic 会话级轮询参数。

    ``fair_rr`` 打开后，淘汰在会话之间轮询挑 victim，并用
    ``RrEvictPolicy.loss_cap`` 限制单会话本轮可承受的 KV 缺口。

    Attributes:
        fair_rr: 会话轮询开关。
        kappa: 单轮缺口上限的比例系数，与 ``min_retain_*`` 一起取较小值。
        min_retain_tokens: 每条会话至少保留的绝对 token 数。
        min_retain_frac: 每条会话至少保留的比例（相对链长 L）。
    """

    fair_rr: bool = False  # Agentic 会话轮询开关
    kappa: float = 0.5  # 单次缺口上限 = min(kappa*L, L-keep)
    min_retain_tokens: int = 512  # 每条会话至少保留的 token
    min_retain_frac: float = 0.3  # 每条会话至少保留的比例

    def __post_init__(self):
        if not 0.0 <= self.kappa <= 1.0:
            raise ValueError(f"kappa must be in [0, 1], got {self.kappa}")
        if self.min_retain_tokens < 0:
            raise ValueError(
                f"min_retain_tokens must be >= 0, got {self.min_retain_tokens}"
            )
        if not 0.0 <= self.min_retain_frac <= 1.0:
            raise ValueError(
                f"min_retain_frac must be in [0, 1], got {self.min_retain_frac}"
            )

    @classmethod
    def from_any(cls, value=None) -> "RrParams":
        return _coerce_params(cls, value, name="rr")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# 跨会话模板的共享会话数阈值。
#
# 语义：共享数 >= N 意味着共 N 个会话用过（创建者首次重读即计 1）。
# 它同时决定两件事，两处必须同源，所以只允许配置一处：
#   * SlruEvictPolicy.applies_to       : 是否受 SLRU 保护（进候选池）
#   * SimpleHybridStrategy.is_one_shot : 是否退出 one-shot 带
DEFAULT_SHARE_MIN = 2


@dataclasses.dataclass
class SlruParams:
    """SLRU 的共享会话数阈值（唯一配置入口）。

    为什么只允许一处配置：同一个数既决定"是否受 SLRU 保护"，又决定"是否退出
    one-shot 带"。两处分别配置必然漂移，最后表现为"受保护的节点被当 one-shot
    清掉"这类难查的错。

    共享数由 ``SlruEvictPolicy`` 记账：每个不同会话 +1，同会话由 bloom 去重。
    ``share_min = 2`` 意味着共 2 个会话用过（创建者 + 1 次外部命中）才算
    跨会话模板。

    Attributes:
        share_min: 受保护的共享会话数阈值，必须 >= 2。
    """

    share_min: int = DEFAULT_SHARE_MIN

    def __post_init__(self):
        if self.share_min < 2:
            raise ValueError(
                "slru.share_min must be >= 2 (>1 个外部会话才算共享), "
                f"got {self.share_min}"
            )

    @classmethod
    def from_any(cls, value=None) -> "SlruParams":
        return _coerce_params(cls, value, name="slru")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MultiTurnParams:
    """多轮对话（QCHAT）带内改写器的参数。

    带内权重取 ``1 - prob(e) = CDF(elapsed; mu, sigma)``，mu/sigma 来自
    ``TASK_CONFIG``（UniCache 原机制）。该权重只用于带内排序，不参与跨队列
    仲裁——跨队列由 ``quota_selector`` 负责。

    默认开启不改变淘汰分布：改写器只在候选落入 QCHAT 带时接管
    （``applies_to`` 门禁），带内语义与全局 ``victim_key`` 等价，
    差异仅在 tie-break 用 LRU。

    Attributes:
        enabled: 是否启用改写器。
    """

    enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"mt.enabled must be bool, got {self.enabled!r}")

    @classmethod
    def from_any(cls, value=None) -> "MultiTurnParams":
        return _coerce_params(cls, value, name="multi-turn")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# 配额覆盖的三级队列。`random` 不在此列：它整行都是 one-shot，
# 在阶段 0 就被清掉，因此永远不会计入配额账本。
QUOTA_QUEUES = ("struct", "agentic", "chat")


@dataclasses.dataclass
class QuotaParams:
    """跨队列配额：下限 rho（保留）+ 上限 k（封顶），中间是弹性区。

    配置形式：
        --radix-eviction-policy-config '{"quota": {
            "caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3},
            "reservations": {"struct": 0.05, "agentic": 0.2, "chat": 0.1}}}'

    两个字典都是**容量比例**，分母恒为池容量 C（allocator.size），
    不随淘汰变化——用当前占用之和作分母会让目标同步塔陷，把淘汰量放大 1/(1-k) 倍。

    caps 必须三个队列全给（漏填等于默认永不封顶，是个静默失效），且 sum(k) <= 1；
    reservations 缺省 0，只需给想保底的队列。chunk_tokens 与 page_size 取大作为
    单次最小减持量；deadzone 与它共同防止在边界上来回点名。
    """

    caps: Dict[str, float] = dataclasses.field(default_factory=dict)
    reservations: Dict[str, float] = dataclasses.field(default_factory=dict)
    deadzone: float = 0.05
    chunk_tokens: int = 0
    pressure_window_s: float = 3.0
    seed: int = 0

    def __post_init__(self):
        if not self.caps:
            # 空 caps = 关闭配额（默认）。只校验标量，不要求三个队列。
            self._validate_scalars()
            if self.reservations:
                raise ValueError(
                    "quota.reservations requires quota.caps; got "
                    f"{sorted(self.reservations)}."
                )
            return

        unknown = set(self.caps) - set(QUOTA_QUEUES)
        if unknown:
            raise ValueError(
                f"quota.caps has unknown queues {sorted(unknown)}; "
                f"expected exactly {list(QUOTA_QUEUES)}."
            )
        missing = set(QUOTA_QUEUES) - set(self.caps)
        if missing:
            raise ValueError(
                f"quota.caps is missing {sorted(missing)}; give all of "
                f"{list(QUOTA_QUEUES)} (a missing cap silently never binds)."
            )
        unknown = set(self.reservations) - set(QUOTA_QUEUES)
        if unknown:
            raise ValueError(
                f"quota.reservations has unknown queues {sorted(unknown)}; "
                f"expected a subset of {list(QUOTA_QUEUES)}."
            )

        for q, cap in self.caps.items():
            if not 0.0 < cap <= 1.0:
                raise ValueError(f"quota.caps[{q!r}] must be in (0, 1], got {cap}")
        total_cap = sum(self.caps.values())
        if total_cap > 1.0 + 1e-9:
            raise ValueError(
                f"quota caps must sum to <= 1, got {total_cap:.4f}; "
                "caps are fractions of the pool capacity."
            )

        for q, res in self.reservations.items():
            if not 0.0 <= res <= self.caps[q]:
                raise ValueError(
                    f"quota.reservations[{q!r}] must be in [0, caps[{q!r}]={self.caps[q]}], "
                    f"got {res}"
                )
        total_res = sum(self.reservations.values())
        if total_res > 1.0 + 1e-9:
            raise ValueError(f"quota reservations must sum to <= 1, got {total_res:.4f}")

        self._validate_scalars()

    def _validate_scalars(self) -> None:
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError(f"quota.deadzone must be in [0, 1), got {self.deadzone}")
        if self.chunk_tokens < 0:
            raise ValueError(f"quota.chunk_tokens must be >= 0, got {self.chunk_tokens}")
        if self.pressure_window_s <= 0:
            raise ValueError(
                f"quota.pressure_window_s must be > 0, got {self.pressure_window_s}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"quota.seed must be int, got {self.seed!r}")

    @property
    def enabled(self) -> bool:
        """未配置 caps 即关闭：此时淘汰行为与引入配额前完全一致。"""
        return bool(self.caps)

    @classmethod
    def from_any(cls, value=None) -> "QuotaParams":
        return _coerce_params(cls, value, name="quota")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@lru_cache(maxsize=8)
def _cached_lognorm_params(mu: float, sigma: float):
    """缓存 lognormal 的派生常数（避免每次命中回调都重算 sqrt）。"""
    return mu, sigma, sigma * math.sqrt(2.0)


@lru_cache(maxsize=8)
def _cached_lognorm_params_v1(mu: float, sigma: float):
    """同上，但多返回两个供概率密度使用的常数（历史实现保留）。"""
    inv_sigma = 1.0 / sigma
    denom = sigma * math.sqrt(2.0)
    norm_const = inv_sigma / math.sqrt(2.0 * math.pi)
    return mu, sigma, inv_sigma, denom, norm_const


def _cached_lognorm_prob(x: float, mu: float, sigma: float) -> float:
    """lognormal 的 CDF（erf 近似）。与 ``_cached_lognorm_cdf`` 同值，保留给旧调用方。"""
    if x <= 0:
        return 0.0
    mu, sigma, inv_sigma, denom, _ = _cached_lognorm_params_v1(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


def _cached_lognorm_cdf(x: float, mu: float, sigma: float) -> float:
    """lognormal 的 CDF（erf 近似，无 SciPy 依赖）。

    x <= 0 直接返回 0：``math.log(0)`` 会抛错，而刚写入的节点必然出现 x=0。

    Args:
        x: 已经过的秒数。
        mu, sigma: 分布参数，来自 ``TASK_CONFIG``。

    Returns:
        ``P(reuse_interval <= x)``，即"复用已经发生过"的概率。
    """
    if x <= 0:
        return 0.0
    mu, sigma, denom = _cached_lognorm_params(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


# 多轮任务的复用间隔分布参数 (mu, sigma)，lognormal（UniCache 原机制）。
# 评分与改写器都必须从这里取值：两边各写一份会让"评分为正、改写为负"。
TASK_CONFIG = {
    SimpleHybridTaskType.MULTI_TURN_CHAT: (4.77, 1.279),
    SimpleHybridTaskType.MULTI_TURN_AGENTIC: (1.81, 1.092),
}

# profile 与记账用的四类标签（顺序即打印顺序）。
# random 不参与跨队列配额：它整行都是 one-shot，在阶段 0 就被清掉。
WORKLOAD_LABELS = ("chat", "agentic", "struct", "random")


def workload_label_of(task_type) -> str:
    """把节点级任务类型映射成四类工作负载标签。

    未知类型回落到 ``"chat"``：多轮是最保守的归类（保留最久），比误判成
    one-shot 直接清掉安全。
    """
    return {
        SimpleHybridTaskType.MULTI_TURN_CHAT: "chat",
        SimpleHybridTaskType.MULTI_TURN_AGENTIC: "agentic",
        SimpleHybridTaskType.STRUCT_SINGLE_TURN: "struct",
        SimpleHybridTaskType.RANDOM_SINGLE_TURN: "random",
    }.get(task_type, "chat")


# ---------------------------------------------------------------------------
# 节点口径的小工具与标签映射
#
# 放在这里而不是各自的策略模块里：evict_policy / victim_override / 指标统计
# 都用同一套口径。重复实现一定会漂移——abs_start 的语义在 HeterCache 迁移中
# 就踩过一次，当时"片段长度"与"绝对偏移"被混用，评分整体偏了一个前缀长度。
# ---------------------------------------------------------------------------


def node_num_tokens(node) -> int:
    """节点的片段长度（radix 片段的 token 数），非驻留节点返回 0。"""
    return len(node.value) if node.value is not None else 0


def node_end_offset(node) -> int:
    """片段在整条序列中的绝对结束偏移（HeterCache 的 num_hashed_tokens）。

    注意这不是片段长度：struct 的位置项要的是"离序列尾部有多远"，用长度会
    把长序列的尾部当成长前缀。
    """
    num_tokens = node_num_tokens(node)
    abs_start = getattr(node, "abs_start", None)
    if abs_start is None:
        return num_tokens
    return abs_start + num_tokens


def task_type_of_tag(tag) -> SimpleHybridTaskType:
    """queue_tag 映射到任务类型；未知/缺失标签按多轮 chat 处理。

    未知标签回落到 chat 而不是 random：random 会被排进 one-shot 带直接清掉，
    多轮归类最多是多留一段时间。
    """
    return {
        SimpleHybridQueueTag.QTRANS: SimpleHybridTaskType.RANDOM_SINGLE_TURN,
        SimpleHybridQueueTag.QSTRUCT: SimpleHybridTaskType.STRUCT_SINGLE_TURN,
        SimpleHybridQueueTag.QAGENT: SimpleHybridTaskType.MULTI_TURN_AGENTIC,
        SimpleHybridQueueTag.QCHAT: SimpleHybridTaskType.MULTI_TURN_CHAT,
    }.get(tag, SimpleHybridTaskType.MULTI_TURN_CHAT)


@dataclasses.dataclass
class SessionState:
    """单条会话的 KV 记账 + 分类证据。

    两套字段故意分开，互不干扰：

    * **KV 记账**：``L``（最近一次 prompt 长度）与 ``d``（当前缺口）。``d`` 只在
      两处变化——会话自己的请求写入时归零（``touch``）、其节点被删时累加
      （``charge``）。RR 的 ``loss_cap`` 直接读这两个值。
    * **分类证据**：``classifier_gaps`` / ``turn_count`` / ``tool_call_count`` /
      ``cross_session_shared``，只用于推 ``session_type``。

    分开的原因是：分类器改一次阈值就会把历史 replay 的分类结果改掉，如果两者
    共用 ``gaps``，连带会把 KV 缺口的口径也改掉。

    Attributes:
        L: 最近一次请求的 prompt 长度（token）。
        d: 当前 KV 缺口（已删但仍在该会话活跃链里的 token）。
        node_count: 仍驻留的节点数。
        last_rid: 用于 chunked prefill 去重——同一 rid 重入不算新的一轮。
    """

    session_id: str
    L: int = 0                  # 最近一次请求的 prompt 长度
    d: int = 0                  # 当前缺口（token）
    node_count: int = 0         # 仍驻留的节点数
    last_active: float = 0.0    # monotonic
    last_rid: str = ""          # 用于 chunked prefill 去重
    gaps: List[float] = dataclasses.field(default_factory=list)
    attached: int = 0           # 累计挂到该会话名下的 token 数（调试用）
    charged: int = 0            # 累计被删的 token 数（调试用）
    turn_count: int = 0
    tool_call_count: int = 0
    cross_session_shared: bool = False
    session_type: SessionType = SessionType.TRANS
    first_context_len: int = 0
    latest_context_len: int = 0
    classifier_gaps: List[float] = dataclasses.field(default_factory=list)
    classifier_last_rid: str = ""
    classifier_last_request_start: Optional[float] = None


logger = logging.getLogger(__name__)

# 调试开关：d 超过 L 时打印具体节点。默认关——这两个分支都在逐节点热路径上，
# 只要打开就会把淘汰循环拖慢一个量级。
_D_BREAK_LOG = os.environ.get("SGLANG_SESSION_D_BREAK_LOG", "0") == "1"
# 只追踪单个会话的每一次 charge，用于核对是否重复计账。
_TRACE_SID = os.environ.get("SGLANG_SESSION_TRACE_SID", "")


class SessionRegistry:
    """会话表：KV 缺口记账 + 会话类型分类。

    职责划分：``observe_request`` / ``record_cross_session_hit`` 只更新分类证据；
    ``touch`` / ``charge`` / ``attach`` 只更新 KV 记账。两边共用一个
    ``SessionState``，但字段不重叠（见该类文档）。

    节点按鸭子类型读取 ``session_id`` 与 ``value``，因此本模块不依赖 radix 树类型。

    **进程级共享**：缓存层统一走 ``global_instance()``，所以

    * 同一进程内所有 cache 实例看到同一份 ``d``/``L``，RR 的 ``loss_cap`` 与
      分类器不会各算一套；
    * ``clear()`` 是全局语义（``/flush_cache`` 清的本来就是整本账）；
    * 需要隔离（单测、并行对照实验）时用 ``reset_global()`` 丢弃单例，或自建
      ``SessionRegistry()`` 后覆盖 ``cache.session_registry``。

    并发：所有调用方都在调度线程（match / insert / evict），``sweep()`` 由
    ``touch()`` 顺带触发，因此不加锁。
    """

    # 进程级单例。用基类名访问，避免子类各持一份。
    _global: Optional["SessionRegistry"] = None

    @classmethod
    def global_instance(cls) -> "SessionRegistry":
        """返回进程级共享账本；首次调用时按默认 TTL 创建。"""
        inst = SessionRegistry._global
        if inst is None:
            inst = SessionRegistry()
            SessionRegistry._global = inst
        return inst

    @classmethod
    def reset_global(cls) -> Optional["SessionRegistry"]:
        """丢弃单例并返回旧实例，下一次 ``global_instance()`` 会新建。

        参数（``idle_ttl`` / ``empty_grace``）在单例创建时就固定了，想换参数
        必须先 reset——这是测试隔离和"换配置重建"的唯一入口。
        """
        old = SessionRegistry._global
        SessionRegistry._global = None
        return old

    GAP_WINDOW = 8          # 只保留最近 8 个轮间隔样本
    SWEEP_EVERY = 256       # 每 N 次 touch 顺带回收一次
    MULTI_TURN_MIN = 5
    GAP_SPLIT_S = 15.0
    AGENTIC_CONTEXT_GROWTH_TOKENS = 1500

    def __init__(self, idle_ttl: float = 3600.0, empty_grace: float = 60.0):
        self._sessions: Dict[str, SessionState] = {}
        self.idle_ttl = idle_ttl
        self.empty_grace = empty_grace
        self._touches = 0

    # ---- 查询 ----
    def get(self, session_id: Optional[str]) -> Optional[SessionState]:
        return self._sessions.get(session_id) if session_id else None

    def __len__(self) -> int:
        return len(self._sessions)

    # ---- 会话分类证据 ----
    def observe_request(
        self,
        session_id: Optional[str],
        rid: str = "",
        prompt_len: int = 0,
        *,
        tool_call=None,
        now: Optional[float] = None,
    ) -> Optional[SessionState]:
        """用当前请求的会话证据更新分类；同一 rid（chunked 重入）只计一次。

        只动分类字段，不碰 KV 记账（``d`` / ``L`` 归 ``touch`` 管）。

        Args:
            session_id: 会话标识；缺失时直接返回 None。
            rid: 请求 id，用于识别 chunked prefill 的重复进入。
            prompt_len: 本次请求的 prompt 长度。
            tool_call: 工具调用证据（存在即视为 agentic 的正向信号）。
            now: 注入时钟，便于离线回放与测试。

        Returns:
            该会话的状态；``session_id`` 为空时返回 None。
        """
        if not session_id:
            return None
        now = time.monotonic() if now is None else now
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id) #获取session状态
        if rid and rid == st.classifier_last_rid:
            return st

        if st.classifier_last_request_start is None:
            st.first_context_len = prompt_len ## 首次请求时记录初始上下文长度
        else:
            st.classifier_gaps.append(max(0.0, now - st.classifier_last_request_start)) ## 记录这个会话的轮间间隔
            if len(st.classifier_gaps) > self.GAP_WINDOW: ## 截断
                del st.classifier_gaps[0]
        st.turn_count += 1
        st.tool_call_count += int(bool(tool_call))
        st.latest_context_len = prompt_len
        st.classifier_last_request_start = now
        st.classifier_last_rid = rid
        st.session_type = self._classify(st)
        return st

    def record_cross_session_hit(
        self, session_id: Optional[str], owner_session_id: Optional[str]
    ) -> Optional[SessionState]:
        """记录"本会话复用了别的会话插入的片段"，作为跨会话共享证据。

        跨会话共享会把单轮 TRANS 升为 STRUCT（结构化负载），因此这是分类证据，
        不参与 KV 记账。

        Args:
            session_id: 复用了别人片段的会话。
            owner_session_id: 片段创建者。两者相同（或任一缺失）时不算证据。

        Returns:
            本会话的状态；无需更新时返回查询结果。
        """
        if not session_id or not owner_session_id or session_id == owner_session_id:
            return self.get(session_id)
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.cross_session_shared = True # 检测是否存在跨会话复用
        st.session_type = self._classify(st)
        return st

    @classmethod
    def _classify(cls, state: SessionState) -> SessionType:
        """按"轮数 -> 工具调用 -> 轮间间隔 -> 上下文增长"的顺序推会话类型。

        顺序不能换：工具调用与短间隔都是 agentic 的正向证据，先判它们可以避免
        一条真正的 agent 会话因为偶尔一次长间隔被判成 chat（chat 的 T-LRU 语义
        完全不同，误判会直接改变保护策略）。

        阈值都放在类属性上，便于离线回放时按 trace 重标定。

        Args:
            state: 会话状态（只读分类字段）。

        Returns:
            推得的会话类型。
        """
        # 轮数不足：按"是否被别的会话复用"区分结构化文档与一次性事务。
        if state.turn_count < cls.MULTI_TURN_MIN:
            return SessionType.STRUCT if state.cross_session_shared else SessionType.TRANS
        if state.tool_call_count:  # 工具调用是 agentic 的强正向证据
            return SessionType.AGENTIC
        if state.classifier_gaps:
            ordered = sorted(state.classifier_gaps)
            middle = len(ordered) // 2
            median_gap = (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
            if median_gap < cls.GAP_SPLIT_S:  # 连续短间隔：agentic
                return SessionType.AGENTIC
            if median_gap > cls.GAP_SPLIT_S:  # 长间隔：人在读，判 chat
                return SessionType.CHAT
        growth = state.latest_context_len - state.first_context_len
        return (
            SessionType.AGENTIC
            # 间隔恰好等于阈值时无法区分，退回"上下文是否持续增长"这一证据。
            if growth >= cls.AGENTIC_CONTEXT_GROWTH_TOKENS
            else SessionType.CHAT
        )

    # ---- 记账 ----
    def touch(
        self,
        session_id: Optional[str],
        rid: str = "",
        prompt_len: int = 0,
        now: Optional[float] = None,
    ) -> Optional[SessionState]:
        """会话自己的请求被写入时调用：更新 L、缺口归零、记录轮间间隔。

        ``d`` 在这里归零而不是"递减"：新一轮 prompt 已经重新 prefill，之前的缺口
        不再代表延迟代价。

        Args:
            session_id: 会话标识；缺失时直接返回。
            rid: 请求 id；与上次相同时说明是 chunked prefill 重入，不计新的一轮。
            prompt_len: 本次 prompt 长度，写入 ``L``。
            now: 注入时钟；``touch`` 同时是 ``sweep`` 的触发点。

        Returns:
            该会话状态；``session_id`` 为空时返回 None。
        """
        if not session_id:
            return None
        now = time.monotonic() if now is None else now
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        elif rid != st.last_rid and st.last_active > 0.0:
            # chunked prefill 会让同一请求多次进入插入路径；只有换 rid 才算新的一轮，
            # 否则会把一个请求切成多"轮"，把 agentic 误判成 chat。
            st.gaps.append(now - st.last_active)
            if len(st.gaps) > self.GAP_WINDOW:
                del st.gaps[0]
        st.last_rid = rid
        st.L = prompt_len
        st.d = 0
        st.last_active = now

        self._touches += 1
        if self._touches % self.SWEEP_EVERY == 0:
            self.sweep(now)
        return st

    def attach(self, session_id: Optional[str], num_tokens: int = 0) -> None:
        """登记一个归属该会话的新节点（token 数与节点数各 +1）。

        只用于观测与 sweep 的存活判据，不参与 ``d`` 的计算。

        Args:
            session_id: 会话标识。
            num_tokens: 新节点的 token 数。
        """
        if not session_id:
            return
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.node_count += 1
        st.attached += num_tokens

    def charge(self, node) -> None:
        """节点被删时调用：所属会话缺口累加，存活节点数减一。

        级联删除会对每个节点各调一次，因此缺口是累加的；本方法同时是
        ``/flush_cache`` 之外唯一会改动 ``d`` 的入口。

        Args:
            node: 被删除的节点（读取 ``session_id`` / ``value``）。
        """
        session_id = getattr(node, "session_id", None)
        if not session_id:
            return
        value = getattr(node, "value", None)
        num_tokens = len(value) if value is not None else 0
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.d += num_tokens
        st.charged += num_tokens
        if st.node_count > 0:
            st.node_count -= 1
        if _TRACE_SID and session_id == _TRACE_SID:
            logger.warning(
                "[session trace charge] nid=%s abs_start=%s key=%d val=%d d=%d L=%d",
                getattr(node, "id", -1), getattr(node, "abs_start", None),
                len(getattr(node, "key", []) or []), num_tokens, st.d, st.L,
            )
        if _D_BREAK_LOG and st.L and st.d > st.L + 32:
            # d 不应超过该会话最近一次写入的链长；一旦超过，说明节点归属或长度
            # 口径出了问题，这条日志是唯一能定位到具体节点的地方。
            logger.warning(
                "[session d break] sid=%s d=%d L=%d +%d abs_start=%s "
                "attached=%d charged=%d live=%d node_count=%d",
                session_id, st.d, st.L, num_tokens,
                getattr(node, "abs_start", None),
                st.attached, st.charged, st.attached - st.charged, st.node_count,
            )
        # 同一会话在树里可能同时存在活路径与孤儿分支（部分清空后从断点重插），
        # 孤儿分支被淘汰时也会计入 d。d 的语义是当前缺口，上限即链长。
        if st.L and st.d > st.L:
            st.d = st.L

    # ---- 回收 ----
    def clear(self) -> None:
        """清空全部会话记账。

        ``/flush_cache`` 与缓存 reset 必须调用：跨 run 复用同一个 server 时，
        残留的 ``d``/``L`` 会让 RR 的 ``loss_cap`` 读到上一轮的缺口，实验之间
        互相污染。
        """
        self._sessions.clear()

    def sweep(self, now: Optional[float] = None) -> int:
        """回收长时间不活跃的会话，返回回收条数。

        由 ``touch`` 每 ``SWEEP_EVERY`` 次顺带触发，不单独起线程：账本只在调度
        线程上被访问，加锁的代价比顺带清理高。

        两种回收条件：超过 ``idle_ttl``；或节点已全部清空且过了 ``empty_grace``
        （后者让刚 flush 完的会话还能保留一小段时间的分类证据）。

        Args:
            now: 注入时钟。

        Returns:
            被回收的会话数。
        """
        now = time.monotonic() if now is None else now
        dead = [
            sid
            for sid, st in self._sessions.items()
            if now - st.last_active > self.idle_ttl
            or (st.node_count <= 0 and now - st.last_active > self.empty_grace)
        ]
        for sid in dead:
            del self._sessions[sid]
        return len(dead)

    # ---- 观测 ----
    def summary(self) -> Dict[str, float]:
        """缺口分布摘要，供 ``[SimpleHybrid profile]`` 打印。

        ``d_nonzero`` 是最有用的一个：它告诉你有多少会话真的处于"被削过"的状态，
        只看 ``d_sum`` 会被少数大缺口带偏。
        """
        ds = sorted(st.d for st in self._sessions.values())
        if not ds:
            return {"sessions": 0, "d_sum": 0, "d_p50": 0, "d_max": 0, "d_nonzero": 0}
        n = len(ds)
        return {
            "sessions": n,
            "d_sum": sum(ds),
            "d_p50": ds[n // 2],
            "d_max": ds[-1],
            "d_nonzero": sum(1 for x in ds if x > 0),
        }
