"""HeterCache 的共享数据模型：任务/会话分类、参数组、会话账本。

三个消费者都依赖这里，所以口径必须单点定义，避免"策略算一套、指标算一套"：

* ``evict_policy``：评分（lognormal 复用概率 / 位置项）与 one-shot 判定；
* ``victim_override``：改写器的作用域、共享数阈值、会话类型；
* ``quota_selector`` / profile：队列标签与占用口径。

命名约定：``HeterCacheNodeTag`` 是**节点级**标签（KV 片段出生时盖章），
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

class SessionType(str, enum.Enum):
    """会话级分类结果，由 ``SessionRegistry._classify`` 依据轮数/工具调用/间隔推出。

    与节点标签分属两层：一个 TRANS 会话插入的片段后来可能被多会话复用，
    那时标签仍是 QTRANS，但已不该算 one-shot。
    """

    TRANS = "trans"
    STRUCT = "struct"
    CHAT = "chat"
    AGENTIC = "agentic"


class HeterCacheNodeTag(str, enum.Enum):
    """节点级标签：KV 片段出生时由会话分类盖章，之后只被少数规则改写。

    与 ``SessionType`` 是两个维度：本枚举回答“这段 KV 是谁创建的”，
    ``SessionType`` 回答“这个会话现在是什么”。两者会不同步，升级规则见
    ``RadixCache._record_node_hit``。评分模型、配额分池、profile 列名都只读本枚举。

    值（``qtrans`` 等）随节点标签快照落盘（JSONL 的 ``tag`` 字段），改值等于改
    数据格式，故保持原样。
    """

    QTRANS = "qtrans"
    QSTRUCT = "qstruct"
    QAGENT = "qagent"
    QCHAT = "qchat"

    @classmethod
    def coerce(cls, tag) -> "HeterCacheNodeTag":
        """把节点上的原始值收敛成枚举；缺失/未知按多轮 chat 处理。

        回落到 chat 而不是 QTRANS：QTRANS 会被排进 one-shot 带直接清掉，
        多轮归类最多是多留一段时间。
        """
        if isinstance(tag, cls):
            return tag
        try:
            return cls(tag)
        except (ValueError, TypeError):
            return cls.QCHAT

    @property
    def queue_name(self) -> str:
        """归属队列名（配额配置的键、profile 的列名）。

        把枚举投影成字符串只是为了对接 CLI 配置（`caps` / `reservations` 的键
        必须是字符串）与日志列名；判定一律用枚举本身。
        """
        return _QUEUE_NAME[self]

    @classmethod
    def from_queue(cls, name) -> "HeterCacheNodeTag | None":
        """队列名反查枚举；没有对应关系时返回 None。

        反向表包含全部四项（含 ``QTRANS`` ↔ ``random``），因此与旧的"队列名
        等于字符串"判定逐字等价。配额层只会传入 ``QUOTA_QUEUES`` 里的三个名字
        （random 由 ``quota_ranking`` 滤掉），其余名字走 None 分支。
        """
        return _QUEUE_TO_TAG.get(name)

    @property
    def is_multi_turn(self) -> bool:
        """是否属于多轮带：只有多轮才有“下一轮复用概率”可言。"""
        return self in (HeterCacheNodeTag.QCHAT, HeterCacheNodeTag.QAGENT)

    @property
    def reuse_config(self) -> Optional[Tuple[float, float]]:
        """复用间隔的 lognormal 参数 ``(mu, sigma)``；无时间维度时返回 None。"""
        return _REUSE_CONFIG.get(self)


def session_bloom_bits(session_id) -> int:
    """把session_id哈希成 64 bit 布隆位图,用于共享数去重."""
    # 1. 序列化
    text = f"sid:{session_id}"
    
    # 2. 基础哈希 (CRC32 结果为无符号32位整数)
    h = zlib.crc32(text.encode("utf-8", "replace"))
    
    # 3. 增强雪崩扩散 (Knuth乘法散列 + 双向异或位移)
    # 解决CRC32线性特征以及低位混合不充分的问题
    h = (h ^ (h >> 16)) * 0x85EBCA6B & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 0xC2B2AE35 & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF

    # 4. 提取4个[0, 63]的索引
    # 分散利用32位的不同区间，避免只用低 24 位
    bits = 0
    bits |= 1 << (h & 63)              # bits 0~5
    bits |= 1 << ((h >> 8) & 63)       # bits 8~13
    bits |= 1 << ((h >> 16) & 63)      # bits 16~21
    bits |= 1 << ((h >> 24) | ((h & 0x3) << 4))  # 组合高位与剩余位，打乱相关性
    
    return bits


# 共享会话数的饱和上界：8 bit，够用且能原样塞进诊断快照。
SESSION_SHARE_MAX = 255


def session_type_to_node_tag(session_type: SessionType) -> HeterCacheNodeTag:
    """把会话分类结果映射成“新写入 KV 该归哪个队列”的标签。

    只在节点出生时使用：出生标签回答“谁创建了这段”，之后的复用事实由共享数
    账本记录，不回写标签。
    """
    return {
        SessionType.TRANS: HeterCacheNodeTag.QTRANS,
        SessionType.STRUCT: HeterCacheNodeTag.QSTRUCT,
        SessionType.AGENTIC: HeterCacheNodeTag.QAGENT,
        SessionType.CHAT: HeterCacheNodeTag.QCHAT,
    }[session_type]


# ---------------------------------------------------------------------------
# HeterCache 的可调参数组
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
        kappa: 单轮缺口上限的比例系数
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
#   * HeterCacheStrategy.is_one_shot : 是否退出 one-shot 带
DEFAULT_SHARE_MIN = 2


@dataclasses.dataclass
class SlruParams:
    """SLRU 的共享会话数阈值（唯一配置入口）。

    为什么只允许一处配置：同一个数既决定"是否受 SLRU 保护"，又决定"是否退出
    one-shot 带"。两处分别配置必然漂移，最后表现为"受保护的节点被当 one-shot
    清掉"这类难查的错。

    共享数由SlruEvictPolicy记录：每个不同会话 +1，同会话由 bloom 去重。
    share_min = 2,意味着共 2 个会话用过（创建者 + 1 次外部命中）才算
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

    带内权重取 1 - prob(e) = CDF(elapsed; mu, sigma)，mu/sigma 来自 HeterCacheNodeTag.reuse_config
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
    """跨队列配额：下限 rho + 上限 k，中间是弹性区。

    配置形式：
        --radix-eviction-policy-config '{"quota": {
            "caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3},
            "reservations": {"struct": 0.05, "agentic": 0.2, "chat": 0.1}}}'

    两个字典都是容量比例，分母恒为池容量C,不随淘汰变化
    chunk_tokens 与 page_size 取大作为单次最小减持量；deadzone与它共同防止在边界上来回颠簸淘汰。
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
        mu, sigma: 分布参数，来自 ``HeterCacheNodeTag.reuse_config``。

    Returns:
        ``P(reuse_interval <= x)``，即"复用已经发生过"的概率。
    """
    if x <= 0:
        return 0.0
    mu, sigma, denom = _cached_lognorm_params(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


# 多轮任务的复用间隔分布参数 (mu, sigma)，lognormal（UniCache 原机制）。
# 评分与改写器都必须从这里取值：两边各写一份会让“评分为正、改写为负”。
_REUSE_CONFIG = {
    HeterCacheNodeTag.QCHAT: (4.77, 1.279),
    HeterCacheNodeTag.QAGENT: (1.81, 1.092),
}

# 节点标签 → 队列名。既是配额配置（caps / reservations）的键，
# 也是 profile 的列名，所以只在这里定义一次。
_QUEUE_NAME = {
    HeterCacheNodeTag.QCHAT: "chat",
    HeterCacheNodeTag.QAGENT: "agentic",
    HeterCacheNodeTag.QSTRUCT: "struct",
    HeterCacheNodeTag.QTRANS: "random",
}
_QUEUE_TO_TAG = {name: tag for tag, name in _QUEUE_NAME.items()}

# profile 的四列，顺序即打印顺序。
# random 不参与跨队列配额：它整行都是 one-shot，在阶段 0 就被清掉。
QUEUE_NAMES = ("chat", "agentic", "struct", "random")



def node_num_tokens(node) -> int:
    """节点的片段长度（radix 片段的 token 数），非驻留节点返回 0。"""
    return len(node.value) if node.value is not None else 0


def node_end_offset(node) -> int:
    """片段在整条序列中的绝对结束偏移"""
    num_tokens = node_num_tokens(node)
    abs_start = getattr(node, "abs_start", None)
    if abs_start is None:
        return num_tokens
    return abs_start + num_tokens





@dataclasses.dataclass
class SessionState:
    """单条会话的 KV 记账 + 分类证据。
    * KV记录：L（最近一次 prompt 长度）与d（当前缺口）。d只在
      两处变化——会话自己的请求写入时归零、其节点被删时累加
      RR 的loss_cap直接读这两个值。
    * 分类依据：classifier_gaps/turn_count/tool_call_count/
      cross_session_shared，只用于推session_type。

    分开的原因是：分类器改一次阈值就会把历史replay的分类结果改掉，如果两者
    共用gaps，连带会把 KV 缺口的口径也改掉。

    Attributes:
        L: 最近一次请求的 prompt 长度（token）。
        d: 当前 KV 缺口。
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
_D_BREAK_LOG = os.environ.get("SGLANG_SESSION_D_BREAK_LOG", "0") == "1"
_TRACE_SID = os.environ.get("SGLANG_SESSION_TRACE_SID", "")
# sweep 耗时采样开关：默认关。一次 sweep 只要两次 perf_counter（约 0.15us），
# 但 sweep 本身在小会话表上也只有微秒级，默认开会让相对开销变难看；
# 真正要看时延的场合显式打开即可。
_SWEEP_PROF = os.environ.get("SGLANG_SESSION_SWEEP_PROFILE", "0") == "1"

class SessionRegistry:
    """会话KV缺口统计：维护 KV Cache 缺口（d/L 记录）并在线推断会话负载类型。

    职责边界：
    - observe_request / record_cross_session_hit：提取特征更新分类（TRANS/STRUCT/AGENTIC/CHAT）。
    - touch / charge / attach：维护 KV 节点增删及缺失（d/L）。


    设计约束：
    1. 进程级单例：多 cache 实例通过 global_instance() 共享同一份状态，防止分类器与调度层策略出现分歧。
    2. 无锁安全：所有入口严格限定在调度主线程（match / insert / evict）执行，禁止跨线程并发。
    3. 鸭子类型：仅读取 node.session_id 与 node.value，与底层 Radix 树实现解耦。
    """

    _global: Optional["SessionRegistry"] = None

    @classmethod
    def global_instance(cls) -> "SessionRegistry":
        """获取进程单例，初次调用按默认参数惰性创建。"""
        inst = SessionRegistry._global
        if inst is None:
            inst = SessionRegistry()
            SessionRegistry._global = inst
        return inst

    @classmethod
    def reset_global(cls) -> Optional["SessionRegistry"]:
        """销毁单例并返回旧对象。用于单测隔离、重置参数或回放对比。"""
        old = SessionRegistry._global
        SessionRegistry._global = None
        return old

    GAP_WINDOW = 8          # 轮间间隔滑动窗口大小（采样最近 8 次）
    SWEEP_EVERY = 256       # 惰性回收周期：每 N 次 touch 触发一次 sweep
    MULTI_TURN_MIN = 5      # 判定多轮的最小 turn 阈值
    GAP_SPLIT_S = 15.0      # Agent 与 Chat 的间隔判定临界值（秒）
    AGENTIC_CONTEXT_GROWTH_TOKENS = 1500  # 边界兜底：判定上下文持续暴涨的 token 阈值
    # 以下三个只服务于 sweep 耗时埋点（SGLANG_SESSION_SWEEP_PROFILE=1 时生效）。
    SWEEP_PROF_EVERY = 64        # 每 N 次 sweep 打一行汇总，太小刷屏、太大漏尖峰
    SWEEP_WARN_US = 2000.0       # 单次 sweep 超过 2ms 就绪告警

    def __init__(self, idle_ttl: float = 3600.0, empty_grace: float = 60.0):
        self._sessions: Dict[str, SessionState] = {}
        self.idle_ttl = idle_ttl          # 会话无请求的最大存活时间
        self.empty_grace = empty_grace    # 节点清空后保留分类证据的宽限期
        self._touches = 0
        # sweep 埋点累加器：只在 _SWEEP_PROF 打开时增长。
        self._sweep_calls = 0
        self._sweep_us_total = 0.0
        self._sweep_us_max = 0.0
        self._sweep_scan = 0              # 最近一次扫描的会话数
        self._sweep_dead = 0              # 最近一次回收的会话数

    # ---- 查询 ----
    def get(self, session_id: Optional[str]) -> Optional[SessionState]:
        return self._sessions.get(session_id) if session_id else None

    def __len__(self) -> int:
        return len(self._sessions)

    # ---- 特征提取与分类 ----
    def observe_request(
        self,
        session_id: Optional[str],
        rid: str = "",
        prompt_len: int = 0,
        *,
        tool_call=None,
        now: Optional[float] = None,
    ) -> Optional[SessionState]:
        """记录请求特征并刷新分类。基于 rid 去重，规避 chunked prefill 重入。

        注意：本方法仅更新分类特征，不更新 d/L 等缓存水位指标。
        """
        if not session_id:
            return None
        now = time.monotonic() if now is None else now
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        # 同一请求的后续 chunk，跳过统计
        if rid and rid == st.classifier_last_rid:
            return st

        if st.classifier_last_request_start is None:
            st.first_context_len = prompt_len
        else:
            st.classifier_gaps.append(max(0.0, now - st.classifier_last_request_start))
            if len(st.classifier_gaps) > self.GAP_WINDOW:
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
        """标记跨会话命中。用于将单轮短会话提升为 STRUCT（共享公共前缀）。"""
        if not session_id or not owner_session_id or session_id == owner_session_id:
            return self.get(session_id)
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.cross_session_shared = True
        st.session_type = self._classify(st)
        return st

    @classmethod
    def _classify(cls, state: SessionState) -> SessionType:
        """会话类型决策树：轮数 -> 工具调用 -> 交互间隔中位数 -> 上下文膨胀量。

        判定优先级：
        1. 轮数不足时，仅凭是否发生跨会话共享区分 STRUCT / TRANS。
        2. 工具调用是强判定信号，命中直接归为 AGENTIC。
        3. 间隔中位数区分机器与人类：短间隔判定为高频自主 Agent，长间隔判定为人工打字交互（CHAT）。
        4. 临界值退避：若间隔处于分界线，依据 prompt 暴涨量区分。
        """
        if state.turn_count < cls.MULTI_TURN_MIN:
            return SessionType.STRUCT if state.cross_session_shared else SessionType.TRANS
        if state.tool_call_count:
            return SessionType.AGENTIC
        if state.classifier_gaps:
            ordered = sorted(state.classifier_gaps)
            middle = len(ordered) // 2
            median_gap = (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
            if median_gap < cls.GAP_SPLIT_S:
                return SessionType.AGENTIC
            if median_gap > cls.GAP_SPLIT_S:
                return SessionType.CHAT
        growth = state.latest_context_len - state.first_context_len
        return (
            SessionType.AGENTIC
            if growth >= cls.AGENTIC_CONTEXT_GROWTH_TOKENS
            else SessionType.CHAT
        )

    # ---- 缓存水位与缺口记账 ----
    def touch(
        self,
        session_id: Optional[str],
        rid: str = "",
        prompt_len: int = 0,
        now: Optional[float] = None,
    ) -> Optional[SessionState]:
        """新请求落盘时调用：重设有效链长 L，缺口 d 清零，重置活跃时钟。

        d 强制置零逻辑：
        新 prompt 已经完整执行预填充，旧缺口带来的重计算开销已被吸收，不再递延。
        """
        if not session_id:
            return None
        now = time.monotonic() if now is None else now
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        elif rid != st.last_rid and st.last_active > 0.0:
            # 过滤 chunked prefill 重入，确保间隔反映真实的轮次交互延时
            st.gaps.append(now - st.last_active)
            if len(st.gaps) > self.GAP_WINDOW:
                del st.gaps[0]
        st.last_rid = rid
        st.L = prompt_len
        st.d = 0
        st.last_active = now

        # 挂在调度热路径上做摊销清理，避免常驻后台回收线程
        self._touches += 1
        if self._touches % self.SWEEP_EVERY == 0:
            self.sweep(now)
        return st

    def attach(self, session_id: Optional[str], num_tokens: int = 0) -> None:
        """KV 树挂载新节点时累加存活计数，供淘汰及 sweep 存活判定使用。"""
        if not session_id:
            return
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.node_count += 1
        st.attached += num_tokens

    def charge(self, node) -> None:
        """KV 节点逐出时调用：累加会话缺口 d，递减节点计数。

        约束说明：
        淘汰孤儿分支或非连续节点时，d 可能会在短时间内高于当前实际链长 L，
        此处将 d 上限截断为 L（缺口最大不会超过 prompt 自身）。
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
            # 告警保护：实际被逐出 token 数远超最近注册的链长，通常是节点 session_id 归属串标
            logger.warning(
                "[session d break] sid=%s d=%d L=%d +%d abs_start=%s "
                "attached=%d charged=%d live=%d node_count=%d",
                session_id, st.d, st.L, num_tokens,
                getattr(node, "abs_start", None),
                st.attached, st.charged, st.attached - st.charged, st.node_count,
            )
        # 裁剪脏缺口，保证物理含义一致性：d <= L
        if st.L and st.d > st.L:
            st.d = st.L

    # ---- 状态回收 ----
    def clear(self) -> None:
        """全量清空账本。在 /flush_cache 或测试用例重置时调用，防止实验间脏数据污染。"""
        self._sessions.clear()

    def sweep(self, now: Optional[float] = None) -> int:
        """基于 TTL 与节点存活状态惰性回收僵尸会话。

        回收判定：
        1. 超过 idle_ttl 未活跃；
        2. 节点全被逐出且超过 empty_grace 宽限期。

        这是本模块唯一会全量遍历会话表的地方（O(会话数)），由 ``touch`` 按
        ``SWEEP_EVERY`` 计数触发。开关 ``SGLANG_SESSION_SWEEP_PROFILE=1`` 时
        记录本次耗时与扫描规模，供高负载下定夺要不要做增量回收。

        Args:
            now: 注入时钟，便于回放与测试。

        Returns:
            本次回收的会话数。
        """
        now = time.monotonic() if now is None else now
        # 计时只包住真正的遍历与删除，不看 ``now`` 的取时钟开销。
        started = time.perf_counter() if _SWEEP_PROF else 0.0
        scanned = len(self._sessions) if _SWEEP_PROF else 0
        dead = [
            sid
            for sid, st in self._sessions.items()
            if now - st.last_active > self.idle_ttl
            or (st.node_count <= 0 and now - st.last_active > self.empty_grace)
        ]
        for sid in dead:
            del self._sessions[sid]
        if _SWEEP_PROF:
            self._record_sweep(time.perf_counter() - started, scanned, len(dead))
        return len(dead)

    def _record_sweep(self, elapsed_s: float, scanned: int, dead: int) -> None:
        """累加一次 sweep 的耗时样本，并按周期或超阈打印。

        同时记录耗时与扫描规模：单看耗时分不清“表小但慢”还是“表大但正常”，
        二者的优化手段完全不同。两者相除就是单会话本，能直接外推到更大规模。

        Args:
            elapsed_s: 本次 sweep 的墙钟耗时（秒）。
            scanned: 本次遍历的会话数。
            dead: 本次回收的会话数。
        """
        us = elapsed_s * 1e6
        self._sweep_calls += 1
        self._sweep_us_total += us
        self._sweep_scan = scanned
        self._sweep_dead = dead
        if us > self._sweep_us_max:
            self._sweep_us_max = us
            # 爆冷门的那一次立即报，不等周期，否则尖峰会被平均值抹平。
            if us > self.SWEEP_WARN_US:
                logger.warning(
                    "[Session sweep] SLOW %.1fus (scan=%d, dead=%d) 阈%.0fus",
                    us,
                    scanned,
                    dead,
                    self.SWEEP_WARN_US,
                )
        if self._sweep_calls % self.SWEEP_PROF_EVERY == 0:
            avg = self._sweep_us_total / self._sweep_calls
            per_session = us / scanned if scanned else 0.0
            logger.info(
                "[Session sweep] n=%d last=%.1fus avg=%.1fus max=%.1fus "
                "scan=%d dead=%d per_session=%.3fus",
                self._sweep_calls,
                us,
                avg,
                self._sweep_us_max,
                scanned,
                dead,
                per_session,
            )

    def sweep_profile(self) -> Dict[str, float]:
        """导出 sweep 耗时统计，供 profile 行或压测脚本直接消费。

        未打开 ``SGLANG_SESSION_SWEEP_PROFILE`` 时返回全零，不区分是“没测”还是
        “零开销”。

        Returns:
            ``calls`` / ``avg_us`` / ``max_us`` / ``last_scan`` / ``last_dead``。
        """
        calls = self._sweep_calls
        return {
            "calls": calls,
            "avg_us": self._sweep_us_total / calls if calls else 0.0,
            "max_us": self._sweep_us_max,
            "last_scan": self._sweep_scan,
            "last_dead": self._sweep_dead,
        }

    # ---- 监控指标 ----
    def summary(self) -> Dict[str, float]:
        """导出会话缺口分布。重点关注 d_nonzero（实际发生截断的受损会话量）。"""
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