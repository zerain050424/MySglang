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

# 单轮 workload 没有「下一轮」，节点级只保留位置项。


class SimpleHybridTaskType(str, enum.Enum):
    MULTI_TURN_CHAT = "multi_turn_chat"
    MULTI_TURN_AGENTIC = "multi_turn_agentic"
    STRUCT_SINGLE_TURN = "struct_single_turn"
    RANDOM_SINGLE_TURN = "random_single_turn"


class SessionType(str, enum.Enum):
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
    """session_id -> 64 bit bloom（4 函数），饱和在 ~8 个会话后误判率上升。

    位布局归共享会话数账本的所有者（SlruEvictPolicy）使用；
    节点上的 ``session_bloom`` 语义因此与任何标签策略无关。
    """
    text = "sid:" + (session_id if isinstance(session_id, str) else str(session_id))
    h = zlib.crc32(text.encode("utf-8", "replace")) & 0xFFFFFFFF
    h = (h * 0x9E3779B1) & 0xFFFFFFFF
    bits = 0
    for i in range(4):
        bits |= 1 << ((h >> (i * 6)) & 63)
    return bits


# 节点共享会话数的饱和上界（8 bit，够用且能塞进诊断快照）。
SESSION_SHARE_MAX = 255


def session_type_to_queue_tag(session_type: SessionType) -> SimpleHybridQueueTag:
    """Map a session classifier result to the queue used for newly owned KV."""
    return {
        SessionType.TRANS: SimpleHybridQueueTag.QTRANS,
        SessionType.STRUCT: SimpleHybridQueueTag.QSTRUCT,
        SessionType.AGENTIC: SimpleHybridQueueTag.QAGENT,
        SessionType.CHAT: SimpleHybridQueueTag.QCHAT,
    }[session_type]


# ---------------------------------------------------------------------------
# SimpleHybrid 的可调参数组
#
#   RrParams        : Agentic 会话级轮询配额（跨会话公平）
#   SlruParams      : SLRU 的共享会话数阈值（决定 one-shot 分界与保护范围）
#   TlruParams      : T-LRU 的尾部预算（QAGENT 候选修正）
#   MultiTurnParams : 多轮对话（QCHAT）带内权重
# 用 dataclass 而不是裸 kwargs，是为了让 `eviction_policy_config` 有层级、能校验、能整体
# 配置只接受 {"rr":{...},"slru":{...},"tlru":{...},"mt":{...}} 分组形式。
# ---------------------------------------------------------------------------


def _coerce_params(cls, value, *, name: str):
    """把 None / dict / 实例统一成参数组实例。"""
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
    """Agentic 会话级轮询（round-robin）配额参数。

    enabled 为真时，淘汰阶段在会话之间轮询挑 victim，并用
    `RrEvictPolicy.loss_cap()` 限制单会话本轮可以承受的 KV 缺口。
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
# 语义：除创建者外还有 >= N 个会话命中过同一节点（即共 >= N+1 个会话用过）。
# 它同时决定两件事，两处必须同源，所以只允许配置一处：
#   * SlruEvictPolicy.applies_to       : 是否受 SLRU 保护（进候选池）
#   * SimpleHybridStrategy.is_one_shot : 是否退出 one-shot 带
DEFAULT_SHARE_MIN = 2


@dataclasses.dataclass
class SlruParams:
    """SLRU 的共享会话数阈值。

    配置形式：``--radix-eviction-policy-config '{"slru": {"share_min": 2}}'``。

    共享数由 `SlruEvictPolicy` 记账（见 victim_override）：每个不同会话 +1，
    创建者在节点插入那一刻就计入，同一会话由`session_bloom` 去重。
    因此 `share_min = 2` 意味着共 2 个会话用过
    ＝创建者 + 1 次外部命中，才算跨会话模板、才开始受保护；
    低于该值的节点（以及 TRANS 会话的节点）进 one-shot 带优先淘汰。
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
class TlruParams:
    """T-LRU 的尾部预算（SimpleHybrid 的 QAGENT 候选修正用）。

    threshold 与 next_prompt_estimate 都是 **token** 数；论文里以 block 为单位，
    移植时需乘 page_size。语义：

        budget = max(L + Q_hat - threshold, 0)

    起点已越过 budget 的片段对下一轮 prefill 的尾部延迟再无贡献（tel-safe），
    可以优先淘汰。threshold 必须 **大于** next_prompt_estimate，否则预算永远
    不小于 L，没有任何节点会 tel-safe，T-LRU 退化成纯 LRU。
    """

    enabled: bool = False
    threshold: int = 0
    next_prompt_estimate: int = 0

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError(f"tlru.threshold must be >= 0, got {self.threshold}")
        if self.next_prompt_estimate < 0:
            raise ValueError(
                "tlru.next_prompt_estimate must be >= 0, "
                f"got {self.next_prompt_estimate}"
            )
        if self.enabled and self.threshold <= self.next_prompt_estimate:
            raise ValueError(
                "tlru.threshold must be > tlru.next_prompt_estimate, got "
                f"{self.threshold} <= {self.next_prompt_estimate}; otherwise no "
                "tokens are ever TEL-safe and T-LRU is exactly LRU."
            )

    @classmethod
    def from_any(cls, value=None) -> "TlruParams":
        return _coerce_params(cls, value, name="tlru")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MultiTurnParams:
    """多轮对话（QCHAT）带内改写器的参数。

    配置形式：``--radix-eviction-policy-config '{"mt": {"enabled": true}}'``。

    带内权重 1 − prob(e) = CDF(elapsed; μ, σ)，μ/σ 取
    ``TASK_CONFIG[MULTI_TURN_CHAT]``（UniCache 原机制，μ=4.77 / σ=1.279）；
    带内排序不乘其他队列的权重——QCHAT 带内语义等价于全局 victim_key。
    enabled 默认 True：mt 改写器只在 victim 落入 QCHAT 带时接管
    （调用链 applies_to 门禁），带内语义与全局 victim_key 等价，
    差异仅在 tie-break=LRU，默认不改变淘汰分布。
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


@lru_cache(maxsize=8)
def _cached_lognorm_params(mu: float, sigma: float):
    """Cache lognormal parameters to avoid recomputation."""
    return mu, sigma, sigma * math.sqrt(2.0)


@lru_cache(maxsize=8)
def _cached_lognorm_params_v1(mu: float, sigma: float):
    """Cache lognormal parameters to avoid recomputation."""
    inv_sigma = 1.0 / sigma
    denom = sigma * math.sqrt(2.0)
    norm_const = inv_sigma / math.sqrt(2.0 * math.pi)
    return mu, sigma, inv_sigma, denom, norm_const


def _cached_lognorm_prob(x: float, mu: float, sigma: float) -> float:
    """CDF of lognormal distribution (erf-based)."""
    if x <= 0:
        return 0.0
    mu, sigma, inv_sigma, denom, _ = _cached_lognorm_params_v1(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


def _cached_lognorm_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of lognormal distribution using fast erf-based approximation."""
    if x <= 0:
        return 0.0
    mu, sigma, denom = _cached_lognorm_params(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


   # (mu, sigma) per multi-turn workload type
TASK_CONFIG = {
    SimpleHybridTaskType.MULTI_TURN_CHAT: (4.77, 1.279),
    SimpleHybridTaskType.MULTI_TURN_AGENTIC: (1.81, 1.092),
}

# profile/记账用的四类工作负载标签（顺序即打印顺序）。
WORKLOAD_LABELS = ("chat", "agentic", "struct", "random")


def workload_label_of(task_type) -> str:
    """把任务类型映射到四类工作负载标签。"""
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
# 都要用同一套口径，重复实现迟早会漂移（abs_start 语义已经在 HeterCache 迁移中踩过一次）。
# ---------------------------------------------------------------------------


def node_num_tokens(node) -> int:
    """片段长度（radix 片段的 token 数）。"""
    return len(node.value) if node.value is not None else 0


def node_end_offset(node) -> int:
    """片段在整条序列中的绝对结束偏移（HeterCache 的 num_hashed_tokens）。"""
    num_tokens = node_num_tokens(node)
    abs_start = getattr(node, "abs_start", None)
    if abs_start is None:
        return num_tokens
    return abs_start + num_tokens


def task_type_of_tag(tag) -> SimpleHybridTaskType:
    """queue_tag -> 任务类型。策略与淘汰改写器共用这一份映射。"""
    return {
        SimpleHybridQueueTag.QTRANS: SimpleHybridTaskType.RANDOM_SINGLE_TURN,
        SimpleHybridQueueTag.QSTRUCT: SimpleHybridTaskType.STRUCT_SINGLE_TURN,
        SimpleHybridQueueTag.QAGENT: SimpleHybridTaskType.MULTI_TURN_AGENTIC,
        SimpleHybridQueueTag.QCHAT: SimpleHybridTaskType.MULTI_TURN_CHAT,
    }.get(tag, SimpleHybridTaskType.MULTI_TURN_CHAT)


@dataclasses.dataclass
class SessionState:
    """一条会话的 KV记账。

    `d` 只在两处变化：会话自己的请求写入时归零、其片段被删时累加。
    `classifier_gaps`、tool-call 与跨会话命中证据只用于更新会话类型；
    原 `gaps` / `L` 仍服务于 KV 账本，避免分类器改变其记账语义。
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

# 调试开关：d 超过 L 时打印具体节点（默认关，零开销）
_D_BREAK_LOG = os.environ.get("SGLANG_SESSION_D_BREAK_LOG", "0") == "1"
# 只追踪单个会话的每一次 charge，用于核对是否存在重复计账
_TRACE_SID = os.environ.get("SGLANG_SESSION_TRACE_SID", "")


class SessionRegistry:
    """
    会话表：独立维护 KV 记账与逐步升级的会话类型。
    `observe_request` / `record_cross_session_hit` 更新分类证据；`touch`、`charge`、
    `attach` 保持 KV 记账职责。节点按鸭子类型读取 `session_id` 与 `value`。

    进程级共享
    ----------
    缓存层通过 `global_instance()` 取用**同一个**账本，因此：

      * 同一进程内所有 radix cache 实例（device/host、重复构造）看到同一份
        `d`/`L`/`gaps`，RR 的 `loss_cap` 与分类器不会各算一套；
      * `clear()` 是全局的 —— `/flush_cache` 清的是整本账（这正是想要的语义）；
      * 需要隔离（单测、对照实验并行）时用 `reset_global()` 丢弃单例，或直接
        `SessionRegistry()` 造私有实例并在构造缓存后覆盖 `cache.session_registry`。

    访问方全部在调度线程（match / insert / evict）里，`sweep()` 由 `touch()` 顺带
    触发，因此这里不加锁。
    """

    # 进程级单例。用基类名访问，避免子类各持一份。
    _global: Optional["SessionRegistry"] = None

    @classmethod
    def global_instance(cls) -> "SessionRegistry":
        """返回进程级共享账本（首次调用时用默认 TTL 创建）。"""
        inst = SessionRegistry._global
        if inst is None:
            inst = SessionRegistry()
            SessionRegistry._global = inst
        return inst

    @classmethod
    def reset_global(cls) -> Optional["SessionRegistry"]:
        """丢弃单例并返回旧实例；下一次 `global_instance()` 会新建。

        用于测试隔离与「换参数重建」：单例一旦创建就固定了 `idle_ttl` /
        `empty_grace`，想换参数必须先 reset。
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
        """用当前请求可见的会话证据更新分类；同一 rid（如 chunked 重入）只计一次。"""
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
        """Record that this session reused a radix segment first inserted by another session."""
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
        if state.turn_count < cls.MULTI_TURN_MIN: # 小于阈值轮数, 认为是单轮 workload，按跨会话共享证据区分结构化与非结构化
            return SessionType.STRUCT if state.cross_session_shared else SessionType.TRANS
        if state.tool_call_count: # 存在Tool_call, 认为是 agentic workload
            return SessionType.AGENTIC
        if state.classifier_gaps:
            ordered = sorted(state.classifier_gaps)
            middle = len(ordered) // 2
            median_gap = (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
            if median_gap < cls.GAP_SPLIT_S: # 轮间间隔小于阈值, 认为是Agentic
                return SessionType.AGENTIC
            if median_gap > cls.GAP_SPLIT_S: # 轮间间隔大于阈值, 认为是Chat
                return SessionType.CHAT
        growth = state.latest_context_len - state.first_context_len
        return (
            SessionType.AGENTIC
            if growth >= cls.AGENTIC_CONTEXT_GROWTH_TOKENS #上下文长度增长超过阈值, 认为是Agentic
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
        """会话自己的请求被写入时调用：记轮间隔、更新 L、缺口归零。"""
        if not session_id:
            return None
        now = time.monotonic() if now is None else now
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        elif rid != st.last_rid and st.last_active > 0.0:
            # chunked prefill 会让同一请求多次进入插入路径，只有换 rid 才算新的一轮
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
        """新增一个归属该会话的节点。"""
        if not session_id:
            return
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.node_count += 1
        st.attached += num_tokens

    def charge(self, node) -> None:
        """节点被删时调用：所属会话缺口累加，存活节点数减一。"""
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
            # d 不应超过该会话最近一次写入的链长；超过说明归属或长度口径有问题
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

        缓存 flush / reset 必须调用：跨 run 复用同一个 server 时，`d`/`L`/`gaps`
        如果残留，RR 的 loss_cap 会读到上一轮的缺口，实验之间互相污染。
        """
        self._sessions.clear()

    def sweep(self, now: Optional[float] = None) -> int:
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
        """给 [SimpleHybrid profile] 用的缺口分布。"""
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
