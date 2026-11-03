import math
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

# 单轮 workload 没有「下一轮」，其队列级权重由 alpha（实测命中效率）承担。


class SimpleHybridTaskType(str, enum.Enum):
    MULTI_TURN_CHAT = "multi_turn_chat"
    MULTI_TURN_AGENTIC = "multi_turn_agentic"
    STRUCT_SINGLE_TURN = "struct_single_turn"
    RANDOM_SINGLE_TURN = "random_single_turn"


class SimpleHybridQueueTag(str, enum.Enum):
    """HeterCache 节点状态标签。

    标签由缓存状态机根据插入、命中间隔和分支拓扑自动维护。
    """

    QTRANS = "qtrans"
    QSTRUCT = "qstruct"
    QAGENT = "qagent"
    QCHAT = "qchat"


# ---------------------------------------------------------------------------
# SimpleHybrid 的三组可调参数
#
#   AlphaParams   : alpha 在线权重的反馈控制（全局、每 update_interval 跑一次）
#   RrParams      : Agentic 会话级轮询配额（跨会话公平）
#   TinyLRUParams : Q_struct 队列内部的 W-TinyLRU（队内驻留/准入，见 STRUCT_QUEUE_WTINYLRU.md）
# 用 dataclass 而不是裸 kwargs，是为了让 `eviction_policy_config` 有层级、能校验、能整体
# 复制/日志/消融；旧的扁平键（alpha_smooth / fair_rr / struct_k ...）已删除，
# 配置只接受 {"alpha":{...},"rr":{...},"tiny_lru":{...}} 分组形式。
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


DEFAULT_ALPHA_BALANCE = {
    "chat": 1.0,
    "agentic": 1.0,
    "struct": 1.0,
    "random": 1.0,
}


@dataclasses.dataclass
class AlphaParams:
    """alpha 在线权重的反馈控制参数。

    `balance` 是各队列的初始权重；运行时由 update_alpha_balance() 按命中效率改写，
    三个稳定性约束（容量份额下限 / 单步限速 / 温度+EMA）都在这里配置。
    """

    update_interval: float = 30.0  # 秒；alpha 重算与 profile 打印的间隔
    min_cap_share: float = 0.02  # eff 分母下限，避免份额趋零时爆炸
    max_alpha_step: float = 3.0  # 单次更新允许的最大倍数变化
    smooth: float = 0.7  # EMA 中旧值保留权重
    temperature: float = 0.5  # (1/temperature) 次幂放大
    balance: Dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_ALPHA_BALANCE)
    )

    def __post_init__(self):
        # 浅拷贝 + 补齐缺省键：外部传入的字典不与自己共享引用（内部会就地改 balance），
        # 只给部分队列（如 {"chat": 2.0}）时其余队列仍回落到 1.0。
        merged = dict(DEFAULT_ALPHA_BALANCE)
        merged.update(self.balance)
        self.balance = merged
        if self.update_interval <= 0:
            raise ValueError(f"update_interval must be > 0, got {self.update_interval}")
        if self.min_cap_share <= 0:
            raise ValueError(f"min_cap_share must be > 0, got {self.min_cap_share}")
        if self.max_alpha_step < 1.0:
            raise ValueError(f"max_alpha_step must be >= 1, got {self.max_alpha_step}")
        if not 0.0 <= self.smooth <= 1.0:
            raise ValueError(f"smooth must be in [0, 1], got {self.smooth}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    @classmethod
    def from_any(cls, value=None) -> "AlphaParams":
        return _coerce_params(cls, value, name="alpha")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


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


@dataclasses.dataclass
class TinyLRUParams:
    """Q_struct 队列内部的 W-TinyLRU（队内淘汰，不做跨队列仲裁）。

    policy="off" 时策略完全不构造，所有钩子判空跳过（零开销）。
    完整语义见 python/sglang/srt/mem_cache/STRUCT_QUEUE_WTINYLRU.md。
    """

    policy: str = "off"  # "off" | "wtinylru"
    k: int = 3  # protected 的 share_count 门限
    window_entries: int = 64  # 观察窗条目配额（无队列预算，按条目而非 token）
    epoch_events: int = 8192  # aging 纪元的请求数
    admission_tie: str = "reject"  # 频数相同时 "reject" | "admit"
    admission_metric: str = "share"  # "share" | "share_per_token"
    inherit_share: bool = True  # 降级叶子是否继承父节点频数

    def __post_init__(self):
        if self.policy not in ("off", "wtinylru"):
            raise ValueError(
                f"Unknown tiny_lru.policy: {self.policy}. Supported: 'off', 'wtinylru'."
            )
        if self.k < 2:
            raise ValueError(f"tiny_lru.k must be >= 2, got {self.k}")
        if self.window_entries < 1:
            raise ValueError(
                f"tiny_lru.window_entries must be >= 1, got {self.window_entries}"
            )
        if self.epoch_events < 1:
            raise ValueError(
                f"tiny_lru.epoch_events must be >= 1, got {self.epoch_events}"
            )
        if self.admission_tie not in ("reject", "admit"):
            raise ValueError(
                f"tiny_lru.admission_tie must be 'reject' or 'admit', got {self.admission_tie}"
            )
        if self.admission_metric not in ("share", "share_per_token"):
            raise ValueError(
                "tiny_lru.admission_metric must be 'share' or 'share_per_token', "
                f"got {self.admission_metric}"
            )

    @classmethod
    def from_any(cls, value=None) -> "TinyLRUParams":
        return _coerce_params(cls, value, name="tiny_lru")

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


   # (mu, sigma, alpha_key) per multi-turn workload type
TASK_CONFIG = {
    SimpleHybridTaskType.MULTI_TURN_CHAT: (4.77, 1.279, "chat"),
    SimpleHybridTaskType.MULTI_TURN_AGENTIC: (1.81, 1.092, "agentic"),
}

# profile/记账用的四类工作负载标签（顺序即打印顺序）。
WORKLOAD_LABELS = ("chat", "agentic", "struct", "random")


def workload_label_of(task_type) -> str:
    """把任务类型映射到四类工作负载标签。"""
    return alpha_key_of(task_type)


def alpha_key_of(task_type) -> str:
    """Map a task type to one of the four workload keys."""
    cfg = TASK_CONFIG.get(task_type)
    if cfg is not None:
        return cfg[2]
    return {
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
    `gaps` 记轮间隔样本（本次请求到达 - 上次请求到达），供后续权重使用。
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


logger = logging.getLogger(__name__)

# 调试开关：d 超过 L 时打印具体节点（默认关，零开销）
_D_BREAK_LOG = os.environ.get("SGLANG_SESSION_D_BREAK_LOG", "0") == "1"
# 只追踪单个会话的每一次 charge，用于核对是否存在重复计账
_TRACE_SID = os.environ.get("SGLANG_SESSION_TRACE_SID", "")


class SessionRegistry:
    """
    会话表,记录每个会话的状态
    只做记账：`touch` 由插入路径调用，`charge`/`attach` 由节点新增与删除调用。
    不依赖 radix tree，节点按鸭子类型读取 `session_id` 与 `value`。
    """

    GAP_WINDOW = 8          # 只保留最近 8 个轮间隔样本
    SWEEP_EVERY = 256       # 每 N 次 touch 顺带回收一次

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
