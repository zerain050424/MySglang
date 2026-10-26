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

# #3：原先这里预计算 exp(-t/600)，供 SINGLE_TURN 的复用分数使用。
# 该时间常数不对应任何生成过程（单轮 workload 没有「下一轮」），已随 #3
# 移除；单轮的队列级权重改由 alpha（实测命中效率）承担，见 evict_policy.py。


class SimpleHybridTaskType(str, enum.Enum):
    MULTI_TURN_TEXT = "multi_turn_text"
    MULTI_TURN_FILE = "multi_turn_file"
    MULTI_TURN_IMAGE = "multi_turn_image"
    MULTI_TURN_SEARCH = "multi_turn_search"
    MULTI_TURN_AGENTIC = "multi_turn_agentic"
    SINGLE_TURN = "single_turn"
    SINGLE_TURN_RANDOM = "single_turn_random"
    DEFAULT = "default"


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
    SimpleHybridTaskType.MULTI_TURN_TEXT: (4.77, 1.279, "text"),
    SimpleHybridTaskType.MULTI_TURN_IMAGE: (4.4449, 1.09066, "image"),
    SimpleHybridTaskType.MULTI_TURN_SEARCH: (4.961663, 1.13877, "search"),
    SimpleHybridTaskType.MULTI_TURN_FILE: (5.07658, 1.06383, "file"),
    SimpleHybridTaskType.MULTI_TURN_AGENTIC: (1.81, 1.092, "agentic"),
}

# profile/记账用的工作负载标签（顺序即打印顺序）。
# 与 alpha_key 的区别：SINGLE_TURN_RANDOM（apps / mind2web）单列成 "random"，
# 不再折进 "single"；alpha 维度仍用 alpha_key_of（策略只调度可调度内容，
# random 单轮整个请求都是 one-shot，不参与 alpha）。
WORKLOAD_LABELS = ("single", "random", "text", "file", "image", "search", "agentic")


def workload_label_of(task_type) -> str:
    """把任务类型映射到**工作负载**标签（记账维度，与淘汰带无关）。"""
    if task_type == SimpleHybridTaskType.SINGLE_TURN_RANDOM:
        return "random"
    return alpha_key_of(task_type)


def classify_simple_hybrid_task_type(
       session_id: Optional[str] = None,
       extra_key: Optional[str] = None,
   ) -> SimpleHybridTaskType:
       """Map a request's session/namespace key to a UniCache workload type.

       Mirrors UniCache's mark_blocks_simple_hybrid_atrribute: the conversation
       key prefix decides the type. sglang carries the analogous string in
       Req.session_id / Req.extra_key.
       """
       key = session_id or extra_key or ""
       if key.startswith(("ShareGPT-multi-turn", "text_ShareGPT-multi-turn")):
           return SimpleHybridTaskType.MULTI_TURN_TEXT
       if key.startswith("image_ShareGPT-multi-turn"):
           return SimpleHybridTaskType.MULTI_TURN_IMAGE
       if key.startswith("search_ShareGPT-multi-turn"):
           return SimpleHybridTaskType.MULTI_TURN_SEARCH
       if key.startswith("file_ShareGPT-multi-turn"):
           return SimpleHybridTaskType.MULTI_TURN_FILE
       if key.startswith("alfred_"):
           return SimpleHybridTaskType.MULTI_TURN_AGENTIC
       if key.startswith(("apps_", "mind2web_")):
           return SimpleHybridTaskType.SINGLE_TURN_RANDOM
       return SimpleHybridTaskType.SINGLE_TURN


def alpha_key_of(task_type) -> str:
    """Map a task type to its alpha-balance key ("single"/"text"/"file"/...)."""
    cfg = TASK_CONFIG.get(task_type)
    if cfg is not None:
        return cfg[2]
    return "single"


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
    """会话表。由 RadixCache 实例持有，不做模块级单例。

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
            # d 不应超过该会话最近一次写入的链长；超了说明归属或长度口径有问题
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
