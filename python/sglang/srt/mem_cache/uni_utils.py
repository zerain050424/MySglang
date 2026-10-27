import math
import dataclasses
import enum
import logging
import os
import statistics
import math
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
    task_type: Optional["SimpleHybridTaskType"] = None
    p: float = 1.0              # 量化后的视界内回归概率
    p_ts: float = 0.0           # p 的上次刷新时刻（节流用）
    attached: int = 0           # 累计挂到该会话名下的 token 数（调试用）
    charged: int = 0            # 累计被删的 token 数（调试用）


logger = logging.getLogger(__name__)

# 调试开关：d 超过 L 时打印具体节点（默认关，零开销）
_D_BREAK_LOG = os.environ.get("SGLANG_SESSION_D_BREAK_LOG", "0") == "1"
# 只追踪单个会话的每一次 charge，用于核对是否存在重复计账
_TRACE_SID = os.environ.get("SGLANG_SESSION_TRACE_SID", "")


class SessionRegistry:
    """
    会话表。
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
        """取会话账本。

        session_id: Req.session_id；None 或未登记时返回 None
        """
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
        task_type: Optional["SimpleHybridTaskType"] = None,
    ) -> Optional[SessionState]:
        """会话自己的请求被写入时调用：记轮间隔、更新 L、缺口归零。

        session_id: Req.session_id；为空则不入表，直接返回 None。
        rid:        请求 rid，用于 chunked prefill 去重（同一请求多次进入插入路径）。
        prompt_len: 本次请求写入的链长（page 对齐后的 key 长度），记为 L_i。
        now:        time.monotonic() 快照；None 时内部取一次。
        task_type:  该请求的 workload 分类，留给 p_reuse 选先验。

        返回更新后的 SessionState；session_id 为空时返回 None。
        """
        if not session_id:
            return None
        now = time.monotonic() if now is None else now
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        elif rid != st.last_rid and st.last_active > 0.0:
            # chunked prefill 会让同一请求多次进入插入路径，只有换 rid 才算新的一轮
            st.gaps.append(now - st.last_active) #轮间间隔
            if len(st.gaps) > self.GAP_WINDOW:
                del st.gaps[0]
        st.last_rid = rid
        st.L = prompt_len
        st.d = 0
        st.last_active = now
        if task_type is not None:
            st.task_type = task_type

        self._touches += 1
        if self._touches % self.SWEEP_EVERY == 0:
            self.sweep(now)
        return st

    def attach(self, session_id: Optional[str], num_tokens: int = 0) -> None:
        """登记一个新节点归属某会话。

        session_id: 该节点的 owner；为空则不记。
        num_tokens: 该节点的片段长度 len(value)，用于调试时核对归属总量。
        """
        if not session_id:
            return
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.node_count += 1
        st.attached += num_tokens

    def charge(self, node: Any) -> None:
        """节点被删时调用：所属会话缺口累加，存活节点数减一。

        node: 刚被 _delete_leaf 摘除的 TreeNode。无session_id的节点不计入。

        副作用：d += len(node.value)，并按 L_i 截断到链长（孤儿分支的重复计账）。
        """
        session_id = getattr(node, "session_id", None)
        if not session_id:
            return
        value = getattr(node, "value", None)
        num_tokens = len(value) if value is not None else 0
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        st.d += num_tokens ## 当前缺口
        st.charged += num_tokens ##累计被删除的token数目
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
        # 孤儿分支被淘汰时也会计入 d  d 的语义是当前缺口，上限即链长。
        if st.L and st.d > st.L:
            st.d = st.L

    # ---- 权重 ----
    P_REFRESH_S = 0.2       # 权重重算节流
    SHRINK_K = 3            # 向 workload 先验收缩的强度

    def p_reuse(
        self,
        session_id: Optional[str],
        now: Optional[float] = None,
        horizon_s: float = 300.0,
        levels: Tuple[float, ...] = (1.0, 0.5, 0.2, 0.05),
    ) -> float:
        """视界内回归概率，量化为档位。

        session_id: 目标会话；未登记时返回 0.0（视为最不值得保留）。
        now:        time.monotonic() 快照；None 时内部取一次。
        horizon_s:  视界 H，池子周转时间的量级（秒）。
        levels:     量化档位，降序；返回值取「不超过 p 的最高档」，p 低于末档时取末档。

        P = F(idle + H) − F(idle)：F 为 lognormal(ln m_i, 1.0) 的 CDF，
        m_i 取该会话轮间隔中位数并向 workload 先验按 SHRINK_K 收缩。
        结果缓存在 state 上，P_REFRESH_S 内重复调用直接返回缓存值。
        """
        st = self._sessions.get(session_id) if session_id else None
        if st is None:
            return 0.0
        now = time.monotonic() if now is None else now
        if st.p_ts and now - st.p_ts < self.P_REFRESH_S:
            return st.p

        cfg = TASK_CONFIG.get(st.task_type) if st.task_type is not None else None
        m_prior = math.exp(cfg[0]) if cfg else 30.0
        n = len(st.gaps)
        if n:
            m = (n * statistics.median(st.gaps) + self.SHRINK_K * m_prior) / (n + self.SHRINK_K)
        else:
            m = m_prior
        mu = math.log(max(m, 1e-3))
        idle = max(now - st.last_active, 0.0)
        p = _cached_lognorm_cdf(idle + horizon_s, mu, 1.0) - _cached_lognorm_cdf(idle, mu, 1.0)
        p = min(max(p, 0.0), 1.0)
        q = levels[-1] if levels else 1.0
        for lv in levels:
            if p >= lv:
                q = lv
                break
        st.p, st.p_ts = q, now
        return q

    # ---- 回收 ----
    def sweep(self, now: Optional[float] = None) -> int:
        """
        回收过期会话账本。
        now: time.monotonic() 快照；None 时内部取一次。

        回收条件：距最后一次请求超过 idle_ttl，或已无存活节点且超过 empty_grace。
        返回被删除的会话数。touch 每 SWEEP_EVERY 次会顺带调一次。
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
        """给 [SimpleHybrid profile] 用的缺口分布。

        返回 {sessions, d_sum, d_p50, d_max, d_nonzero}：
          sessions  登记在册的会话数
          d_sum     各会话当前缺口之和（与淘汰总量同量纲，可互相校验）
          d_p50/max 缺口的中位数 / 最大值
          d_nonzero 缺口大于 0 的会话数（被淘汰过的会话数）
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
