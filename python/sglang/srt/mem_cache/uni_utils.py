import math
import enum
import math
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

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
