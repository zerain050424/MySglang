"""仿真核心基础设施：环境装载、虚拟时钟、分配器、请求替身与生命周期管理。"""

from sim.core.allocator import OutOfTokens, TokenAllocator, tree_tokens
from sim.core.bootstrap import REPO_ROOT, SGLANG_PY, load, status, stubs
from sim.core.clock import VirtualClock
from sim.core.config import SimConfig, build_policy_config
from sim.core.lifecycle import Matched, Pending, RequestLifecycle
from sim.core.pricing import PromptPricer
from sim.core.request import (
    MAX_SENTINEL_REQUESTS,
    OUTPUT_SENTINEL_BASE,
    FakeReqToTokenPool,
    SimKv,
    SimReq,
)

__all__ = [
    "FakeReqToTokenPool",
    "MAX_SENTINEL_REQUESTS",
    "Matched",
    "OUTPUT_SENTINEL_BASE",
    "OutOfTokens",
    "Pending",
    "PromptPricer",
    "REPO_ROOT",
    "RequestLifecycle",
    "SGLANG_PY",
    "SimConfig",
    "SimKv",
    "SimReq",
    "TokenAllocator",
    "VirtualClock",
    "build_policy_config",
    "load",
    "status",
    "stubs",
    "tree_tokens",
]
