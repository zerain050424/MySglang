"""CPU 仿真环境下的生产前缀缓存装载层。"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

# bootstrap.py 位于 sim/core/ 下，仓库根是其上三级目录。
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SGLANG_PY = os.path.join(REPO_ROOT, "python")

_STUB_REASONS: Dict[str, str] = {}
_LOADED: Optional[Dict[str, Any]] = None


def _raise(what: str):
    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            f"{what} 在 CPU 仿真下不可用。"
            "请将 CacheInitParams.enable_metrics 设为 False 或完善依赖后重试。"
        )

    return _unavailable


def _stub_memory_pool() -> types.ModuleType:
    """ReqToTokenPool 仅用于 base_prefix_cache 类型注解，替身阻止拉起 transformers。"""
    module = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    module.ReqToTokenPool = type(
        "ReqToTokenPool", (), {"__init__": _raise("ReqToTokenPool")}
    )
    return module


def _stub_metrics_collector() -> types.ModuleType:
    """metrics_collector 仅用于指标路径；调用时直接抛错以避免静默丢指标。"""
    module = types.ModuleType("sglang.srt.observability.metrics_collector")
    module.STAT_LOGGER_ROLE_RADIX_CACHE = "radix_cache"
    module.RadixCacheMetricsCollector = type(
        "RadixCacheMetricsCollector",
        (),
        {"__init__": _raise("RadixCacheMetricsCollector")},
    )
    module.radix_cache_metric_labels = _raise("radix_cache_metric_labels")
    module.resolve_collector_class = _raise("resolve_collector_class")
    return module


_STUBS = {
    "sglang.srt.mem_cache.memory_pool": _stub_memory_pool,
    "sglang.srt.observability.metrics_collector": _stub_metrics_collector,
}


def _ensure_sys_path() -> None:
    if SGLANG_PY not in sys.path:
        sys.path.insert(0, SGLANG_PY)


def _import_or_stub(name: str, factory) -> Tuple[Any, bool]:
    """优先尝试真实模块，仅捕获 ImportError 后再退化为替身。"""
    try:
        return importlib.import_module(name), False
    except ImportError as exc:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]
        _STUB_REASONS[name] = f"{type(exc).__name__}: {exc}".splitlines()[0]
        stub = factory()
        sys.modules[name] = stub
        return stub, True


def load(force: bool = False) -> Dict[str, Any]:
    """导入生产前缀缓存与策略符号表。"""
    global _LOADED
    if _LOADED is not None and not force:
        return _LOADED

    if not os.path.isdir(SGLANG_PY):
        raise RuntimeError(f"找不到 sglang 源码目录：{SGLANG_PY}")
    _ensure_sys_path()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CPU 仿真需要 torch（用 CPU tensor 承载节点 value）：pip install torch"
        ) from exc

    for name, factory in _STUBS.items():
        _import_or_stub(name, factory)

    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        EvictResult,
        InsertParams,
        InsertResult,
        MatchPrefixParams,
        MatchResult,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy
    from sglang.srt.mem_cache.heter_cache.occupancy import QueueOccupancyTracker
    from sglang.srt.mem_cache.heter_cache.radix_cache import HeterCacheRadixCache
    from sglang.srt.mem_cache.heter_cache.utils import (
        HeterCacheNodeTag,
        QueueValueParams,
        SessionRegistry,
        SessionType,
        VALUE_QUEUES,
    )
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

    _LOADED = {
        "RadixCache": RadixCache,
        "HeterCacheRadixCache": HeterCacheRadixCache,
        "RadixKey": RadixKey,
        "TreeNode": TreeNode,
        "CacheInitParams": CacheInitParams,
        "InsertParams": InsertParams,
        "InsertResult": InsertResult,
        "MatchPrefixParams": MatchPrefixParams,
        "MatchResult": MatchResult,
        "EvictParams": EvictParams,
        "EvictResult": EvictResult,
        "HeterCacheStrategy": HeterCacheStrategy,
        "QueueValueParams": QueueValueParams,
        "QueueOccupancyTracker": QueueOccupancyTracker,
        "SessionRegistry": SessionRegistry,
        "SessionType": SessionType,
        "HeterCacheNodeTag": HeterCacheNodeTag,
        "VALUE_QUEUES": VALUE_QUEUES,
    }
    return _LOADED


def stubs() -> Dict[str, str]:
    """返回本次运行装入的替身模块及其原因（空 dict 表示真实依赖完整）。"""
    return dict(_STUB_REASONS)


def status() -> Dict[str, Any]:
    """返回装载状态摘要，供环境自检输出。"""
    return {
        "repo_root": REPO_ROOT,
        "sglang_py": SGLANG_PY,
        "python": sys.executable,
        "torch": _torch_version(),
        "stubs": stubs(),
    }


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "<missing>"


__all__: List[str] = ["load", "stubs", "status", "REPO_ROOT", "SGLANG_PY"]
