"""CPU 上的 sglang 前缀缓存装载层（M0）。

目标：**不修改 python/sglang/**，让生产模块在本机（base conda，huggingface-hub
与 transformers 版本冲突未解）按原样导入。

做法只有一件事——为两个「只出现在类型注解 / 指标路径上」的重依赖预置替身：

    sglang.srt.mem_cache.memory_pool
        base_prefix_cache.py:23 只为 :58 的 `req_to_token_pool: ReqToTokenPool`
        注解而导入，却把 memory_pool -> srt.configs -> transformers 整条链拉起来。
    sglang.srt.observability.metrics_collector
        base_prefix_cache.py:25-30 的四个符号只在 init_metrics_collector() 里用。

候选条件：**先试真模块**，导入成功就完全不替换（此时仿真与线上同源）；只有抛
``ImportError`` 才装替身，并记下原因。这样环境一旦修好，替身自动退场，不会出现
「以为在跑真代码，其实跑的是空壳」这种最难查的偏差。

替身的能力边界（都是有意的）：
  * 指标采集不可用：``enable_metrics`` 必须为 False，替身一旦被真正调用就抛错；
  * ``req_to_token_pool`` 由调用方提供（仿真里是 sim/req.py 的假池）。

时钟、分配器、请求驱动都不在这里，见 sim/README.md 的里程碑。
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

# 仓库根 = sim/ 的上一级。生产包在 <root>/python 下，不依赖安装。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGLANG_PY = os.path.join(REPO_ROOT, "python")

# 装过替身的模块名 -> 当初失败原因。空 dict 表示全程用的真模块。
_STUB_REASONS: Dict[str, str] = {}
_LOADED: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------
def _raise(what: str):
    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            f"{what} 在 CPU 仿真下不可用。"
            "要么把 CacheInitParams.enable_metrics 设为 False，"
            "要么修好环境后重新运行（见 sim/bootstrap.py 顶部说明）。"
        )

    return _unavailable


def _stub_memory_pool() -> types.ModuleType:
    """`ReqToTokenPool` 只在 base_prefix_cache 的类型注解里出现，实例化即报错。"""

    module = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    module.ReqToTokenPool = type(
        "ReqToTokenPool", (), {"__init__": _raise("ReqToTokenPool")}
    )
    module.__doc__ = "sim stub: 真实 memory_pool 拉 transformers，CPU 仿真用不到。"
    return module


def _stub_metrics_collector() -> types.ModuleType:
    """四个符号只被 init_metrics_collector() 使用；调用即报错，避免静默丢指标。"""

    module = types.ModuleType("sglang.srt.observability.metrics_collector")
    module.STAT_LOGGER_ROLE_RADIX_CACHE = "radix_cache"
    module.RadixCacheMetricsCollector = type(
        "RadixCacheMetricsCollector",
        (),
        {"__init__": _raise("RadixCacheMetricsCollector")},
    )
    module.radix_cache_metric_labels = _raise("radix_cache_metric_labels")
    module.resolve_collector_class = _raise("resolve_collector_class")
    module.__doc__ = "sim stub: CPU 仿真不采集 metrics（enable_metrics=False）。"
    return module


_STUBS = {
    "sglang.srt.mem_cache.memory_pool": _stub_memory_pool,
    "sglang.srt.observability.metrics_collector": _stub_metrics_collector,
}


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------
def _ensure_sys_path() -> None:
    if SGLANG_PY not in sys.path:
        sys.path.insert(0, SGLANG_PY)


def _import_or_stub(name: str, factory) -> Tuple[Any, bool]:
    """真模块优先；仅 ImportError 才退化成替身（其它异常原样抛出）。"""

    try:
        return importlib.import_module(name), False
    except ImportError as exc:
        # 清掉可能残留的半成品（失败的导入一般已自动移除，这里防御性再扫一遍）
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]
        _STUB_REASONS[name] = f"{type(exc).__name__}: {exc}".splitlines()[0]
        stub = factory()
        sys.modules[name] = stub
        return stub, True


def load(force: bool = False) -> Dict[str, Any]:
    """导入生产缓存模块，返回仿真需要的符号表。

    Args:
        force: 重新装载（默认进程内只装一次）。

    Returns:
        ``{"HeterCacheRadixCache": ..., "RadixKey": ..., ...}``。

    Raises:
        RuntimeError: torch 不可用，或 sglang 源码路径不存在。
    """

    global _LOADED
    if _LOADED is not None and not force:
        return _LOADED

    if not os.path.isdir(SGLANG_PY):
        raise RuntimeError(f"找不到 sglang 源码目录：{SGLANG_PY}")
    _ensure_sys_path()

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError(
            "CPU 仿真仍需要 torch（只用 CPU tensor 承载节点 value）：pip install torch"
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
    from sglang.srt.mem_cache.heter_cache.radix_cache import HeterCacheRadixCache
    from sglang.srt.mem_cache.heter_cache.utils import (
        HeterCacheNodeTag,
        QueueValueParams,
        SessionRegistry,
        SessionType,
        VALUE_QUEUES,
    )
    from sglang.srt.mem_cache.heter_cache.occupancy import QueueOccupancyTracker
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

    _LOADED = {
        # 缓存
        "RadixCache": RadixCache,
        "HeterCacheRadixCache": HeterCacheRadixCache,
        "RadixKey": RadixKey,
        "TreeNode": TreeNode,
        "CacheInitParams": CacheInitParams,
        # 参数对象
        "InsertParams": InsertParams,
        "InsertResult": InsertResult,
        "MatchPrefixParams": MatchPrefixParams,
        "MatchResult": MatchResult,
        "EvictParams": EvictParams,
        "EvictResult": EvictResult,
        # 策略与账本
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
    """本次装载装了哪些替身（空 dict = 真实依赖齐全）。"""

    return dict(_STUB_REASONS)


def status() -> Dict[str, Any]:
    """装载状态摘要，供 ``python -m sim`` 打印。"""

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
    except Exception:  # pragma: no cover
        return "<missing>"


__all__: List[str] = ["load", "stubs", "status", "REPO_ROOT", "SGLANG_PY"]
