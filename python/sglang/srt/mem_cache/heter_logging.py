"""HeterCache 策略日志桥：把 sglang 侧的观测事件接到 Uni-Bench 的日志/账本实现。

策略代码（``evict_policy`` / ``radix_cache`` 等）只写 ``heterlog.xxx(...)`` 并传
原始数据；格式化、序列化、级别门槛全部收敛在 ``Uni-Bench/log`` 的 HeterLogger 里，
策略侧不持有文件句柄。

为什么用相对路径而不是 import 包：``Uni-Bench`` 是随仓库走的实验脚手架，不随
sglang 安装分发，只能按仓库根定位（本文件向上 4 级）。

环境变量：
    STRAT_LOG_LEVEL:        quiet/info/debug，默认 INFO。
    HETER_STRATEGY_EVENTS:  给定 JSONL 路径则开启策略事件流落盘（实验用）。

降级：``Uni-Bench/log`` 不在旁边时（单独打包/迁移），退化为命名空间 logger，
不落盘也不抛错——策略调用点因此不需要判空。
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..",
                 "Uni-Bench", "log"))

# 观测账本目录：HeterProfiler 与 HeterLogger 同构落位（都跑在 scheduler 进程内）。
_PROFILE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..",
                 "Uni-Bench", "profile"))

if os.path.isdir(_LOG_DIR):
    if _LOG_DIR not in sys.path:
        sys.path.insert(0, _LOG_DIR)
    from heter_logger import HeterLogger, heterlog  # noqa: F401, E402

    if os.environ.get("HETER_STRATEGY_EVENTS"):
        heterlog.start_capture(os.environ["HETER_STRATEGY_EVENTS"])
    # HeterProfiler：SimpleHybridStrategy 的观测账本与渲染器（Uni-Bench/profile）。
    if os.path.isdir(_PROFILE_DIR):
        if _PROFILE_DIR not in sys.path:
            sys.path.insert(0, _PROFILE_DIR)
        from heter_profiler import HeterProfiler  # noqa: F401, E402
        heterprofiler = HeterProfiler()
else:
    # 兜底路径：Uni-Bench/log 不存在（单独打包/迁移）时退化为普通 logger。
    _log = logging.getLogger("hetercache.strategy")
    _log.warning(
        "Uni-Bench/log not found at %s; HeterLogger degraded "
        "(text log only, no JSONL capture).", _LOG_DIR)

    class _FallbackHeterLogger:
        """与 HeterLogger 同签名的降级实现：文本日志照打，事件流静默丢弃。

        用 ``__getattr__`` 兜住所有事件方法，这样上游新增事件类型时这里不用跟改；
        丢弃而非报错，是因为观测失败不该影响淘汰主流程。
        """

        @property
        def enabled(self) -> bool:
            return _log.isEnabledFor(logging.DEBUG)

        def set_level(self, level: str) -> None:
            _log.setLevel(str(level).upper())

        def start_capture(self, path: str) -> None:
            pass

        def end_capture(self) -> None:
            pass

        def __getattr__(self, name):
            def _any(*args, **kwargs):
                _log.debug("[%s] %s", name, args)

            return _any

    heterlog = _FallbackHeterLogger()  # noqa: F811
