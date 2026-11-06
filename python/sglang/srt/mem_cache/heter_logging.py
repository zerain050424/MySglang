# HeterCache 策略日志桥（相对路径引用）
#
# 定位：策略模块（evict_policy / radix_cache 等）一律
#   from sglang.srt.mem_cache.heter_logging import heterlog
# 由本模块负责解析 Uni-Bench/log/HeterLogger 的相对路径并导出单例。
# 策略代码只传原始数据；格式化/序列化/级别门槛全部收敛在 HeterLogger 内。
#
# 环境变量：
#   STRAT_LOG_LEVEL       : quiet/info/debug（默认 INFO，行为与改造前一致）
#   HETER_STRATEGY_EVENTS : 指定 JSONL 路径则开启策略事件流落盘（用于实验）
#
# 路径：本文件位于 python/sglang/srt/mem_cache/，
#       向上 4 级到仓库根，再进 Uni-Bench/log。
from __future__ import annotations

import logging
import os
import sys

_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..",
                 "Uni-Bench", "log"))

if os.path.isdir(_LOG_DIR):
    if _LOG_DIR not in sys.path:
        sys.path.insert(0, _LOG_DIR)
    from heter_logger import HeterLogger, heterlog  # noqa: F401, E402

    if os.environ.get("HETER_STRATEGY_EVENTS"):
        heterlog.start_capture(os.environ["HETER_STRATEGY_EVENTS"])
else:
    # 兜底：mem_cache 被单独打包/迁移、Uni-Bench/log 不在旁边时，
    # 退化为普通命名空间 logger（不落盘、不异常，策略调用点无须判空）。
    _log = logging.getLogger("hetercache.strategy")
    _log.warning(
        "Uni-Bench/log not found at %s; HeterLogger degraded "
        "(text log only, no JSONL capture).", _LOG_DIR)

    class _FallbackHeterLogger:
        """与 HeterLogger 同签名降级实现：文本日志兜底，事件流静默丢弃。"""

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
