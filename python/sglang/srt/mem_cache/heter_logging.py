"""sglang ↔ Heter-Bench 的观测接缝：定位、导入、降级。

本模块**只做装配**，不含任何观测逻辑。所有观测实现都在 ``Heter-Bench``：

    Heter-Bench/log/heter_logger.py         文本日志 + JSONL 事件流
    Heter-Bench/profile/heter_profiler.py   账本与 profile 渲染
    Heter-Bench/profile/cache_telemetry.py  遥测门面（多写收口、时钟、去重）
    Heter-Bench/profile/node_label_logger.py 节点级标签快照

为什么按仓库根定位而不是 import 包：``Heter-Bench`` 是随仓库走的实验脚手架，
不随 sglang 安装分发，只能从本文件向上 4 级找。

降级（Heter-Bench 不在旁边，例如单独打包或迁移）：下面三个名字仍可用，但都是
空实现——``heterlog`` 退化为命名空间 logger，``HeterProfiler`` 变成不累计的
null 对象，``CacheTelemetry`` 与 ``NodeLabelLogger`` 变成同签名空壳。策略调用点
因此不需要判空，也不会因为缺少实验脚手架而导入失败。

环境变量：
    STRAT_LOG_LEVEL:        quiet/info/debug/trace，不分大小写，默认 info。
    HETER_STRATEGY_EVENTS:  给定 JSONL 路径则开启结构化事件流落盘。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
_LOG_DIR = os.path.join(_ROOT, "Heter-Bench", "log")
_PROFILE_DIR = os.path.join(_ROOT, "Heter-Bench", "profile")


def _add_path(path: str) -> bool:
    """目录存在则加入 sys.path；返回是否可用。"""
    if not os.path.isdir(path):
        return False
    if path not in sys.path:
        sys.path.insert(0, path)
    return True


_LOG_READY = _add_path(_LOG_DIR)
_PROFILE_READY = _add_path(_PROFILE_DIR)

if _LOG_READY:
    from heter_logger import HeterLogger, heterlog  # noqa: F401, E402

    if os.environ.get("HETER_STRATEGY_EVENTS"):
        heterlog.start_capture(os.environ["HETER_STRATEGY_EVENTS"])
else:
    # 兜底：命名空间 logger，不落盘也不抛错。
    _log = logging.getLogger("hetercache.strategy")
    _log.warning(
        "Heter-Bench/log not found at %s; HeterLogger degraded "
        "(text log only, no JSONL capture).",
        _LOG_DIR,
    )

    class _FallbackHeterLogger:
        """与 HeterLogger 同签名的降级实现：文本日志照打，事件流静默丢弃。

        用 ``__getattr__`` 兜住所有事件方法，这样上游新增事件类型时这里不用跟改；
        丢弃而非报错，是因为观测失败不该影响淘汰主流程。
        """

        @property
        def enabled(self) -> bool:
            return _log.isEnabledFor(logging.DEBUG)

        @property
        def capturing(self) -> bool:
            """降级态永远不会捕获事件流。

            必须显式定义：本类的 ``__getattr__`` 会把任何未定义属性变成一个
            可调用对象（真值），不写这一行的话 ``heterlog.capturing`` 会假报
            "正在捕获"，让调用方白构造一份被丢弃的事件。
            """
            return False

        def set_level(self, level) -> None:
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


if _PROFILE_READY:
    from cache_telemetry import CacheTelemetry  # noqa: F401, E402
    from heter_profiler import HeterProfiler  # noqa: F401, E402
    from node_label_logger import (  # noqa: F401, E402
        NodeLabelLogger,
        start_from_env as start_node_label_logger,
    )
else:
    # 兜底：同签名空实现，保证策略与缓存层的导入路径永远成立。
    _log.warning(
        "Heter-Bench/profile not found at %s; profiler and telemetry degraded.",
        _PROFILE_DIR,
    )

    class HeterProfiler:  # noqa: D101 - 与真实实现同签名
        def record_hit(self, queue, hit_tokens) -> None:
            pass

        def record_evict(self, queue, tokens, one_shot) -> None:
            pass

        def record_victim_source(self, source, tokens) -> None:
            pass

        def summary(self, *args, **kwargs) -> str:
            return ""

        def reset(self) -> None:
            pass

    class CacheTelemetry:  # noqa: D101 - 与真实实现同签名
        PROFILE_INTERVAL = 30.0

        def __init__(self, profiler) -> None:
            self.profiler = profiler

        def on_hit(self, tag, hit_tokens: int, req_id=None) -> None:
            pass

        def on_evict(self, node_id: int, tag, tokens: int, one_shot: bool) -> None:
            pass

        def on_victim_source(self, source: str, tokens: float) -> None:
            pass

        def wants_profile(self) -> bool:
            return False

        def observe_profile(self, *args, **kwargs) -> None:
            pass

        def reset(self) -> None:
            pass

    class NodeLabelLogger:  # noqa: D101 - 与真实实现同签名
        def __init__(self, cache, path: str, interval: float = 30.0) -> None:
            raise RuntimeError(
                "NodeLabelLogger needs Heter-Bench/profile; "
                f"missing at {_PROFILE_DIR}."
            )

    def start_node_label_logger(cache) -> Optional[object]:
        """降级态不采样：返回 None，调用方只留一个空句柄。"""
        return None
