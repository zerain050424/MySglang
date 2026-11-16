"""sglang ↔ Heter-Bench 的装配接缝：定位并导入观测实现。

本模块**只做装配**，不含任何观测逻辑。所有观测实现都在 ``Heter-Bench``：

    Heter-Bench/log/heter_logger.py          文本日志 + JSONL 事件流
    Heter-Bench/profile/heter_profiler.py    账本与 profile 渲染
    Heter-Bench/profile/cache_telemetry.py   遥测门面（多写收口、时钟、去重）
    Heter-Bench/profile/node_label_logger.py 节点级标签快照

按仓库根定位而不是 import 包：``Heter-Bench`` 是随仓库走的实验脚手架，不随
sglang 安装分发，只能从本文件向上 4 级找。

**缺席即报错。** 这里刻意不做降级实现：静默空转会让「profile 没数据」和
「根本没采集」看起来一样，排查时误导性极强。Heter-Bench 不在场时，服务启动阶段
就带着期望路径失败，问题第一时间暴露。

环境变量：
    STRAT_LOG_LEVEL:        quiet/info/debug/trace，不分大小写，默认 info。
    HETER_STRATEGY_EVENTS:  给定 JSONL 路径则开启结构化事件流落盘。
    HETER_REGRET:           置 1 则开启淘汰后悔率账本（默认关：它要算内容键、
                            存坟场，是跑对照实验时才需要的开销）。
    HETER_REGRET_OUT:       给定 JSON 路径则把 regret 报告原子落盘。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
_LOG_DIR = os.path.join(_ROOT, "Heter-Bench", "log")
_PROFILE_DIR = os.path.join(_ROOT, "Heter-Bench", "profile")


def _add_path(path: str) -> None:
    missing = not os.path.isdir(path)
    if missing:
        raise ImportError(
            f"HeterCache 观测实现缺失：{path} 不存在。\n"
            "观测代码随仓库分发在 Heter-Bench/ 下，HeterCache 策略依赖它。"
            "若单独打包或迁移 sglang，请一并带上该目录。"
        )
    if path not in sys.path:
        sys.path.insert(0, path)


_add_path(_LOG_DIR)
_add_path(_PROFILE_DIR)

# 路径就位后的真实导入；顺序即依赖顺序，勿随意调整。
from cache_telemetry import CacheTelemetry  # noqa: E402
from heter_logger import heterlog  # noqa: E402
from heter_profiler import HeterProfiler  # noqa: E402
from node_label_logger import NodeLabelLogger  # noqa: E402
from node_label_logger import start_from_env as start_node_label_logger  # noqa: E402
from regret_ledger import (  # noqa: E402
    RegretLedger,
    content_key,
    namespace_key,
    queue_horizons_from_policy,
)

# 事件流必须在导入期开启：晚于此刻产生的事件不会被捕获。
if os.environ.get("HETER_STRATEGY_EVENTS"):
    heterlog.start_capture(os.environ["HETER_STRATEGY_EVENTS"])

_TRUTHY = {"1", "true", "yes", "on"}


def regret_from_env(now=None):
    """按环境变量装配 regret 账本；未开启时返回 ``None``（调用方靠它短路）。

    Args:
        now: 时间源；必须与缓存层同源，否则 stamp 的年龄与 H 不同量级。

    Returns:
        ``RegretLedger`` 或 ``None``。
    """
    flag = os.environ.get("HETER_REGRET", "").strip().lower()
    if flag not in _TRUTHY:
        return None
    return RegretLedger(
        now=now,
        queue_horizon=queue_horizons_from_policy(),
        out_path=os.environ.get("HETER_REGRET_OUT") or None,
    )
