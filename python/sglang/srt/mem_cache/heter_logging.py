"""HeterCache 策略日志桥：把 sglang 侧的观测事件接到 Heter-Bench 的日志/账本实现。

策略代码（``evict_policy`` / ``radix_cache`` 等）只写 ``heterlog.xxx(...)`` 并传
原始数据；格式化、序列化、级别门槛全部收敛在 ``Heter-Bench/log`` 的 HeterLogger 里，
策略侧不持有文件句柄。

为什么用相对路径而不是 import 包：``Heter-Bench`` 是随仓库走的实验脚手架，不随
sglang 安装分发，只能按仓库根定位（本文件向上 4 级）。

环境变量：
    STRAT_LOG_LEVEL:        quiet/info/debug/trace，不分大小写，默认 info。
                            决定是否输出高频文本行（``enabled``）。
    HETER_STRATEGY_EVENTS:  给定 JSONL 路径则开启结构化事件流落盘（``capturing``）。

两个出口互相独立：只想拿事件流不必开 debug，只想看文本行也不必开 capture。
``CacheTelemetry`` 按 ``capturing or enabled`` 判断是否值得构造事件。

降级：``Heter-Bench/log`` 不在旁边时（单独打包/迁移），退化为命名空间 logger，
不落盘也不抛错——策略调用点因此不需要判空。
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..",
                 "Heter-Bench", "log"))

# 观测账本目录：HeterProfiler 与 HeterLogger 同构落位（都跑在 scheduler 进程内）。
_PROFILE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..",
                 "Heter-Bench", "profile"))

if os.path.isdir(_LOG_DIR):
    if _LOG_DIR not in sys.path:
        sys.path.insert(0, _LOG_DIR)
    from heter_logger import HeterLogger, heterlog  # noqa: F401, E402

    if os.environ.get("HETER_STRATEGY_EVENTS"):
        heterlog.start_capture(os.environ["HETER_STRATEGY_EVENTS"])
    # HeterProfiler：HeterCacheStrategy 的观测账本与渲染器（Heter-Bench/profile）。
    if os.path.isdir(_PROFILE_DIR):
        if _PROFILE_DIR not in sys.path:
            sys.path.insert(0, _PROFILE_DIR)
        from heter_profiler import HeterProfiler  # noqa: F401, E402
else:
    # 兜底路径：Heter-Bench/log 不存在（单独打包/迁移）时退化为普通 logger。
    _log = logging.getLogger("hetercache.strategy")
    _log.warning(
        "Heter-Bench/log not found at %s; HeterLogger degraded "
        "(text log only, no JSONL capture).", _LOG_DIR)

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


class CacheTelemetry:
    """策略侧观测的唯一入口：把账本与事件流的多写、字段映射、开关判断收在一处。

    业务层只表达意图（发生了一次命中 / 一次逐出 / 一次裁决），不再关心
    "要不要写 profiler、要不要判开关、参数怎么拼"。上层的三个 ``record_*``
    因此各剩一行调用。

    放在本模块而不是策略模块：这里是 sglang 与 Heter-Bench 观测实现的唯一接缝，
    装配与降级都发生在这里，收敛点也应该在这里。

    开关判据是 ``capturing or enabled``，两者对应不同出口：

    * ``capturing``：事件流落盘（``HETER_STRATEGY_EVENTS``），默认关；
    * ``enabled``：文本 debug 日志（``STRAT_LOG_LEVEL=debug``），默认关。

    只判 ``enabled`` 会让开了事件流但没开 debug 的跑法丢掉全部事件；只判
    ``capturing`` 又会让纯文本排查时看不到事件行。两个都不满足时直接返回，
    避免白构造一份会被丢弃的事件字典。
    """

    __slots__ = ("profiler",)

    def __init__(self, profiler):
        """绑定账本。

        Args:
            profiler: HeterProfiler 实例；策略侧同时保留引用（外部消费者直接读它）。
        """
        self.profiler = profiler

    @staticmethod
    def _emitting() -> bool:
        """是否值得进入事件构造函数。"""
        return heterlog.capturing or heterlog.enabled

    def on_hit(self, tag, hit_tokens: int, req_id=None) -> None:
        """一次前缀命中。"""
        self.profiler.record_hit(tag.queue_name, hit_tokens)
        if self._emitting():
            heterlog.hit(
                queue=tag.queue_name,
                tag=tag.value,
                hit_tokens=hit_tokens,
                req=req_id,
            )

    def on_evict(self, node_id: int, tag, tokens: int, one_shot: bool) -> None:
        """一次节点逐出（级联删除时逐点上报）。"""
        self.profiler.record_evict(tag.queue_name, tokens, one_shot=one_shot)
        if self._emitting():
            heterlog.evict(
                node_id=node_id,
                queue=tag.queue_name,
                tag=tag.value,
                tokens=tokens,
                one_shot=one_shot,
            )

    def on_victim_source(self, source: str, tokens: float) -> None:
        """一次淘汰裁决来源。"""
        self.profiler.record_victim_source(source, tokens)
        if self._emitting():
            heterlog.victim_source(source, tokens)
