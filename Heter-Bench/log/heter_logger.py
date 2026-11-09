"""HeterLogger: HeterCache 策略专用日志与打点组件。

【模块定位】
统一管理 HeterCache 策略层（evictor / block tracker / wa_helper 等）的日志输出与事件追踪，
解决传统 print/logger 导致的代码侵入性高、日志格式不统一以及底层框架日志刷屏的问题。

【核心设计】
1. 业务逻辑解耦：
   - 策略决策代码只需单行调用高层 API（例如 `heterlog.evict(...)`、`heterlog.mark(...)`）并传入原生变量。
   - 字符串拼接、字段命名规范、日志门槛判断和 JSON 序列化统一在此处执行。
2. 双通道输出支持：
   - 文本日志：用于控制台实时观察和关键词 grep 过滤。
   - JSONL 事件流（可选）：以结构化数据落盘，便于后续离线分析、评测矩阵复现与可视化绘图。

【配置说明】
    - 环境变量: STRAT_LOG_LEVEL
      * 设为 `DEBUG`：打印详细的策略决策过程与细节打点。
      * 设为 `INFO`（默认值）：保持终端清爽，避免高频输出影响性能与观测。
"""

import json
import logging
import os
import sys
import threading


class HeterLogger:
    """HeterCache 策略日志门面（单实例 heterlog）。

    红线约定：本类只做日志，绝不承载任何业务判断/状态推进。
    所有“是否该记日志”的策略语义判断保留在调用点。
    """

    NAMESPACE = "hetercache.strategy"

    # 级别名 → logging 级别。直接用 setLevel("debug") 会抛 ValueError：
    # 标准库只认大写或整数，而文档与实验脚本习惯写小写，故在这里统一折算。
    _LEVELS = {
        "quiet": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "trace": logging.DEBUG,
    }

    def __init__(self):
        self._log = logging.getLogger(self.NAMESPACE)
        self._log.propagate = False
        if not self._log.handlers:
            _h = logging.StreamHandler(sys.stdout)
            _h.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)-7s %(message)s"))
            self._log.addHandler(_h)
        self.set_level(os.environ.get("STRAT_LOG_LEVEL", "info"))
        self._fp = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 开关/状态
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """是否处于 DEBUG 级别（供高频路径在进入循环前短路）。"""
        return self._log.isEnabledFor(logging.DEBUG)

    @property
    def capturing(self) -> bool:
        """是否正在往 JSONL 落事件流（与文本日志级别无关）。

        与 ``enabled`` 是两个出口：调试级别影响文本行，本属性影响结构化事件。
        调用方需要任一出口时用 ``capturing or enabled``，两者都不满足时应直接
        返回，不要白构造一份会被 ``_record`` 丢弃的字典。
        """
        return self._fp is not None

    def set_level(self, level) -> None:
        """quiet/info/debug/trace（大小写不敏感，也接受 logging 级别整数）。

        Raises:
            ValueError: 未知级别名。启动即失败，好过静默停在默认级别。
        """
        if isinstance(level, int):
            self._log.setLevel(level)
            return
        name = str(level).strip().lower()
        if name not in self._LEVELS:
            raise ValueError(
                f"unknown STRAT_LOG_LEVEL {level!r}; "
                f"expected one of {sorted(self._LEVELS)}"
            )
        self._log.setLevel(self._LEVELS[name])

    # ------------------------------------------------------------------
    # JSONL 事件捕获
    # ------------------------------------------------------------------
    def start_capture(self, path: str) -> None:
        """开启 JSONL 事件流落盘；重复调用会先关闭旧的。"""
        with self._lock:
            self.end_capture()
            self._fp = open(path, "a", encoding="utf-8")

    def end_capture(self) -> None:
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def _record(self, payload: dict) -> None:
        if self._fp is None:
            return
        with self._lock:
            self._fp.write(
                json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._fp.flush()

    # ------------------------------------------------------------------
    # 结构化事件 API（文本 debug + JSONL 同源双写）
    # ------------------------------------------------------------------
    def evict(self, node_id, queue, tokens, tag=None, source=None,
              one_shot=None, step=None):
        """一次驱逐事件（sglang-radix 节点语义：node_id/queue/tokens）。"""
        self._log.debug(
            "[Evict] node=%s queue=%s tokens=%s tag=%s source=%s one_shot=%s step=%s",
            node_id, queue, tokens, tag, source, one_shot, step)
        self._record({
            "ev": "evict",
            "node": node_id,
            "queue": queue,
            "tokens": tokens,
            "tag": str(tag) if tag is not None else None,
            "source": source,
            "one_shot": one_shot,
            "step": step,
        })

    def hit(self, queue=None, tag=None, hit_tokens=0, req=None):
        """一次前缀命中事件"""
        self._log.debug("[Hit] queue=%s tag=%s tokens=%s req=%s",
                        queue, tag, hit_tokens, req)
        self._record({
            "ev": "hit",
            "queue": queue,
            "tag": str(tag) if tag is not None else None,
            "tokens": hit_tokens,
            "req": req,
        })

    def victim_source(self, source, tokens):
        """本次淘汰的裁决来源记账（record_victim_source 收口）。"""
        self._log.debug("[VictimSource] source=%s tokens=%s", source, tokens)
        self._record({
            "ev": "victim_source",
            "source": source,
            "tokens": tokens,
        })

    # ------------------------------------------------------------------
    # 通用透传（未形成专属事件前的过渡出口；调用点自行控制频率）
    # ------------------------------------------------------------------
    def debug(self, tag, msg):
        if self._log.isEnabledFor(logging.DEBUG):
            self._log.debug("[%s] %s", tag, msg)
        self._record({"ev": "debug", "tag": tag, "msg": msg})

    def info(self, tag, msg):
        self._log.info("[%s] %s", tag, msg)
        self._record({"ev": "info", "tag": tag, "msg": msg})

    def warning(self, tag, msg):
        self._log.warning("[%s] %s", tag, msg)
        self._record({"ev": "warning", "tag": tag, "msg": msg})

    def error(self, tag, msg):
        self._log.error("[%s] %s", tag, msg)
        self._record({"ev": "error", "tag": tag, "msg": msg})


heterlog = HeterLogger()
