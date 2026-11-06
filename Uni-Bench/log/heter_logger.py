# HeterLogger — HeterCache 策略调试日志唯一出口
#
# 目标（解耦契约）：
#   1. 策略代码（evictor / block tracker / wa_helper）只做一件事：
#      在决策出口调用 heterlog.evict(...) / heterlog.mark(...) 等一行式 API，
#      传入原始数据；f-string 组装、字段命名、级别门槛、JSON 序列化全部收敛在本类。
#   2. 与 vLLM 全局日志栈隔离：独立命名空间 + propagate=False，
#      开 DEBUG 不再连带打开整条 vLLM 内部日志。
#   3. 双写：文本日志（可过滤）+ 可选 JSONL 事件流（实验分析、结果矩阵复用）。
#
# 用法（策略模块中）：
#   import sys
#   _HETER_LOG_DIR = "/home/rain/MySglang/Uni-Bench/log"
#   if _HETER_LOG_DIR not in sys.path:
#       sys.path.insert(0, _HETER_LOG_DIR)
#   from heter_logger import heterlog
#
# 级别控制：
#   环境变量 STRAT_LOG_LEVEL=DEBUG 打开策略 debug；
#   默认 INFO（与改造前行为一致：默认终端无多余输出）。

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

    def __init__(self):
        self._log = logging.getLogger(self.NAMESPACE)
        self._log.propagate = False  # 关键：不冒泡到 vllm / root logger
        if not self._log.handlers:
            _h = logging.StreamHandler(sys.stdout)
            _h.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)-7s %(message)s"))
            self._log.addHandler(_h)
        self._log.setLevel(os.environ.get("STRAT_LOG_LEVEL", "INFO"))
        self._fp = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 开关/状态
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """是否处于 DEBUG 级别（供高频路径在进入循环前短路）。"""
        return self._log.isEnabledFor(logging.DEBUG)

    def set_level(self, level: str) -> None:
        """quiet/info/debug/trace -> logging 级别。"""
        self._log.setLevel(str(level).upper())

    # ------------------------------------------------------------------
    # JSONL 事件捕获（实验留存）
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
    def evict(self, node_id, workload, tokens, task_type=None, source=None,
              one_shot=None, step=None):
        """一次驱逐事件（sglang-radix 节点语义：node_id/workload/tokens）。"""
        self._log.debug(
            "[Evict] node=%s workload=%s tokens=%s task_type=%s source=%s one_shot=%s step=%s",
            node_id, workload, tokens, task_type, source, one_shot, step)
        self._record({
            "ev": "evict",
            "node": node_id,
            "workload": workload,
            "tokens": tokens,
            "task_type": str(task_type) if task_type is not None else None,
            "source": source,
            "one_shot": one_shot,
            "step": step,
        })

    def hit(self, workload=None, task_type=None, hit_tokens=0, req=None):
        """一次前缀命中事件（record_hit 收口）。"""
        self._log.debug("[Hit] workload=%s task_type=%s tokens=%s req=%s",
                        workload, task_type, hit_tokens, req)
        self._record({
            "ev": "hit",
            "workload": workload,
            "task_type": str(task_type) if task_type is not None else None,
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

    def admit(self, block_id, task_type, queue):
        """block 进入驱逐候选队列（add 路径出口）。"""
        self._log.debug("block=%s admitted: task_type=%s queue=%s",
                        block_id, task_type, queue)
        self._record({
            "ev": "admit",
            "block": block_id,
            "task_type": str(task_type),
            "queue": queue,
        })

    def hit_interval(self, task_type, interval, block_id=None, now=None,
                     last=None):
        """WA 命中区间采集事件。"""
        self._log.debug(
            "[WA Hit] task=%s interval=%.3f block=%s now=%s last=%s",
            task_type, interval, block_id, now, last)
        self._record({
            "ev": "hit",
            "task": str(task_type),
            "interval": interval,
            "block": block_id,
            "now": now,
            "last": last,
        })

    def mark(self, block_id, attr, value):
        """block 属性标记（task_type / block_type / last_hit 等）。"""
        self._log.debug("block=%s set %s=%s", block_id, attr, value)
        self._record({
            "ev": "mark",
            "block": block_id,
            "attr": attr,
            "value": str(value),
        })

    def classify_error(self, key_id, detail):
        """任务分类失败诊断（统一替代散落的裸 print）。"""
        self._log.error("task classification failed: key_id=%s detail=%s",
                        key_id, detail)
        self._record({
            "ev": "classify_error",
            "key_id": key_id,
            "detail": str(detail),
        })

    def lecar_arc_miss(self, content_hash):
        """LeCaR/ARC 缓存未命中事件。"""
        self._log.debug("LeCaR/ARC miss hash=%s", content_hash)
        self._record({
            "ev": "lecar_arc_miss",
            "hash": content_hash,
        })

    # ------------------------------------------------------------------
    # 通用透传（供 wa_helper 等还没有专属事件的模块过渡使用）
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
