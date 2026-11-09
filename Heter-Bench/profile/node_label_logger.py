"""节点级标签快照：把整棵 radix 树的逐节点标签周期性追加到 JSONL。

从 ``RadixCache`` 抽出的观测组件。职责单一——只读遍历、不改树、不参与淘汰：

* 每 ``interval`` 秒遍历一次树，把每个非 evicted 节点的一行记录追加到文件；
* 空树不落盘（某些构造路径会额外建一个空的 RadixCache，写进去会污染快照）；
* 采样失败只记日志，不影响主流程。

与 ``[HeterCache profile]`` 的分工：那张表只给聚合占比，这里给**逐节点**的
(标签, 真实归属 session_id, token 数, 命中次数, 距上次访问时间, 子节点数,
是否 decode 段, 绝对起始偏移)。数据消费方是 Heter-Bench 的节点级标注准确率
统计（``evaluate_node_labels.py`` / ``evaluate_node_count.py``）。

本模块属于实验脚手架，**不随 sglang 安装分发**：由
``sglang/srt/mem_cache/heter_logging.py`` 按仓库根定位后加载；Heter-Bench
不在场时该桥退化为空实现。

环境变量：
    SGLANG_NODE_LABEL_DUMP:     给定 JSONL 路径则开启，空表示不开启（默认）。
    SGLANG_NODE_LABEL_INTERVAL: 采样间隔（秒），默认 30。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

from heter_logger import heterlog

from sglang.srt.mem_cache.heter_utils import SessionType

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查，避免与缓存模块成环
    from sglang.srt.mem_cache.radix_cache import RadixCache

logger = logging.getLogger(__name__)

ENV_PATH = "SGLANG_NODE_LABEL_DUMP"
ENV_INTERVAL = "SGLANG_NODE_LABEL_INTERVAL"
DEFAULT_INTERVAL_S = 30.0


class NodeLabelLogger:
    """把一棵 radix 树的逐节点标签周期性落盘为 JSONL。

    Args:
        cache: 被观测的缓存；只读它的 ``root_node`` / ``session_registry`` /
            ``eviction_policy``，不调用任何会改树的方法。
        path: 输出文件路径，追加写。
        interval: 采样间隔（秒）。
    """

    def __init__(
        self,
        cache: "RadixCache",
        path: str,
        interval: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self._cache = cache
        self.path = path
        self.interval = interval
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """起一个 daemon 线程周期性采样；重复调用不会起第二个线程。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="node-label-dumper",
        )
        self._thread.start()
        heterlog.info(
            "NodeLabel",
            f"dump -> {self.path} every {self.interval:.0f}s "
            f"(pid={os.getpid()} policy={self._cache.eviction_policy})",
        )

    def _loop(self) -> None:
        """后台循环：异常只记日志，采样失败不能影响淘汰主流程。"""
        tick = 0
        while True:
            time.sleep(self.interval)
            tick += 1
            try:
                dumped = self.dump_once(tick)
                heterlog.info("NodeLabel", f"tick={tick} dumped {dumped} nodes")
            except Exception:  # noqa: BLE001 - 采样失败不影响主流程
                logger.exception("node label dump failed")

    def dump_once(self, tick: int) -> int:
        """立即采一次快照并追加写入，返回写入的节点数。

        空树返回 0 且不落盘：否则「刚刚新建、还没插入任何东西」的缓存会留下
        一行 ``nodes: []``，离线统计会把它当成一次有效采样。

        Args:
            tick: 采样序号，原样写进记录，便于与日志行对齐。
        """
        cache = self._cache
        now_mono = time.monotonic()
        rows = []
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
                session_state = cache.session_registry.get(child.session_id)
                rows.append(
                    {
                        # 队列标签和会话分类分别记录，避免把分叉角色当成会话真值。
                        "tag": getattr(child.queue_tag, "value", str(child.queue_tag)),
                        "stype": (
                            session_state.session_type.value
                            if session_state is not None
                            else SessionType.TRANS.value
                        ),
                        # 真实归属：由插入该片段的请求携带，前缀即负载类别
                        "sid": child.session_id,
                        "tok": len(child.key),
                        "hits": child.hit_count,
                        "dt": round(now_mono - child.last_access_time, 3),
                        "nch": len(child.children),
                        "dec": bool(child.is_decode),
                        "abs": child.abs_start,
                    }
                )
        if not rows:
            return 0
        record = {
            "ts": round(time.time(), 3),
            "tick": tick,
            "policy": cache.eviction_policy,
            "nodes": rows,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(rows)


def start_from_env(cache: "RadixCache") -> Optional[NodeLabelLogger]:
    """按环境变量装配并启动；未配置时返回 None（零开销）。

    Args:
        cache: 被观测的缓存。

    Returns:
        已启动的 logger；``SGLANG_NODE_LABEL_DUMP`` 为空时返回 None。
    """
    path = os.environ.get(ENV_PATH, "")
    if not path:
        return None
    interval = float(os.environ.get(ENV_INTERVAL, str(DEFAULT_INTERVAL_S)))
    label_logger = NodeLabelLogger(cache, path, interval)
    label_logger.start()
    return label_logger
