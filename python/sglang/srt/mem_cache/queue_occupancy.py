"""配额账本：各队列可调度占用的唯一写入口。

只做状态，不含策略、不碰树、不读时钟：

* 写入口两条——``insert`` / ``evict``（节点级增量）；
* 加上一条全量校准 ``reconcile``（吸收标签变更、one-shot 晋升等增量漂移）；
* 读侧给 ``occ``（活引用，调用方约定只读）与 ``snapshot``（副本，供渲染）。

为什么单独成对象：占用是唯一被"记账方"与"决策方"同时需要的状态。放在策略里，
策略就同时扮演账本与决策器；放在缓存层，又会让驱逐循环直接摸账本内部。抽出来
之后，谁写（缓存层）、谁读（配额仲裁）在类型层面就分开了。
"""

from __future__ import annotations

from typing import Dict

from sglang.srt.mem_cache.heter_utils import QUOTA_QUEUES


class QueueOccupancyTracker:
    """三队列配额账本。

    口径：只计"可调度"内容——one-shot 节点在写入方就已被排除，因此这里
    不再判断 one-shot；``reconcile`` 用全树快照的 ``total - one_shot`` 对齐。
    """

    __slots__ = ("_occ", "_capacity")

    def __init__(self) -> None:
        self._occ: Dict[str, float] = {q: 0.0 for q in QUOTA_QUEUES}
        self._capacity = 0.0

    @property
    def occ(self) -> Dict[str, float]:
        """活引用，供热路径只读遍历；调用方不得修改。"""
        return self._occ

    @property
    def capacity(self) -> float:
        """最近一次校准得到的池容量（Token）。"""
        return self._capacity

    def snapshot(self) -> Dict[str, float]:
        """副本，供渲染或跨线程使用。"""
        return dict(self._occ)

    def insert(self, queue: str, tokens: int) -> None:
        """新节点挂载时增量计入。

        Args:
            queue: 队列名；不在配额三队列内时忽略（random 不参与配额）。
            tokens: 该节点的 token 数。
        """
        if queue in self._occ:
            self._occ[queue] += tokens

    def evict(self, queue: str, tokens: int) -> None:
        """节点逐出时增量扣减，下界为 0。

        必须早于同一轮淘汰循环里的下一次 ``_pick_victim``：仲裁按这份占用选队列，
        扣减晚了就会按虚高的水位点名，多削一轮。
        """
        if queue in self._occ:
            self._occ[queue] = max(0.0, self._occ[queue] - tokens)

    def reconcile(self, snapshot: dict, capacity: float) -> None:
        """用全树快照校准，吸收增量维护跟不上的漂移。

        Args:
            snapshot: ``{queue: {"total": t, "one_shot": o}}``，来自全树遍历。
            capacity: 本次拿到的池容量（Token）；拿不到时传 0，容量保持不变。
        """
        for q in QUOTA_QUEUES:
            rec = snapshot.get(q) or {}
            total = float(rec.get("total", 0.0))
            one_shot = float(rec.get("one_shot", 0.0))
            self._occ[q] = max(0.0, total - one_shot)
        if capacity > 0:
            self._capacity = float(capacity)

    def reset(self) -> None:
        """清空账本（缓存 flush / 跨 run 复用同一进程时调用）。"""
        for q in QUOTA_QUEUES:
            self._occ[q] = 0.0
