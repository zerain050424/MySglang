from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.heter_utils import VALUE_QUEUES


def queue_order(
    occ: Dict[str, float], value_density: Dict[str, float]
) -> List[str]:
    """按 ``occ_q / (1 + VD_q)`` 降序返回候选队列，最前的先削。

    比值语义：每单位已兑现价值对应的驻留量，越大越应该削。用 ``1 + VD`` 而不是
    ``VD`` 做分母有两个原因：

    * VD=0 表示"τ 内没有命中证据"，不等于"没有价值"。裸用 ``argmin VD`` 会把
      刚写入、还没来得及复用的前缀排到最前，造成写入即淘汰的抖动；这里退化为
      按占用降序（与旧的等权份额同形）。
    * 分母恒 ≥ 1，不需要为除零加特判。

    占用为 0 的队列不进入候选；同比值按占用降序，再按 ``VALUE_QUEUES`` 声明
    顺序（确定性）。

    Args:
        occ: 各队列可调度占用（Token），来自 ``QueueOccupancyTracker`` 只读投影。
        value_density: 各队列的已兑现价值密度（VD），来自 ``QueueValueTracker``。

    Returns:
        队列名列表。
    """
    ranked = [
        (
            -(occ[q] / (1.0 + value_density.get(q, 0.0))),
            -occ[q],
            index,
            q,
        )
        for index, q in enumerate(VALUE_QUEUES)
        if occ.get(q, 0.0) > 0.0
    ]
    ranked.sort()
    return [q for *_, q in ranked]


def pick_victim(*, strategy, leaves):
    """两级淘汰决策：Stage 0 one-shot → Stage 1 VD 仲裁 → 兜底。

    Args:
        strategy: HeterCacheStrategy，提供 one-shot 判据、队列排序与队内选点。
        leaves: 当前可驱逐叶子节点集合。

    Returns:
        ``(victim, source)``；source ∈ {"one_shot", "value:<queue>",
        "value:fallback"}。无候选时返回 ``(None, None)``。
    """
    one_shot = [n for n in leaves if strategy.is_one_shot(n)]
    if one_shot:
        return min(one_shot, key=strategy.one_shot_key), "one_shot"
    if not leaves:
        return None, None

    for queue in strategy.queue_order():
        victim = strategy.queue_victim(queue, leaves)
        if victim is not None:
            return victim, f"value:{queue}"
    return strategy.fallback(leaves), "value:fallback"


def struct_tier(node, share_of, share_min: int) -> Tuple[int, float]:
    """生成 STRUCT（结构化/模板）队列的淘汰优先级 Key：优先逐出低复用模板，保护高复用模板。
    Args:
        node: 目标缓存节点。
        share_of: 读取节点共享会话数的函数指针。
        share_min: 模板判定的最小共享数阈值。

    Returns:
        可直接用于 min() 比较的排序元组 (tier, last_access_time, id)。

    末尾的 ``node.id`` 是平局判据：虚拟时钟下同一 tick 写入的节点时间戳全等，
    而 ``min()`` 在完全同分时取迭代顺序（set 的 id 哈希序，跨进程不稳定）。
    """
    tier = 0 if share_of(node) <= share_min else 1
    return (tier, node.last_access_time, node.id)