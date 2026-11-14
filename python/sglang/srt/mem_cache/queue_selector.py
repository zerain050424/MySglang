from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.heter_utils import WEIGHT_QUEUES


def queue_pressure(occ: Dict[str, float], weights: Dict[str, float]) -> List[str]:
    """按occ_q / w_q降序返回需要优先削减的队列。

    用比值而不是occ_q - w_q * C：权重是比例语义，池容量不参与排序，
    队列大小量级不同时也不会被绝对值主导。占用为 0 的队列不会进入候选。

    Args:
        occ: 各队列可调度占用（Token，来自 QueueOccupancyTracker，只读）。
        weights: ``{queue: w}``，键必须覆盖 ``WEIGHT_QUEUES``。

    Returns:
        队列名列表；压力相同保持 ``WEIGHT_QUEUES`` 的声明顺序（确定性）。
    """
    ranked = [
        (occ.get(q, 0.0) / weights[q], q)
        for q in WEIGHT_QUEUES
        if occ.get(q, 0.0) > 0.0
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [q for _, q in ranked]


def pick_victim(*, strategy, leaves):
    """两级淘汰决策：Stage 0 one-shot → Stage 1 权重队列 → 兜底。

    Args:
        strategy: HeterCacheStrategy，提供 one-shot 判据、权重排序与队内选点。
        leaves: 当前可驱逐叶子节点集合。

    Returns:
        ``(victim, source)``；source ∈ {"one_shot", "weight:<queue>",
        "weight:fallback"}。无候选时返回 ``(None, None)``。
    """
    one_shot = [n for n in leaves if strategy.is_one_shot(n)]
    if one_shot:
        return min(one_shot, key=strategy.one_shot_key), "one_shot"
    if not leaves:
        return None, None

    for queue in strategy.pressure_ranking():
        victim = strategy.queue_victim(queue, leaves)
        if victim is not None:
            return victim, f"weight:{queue}"
    return strategy.fallback(leaves), "weight:fallback"


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