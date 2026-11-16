from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.heter_cache.utils import VALUE_QUEUES


def queue_order(
    occ: Dict[str, float], value_density: Dict[str, float]
) -> List[str]:
    """按已兑现价值密度 VD 升序返回候选队列，最前的先削。

    裁决只读 VD：每单位驻留量在视野内换回多少命中，VD 小的队列就是“更无用”的队列。
    占用不出现在排序键里：把它当分子乘进去（旧的 ``occ_q/(1+VD_q)``）等于让队列
    规模替价值做决定——占用大 10 倍、密度只高 1 倍的队列会被排到后面，效果是一个
    隐式的按规模配额（已删除的 caps/reservations 换个地方又回来了）。

    VD 相同时（含冷启动三队全为 0、尚无命中证据）按占用降序兜底：同样没用时，
    先削大的——一次决策腾出更多 token。占用为 0 的队列不进入候选。

    Args:
        occ: 各队列可调度占用（Token）；只用于候选过滤与同 VD 的兜底次序。
        value_density: 各队列的已兑现价值密度（VD_q = H_q / max(occ_q, floor)）
    Returns:
        队列名列表。
    """
    ranked = [
        (
            value_density.get(q, 0.0),
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
        (victim, source);ource ∈ {"one_shot", "value:<queue>",
        "value:fallback"}。无候选时返回(None, None)。
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
    """
    tier = 0 if share_of(node) <= share_min else 1
    return (tier, node.last_access_time, node.id)