"""按权重的跨队列淘汰决策。

HeterCache 把 KV 池按业务类型切成三个队列（struct / agentic / chat）。配额机制
（caps / reservations / 充水 target / deadzone / chunk）已整体删除：淘汰顺序现在
只由一个量决定——**归一化占用** ``occ_q / w_q``。``w_q`` 越大，该队列越晚被削。

两层结构保持不变：
1. Stage 0：one-shot 带（含按类型老化的节点）无条件先清，不计入权重账本；
2. Stage 1：按 ``occ_q / w_q`` 从大到小逐队取 victim；某队弃权就跳下一队，全部
   弃权时用策略私有的伪随机兜底。

本模块只做纯计算（排序 + 决策），不碰树、不读时钟；队内选点仍由
``victim_override`` 的改写器负责。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.heter_utils import WEIGHT_QUEUES


def queue_pressure(occ: Dict[str, float], weights: Dict[str, float]) -> List[str]:
    """按 ``occ_q / w_q`` 降序返回需要优先削减的队列。

    用比值而不是 ``occ_q - w_q * C``：权重是比例语义，池容量不参与排序，
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

    进入可调度集合的结构化节点满足 share >= share_min（更低复用度已被 Stage 0 过滤），
    因此只需划分两档保护梯队：
      - Tier 0（低度复用）：share == share_min，仅达到准入门槛，置换成本最低，优先逐出；
      - Tier 1（深度复用）：share > share_min，已被多次跨会话复用，置换代价极高，延迟淘汰。

    同阶梯内按 last_access_time 升序排列（最久未访问者优先），保持段内 LRU 语义。

    Args:
        node: 目标缓存节点。
        share_of: 读取节点共享会话数的函数指针。
        share_min: 模板判定的最小共享数阈值。

    Returns:
        可直接用于 min() 比较的排序元组 (tier, last_access_time)。
    """
    tier = 0 if share_of(node) <= share_min else 1
    return (tier, node.last_access_time)