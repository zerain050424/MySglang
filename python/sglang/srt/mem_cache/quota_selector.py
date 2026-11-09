"""跨队列配额仲裁与淘汰决策算法。

HeterCache 后端将 KV Cache 资源池按业务类型划分为三个队列（struct / agentic / chat）。
本模块提供无状态的纯计算逻辑，不感知底层 Radix 树拓扑与节点数据，专注解决两个调度问题：

1. 各队列当前的有效配额目标（基于 Max-Min 充水算法与上下限截断计算 target）；
2. 本轮淘汰优先由哪个队列释放容量（超上限队列优先，其次按超出保底预留量排序）。

架构设计：
抽离为独立纯函数模块，脱离 PyTorch 运行时及全局缓存状态，便于算法边界的确定性单测。
队列内部的具体淘汰策略（如 SLRU 分级、会话轮询、保活概率）交由 victim_override 执行。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.heter_utils import QUOTA_QUEUES, QuotaParams

# 浮点比较容差，防止乘算容量比例时的舍入误差导致临界点误判。
_EPS = 1e-9


def effective_targets(
    occ: Dict[str, float],
    capacity: float,
    params: QuotaParams,
) -> Dict[str, float]:
    """基于 Max-Min 充水算法（Water-Filling）计算各队列的动态目标配额（Token）。

    基准容量必须固定为总池大小（allocator.size），而非各队列实时占用总和：
    若使用动态占用作为分母，每释放 1 个 Token 都会导致分母收缩，形成负反馈放大
    （例如上限比例为 0.5、超出 300 Token 时，实际需要淘汰 600 Token 才能达成平衡），
    会导致常规配额保护退化为过激的深度逐出。

    Args:
        occ: 各队列的可调度占用量（Token）。统计口径与 QueueOccupancyTracker 一致
            （排除 one-shot，包含 locked）。
        capacity: 缓存池总容量（Token），取自 token_to_kv_pool_allocator.size。
        params: 配额参数，内部将 caps（上限比例）和 reservations（保底比例）换算为 Token。

    Returns:
        各队列的有效配额字典 {queue: target}，键恒包含全部 QUOTA_QUEUES。
        当所有队列均触及上限时，sum(target) < capacity，剩余未分配容量属于正常空闲。

    充水分配规则：
      1. 保护未触底队列：当前占用未达到保底预留（reservation）的队列，维持其实际占用，
         不削减也不参与剩余容量的再分配；
      2. 剩余配额均分：其余活跃队列按 caps 权重均分剩余空闲容量。若分配值超过自身硬上限（cap），
         则截断至 cap，溢出的额度在剩余活跃队列中继续迭代分配（3 个队列最多迭代 3 轮收敛）；
      3. 极端欠载处理：当 Locked Token 过多导致计算出的剩余容量为负时，仍按比例缩减，
         允许 target 算得低于保底线。此状态视为全局不可释放信号，由上层调度降级处理。
    """
    total_capacity = max(float(capacity), 0.0)
    if total_capacity <= 0.0:
        return {q: 0.0 for q in QUOTA_QUEUES}
    cap = {q: params.caps[q] * total_capacity for q in QUOTA_QUEUES} ## 上限
    rho = {q: params.reservations.get(q, 0.0) * total_capacity for q in QUOTA_QUEUES} ## 下限

    fixed = [q for q in QUOTA_QUEUES if occ.get(q, 0) <= rho[q]] ## 安全队列
    active = [q for q in QUOTA_QUEUES if q not in fixed] ## 竞争队列

    target: Dict[str, float] = {q: float(occ.get(q, 0)) for q in fixed}
    residual = total_capacity - sum(target.values()) ## 先满足fixed队列的现有占用
    residual -= sum(rho[q] for q in active) ## 再满足active队列的底线配额

    while active:
        weight = sum(params.caps[q] for q in active)
        share = {q: rho[q] + residual * params.caps[q] / weight for q in active}
        clamped = [q for q in active if share[q] > cap[q] + _EPS]
        if not clamped:
            target.update(share)
            break
        for q in clamped:
            target[q] = cap[q]
            residual -= cap[q] - rho[q]
        active = [q for q in active if q not in clamped]

    return target


def over_quota_queues(
    occ: Dict[str, float],
    targets: Dict[str, float],
    params: QuotaParams,
    chunk: float,
) -> List[str]:
    """筛选超出配额目标的队列，按超额绝对值降序返回。

    设置双重门限过滤，防止在阈值临界点频繁颠簸（Thrashing）：
      - 相对超额门限（deadzone）：设置死区缓冲，过滤刚好压线的浮点波动；
      - 绝对超额门限（chunk）：设置最小削减批次，避免每次只驱逐极少 Token 的微碎抖动
        （在小页尺寸或单 Token 实验场景下尤为关键）。

    Args:
        occ: 各队列实时可调度占用（Token）。
        targets: effective_targets 计算得出的目标配额。
        params: 配额参数，读取 deadzone 容差。
        chunk: 单次淘汰最小批次（Token），由 resolve_chunk 确定。

    Returns:
        超出配额且突破死区的队列名列表，按 (occ - target) 差值降序排列；无则返回空列表。
    """
    over = []
    for q in QUOTA_QUEUES:
        used = float(occ.get(q, 0))
        target = targets[q]
        if used - target < chunk:
            continue
        if used <= target * (1.0 + params.deadzone) + _EPS:
            continue
        over.append(q)
    over.sort(key=lambda q: occ.get(q, 0) - targets[q], reverse=True)
    return over


def queue_ranking(
    occ: Dict[str, float],
    targets: Dict[str, float],
    params: QuotaParams,
    chunk: float,
    capacity: float,
) -> List[str]:
    """生成本轮淘汰候选队列的仲裁优先级序列，供逐级尝试。

    采用两阶段优先级倾斜机制：
      1. 超上限队列优先：严格约束超限，按超额量绝对值降序排列（强硬执行阶段）；
      2. 弹性借用队列次之：未超上限但超出保底预留量的队列，按超出预留线的富余量降序排列。

    若返回空列表，表明当前没有任何队列高于保底线：
    说明当前资源受困于 Locked 或 One-shot 占用，或者预留参数配置过大，
    此时配额层不予强行干预，交由底层兜底逻辑接管。

    Args:
        occ: 各队列实时可调度占用（Token）。
        targets: effective_targets 计算得出的目标配额。
        params: 配额参数，读取 reservations 配置。
        chunk: 单次淘汰最小批次（Token）。
        capacity: 缓存池总容量（Token），用于将预留比例换算为绝对值。

    Returns:
        排序后的队列名列表（无重复）；无候选时返回空列表。
    """
    ranking = over_quota_queues(occ, targets, params, chunk)
    elastic = [
        q
        for q in QUOTA_QUEUES
        if q not in ranking
        and occ.get(q, 0) > params.reservations.get(q, 0.0) * max(float(capacity), 0.0)
    ]
    elastic.sort(
        key=lambda q: occ.get(q, 0)
        - params.reservations.get(q, 0.0) * max(float(capacity), 0.0),
        reverse=True,
    )
    return ranking + elastic


def queue_capacities(params: QuotaParams, capacity: float) -> Dict[str, float]:
    """将各队列的上限比例（caps）转换为实际 Token 容量绝对值，用于观测统计。"""
    return {q: params.caps[q] * max(float(capacity), 0.0) for q in QUOTA_QUEUES}


def resolve_chunk(page_size: int, params: QuotaParams) -> int:
    """计算单次淘汰的对齐粒度：取 max(page_size, chunk_tokens)。

    设计原因：
    Page 是底层的物理分配最小单元。若削减步长小于单个 Page，实际回收仍会释放整页，
    导致算法计算的滞后门限（Deadzone）与底层实际释放的物理内存量产生偏差。
    """
    return max(int(page_size or 1), int(params.chunk_tokens))


def pick_victim(
    *,
    strategy,
    leaves,
    occ: dict,
    capacity: float,
    page_size: int,
):
    """两级淘汰决策主入口（仅执行裁决，不产生实际内存释放或树结构修改）。

    执行阶段：
      - Stage 0（无复用价值清理）：One-shot 临时节点没有跨请求复用收益，走内部
        分段 LRU 无条件淘汰，不计入配额开销；
      - Stage 1（配额仲裁）：按 queue_ranking 的优先级逐队找候选；某队列全被锁定、
        均为 one-shot 或队内策略弃权就跳下一队；所有队列都无候选时用策略私有
        伪随机兜底，保证淘汰指针单调步进。

    没有“未开配额”的降级路径：配额必填，候选只能由队列仲裁产生。

    Args:
        strategy: HeterCacheStrategy 策略实例，提供判决接口。
        leaves: 当前可驱逐的叶子节点集合。
        occ: 配额口径的各队列占用（来自 QueueOccupancyTracker，只读）。
        capacity: 缓存池总容量（Token），用于配额计算。
        page_size: 内存页大小，用于计算对齐粒度 chunk。

    Returns:
        二元组 (victim, source)。source 标识决策来源，记录于监控打点：
        "one_shot" / "quota:<queue>" / "quota:fallback"。
        无候选可淘汰时返回 (None, None)，驱动上层结束回收循环。
    """
    one_shot = [n for n in leaves if strategy.is_one_shot(n)]
    if one_shot:
        return min(one_shot, key=strategy.one_shot_key), "one_shot"
    if not leaves:
        return None, None

    for queue in strategy.quota_ranking(occ, capacity, page_size):
        victim = strategy.quota_victim(queue, leaves, occ, capacity)
        if victim is not None:
            return victim, f"quota:{queue}"
    return strategy.quota_fallback(leaves), "quota:fallback"


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