"""跨队列配额的仲裁数学。

HeterCache 在 ``simple_hybrid`` 后端上按队列（struct / agentic / chat）分配 KV 池。
本模块只回答两个问题，不碰树、不碰节点内容：

1. 各队列此刻的**有效目标** ``target`` 是多少（max-min 水填充 + 上下限夹紧）；
2. 这一轮该**点名哪个队列**承压（超上限优先，其次弹性区按超出下限最多）。

之所以独立成模块而不是塞进策略类：这套计算是纯函数，可以脱离 torch 与缓存状态
单测（见 ``test/manual/test_quota_selector.py``）。

队内排序（SLRU 分档 / 会话轮询 / 生存概率）不在这里，属于 ``victim_override``。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.uni_utils import QUOTA_QUEUES, QuotaParams

# 额度是浮点比例，比较时留容差，避免 0.5*C 的舍入把"刚好达标"误判成超标。
_EPS = 1e-9


def effective_targets(
    occ: Dict[str, float],
    capacity: float,
    params: QuotaParams,
) -> Dict[str, float]:
    """用 max-min 水填充算出各队列此刻的有效目标（单位 token）。

    分母恒为**池容量** ``capacity``（``allocator.size``），不是当前占用之和：
    每淘汰 1 token 分母也会同步少 1，会把"降到目标"实际需要的淘汰量放大
    1/(1-k) 倍（k=0.5、超出 300 时实际要淘汰 600），使上限型配额退化成激进清理。

    Args:
        occ: 各队列的可调度占用（token）。口径为「排除 one-shot、包含 locked」，
            与 ``SimpleHybridStrategy._occ`` 一致。
        capacity: 池容量（token），采样自 ``token_to_kv_pool_allocator.size``。
        params: 配额参数，``caps`` / ``reservations`` 为比例，函数内换算成 token。

    Returns:
        ``{queue: target}``，键恒为 ``QUOTA_QUEUES`` 全集。
        全部队列触顶时 ``sum(target) < capacity``：空闲容量无人认领，属正常结果。

    Raises:
        ValueError: 配额未启用（``caps`` 为空）。调用方应先判 ``params.enabled``，
            否则说明接线错误，静默返回 0 会把问题藏起来。

    水填充的三条规则：
      1. 占用不高于下限的队列判为"未触底"，保住现有占用、不参与再分配；
      2. 其余队列按 ``caps`` 比例分剩余容量，分到超过自己上限的夹到上限，
         夹紧释放出的容量再分给剩下的队列（队列数固定为 3，最多 3 轮）；
      3. 剩余容量为负（locked token 占住了池子）时仍按比例分，结果可能低于下限。
         这是"所有队列都不可回收"的信号，交由调用方的跳队/兜底吸收，不做特判。
    """
    if not params.enabled:
        raise ValueError(
            "effective_targets needs quota.caps; QuotaParams.enabled is False "
            "(the strategy must skip the selector entirely when quota is off)."
        )

    total_capacity = max(float(capacity), 0.0)
    if total_capacity <= 0.0:
        return {q: 0.0 for q in QUOTA_QUEUES}
    cap = {q: params.caps[q] * total_capacity for q in QUOTA_QUEUES}
    rho = {q: params.reservations.get(q, 0.0) * total_capacity for q in QUOTA_QUEUES}

    fixed = [q for q in QUOTA_QUEUES if occ.get(q, 0) <= rho[q]]
    active = [q for q in QUOTA_QUEUES if q not in fixed]

    target: Dict[str, float] = {q: float(occ.get(q, 0)) for q in fixed}
    residual = total_capacity - sum(target.values())
    residual -= sum(rho[q] for q in active)

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
    """筛出真正要减持的队列，按超出量降序返回。

    两个门禁必须同时满足，缺一个都会在边界上抖动：
      * 相对超限 > ``deadzone``：拦住"刚好压线"的浮点抖动；
      * 绝对超限 >= ``chunk``：拦住每次只腾出一个 token 的逐点颠簸
        （``page_size=1`` 的实验配置下尤其明显）。

    Args:
        occ: 各队列可调度占用（token）。
        targets: ``effective_targets`` 的输出。
        params: 配额参数，只用到 ``deadzone``。
        chunk: 单次最小减持量（token），见 ``resolve_chunk``。

    Returns:
        满足门禁的队列名，按 ``occ - target`` 降序；无队列满足时返回空列表。
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
    """给出本轮点名的完整优先级，供调用方逐个尝试（第一个有候选的即为 Q*）。

    两段式优先级：
      1. 超上限者在前，按超出量降序 —— 这是"封顶"的强制执行部分；
      2. 弹性区在后，按超出下限降序 —— 配额不禁止队列用得多，只要求排队。

    返回空列表表示"没有任何队列高于下限"，即配额对这次淘汰没有意见：
    此时要么池子并未真被这三个队列占满（locked/one-shot 占着），要么配置的
    下限之和过大，两种情形都该交给兜底，而不是由数学层猜。

    Args:
        occ: 各队列可调度占用（token）。
        targets: ``effective_targets`` 的输出。
        params: 配额参数，只用到 ``reservations``。
        chunk: 单次最小减持量（token）。
        capacity: 池容量（token），用于把下限比例换算成 token。

    Returns:
        队列名列表，可能为空；不含重复项。
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
    """把 ``caps`` 比例换算成各队列的硬上限（token），仅用于观测。"""
    return {q: params.caps[q] * max(float(capacity), 0.0) for q in QUOTA_QUEUES}


def resolve_chunk(page_size: int, params: QuotaParams) -> int:
    """取单次最小减持量：``max(page_size, chunk_tokens)``。

    为什么不只用 ``chunk_tokens``：page 是 KV 分配的最小粒度，减持量小于一个
    page 时淘汰循环仍会释放整页，死区判定与实际释放量对不上。
    """
    return max(int(page_size or 1), int(params.chunk_tokens))


def pick_victim(
    *,
    strategy,
    leaves,
    capacity: float,
    page_size: int,
    overrides,
    session_registry,
):
    """三级淘汰的候选选择（纯决策，不释放 KV、不改树）。

    阶段 0  one-shot 带永远先清空（带内 LRU），不计配额 —— 清它没有机会成本。
    阶段 1  配额启用：按 ``quota_ranking`` 的顺序逐个问队内策略，某队没有候选
            就跳下一队（全 locked / 全是 one-shot / 标签全场集中到别处）；
            全部无候选才用策略私有 RNG 兜底，保证淘汰单调推进、不 spin。
    阶段 2  未配置配额：保留原路径（全局 ``victim_key`` + 改写器链），行为不变。

    Args:
        strategy: ``SimpleHybridStrategy``，提供 is_one_shot / *_key / quota_* 接口。
        leaves: 当前可淘汰叶子集合（已排除 locked 节点）。
        capacity: 池容量（token），仅供配额仲裁使用。
        page_size: 页大小，用于推导 chunk。
        overrides: 改写器链（RR / SLRU / MT），仅阶段 2 使用。
        session_registry: 会话账本，改写器的 ``pick`` 需要它。

    Returns:
        ``(victim, source)``。``source`` 会进 ``victim_src`` 观测：
        ``"one_shot"`` / ``"quota:<queue>"`` / ``"quota:fallback"`` / ``"global"``。

    无候选时返回 ``(None, None)``，调用方据此结束淘汰循环。
    """
    one_shot = [n for n in leaves if strategy.is_one_shot(n)]
    if one_shot:
        return min(one_shot, key=strategy.one_shot_key), "one_shot"
    if not leaves:
        return None, None

    if strategy.quota.enabled:
        for queue in strategy.quota_ranking(capacity, page_size):
            victim = strategy.quota_victim(queue, leaves, capacity)
            if victim is not None:
                return victim, f"quota:{queue}"
        return strategy.quota_fallback(leaves), "quota:fallback"

    victim = min(leaves, key=strategy.victim_key)
    source = "global"
    for override in overrides:
        if override.applies_to(victim):
            alt = override.pick(leaves, session_registry)
            if alt is not None:
                victim = alt
                source = (
                    override.queue.value
                    if override.queue is not None
                    else override.name
                )
    return victim, source


def struct_tier(node, share_of, share_min: int) -> Tuple[int, float]:
    """STRUCT 队列的减持顺序键：先牺牲"刚够保护"的模板，再动高复用模板。

    可调度带里的 struct 节点必然 ``share >= share_min`` —— 更低的那些属于
    one-shot，在阶段 0 就被清掉了。所以能分出的梯度只有两档：

      * 0：``share == share_min``，只被最小数量的会话用过，最廉价；
      * 1：``share > share_min``，被多会话复用，最后才牺牲。

    同档内按 ``last_access_time`` 升序（最久未访问先走），与 SLRU 段内 LRU 同口径。

    Args:
        node: 树节点。
        share_of: 读取节点共享会话数的可调用对象（``SlruEvictPolicy.share_of``）。
        share_min: 保护阈值（``SlruParams.share_min``）。

    Returns:
        可直接用于 ``min()`` 的 ``(tier, last_access_time)``。
    """
    tier = 0 if share_of(node) <= share_min else 1
    return (tier, node.last_access_time)
