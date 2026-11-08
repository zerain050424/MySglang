"""跨队列配额的仲裁数学（纯函数，无全局状态）。

本模块只回答两个问题，不碰树、不碰节点内容：

  1. 给定各队列占用 `occ`、池容量 `C`、下限 `rho`、上限 `k`，每个队列此刻的
     有效目标 `target` 是多少（max-min 水填充 + 上下限夹紧）；
  2. 这一轮该点名哪个队列（超上限优先，其次弹性区按超出下限最多）。

分母恒为池容量 `C`（`allocator.size`），**不是** `sum(occ)`：每淘汰 1 token
分母也跟着少 1，会把"降到目标"实际所需的淘汰量放大 1/(1-k) 倍
（k=0.5、超出 300 时实际要淘汰 600），使上限型配额退化成激进清理。

队内排序、会话公平、生存概率不在这里——那是 victim_override 三个插件的职责。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.uni_utils import QUOTA_QUEUES, QuotaParams

# 额度计算用浮点比例，比较留一点容差（避免 0.5*C 的舍入把"刚好达标"判成超标）。
_EPS = 1e-9


def effective_targets(
    occ: Dict[str, float],
    capacity: float,
    params: QuotaParams,
) -> Dict[str, float]:
    """Max-min 水填充：返回各队列此刻的有效目标（单位 token，Σtarget <= capacity）。

    规则：
      * 占用不高于下限的队列"未触底"，免死——保住现有占用，不参与再分配；
      * 其余队列按上限比例 k 分剩余容量，分到超过自己上限的夹到上限，
        夹紧释放出来的容量再分给剩下的队列（最多 3 轮，队列数固定为 3）；
      * 若所有剩余队列都触顶，Σtarget 会小于 capacity：这是"上限全部生效"的
        正常结果（空闲容量无人认领），不是错误；
      * 剩余容量为负（锁定 token 把池子占住导致）时，仍按 k 比例分，
        结果可能低于下限——这是"所有队列都不可回收"的信号，由调用方的
        跳队/兜底逻辑吸收，不在数学层做特判。

    `caps` / `reservations` 是比例，这里统一换算成 token 后再参与加减。
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
    """超出有效目标、且超过死区与最小减持量的队列，按超出量降序。"""
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
    """本轮点名的完整优先级：超上限者（按超出量降序）→ 弹性区（按超出下限降序）。

    调用方按此顺序逐个尝试，第一个拿得出可淘汰叶子的队列即为 Q*；
    全部拿不出才进兜底。这样"跳队"只需要遍历这个列表，不必重复算数学。

    第三个条件（无任何队列高于下限）返回空列表 = 配额没有意见。
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
    """各队列的硬上限（tokens），用于观测列 `quota_cap`。"""
    return {q: params.caps[q] * max(float(capacity), 0.0) for q in QUOTA_QUEUES}


def resolve_chunk(page_size: int, params: QuotaParams) -> int:
    """单次最小减持量：至少一个 page，避免 page_size=1 时逐 token 颠簸。"""
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
    """三级淘汰的候选选择：返回 (victim, source)；无可淘汰对象时 (None, None)。

    阶段 0  one-shot 带永远先清空（带内 LRU），不计配额。
    阶段 1  配额启用：按点名声次逐个尝试，队内无候选就跳下一个；全部无候选才
            用策略私有 RNG 兜底（同 seed 可复现）。
    阶段 2  未配置配额时保留原路径：全局 victim_key 竞争 + 改写器链。

    这里是纯决策，不释放 KV、不改树；副作用留在缓存层。
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
    """STRUCT 队列的减持顺序键（Q1=b）：先"刚好够保护"，再"高出阈值"，同档 LRU。

    可调度带里的 struct 节点必然 share >= share_min（更低的那些属于 one-shot，
    在阶段 0 就被清了）。因此分档只有两档：
      0 = share == share_min（刚够保护，最廉价）
      1 = share >  share_min（被多会话复用，最后牺牲）
    """
    tier = 0 if share_of(node) <= share_min else 1
    return (tier, node.last_access_time)
