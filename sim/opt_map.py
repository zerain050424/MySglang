"""离线 use map：把会话 trace 变成 ``{block_hash: [使用时刻]}``。

这是 OPT 上界（``mem_cache/evict_policy.OPTStrategy``）的唯一输入。生成规则
必须与缓存层写进 ``TreeNode.hash_value`` 的那套逐字一致，否则每个块都查不到，
OPT 会静默退化成「全 +∞」的任意序。

一致性来自**复用同一个底层函数**：这里逐 page 调用
``mem_cache/utils.get_storage_hash_str(key, prior_hash)``，而
``compute_node_hash_values`` 也是「父 hash + 本段 tokens」按 page 累进。区别只在
调用方：它从树上取 ``node.key``，本模块直接从 trace 的 prompt 切段——所以
use map 可以在跑仿真之前独立生成，不需要先跑一遍淘汰。

时间轴约定：存的是 ``Turn.t``（虚拟秒），与 ``RadixCache._now`` 同源。因此不需要
UniCache 那种全局 ``PREFILL_CNT`` 游标，``OPTStrategy`` 用二分查询即可。

``turn.prompt`` 是完整历史。离线回放保留 ``Turn.response`` 时，上一轮 response
已经作为下一轮 prompt 的一部分出现，所以 response 块会自然拿到有限的下次使用；
只给长度（合成负载）的那一支用 fallback 哨兵，这些页必须靠
``with_response=True`` 补进表。

用法（**闭环调度**下必须先从一次 run 冻结时间轴，详见 ``use_map_from_run``）::

    run = Simulator(SimConfig(page_size=8, eviction_policy="lru")).run_scheduled(
        sessions, specs, service=service
    )
    write_use_map("/tmp/use_map.json",
                  use_map_from_run(run, sessions, page_size=8))

    sim = Simulator(SimConfig(page_size=8, eviction_policy="opt",
                              opt_use_map="/tmp/use_map.json"))
    sim.run(run.freeze_sessions(sessions))

只有 **开环** trace（``Turn.t`` 已经是决策时刻）才能直接
``build_use_map(sessions, page_size=...)``。
"""

from __future__ import annotations

import json
from array import array
from typing import Dict, Iterable, List, Optional


def _page_hashes(
    tokens: List[int], page_size: int, hash_fn, key_cls, extra_key=None
) -> List[str]:
    """按 ``page_size`` 切段并累进 hash（与 ``compute_node_hash_values`` 同规则）。"""

    hashes: List[str] = []
    prior: Optional[str] = None
    # 尾段不足一页时丢弃：缓存层也只对整页算 hash（``page_aligned``）。
    usable = len(tokens) - len(tokens) % page_size
    for start in range(0, usable, page_size):
        segment = array("i", tokens[start : start + page_size])
        current = hash_fn(key_cls(segment, extra_key=extra_key), prior)
        if not isinstance(current, str):
            raise TypeError(
                "get_storage_hash_str 必须返回单个 hash（本函数逐页调用）；"
                f"got {type(current).__name__}"
            )
        hashes.append(current)
        prior = current
    return hashes


def build_use_map(
    sessions: Iterable,
    page_size: int = 8,
    *,
    extra_key: Optional[str] = None,
    with_response: bool = False,
    response_tokens: Optional[Dict[tuple, int]] = None,
) -> Dict[str, List[float]]:
    """把会话 trace 展开成 ``{block_hash: [使用时刻...]}``。

    Args:
        sessions: ``sim.trace.Session`` 序列（只需 ``.turns[].t`` 与
            ``.turns[].prompt``）。
        page_size: 切块粒度（token），必须与仿真里的 ``SimConfig.page_size``
            和缓存的 ``page_size`` 相同。
        extra_key: 命名空间；与 ``CacheInitParams`` 的 ``extra_key`` 一致时
            hash 才能对上。仿真默认 ``None``。
        with_response: 把 response 段也写进表。树里
            ``_adopt(prompt + output)`` 确实存了它们，不写就会让这批页永远查不到
            （未命中率虚高、且「粒度错配」告警失去判别力）。**闭环 trace 必须开**。
            ``Turn.response`` 有真实 token 时用它；只给长度时用 fallback 哨兵。
        response_tokens: ``{(session_id, 轮次): 哨兵}``，来自
            ``SimResult.response_tokens()``。哨兵是每请求唯一的（见 ``sim/req.py``），
            所以只能从 run 里取，不能从 trace 推。``Turn.response`` 齐全时可不传。

    Returns:
        ``{hash: [t, ...]}``，每个块的使用时刻按请求顺序追加（可能重复、可能
        乱序——``load_use_map`` 会排序去重）。

    Raises:
        ValueError: ``page_size`` 非正。
    """

    if page_size <= 0:
        raise ValueError(f"page_size must be > 0, got {page_size}")

    # 延迟导入：这两个模块会拉起 torch，必须先在 bootstrap 里装好替身。
    from sim import bootstrap

    bootstrap.load()
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    def hash_fn(key, prior):
        # extra_key 由 RadixKey 携带；prior=None 时 get_storage_hash_str 会
        # 用 storage_namespace_seed(extra_key, cache_salt) 起链，与树根一致。
        return get_storage_hash_str(key, prior)

    sessions = list(sessions)
    if with_response:
        # 只给长度的那一支（合成负载 / 老 trace）没有 response token，只能拿 run
        # 里记下的 fallback 哨兵；有 Turn.response 的走真实 token。
        needs_fallback = any(
            turn.out and getattr(turn, "response", None) is None
            for session in sessions
            for turn in session.turns
        )
        if needs_fallback and response_tokens is None:
            raise ValueError(
                "with_response=True 且 trace 只给 response 长度时需要 response_tokens"
                "（取自 SimResult.response_tokens()：fallback 哨兵每请求唯一，"
                "trace 里没有）"
            )
        response_tokens = response_tokens or {}

    table: Dict[str, List[float]] = {}
    for session in sessions:
        for idx, turn in enumerate(session.turns):
            tokens = list(turn.prompt)
            if with_response and turn.out:
                real = getattr(turn, "response", None)
                if real is not None:
                    tokens.extend(int(token) for token in real)
                else:
                    token = response_tokens.get((session.session_id, idx))
                    if token is None:
                        raise KeyError(
                            f"缺 response 哨兵：session={session.session_id} "
                            f"turn={idx}；用 SimResult.response_tokens() 提供"
                        )
                    tokens.extend([int(token)] * int(turn.out))
            if not tokens:
                continue
            for block_hash in _page_hashes(
                tokens, page_size, hash_fn, RadixKey, extra_key
            ):
                table.setdefault(block_hash, []).append(float(turn.t))
    if not table:
        raise ValueError(
            "use map is empty: every prompt was shorter than one page "
            f"(page_size={page_size})?"
        )
    return table


def use_map_from_run(
    result,
    sessions: Iterable,
    *,
    page_size: int = 8,
) -> Dict[str, List[float]]:
    """一次闭环 run → 它的 use map（冻结时间轴 + 覆盖 response 页）。

    这是 sim 里出 OPT 上界的唯一入口：早一步建表会拿到全零时间轴（数据集
    ``Turn.t`` 是 0、闭环时刻不写回），``OPTStrategy`` 于是静默退化成任意序。

    Args:
        result: 一次 ``SimResult``（``run`` / ``run_scheduled`` / ``run_engine``
            都带 ``dispatched``）。
        sessions: 那次 run 用的会话。
        page_size: 必须等于重放时的 ``SimConfig.page_size``。

    Returns:
        ``{block_hash: [使用时刻...]}``。

    Raises:
        ValueError: 冻结失败（run 没记录派发时刻）或 response 哨兵缺失。
    """

    frozen = result.freeze_sessions(list(sessions))
    return build_use_map(
        frozen,
        page_size=page_size,
        with_response=True,
        response_tokens=result.response_tokens(),
    )


def write_use_map(path: str, table: Dict[str, List[float]]) -> None:
    """写 JSON（键排序，便于 diff 与人工比对）。"""

    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: table[k] for k in sorted(table)}, f)
