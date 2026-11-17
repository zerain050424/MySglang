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

``turn.prompt`` 是完整历史；离线回放保留 ``Turn.response`` 时，上一轮 response
已经作为下一轮 prompt 的一部分出现，所以 response 块会自然拿到有限的下次使用。

用法::

    from sim.opt_map import build_use_map, write_use_map
    write_use_map("/tmp/use_map.json", build_use_map(sessions, page_size=8))

    sim = Simulator(SimConfig(page_size=8, eviction_policy="opt",
                              opt_use_map="/tmp/use_map.json"))
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
) -> Dict[str, List[float]]:
    """把会话 trace 展开成 ``{block_hash: [使用时刻...]}``。

    Args:
        sessions: ``sim.trace.Session`` 序列（只需 ``.turns[].t`` 与
            ``.turns[].prompt``）。
        page_size: 切块粒度（token），必须与仿真里的 ``SimConfig.page_size``
            和缓存的 ``page_size`` 相同。
        extra_key: 命名空间；与 ``CacheInitParams`` 的 ``extra_key`` 一致时
            hash 才能对上。仿真默认 ``None``。

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

    table: Dict[str, List[float]] = {}
    for session in sessions:
        for turn in session.turns:
            tokens = list(turn.prompt)
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


def write_use_map(path: str, table: Dict[str, List[float]]) -> None:
    """写 JSON（键排序，便于 diff 与人工比对）。"""

    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: table[k] for k in sorted(table)}, f)
