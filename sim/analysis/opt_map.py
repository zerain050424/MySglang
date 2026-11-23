"""为离线 OPT (Belady) 策略构建与树同源的块级使用时刻表。"""

from __future__ import annotations

import json
from array import array
from typing import Dict, Iterable, List, Optional


def _page_hashes(
    tokens: List[int], page_size: int, hash_fn, key_cls, extra_key=None
) -> List[str]:
    """按 page_size 切块并累进计算前缀 hash（不足一整页的尾部按缓存约定丢弃）。"""
    hashes: List[str] = []
    prior: Optional[str] = None
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
    capacity_blocks: Optional[int] = None,
    cost_model=None,
    cost_of: Optional[Dict[str, int]] = None,
) -> Dict[str, List[float]]:
    """根据已赋予虚拟时间戳的会话构造 {block_hash: [t, ...]} 未来使用表。"""
    if page_size <= 0:
        raise ValueError(f"page_size must be > 0, got {page_size}")
    if capacity_blocks is not None and capacity_blocks <= 0:
        raise ValueError(
            f"capacity_blocks must be > 0 or None, got {capacity_blocks}"
        )

    from sim.core import bootstrap

    bootstrap.load()
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    sessions = list(sessions)
    if with_response:
        needs_fallback = any(
            turn.out and getattr(turn, "response", None) is None
            for session in sessions
            for turn in session.turns
        )
        if needs_fallback and response_tokens is None:
            raise ValueError(
                "with_response=True 且 trace 只给 response 长度时需要 response_tokens"
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
            if capacity_blocks is not None:
                total_tokens = len(tokens) + turn.response_len
                total_blocks = -(-total_tokens // page_size)
                if total_blocks > capacity_blocks:
                    continue
            for block_index, block_hash in enumerate(
                _page_hashes(tokens, page_size, get_storage_hash_str, RadixKey, extra_key)
            ):
                table.setdefault(block_hash, []).append(float(turn.t))
                if cost_of is not None and cost_model is not None:
                    cost_of[block_hash] = int(cost_model.cost(block_index))

    if not table:
        raise ValueError(
            "use map is empty: every prompt was shorter than one page "
            f"(page_size={page_size}) or every turn exceeded "
            f"capacity_blocks={capacity_blocks}"
        )
    return table


def use_map_from_run(
    result,
    sessions: Iterable,
    *,
    page_size: int = 8,
    cost_model=None,
    cost_of: Optional[Dict[str, int]] = None,
) -> Dict[str, List[float]]:
    """从闭环仿真结果冻结派发时间轴，并包含 response 段页生成 OPT use map。"""
    frozen = result.freeze_sessions(list(sessions))
    return build_use_map(
        frozen,
        page_size=page_size,
        with_response=True,
        response_tokens=result.response_tokens(),
        cost_model=cost_model,
        cost_of=cost_of,
    )


def write_use_map(path: str, table: Dict[str, List[float]]) -> None:
    """按排序后的键写出 JSON，便于人工 diff 与排查。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: table[k] for k in sorted(table)}, f)


def write_block_costs(path: str, cost_of: Dict[str, int]) -> None:
    """写出代价加权 OPT 使用的 {block_hash: cost} 映射。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: cost_of[k] for k in sorted(cost_of)}, f)
