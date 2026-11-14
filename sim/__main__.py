"""M0 自检：``python -m sim``。

分两级，任一级失败都以非 0 退出：

1. **装载**：两个重依赖替身就位后，生产模块能导入，符号表齐全；
2. **可运行**：用一次最小的 match → insert → evict 证明导入的不是空壳
   （树、标签、权重仲裁、账本 reconcile 都真的在动）。

第 2 级里的 ``_CheckAllocator`` 只是自检用的最小分配器，M2 会被
``sim/allocator.py`` 的 ``TokenAllocator`` 取代，不要在这里加功能。
"""

from __future__ import annotations

import argparse
import sys
from array import array

from sim import bootstrap


class _CheckAllocator:
    """自检用：只满足 evict() 对 allocator 的三处读取。"""

    def __init__(self, capacity: int) -> None:
        self.size = float(capacity)
        self.used = 0.0
        self.device = "cpu"  # 字符串也认，RadixCache 会折成 torch.device("cpu")

    def available_size(self) -> float:
        return self.size - self.used

    def free_segment(self, indices=None, start_pos: int = 0) -> None:
        self.used -= len(indices) - start_pos


class _CheckReq:
    """自检用：match_prefix 只读 rid / session_id / origin_input_ids。"""

    def __init__(self, rid: str, session_id: str, tokens) -> None:
        self.rid = rid
        self.session_id = session_id
        self.origin_input_ids = tokens
        self.tool_call = None


def _check_import(sym) -> list:
    print("[1] 装载")
    info = bootstrap.status()
    print(f"    repo      : {info['repo_root']}")
    print(f"    python    : {info['python']}")
    print(f"    torch     : {info['torch']}")
    if info["stubs"]:
        for name, reason in info["stubs"].items():
            print(f"    stub      : {name}")
            print(f"                ^ {reason}")
    else:
        print("    stub      : 无（真实依赖齐全）")

    missing = [
        key
        for key in (
            "RadixKey",
            "TreeNode",
            "CacheInitParams",
            "HeterCacheRadixCache",
            "HeterCacheStrategy",
            "QueueOccupancyTracker",
            "SessionRegistry",
            "WEIGHT_QUEUES",
        )
        if key not in sym
    ]
    if missing:
        print(f"    FAIL 缺少符号：{missing}")
        return ["符号表不完整"]

    print(f"    cache     : {sym['HeterCacheRadixCache'].__name__}")
    print(f"    strategy  : {sym['HeterCacheStrategy'].__name__}")
    print(f"    queues    : {list(sym['WEIGHT_QUEUES'])}")
    return []


def _check_run(sym) -> list:
    print("[2] 可运行（match → insert → evict）")
    torch = __import__("torch")
    fails = []

    alloc = _CheckAllocator(capacity=1200)
    cache = sym["HeterCacheRadixCache"](
        sym["CacheInitParams"](
            disable=False,
            req_to_token_pool=None,  # 本自检不走 _adopt（M3 才需要假池）
            token_to_kv_pool_allocator=alloc,
            page_size=1,
            eviction_policy="heter_cache",
            eviction_policy_config={
                "weight": {"weights": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}}
            },
        )
    )
    print(f"    built     : {type(cache).__name__} / {type(cache.eviction_strategy).__name__}")

    RadixKey = sym["RadixKey"]
    shared = list(range(1000, 1300))

    def request(rid: str, sid: str, tokens: list) -> tuple:
        key = RadixKey(array("q", tokens))
        result = cache.match_prefix(
            sym["MatchPrefixParams"](key=key, req=_CheckReq(rid, sid, tokens))
        )
        hit = result.device_indices.numel()
        alloc.used += len(tokens) - hit
        cache.session_registry.touch(sid, rid, len(tokens))
        cache.insert(
            sym["InsertParams"](
                key=RadixKey(array("q", tokens)),
                value=torch.arange(len(tokens), dtype=torch.int64),
                decode_start=len(tokens),
                session_id=sid,
            )
        )
        evicted = 0
        if alloc.used > alloc.size:
            evicted = cache.evict(
                sym["EvictParams"](num_tokens=int(alloc.used - alloc.size))
            ).num_tokens_evicted
        return hit, evicted

    # (rid, session, tokens, 期望命中)：r1 与 r0 完全相同 -> 全中；r2 只共享
    # 300 token 的前缀 -> 命中 300；r3 与前面无交集 -> 0。
    plan = [
        ("r0", "ShareGPT-1", shared + [0] * 200, 0),
        ("r1", "ShareGPT-1", shared + [0] * 200, 500),
        ("r2", "Code-0", shared + [1] * 200, 300),
        ("r3", "alfred-1", [7] * 900, 0),
    ]
    for rid, sid, tokens, want_hit in plan:
        hit, evicted = request(rid, sid, tokens)
        balanced = alloc.used == cache.evictable_size() + cache.protected_size()
        flag = "" if hit == want_hit else "  <-- 命中不符"
        print(
            f"    {rid:3s} {sid:11s} tokens={len(tokens):4d} hit={hit:4d} "
            f"want={want_hit:4d} evicted={evicted:4d} used={alloc.used:6.0f} "
            f"balanced={balanced}{flag}"
        )
        if hit != want_hit:
            fails.append(f"{rid}: 期望命中 {want_hit}，实际 {hit}")
        if not balanced:
            fails.append(
                f"{rid}: 池账不平 used={alloc.used} "
                f"tree={cache.evictable_size() + cache.protected_size()}"
            )

    profiler = cache.eviction_strategy.profiler
    sources = dict(profiler._victim_sources)
    print(f"    hit/queue : {dict(profiler._total_hit_tokens)}")
    print(f"    victim src: {sources}")
    print(f"    sched occ : {cache.occupancy.snapshot()} cap={cache.occupancy.capacity}")
    if not sources:
        fails.append("没有产生任何 victim 来源记录（淘汰路径没走到）")
    if cache.occupancy.capacity <= 0:
        fails.append("账本 capacity 仍为 0（reconcile 没跑到）")
    return fails


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim", description="M0 自检")
    parser.add_argument(
        "--import-only", action="store_true", help="只验证装载，不构造缓存"
    )
    args = parser.parse_args(argv)

    try:
        sym = bootstrap.load()
    except Exception as exc:
        print(f"[1] 装载 FAIL: {type(exc).__name__}: {exc}")
        return 2

    fails = _check_import(sym)
    if args.import_only:
        return 1 if fails else 0

    fails += _check_run(sym)
    if fails:
        print("\nM0 FAIL")
        for item in fails:
            print(f"  - {item}")
        return 1
    print("\nM0 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
