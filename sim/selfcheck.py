"""离线仿真器运行环境自检：校验模块装载、最小缓存回路与虚拟时钟边界。"""

from __future__ import annotations

import argparse
import os
import sys
from array import array
from typing import List

from sim.core import bootstrap
from sim.core.allocator import TokenAllocator, tree_tokens
from sim.core.clock import VirtualClock

VALUE = {"horizon_s": 120.0, "floor_tokens": 0.0}


class _CheckReq:
    """自检用最小请求，满足 match_prefix 读取的四个属性。"""

    def __init__(self, rid: str, session_id: str, tokens) -> None:
        self.rid = rid
        self.session_id = session_id
        self.origin_input_ids = tokens
        self.tool_call = None


def _build_cache(sym, capacity: int = 1200, now=None):
    alloc = TokenAllocator(capacity)
    cache = sym["HeterCacheRadixCache"].create_simulated(
        mock_allocator=alloc,
        page_size=1,
        eviction_policy="heter_cache",
        eviction_policy_config={"value": dict(VALUE)},
        now=now,
    )
    return cache, alloc


def _insert_only(sym, cache, tokens, session_id: str):
    torch = __import__("torch")
    result = cache.insert(
        sym["InsertParams"](
            key=sym["RadixKey"](array("q", tokens)),
            value=torch.arange(len(tokens), dtype=torch.int64),
            decode_start=len(tokens),
            session_id=session_id,
        )
    )
    return result.last_device_node


def _check_import(sym) -> List[str]:
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

    required = (
        "RadixKey",
        "TreeNode",
        "CacheInitParams",
        "HeterCacheRadixCache",
        "HeterCacheStrategy",
        "QueueOccupancyTracker",
        "SessionRegistry",
        "VALUE_QUEUES",
    )
    missing = [key for key in required if key not in sym]
    if missing:
        print(f"    FAIL 缺少符号：{missing}")
        return ["符号表不完整"]

    print(f"    cache     : {sym['HeterCacheRadixCache'].__name__}")
    print(f"    strategy  : {sym['HeterCacheStrategy'].__name__}")
    print(f"    queues    : {list(sym['VALUE_QUEUES'])}")
    return []


def _check_smoke_loop(sym) -> List[str]:
    print("[2] 最小回路（match -> insert -> evict）")
    torch = __import__("torch")
    fails: List[str] = []

    cache, alloc = _build_cache(sym, capacity=1200)
    print(
        f"    built     : {type(cache).__name__} / "
        f"{type(cache.eviction_strategy).__name__} / now={cache._now.__name__}"
    )

    shared = list(range(1000, 1300))

    def request(rid: str, sid: str, tokens: list):
        key = sym["RadixKey"](array("q", tokens))
        result = cache.match_prefix(
            sym["MatchPrefixParams"](key=key, req=_CheckReq(rid, sid, tokens))
        )
        hit = result.device_indices.numel()
        need = len(tokens) - hit
        evicted = 0
        shortfall = need - alloc.available_size()
        if shortfall > 0:
            evicted = cache.evict(
                sym["EvictParams"](num_tokens=shortfall)
            ).num_tokens_evicted
        alloc.alloc(need)
        cache.session_registry.touch(sid, rid, len(tokens))
        cache.insert(
            sym["InsertParams"](
                key=sym["RadixKey"](array("q", tokens)),
                value=torch.arange(len(tokens), dtype=torch.int64),
                decode_start=len(tokens),
                session_id=sid,
            )
        )
        return hit, evicted

    plan = [
        ("r0", "ShareGPT-1", shared + [0] * 200, 0),
        ("r1", "ShareGPT-1", shared + [0] * 200, 500),
        ("r2", "Code-0", shared + [1] * 200, 300),
        ("r3", "alfred-1", [7] * 900, 0),
    ]
    for rid, sid, tokens, want_hit in plan:
        hit, evicted = request(rid, sid, tokens)
        balanced = alloc.used == tree_tokens(cache)
        flag = "" if hit == want_hit else "  <-- 命中不符"
        print(
            f"    {rid:3s} {sid:11s} tokens={len(tokens):4d} hit={hit:4d} "
            f"want={want_hit:4d} evicted={evicted:4d} used={alloc.used:6.0f} "
            f"balanced={balanced}{flag}"
        )
        if hit != want_hit:
            fails.append(f"{rid}: 期望命中 {want_hit}，实际 {hit}")
        if not balanced:
            fails.append(f"{rid}: 池账不平 used={alloc.used} tree={tree_tokens(cache)}")

    profiler = cache.eviction_strategy.profiler
    sources = dict(profiler._victim_sources)
    print(f"    hit/queue : {dict(profiler._total_hit_tokens)}")
    print(f"    victim src: {sources}")
    print(f"    sched occ : {cache.occupancy.snapshot()} cap={cache.occupancy.capacity}")
    if not sources:
        fails.append("没有产生任何 victim 来源记录")
    if cache.occupancy.capacity <= 0:
        fails.append("账本 capacity 仍为 0（reconcile 未执行）")
    return fails


def _check_virtual_clock(sym) -> List[str]:
    print("[3] 虚拟时钟边界")
    fails: List[str] = []
    Tag = sym["HeterCacheNodeTag"]

    clock = VirtualClock(0.0)
    cache, _ = _build_cache(sym, capacity=4096, now=clock)
    leaf = _insert_only(sym, cache, list(range(0, 64)), "ShareGPT-a")
    stamps = (leaf.last_access_time, leaf.creation_time)
    ok = stamps == (0.0, 0.0)
    print(f"    node ts   : last/creation={stamps} (want (0.0, 0.0))")
    if not ok:
        fails.append(f"新节点时间戳未走虚拟时钟：{stamps}")

    clock.advance_to(5.0)
    cache.match_prefix(
        sym["MatchPrefixParams"](
            key=sym["RadixKey"](array("q", list(range(0, 64)))),
            req=_CheckReq("r-touch", "ShareGPT-a", list(range(0, 64))),
        )
    )
    if leaf.last_access_time != 5.0:
        fails.append(f"match 未更新虚拟时钟：{leaf.last_access_time}")

    strategy = cache.eviction_strategy
    leaf.queue_tag = Tag.QCHAT
    chat_ttl = strategy.share.ageing_ttl_s["chat"]
    base = leaf.last_access_time
    clock.advance_to(base + chat_ttl - 0.1)
    before = strategy.is_aged(leaf)
    clock.advance_to(base + chat_ttl)
    after = strategy.is_aged(leaf)
    if before or not after:
        fails.append(f"chat 老化边界错：{before} -> {after}")

    from sglang.srt.mem_cache.heter_cache.utils import ONE_SHOT_IDLE_S as idle_s

    clock = VirtualClock(0.0)
    cache, _ = _build_cache(sym, capacity=4096, now=clock)
    strategy = cache.eviction_strategy
    cache.session_registry.observe_request("apps-1", "r-one", 400, now=clock())
    one = _insert_only(sym, cache, list(range(2000, 2100)), "apps-1")
    one.queue_tag = Tag.QTRANS
    one.hit_count = 0
    clock.advance_to(idle_s - 0.1)
    before_idle = strategy.is_one_shot(one)
    clock.advance_to(idle_s)
    after_idle = strategy.is_one_shot(one)
    if before_idle or not after_idle:
        fails.append(f"one-shot 豁免边界错：{before_idle} -> {after_idle}")

    clock = VirtualClock(0.0)
    cache, _ = _build_cache(sym, capacity=1024, now=clock)
    tracker = cache.eviction_strategy.occupancy
    telemetry = cache.eviction_strategy.telemetry
    tracker.reconcile({}, 1024.0)
    clock.advance_to(2.9)
    early = tracker.wants_reconcile(
        0.0, 1024.0, normal_window_s=telemetry.PROFILE_INTERVAL, has_capacity=True
    )
    clock.advance_to(3.0)
    due = tracker.wants_reconcile(
        0.0, 1024.0, normal_window_s=telemetry.PROFILE_INTERVAL, has_capacity=True
    )
    clock.advance_to(30.0)
    prof = telemetry.wants_profile()
    if early or not due or not prof:
        fails.append(f"虚拟窗口判定错：early={early} due={due} prof={prof}")
    else:
        print(f"    clock ok  : chat_ttl={chat_ttl}s one_shot_idle={idle_s}s pressure=3.0s profile=30.0s")
    return fails


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim", description="仿真器环境自检")
    parser.add_argument("--import-only", action="store_true", help="仅验证装载，不构造缓存")
    parser.add_argument("--skip-clock", action="store_true", help="跳过虚拟时钟边界检查")
    parser.add_argument("--verbose", action="store_true", help="保留策略 profile 日志")
    args = parser.parse_args(argv)

    if not args.verbose:
        os.environ.setdefault("STRAT_LOG_LEVEL", "quiet")

    try:
        sym = bootstrap.load()
    except Exception as exc:
        print(f"[1] 装载 FAIL: {type(exc).__name__}: {exc}")
        return 2

    fails = _check_import(sym)
    if args.import_only:
        return 1 if fails else 0

    fails += _check_smoke_loop(sym)
    if not args.skip_clock:
        fails += _check_virtual_clock(sym)

    if fails:
        print("\nFAIL")
        for item in fails:
            print(f"  - {item}")
        return 1
    print("\nOK")
    return 0
