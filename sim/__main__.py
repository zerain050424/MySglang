"""自检：``python -m sim``。

分三段，任一段失败都以非 0 退出：

1. **装载**（M0）：两个重依赖替身就位后，生产模块能导入，符号表齐全；
2. **可运行**（M0）：一次最小的 match → insert → evict 证明导入的不是空壳
   （树、标签、权重仲裁、账本 reconcile 都真的在动）；
3. **虚拟时钟**（M1）：节点时间戳、老化 TTL、one-shot 豁免、承压校准窗口、
   profile 间隔全部由注入的时钟推进，且精确在边界翻转。

第 2/3 段用 ``sim/allocator.py`` 的 ``TokenAllocator`` 充当替身分配器。
"""

from __future__ import annotations

import argparse
import os
import sys
from array import array

from sim import bootstrap
from sim.allocator import OutOfTokens, TokenAllocator, tree_tokens
from sim.clock import VirtualClock

WEIGHTS = {"struct": 0.2, "agentic": 0.5, "chat": 0.3}


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


def _build(sym, capacity: int = 1200, now=None):
    """按仿真口径建一个 HeterCache 缓存（page_size=1，权重三队齐全）。"""

    alloc = TokenAllocator(capacity)
    cache = sym["HeterCacheRadixCache"].create_simulated(
        mock_allocator=alloc,
        page_size=1,
        eviction_policy="heter_cache",
        eviction_policy_config={"weight": {"weights": dict(WEIGHTS)}},
        now=now,
    )
    return cache, alloc


def _insert(sym, cache, tokens, session_id: str):
    """只插入（不 match / 不 touch），返回落盘的叶子节点。"""

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

    cache, alloc = _build(sym, capacity=1200)
    print(
        f"    built     : {type(cache).__name__} / "
        f"{type(cache.eviction_strategy).__name__} / now={cache._now.__name__}"
    )

    shared = list(range(1000, 1300))

    def request(rid: str, sid: str, tokens: list) -> tuple:
        key = sym["RadixKey"](array("q", tokens))
        result = cache.match_prefix(
            sym["MatchPrefixParams"](key=key, req=_CheckReq(rid, sid, tokens))
        )
        hit = result.device_indices.numel()
        # 生产顺序：短缺 -> 淘汰 -> 预留 -> 写入。
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
            fails.append(
                f"{rid}: 池账不平 used={alloc.used} tree={tree_tokens(cache)}"
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


def _check_allocator() -> list:
    """M2 验收：容量守卫、二次释放守卫、归还后水位回零。"""

    print("[2b] 分配器守卫")
    fails = []
    alloc = TokenAllocator(100)
    alloc.alloc(60)
    over = False
    try:
        alloc.alloc(41)
    except OutOfTokens:
        over = True
    double_free = False
    try:
        alloc.free_segment([0] * 70)
    except RuntimeError:
        double_free = True
    alloc.free_segment([0] * 60)
    drained = alloc.used == 0 and alloc.available_size() == 100
    ok = over and double_free and drained
    print(
        f"    over-alloc 拦={over} 二次释放拦={double_free} "
        f"归还后水位={alloc.used}{'' if ok else '  <-- 守卫失效'}"
    )
    if not ok:
        fails.append(f"分配器守卫错：over={over} double_free={double_free} drained={drained}")
    return fails


def _check_clock(sym) -> list:
    """M1 验收：所有时间判据都由注入的时钟推进，且精确在边界翻转。"""

    print("[3] 虚拟时钟（M1）")
    fails = []
    Tag = sym["HeterCacheNodeTag"]

    # ---- 3.1 节点时间戳来自缓存持有的时钟，而不是墙钟 ----
    clock = VirtualClock(0.0)
    cache, _ = _build(sym, capacity=4096, now=clock)
    leaf = _insert(sym, cache, list(range(0, 64)), "ShareGPT-a")
    stamps = (leaf.last_access_time, leaf.creation_time)
    ok = stamps == (0.0, 0.0)
    print(f"    node ts   : last/creation={stamps} (want (0.0, 0.0)){'  <-- 墙钟' if not ok else ''}")
    if not ok:
        fails.append(f"新节点时间戳没走虚拟时钟：{stamps}")

    # match 也必须盖同一个时钟
    clock.advance_to(5.0)
    cache.match_prefix(
        sym["MatchPrefixParams"](
            key=sym["RadixKey"](array("q", list(range(0, 64)))),
            req=_CheckReq("r-touch", "ShareGPT-a", list(range(0, 64))),
        )
    )
    ok = leaf.last_access_time == 5.0
    print(f"    match ts  : {leaf.last_access_time} (want 5.0){'' if ok else '  <-- 墙钟'}")
    if not ok:
        fails.append(f"match 没走虚拟时钟：{leaf.last_access_time}")

    # ---- 3.2 老化 TTL 在边界翻转 ----
    chat_ttl = cache.eviction_strategy.slru.ageing_ttl_s["chat"]
    agent_ttl = cache.eviction_strategy.slru.ageing_ttl_s["agentic"]
    strategy = cache.eviction_strategy

    leaf.queue_tag = Tag.QCHAT
    chat_base = leaf.last_access_time   # match 在 t=5 刷过访问时间
    clock.advance_to(chat_base + chat_ttl - 0.1)
    before = strategy.is_aged(leaf)
    clock.advance_to(chat_base + chat_ttl)
    after = strategy.is_aged(leaf)
    ok = (not before) and after
    print(f"    aging chat {chat_ttl:>5.1f}s: t-0.1 aged={before} t aged={after}{'' if ok else '  <-- 边界错'}")
    if not ok:
        fails.append(f"chat 老化边界错：{before} -> {after}")

    leaf2 = _insert(sym, cache, list(range(5000, 5064)), "alfred-a")
    leaf2.queue_tag = Tag.QAGENT
    base = leaf2.last_access_time
    clock.advance_to(base + agent_ttl - 0.1)
    before = strategy.is_aged(leaf2)
    clock.advance_to(base + agent_ttl)
    after = strategy.is_aged(leaf2)
    ok = (not before) and after
    print(f"    aging agt  {agent_ttl:>5.1f}s: t-0.1 aged={before} t aged={after}{'' if ok else '  <-- 边界错'}")
    if not ok:
        fails.append(f"agentic 老化边界错：{before} -> {after}")

    # ---- 3.3 one-shot 时间豁免（ONE_SHOT_IDLE_S=90s）----
    clock = VirtualClock(0.0)
    cache, _ = _build(sym, capacity=4096, now=clock)
    strategy = cache.eviction_strategy   # 新缓存新策略，时钟也跟着换了
    # 单轮、无跨会话共享 -> TRANS；observe_request 只建分类证据，
    # last_active 由 touch 设置，这里两者都停在 0。
    cache.session_registry.observe_request("apps-1", "r-one", 400, now=clock())
    one = _insert(sym, cache, list(range(2000, 2100)), "apps-1")
    one.queue_tag = Tag.QTRANS
    one.hit_count = 0

    clock.advance_to(89.9)
    before = strategy.is_one_shot(one)
    clock.advance_to(90.0)
    after = strategy.is_one_shot(one)
    ok = (not before) and after
    print(f"    one-shot  : t=89.9 {before} / t=90.0 {after} (want False/True){'' if ok else '  <-- 边界错'}")
    if not ok:
        fails.append(f"one-shot 豁免边界错：{before} -> {after}")

    # 未达 share_min 的 struct 不参与老化（文档里的 protection guard）
    one.queue_tag = Tag.QSTRUCT
    guarded = not strategy.is_aged(one)
    print(f"    struct 无共享不老化: is_aged={not guarded and 'True' or 'False'}{'' if guarded else '  <-- 保护失效'}")
    if not guarded:
        fails.append("未达 share_min 的 struct 被判定老化")

    # ---- 3.4 承压校准窗口（3s）与 profile 间隔（30s）按虚拟秒推进 ----
    clock = VirtualClock(0.0)
    cache, _ = _build(sym, capacity=1024, now=clock)
    tracker = cache.eviction_strategy.occupancy
    telemetry = cache.eviction_strategy.telemetry
    first = telemetry._last_profile == 0.0

    tracker.reconcile({}, 1024.0)  # 把上一行校准时刻钉在 t=0
    clock.advance_to(2.9)
    pressure_early = tracker.wants_reconcile(
        0.0, 1024.0, normal_window_s=telemetry.PROFILE_INTERVAL, has_capacity=True
    )
    clock.advance_to(3.0)
    pressure_due = tracker.wants_reconcile(
        0.0, 1024.0, normal_window_s=telemetry.PROFILE_INTERVAL, has_capacity=True
    )
    relaxed = tracker.wants_reconcile(
        512.0, 1024.0, normal_window_s=telemetry.PROFILE_INTERVAL, has_capacity=True
    )
    clock.advance_to(30.0)
    profile_due = telemetry.wants_profile()

    ok = (
        first
        and not pressure_early
        and pressure_due
        and not relaxed
        and profile_due
    )
    print(
        f"    windows   : 承压 t=2.9 {pressure_early} / t=3.0 {pressure_due} / "
        f"非承压 {relaxed} / profile t=30 {profile_due}{'' if ok else '  <-- 窗口错'}"
    )
    if not ok:
        fails.append(
            f"虚拟窗口错：first={first} pressure={pressure_early}/{pressure_due} "
            f"relaxed={relaxed} profile={profile_due}"
        )
    return fails


def _check_driver() -> list:
    """M3 验收：驱动跑通、同 seed 可复现、one-shot 由虚拟时钟推进触发。"""

    print("[4] 请求驱动（M3）")
    from sim.runner import SimConfig, Simulator, run_synthetic
    from sim.trace import Session, Turn

    fails = []
    first = run_synthetic(SimConfig(capacity=4096, seed=7), seed=3, chat_turns=5)
    second = run_synthetic(SimConfig(capacity=4096, seed=7), seed=3, chat_turns=5)
    summary = first.summary
    types = sorted(summary.get("per_type", {}))
    print(
        f"    synthetic : requests={summary['requests']} "
        f"hit_rate={summary['hit_rate']:.1%} "
        f"evicted={summary['evicted_tokens']} types={types}"
    )
    same = first.fingerprint() == second.fingerprint()
    print(
        f"    同 seed   : {'逐字一致' if same else '不一致'}"
        f" ({first.fingerprint()})"
    )
    if not same:
        fails.append(
            f"同 seed 两次 run 指纹不同：{first.fingerprint()} vs {second.fingerprint()}"
        )
    if not 0.0 < summary["hit_rate"] < 1.0:
        fails.append(f"命中率越界：{summary['hit_rate']:.3f}")
    for expected in ("struct", "trans", "chat", "agentic"):
        if expected not in summary["per_type"]:
            fails.append(f"四类会话未全部出现，缺 {expected}（合成 trace 没跑开）")
    if not summary["victim_sources"]:
        fails.append("没有任何 victim 来源记录")

    # one-shot 时间豁免：只有虚拟时钟真的推到 90s 后才会成立。
    # t=0 写入一条单轮会话（400 token），t=200 来一条 800 token 的请求，
    # 池 1024 -> 短缺 176 -> 只能拿已经放凉的单轮节点开刀。
    sim = Simulator(SimConfig(capacity=1024, seed=0))
    trace = [
        Session("apps_0", (Turn(t=0.0, prompt=array("q", range(400)), out=0),)),
        Session(
            "apps_1", (Turn(t=200.0, prompt=array("q", range(5000, 5800)), out=0),)
        ),
    ]
    result = sim.run(trace)
    sources = result.summary["victim_sources"]
    one_shot = sources.get("one_shot", 0.0)
    ok = one_shot >= 176
    print(
        f"    one-shot  : 短缺 176 -> victim_sources={dict(sources)}"
        f"{'  <-- 未触发' if not ok else ''}"
    )
    if not ok:
        fails.append(
            "虚拟时钟下 one-shot 未被触发（SessionRegistry 的 last_active "
            f"是否仍取墙钟？）：{sources}"
        )
    return fails


def _check_fidelity() -> list:
    """M4 验收：逐请求拓扑不变量 + 账本对拍 + 同 seed 复现。"""

    print("[5] 保真对拍（M4）")
    from sim.fidelity import run_fidelity
    from sim.runner import SimConfig

    fails = []
    report = run_fidelity(SimConfig(capacity=2048, seed=11))
    print(
        f"    压力 trace: {report.requests} 请求，拓扑违反={len(report.violations)}"
    )
    for item in report.violations[:3]:
        print(f"      - {item}")
    if report.violations:
        fails.append(f"拓扑不变量被破坏：{report.violations[0]}")

    ledger = report.ledger or {}
    before = ledger.get("drift_before_reconcile", {})
    after = ledger.get("drift_after_reconcile", {})
    exact = ledger.get("exact_after_reconcile", False)
    drift_txt = " ".join(f"{q}={v:+.0f}" for q, v in sorted(before.items()))
    print(f"    账本漂移   : reconcile 前 {drift_txt} -> 后 {after}")
    print(
        f"    账本对拍   : {'reconcile 后逐键相等' if exact else '仍有漂移'}"
        f"{'' if exact else '  <-- 口径不一致'}"
    )
    if not exact:
        fails.append(f"reconcile 后账本仍与全树快照不一致：{after}")

    print(
        f"    同 seed   : {'逐字一致' if report.matched else '不一致'}"
        f" ({report.fingerprint})"
    )
    if not report.matched:
        fails.append("压力 trace 两次 run 的逐请求指纹不同")
    return fails


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim", description="M0/M1 自检")
    parser.add_argument(
        "--import-only", action="store_true", help="只验证装载，不构造缓存"
    )
    parser.add_argument(
        "--skip-clock", action="store_true", help="跳过 M1 虚拟时钟检查"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="保留策略 profile 日志（默认 quiet）"
    )
    args = parser.parse_args(argv)

    # 策略 profile 每 30 虚拟秒打一行，自检里纯噪声；用户显式设了就不覆盖。
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

    fails += _check_run(sym)
    fails += _check_allocator()
    if not args.skip_clock:
        fails += _check_clock(sym)
    fails += _check_driver()
    fails += _check_fidelity()
    if fails:
        print("\nFAIL")
        for item in fails:
            print(f"  - {item}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
