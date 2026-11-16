"""RegretLedger（淘汰后悔率账本）的回归测试（不需要 torch）。

钉三件事：

1. **内容键两侧同值**：请求侧（全量 ids）与淘汰侧（祖先链尾部窗口）算出的
   键必须相等——这是 join 的全部前提，对不上号时 regret 恒 0 且不报错。
2. **一次决策一个 stamp，归因只给最浅页**：``_evict_one`` 级联清掉整条
   One-shot 私有链时只落一个 stamp，``span`` = 全链腾出量；断点在更深页的
   请求不允许 join（first-miss 归因，深页的损失是免费的）。
3. **删失显式**：没到 H 的 stamp 计 censored，不冒充「没回访」；join_rate
   过低时渲染必须带 WARN，regret=0 不能看起来像策略完美。

    python test/manual/test_regret_ledger.py
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "..", "python")
ROOT = os.path.join(HERE, "..", "..")
PROFILE_DIR = os.path.join(ROOT, "Heter-Bench", "profile")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeClock:
    """可手动推进的单调时钟（虚拟秒）。"""

    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# --------------------------------------------------------------------------
# 装载：stub 掉 sglang 包链，把真正的 utils.py / regret_ledger.py 按路径装进来
# --------------------------------------------------------------------------

HU = "sglang.srt.mem_cache.heter_cache.utils"
LEDGER = "regret_ledger"


def _bootstrap_ledger():
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.heter_cache",
    ):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    _load(HU, os.path.join(PY, "sglang/srt/mem_cache/heter_cache/utils.py"))
    return _load(LEDGER, os.path.join(PROFILE_DIR, "regret_ledger.py"))


# --------------------------------------------------------------------------
# 假树节点（驱动层接线用）
# --------------------------------------------------------------------------


class FakeKey:
    """只提供 regret 需要的 RadixKey 读法。"""

    def __init__(self, tokens, extra_key=None):
        self.token_ids = list(tokens)
        self.is_bigram = False
        self.extra_key = extra_key
        self.cache_salt = None

    def raw_token_ids(self):
        return self.token_ids

    def __len__(self):
        return len(self.token_ids)

    def child_key(self, page_size):
        return f"k{self.token_ids[:1]}"


class FakeNode:
    _next = 0

    def __init__(self, tokens, abs_start, session_id, tag, last_access=0.0):
        FakeNode._next += 1
        self.id = FakeNode._next
        self.ckey = f"n{self.id}"
        self.key = FakeKey(list(range(abs_start, abs_start + tokens)))
        self.value = [0] * tokens
        self.abs_start = abs_start
        self.session_id = session_id
        self.queue_tag = tag
        self.last_access_time = last_access
        self.hit_count = 0
        self.reuse_count = 0
        self.share_sessions = 0
        self.session_bloom = 0
        self.parent = None
        self.children = {}
        self.lock_ref = 0
        self.is_decode = False


class FakeAllocator:
    def __init__(self):
        self.freed = []
        self.size = 1000

    def free_segment(self, value, *, start_pos=0):
        self.freed.append(tuple(value))

    def available_size(self):
        return 800


class FakeEvents:
    def record_remove(self, node):
        pass


# --------------------------------------------------------------------------
# 用例一：账本（纯逻辑）
# --------------------------------------------------------------------------

rl = _bootstrap_ledger()
hu = sys.modules[HU]
HORIZONS = (30.0, 120.0)


def _ledger(now=None, **kw):
    clock = now if isinstance(now, FakeClock) else FakeClock(now or 0.0)
    led = rl.RegretLedger(
        now=clock,
        horizons=HORIZONS,
        queue_horizon=rl.queue_horizons_from_policy(),
        **kw,
    )
    return led, clock


def _key(window, page_start, page_size, base=0, ns=0):
    return rl.content_key(
        window, page_start, page_size, base=base, ns=ns, context=rl.CONTEXT_TOKENS
    )


def test_content_key_symmetry():
    """请求侧（全量 ids）与淘汰侧（尾部窗口）必须算出同一个键。"""
    print("[1] content_key: request side == evict side")
    ids = list(range(1000, 1100))
    page_size = 1
    ns = rl.namespace_key(None, None)
    for matched in (0, 1, 16, 17, 50, 99):
        want = _key(ids, matched, page_size, base=0, ns=ns)
        tail_start = max(0, matched - rl.CONTEXT_TOKENS)
        tail = ids[tail_start : matched + page_size]
        got = _key(tail, matched, page_size, base=tail_start, ns=ns)
        check(
            got == want,
            f"matched={matched}: evict key != request key ({got} vs {want})",
        )
    # 命名空间不同 → 键不同（跨命名空间不得误 join）
    check(
        _key(ids, 50, page_size, ns=rl.namespace_key("a", None))
        != _key(ids, 50, page_size, ns=rl.namespace_key("b", None)),
        "different extra_key must not share a content key",
    )
    # 位置不同 → 键不同（同内容不同位置不得误 join）
    check(
        _key(ids, 50, page_size, ns=ns) != _key(ids, 51, page_size, ns=ns),
        "same content at different position must not share a key",
    )


def test_one_stamp_per_decision_and_first_miss():
    """级联只落一个 stamp，span=全链；深页不允许 join。"""
    print("[2] one stamp per decision; only the shallowest page can join")
    led, clock = _ledger()
    ids = list(range(100))
    ns = rl.namespace_key(None, None)
    # 一次决策腾出 [16, 48)：浅节点 [16,32) + 级联清理的 [32,48)
    window = ids[0:17]
    led.on_evict(
        key=_key(window, 16, 1, base=0, ns=ns),
        span=32,
        source="one_shot",
        queue="random",
        share=0,
        chain=2,
        depth=16,
        elapsed=1000.0,
        ts=0.0,
    )
    # 深页断点（matched=32）不得 join：它对请求是免费的
    joined = led.on_request(
        key=_key(ids, 32, 1, ns=ns), recompute=68, prompt=100, ts=1.0
    )
    check(joined is False, "deep boundary must not join a cascaded stamp")
    check(led._requests == 1 and led._recompute_tokens == 68, "tax must still count")
    # 浅页断点（matched=16）join 到整链 span
    joined = led.on_request(
        key=_key(ids, 16, 1, ns=ns), recompute=84, prompt=100, ts=2.0
    )
    check(joined is True, "shallow boundary must join")
    rep = led.report()
    check(rep["tax"]["joined"] == 1, f"joined={rep['tax']['joined']}, want 1")
    cell = rep["groups"]["source"]["own"]["one_shot"]
    check(cell["evicted"] == 32, f"evicted={cell['evicted']}, want 32")
    check(cell["regret"] == 32, f"regret={cell['regret']}, want 32 (span, not recompute)")
    check(cell["regret_pct"] == 100.0, "hit within every horizon -> 100%")


def test_horizon_curve_and_censoring():
    """没到 H 的 stamp 是 censored，不是「没回访」。"""
    print("[3] censoring must be explicit; own-H uses the queue's T50")
    led, clock = _ledger()
    ids = list(range(50))
    ns = rl.namespace_key(None, None)
    # agentic：own H = T50 ≈ 6.11s；同一 stamp 在 H=30 档必然也命中
    led.on_evict(
        key=_key(ids, 8, 1, ns=ns), span=10, source="value:agentic",
        queue="agentic", share=0, chain=1, depth=8, elapsed=1.0, ts=0.0,
    )
    led.on_request(key=_key(ids, 8, 1, ns=ns), recompute=42, prompt=50, ts=3.0)
    rep = led.report()
    cell = rep["groups"]["queue"]["own"]["agentic"]
    check(abs(rep["queue_horizon"]["agentic"] - hu.reuse_interval_median(
        hu.HeterCacheNodeTag.QAGENT)) < 1e-9, "own H must be the queue's T50")
    check(cell["regret_pct"] == 100.0, "hit at age 3 <= T50(6.1) -> 100%")
    check(cell["censored_pct"] == 0.0, "joined stamp is not censored")
    # chat 的 own H ≈ 118s：一个 3s 内没回访的 stamp 在 H=30 档还是 censored
    led2, clock2 = _ledger()
    led2.on_evict(
        key=_key(ids, 8, 1, ns=ns), span=10, source="value:chat",
        queue="chat", share=0, chain=1, depth=8, elapsed=1.0, ts=0.0,
    )
    clock2.advance(10.0)
    rep = led2.report()
    cell = rep["groups"]["queue"]["own"]["chat"]
    check(cell["regret_pct"] == 0.0, "no revisit yet -> 0 regret")
    check(cell["censored_pct"] == 100.0,
          f"age 10s < T50(118s) must be censored, got {cell['censored_pct']}")
    check(rep["groups"]["queue"][30.0]["chat"]["censored_pct"] == 100.0,
          "age 10s < H=30 is censored too")
    clock2.advance(1000.0)
    rep = led2.report()
    cell = rep["groups"]["queue"]["own"]["chat"]
    check(cell["censored_pct"] == 0.0 and cell["regret_pct"] == 0.0,
          "past H with no revisit -> observed negative, not censored")


def test_overwrite_and_expiry():
    """同一内容再次逐出覆盖旧 stamp；超出保留时长的样本按过期计。"""
    print("[4] re-eviction overwrites; retention expires old stamps")
    led, clock = _ledger(retain_s=50.0)
    ids = list(range(50))
    ns = rl.namespace_key(None, None)
    key = _key(ids, 8, 1, ns=ns)
    led.on_evict(key=key, span=10, source="one_shot", queue="random",
                 share=0, chain=1, depth=8, elapsed=1.0, ts=0.0)
    led.on_evict(key=key, span=20, source="value:chat", queue="chat",
                 share=0, chain=1, depth=8, elapsed=1.0, ts=5.0)
    check(led._dropped == 1, f"old stamp overwritten, dropped={led._dropped}")
    # 没过保留期：join 归给第二次决策（span=20）
    led.on_request(key=key, recompute=40, prompt=50, ts=6.0)
    cell = led.report()["groups"]["queue"]["own"]["chat"]
    check(cell["regret"] == 20, f"regret must follow the newest decision, got {cell}")
    # 过期：旧内容逐出后 100s 才回访，坟场里已经没有它
    led2, clock2 = _ledger(retain_s=50.0)
    led2.on_evict(key=key, span=10, source="one_shot", queue="random",
                  share=0, chain=1, depth=8, elapsed=1.0, ts=0.0)
    clock2.advance(100.0)
    joined = led2.on_request(key=key, recompute=40, prompt=50)
    check(joined is False, "expired stamp must not join")
    check(led2._expired == 1, f"expired must be counted, got {led2._expired}")


def test_join_rate_zero_warns():
    """join_rate=0 时渲染必须带 WARN：regret=0 更可能是没对上号。"""
    print("[5] join_rate=0 must render a WARN line")
    led, clock = _ledger()
    led.on_request(key=None, recompute=10, prompt=100)
    text = led.emit()
    check("WARN" in text, f"render must warn when nothing joins:\n{text}")


def test_calibration_uses_queue_distribution():
    """校准：预测值来自该队列的复用分布，实测值来自 join。"""
    print("[6] calibration: predicted (lognormal) vs realized")
    led, clock = _ledger()
    ids = list(range(50))
    ns = rl.namespace_key(None, None)
    t50 = hu.reuse_interval_median(hu.HeterCacheNodeTag.QAGENT)
    # elapsed=0、H=T50：P(reuse in (0, T50]) = CDF(T50) = 0.5
    led.on_evict(key=_key(ids, 8, 1, ns=ns), span=10, source="value:agentic",
                 queue="agentic", share=0, chain=1, depth=8, elapsed=0.0, ts=0.0)
    led.on_request(key=_key(ids, 8, 1, ns=ns), recompute=40, prompt=50, ts=t50 / 2)
    calib = led.report()["calibration"]["own"]
    bucket = next(iter(calib.values()))
    check(abs(bucket["predicted"] - 0.5) < 0.01,
          f"predicted must be ~0.5 at elapsed=0,H=T50, got {bucket['predicted']}")
    check(bucket["realized"] == 1.0, f"realized must be 1.0, got {bucket['realized']}")
    check(bucket["tokens"] == 10, f"token-weighted, got {bucket['tokens']}")


def test_reset_and_cap():
    """reset 清空一切；触顶丢最老的样本并计数。"""
    print("[7] reset clears; capacity cap drops oldest")
    led, clock = _ledger(max_stamps=2)
    ids = list(range(50))
    ns = rl.namespace_key(None, None)
    for i in range(3):
        led.on_evict(key=_key(ids, i, 1, ns=ns), span=1, source="one_shot",
                     queue="random", share=0, chain=1, depth=i,
                     elapsed=1.0, ts=float(i))
    check(len(led._stamps) == 2 and led._dropped == 1,
          f"cap must keep newest 2, stamps={len(led._stamps)} dropped={led._dropped}")
    led.reset()
    rep = led.report()
    check(rep["tax"]["stamps"] == 0 and rep["tax"]["requests"] == 0,
          "reset must clear stamps and counters")


# --------------------------------------------------------------------------
# 用例二：驱动层接线（假树 + 真 strategy + 真 radix_cache）
# --------------------------------------------------------------------------

evl = importlib.util.spec_from_file_location(
    "test_evict_loop", os.path.join(HERE, "test_evict_loop.py")
)
_evl = importlib.util.module_from_spec(evl)
sys.modules["test_evict_loop"] = _evl
evl.loader.exec_module(_evl)
ep, shc, base = _evl._bootstrap()


def _strategy(clock, sessions):
    hu_mod = sys.modules[HU]
    st = ep.HeterCacheStrategy(now=clock)
    reg = hu_mod.SessionRegistry()
    for sid, stype in sessions.items():
        reg._sessions[sid] = hu_mod.SessionState(
            session_id=sid, session_type=stype
        )
    st.session_registry = reg
    return st


def _cache(st, nodes, led, clock):
    cache = shc.HeterCacheRadixCache.__new__(shc.HeterCacheRadixCache)
    cache.disable = False
    cache.eviction_strategy = st
    cache.session_registry = st.session_registry
    cache.node_share = st.node_share
    cache.occupancy = st.occupancy
    cache.telemetry = st.telemetry
    cache.realized = st.realized
    cache._needs_capacity = False
    cache.page_size = 1
    cache.is_eagle = False
    cache._now = clock
    cache._regret = led
    cache._regret_source = "unknown"
    cache._hit_recorder = None
    cache._record_insert = None
    cache.force_agentic = False
    cache.token_to_kv_pool_allocator = FakeAllocator()
    cache.kv_events = FakeEvents()
    cache.metrics = []
    cache.deleted = []
    hu_mod = sys.modules[HU]
    cache.root_node = FakeNode(0, 0, None, hu_mod.HeterCacheNodeTag.QTRANS)
    cache.root_node.ckey = "root"
    cache.evictable_leaves = set()
    for node in nodes:
        node.parent = cache.root_node
        cache.root_node.children[node.ckey] = node
        cache.evictable_leaves.add(node)
    # one-shot 节点不进占用账本（record_insert 自己会跳过）
    for node in nodes:
        if not st.is_one_shot(node):
            st.record_insert(node)
    shc.HeterCacheRadixCache._delete_leaf = _evl._delete_leaf
    shc.HeterCacheRadixCache.update_eviction_metrics = _evl._update_eviction_metrics
    shc.HeterCacheRadixCache._per_queue_token_count = _evl._per_queue_token_count
    return cache


def test_driver_stamps_one_decision():
    """one-shot 级联：一个 stamp、span=全链、source/queue 取牺牲节点口径。"""
    print("[8] driver: cascade eviction produces exactly one stamp")
    clock = FakeClock(1000.0)
    st = _strategy(clock, {"t1": hu.SessionType.TRANS})
    led, _ = _ledger(now=clock)
    parent = FakeNode(16, 0, "t1", hu.HeterCacheNodeTag.QTRANS, last_access=0.0)
    leaf = FakeNode(16, 16, "t1", hu.HeterCacheNodeTag.QTRANS, last_access=0.0)
    parent.children[leaf.ckey] = leaf
    leaf.parent = parent
    cache = _cache(st, [parent], led, clock)
    cache.evictable_leaves = {leaf}
    check(st.is_one_shot(leaf) is True, "sanity: idle trans leaf is one-shot")

    result = cache.evict(base.EvictParams(num_tokens=8))
    check(result.num_tokens_evicted == 32,
          f"cascade must free the whole chain, got {result.num_tokens_evicted}")
    check(len(led._stamps) == 1, f"one stamp per decision, got {len(led._stamps)}")
    stamp = next(iter(led._stamps.values()))
    # stamp: [ts, span, source, queue, rule, share, chain, depth, elapsed, hit_age]
    check(stamp[1] == 32, f"span must be the whole chain, got {stamp[1]}")
    check(stamp[2] == "one_shot", f"source must follow the arbitration, got {stamp[2]}")
    check(stamp[4] == "one_shot_key", f"rule derived from source, got {stamp[4]}")
    check(stamp[6] == 2, f"chain must count both nodes, got {stamp[6]}")
    check(stamp[7] == 0, f"depth is the break page (parent's), got {stamp[7]}")

    # 回访：断点在浅页（matched=0）→ join 到整链 span
    ids = parent.key.raw_token_ids() + leaf.key.raw_token_ids()
    joined = led.on_request(
        key=rl.content_key(ids, 0, 1, ns=rl.namespace_key(None, None)),
        recompute=32, prompt=32,
    )
    check(joined is True, "returning request at the break page must join")
    cell = led.report()["groups"]["source"]["own"]["one_shot"]
    check(cell["regret_pct"] == 100.0, f"regret must be 100%, got {cell['regret_pct']}")


def test_driver_without_ledger_is_inert():
    """没装配账本（默认 HETER_REGRET 关）时，淘汰路径照常工作。"""
    print("[9] driver: no ledger -> hooks are inert")
    clock = FakeClock(1000.0)
    st = _strategy(clock, {"t1": hu.SessionType.TRANS})
    parent = FakeNode(16, 0, "t1", hu.HeterCacheNodeTag.QTRANS, last_access=0.0)
    leaf = FakeNode(16, 16, "t1", hu.HeterCacheNodeTag.QTRANS, last_access=0.0)
    parent.children[leaf.ckey] = leaf
    leaf.parent = parent
    cache = _cache(st, [parent], None, clock)
    cache.evictable_leaves = {leaf}
    result = cache.evict(base.EvictParams(num_tokens=8))
    check(result.num_tokens_evicted == 32,
          f"eviction must be unaffected, got {result.num_tokens_evicted}")


def test_driver_window_walks_parent_chain():
    """_regret_window 从祖先链取页前上下文，与请求侧键等值。"""
    print("[10] driver: window walk == request-side content key")
    clock = FakeClock(1000.0)
    st = _strategy(clock, {"a1": hu.SessionType.AGENTIC})
    led, _ = _ledger(now=clock)
    head = FakeNode(16, 0, "a1", hu.HeterCacheNodeTag.QAGENT, last_access=1000.0)
    mid = FakeNode(16, 16, "a1", hu.HeterCacheNodeTag.QAGENT, last_access=1000.0)
    tail = FakeNode(16, 32, "a1", hu.HeterCacheNodeTag.QAGENT, last_access=1000.0)
    head.children[mid.ckey] = mid
    mid.parent = head
    mid.children[tail.ckey] = tail
    tail.parent = mid
    cache = _cache(st, [head], led, clock)

    window, base = cache._regret_window(tail)
    check(base == 16, f"base must back off to the context start, got {base}")
    ids = head.key.raw_token_ids() + mid.key.raw_token_ids() + tail.key.raw_token_ids()
    want = rl.content_key(ids, 32, 1, ns=rl.namespace_key(None, None))
    got = rl.content_key(
        window, 32, 1, base=base, context=led.context,
        ns=rl.namespace_key(None, None),
    )
    check(got == want, f"driver window must match the request-side key ({got} vs {want})")


def main():
    tests = [
        test_content_key_symmetry,
        test_one_stamp_per_decision_and_first_miss,
        test_horizon_curve_and_censoring,
        test_overwrite_and_expiry,
        test_join_rate_zero_warns,
        test_calibration_uses_queue_distribution,
        test_reset_and_cap,
        test_driver_stamps_one_decision,
        test_driver_without_ledger_is_inert,
        test_driver_window_walks_parent_chain,
    ]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILS.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)} problems)")
        for item in FAILS:
            print("  -", item)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
