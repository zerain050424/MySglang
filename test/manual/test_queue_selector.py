"""按权重的跨队列淘汰：纯函数与策略接线单测（不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/queue_selector.py`
  queue_pressure  —— occ_q / w_q 降序
  pick_victim     —— Stage 0 one-shot → 权重队列 → 兜底
  struct_tier     —— STRUCT 两遍降级键

配额机制（caps / reservations / target / deadzone / chunk）已删除，对应的
水填充与门限测试一并移除；账本记账仍在 QueueOccupancyTracker。
"""

import importlib.util
import os
import sys
import time
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CACHE_DIR = os.path.join(_ROOT, "python", "sglang", "srt", "mem_cache")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


for _pkg in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
    _stub_package(_pkg)

hu = _load("sglang.srt.mem_cache.heter_utils", os.path.join(_CACHE_DIR, "heter_utils.py"))
# 占用账本已独立成模块，evict_policy 直接依赖它
_load("sglang.srt.mem_cache.queue_occupancy", os.path.join(_CACHE_DIR, "queue_occupancy.py"))
qs = _load("sglang.srt.mem_cache.queue_selector", os.path.join(_CACHE_DIR, "queue_selector.py"))
_load("sglang.srt.mem_cache.victim_override", os.path.join(_CACHE_DIR, "victim_override.py"))
_load("sglang.srt.mem_cache.heter_bench_bridge", os.path.join(_CACHE_DIR, "heter_bench_bridge.py"))
ep = _load("sglang.srt.mem_cache.evict_policy", os.path.join(_CACHE_DIR, "evict_policy.py"))

WEIGHTS = dict(hu.DEFAULT_QUEUE_WEIGHTS)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class _Leaf:
    """极简叶子替身：只带权重仲裁 / 记账 / 老化用到的字段。

    必须是普通类而不是 SimpleNamespace —— SLRU 的共享数账本是
    WeakKeyDictionary，需要节点可弱引用。
    """

    def __init__(self, tag, tokens, ts=None, session_id=None):
        self.queue_tag = tag
        self.value = [0] * tokens
        self.key = [0] * tokens
        self.last_access_time = time.monotonic() if ts is None else ts
        self.session_id = session_id


def _leaf(tag, tokens, ts=None, session_id=None):
    return _Leaf(tag, tokens, ts, session_id)


TAG = {
    "struct": hu.HeterCacheNodeTag.QSTRUCT,
    "agentic": hu.HeterCacheNodeTag.QAGENT,
    "chat": hu.HeterCacheNodeTag.QCHAT,
}


def _strategy(**kw):
    return ep.HeterCacheStrategy(weight={"weights": dict(WEIGHTS)}, **kw)


def test_pressure_orders_by_weighted_occupancy():
    """occ/w 降序：权重越大越晚被削；占用为 0 的队列不进入候选。"""
    print("[1] pressure = occ/w, zero-occupancy queues are skipped")
    occ = {"struct": 200, "agentic": 500, "chat": 300}
    check(qs.queue_pressure(occ, WEIGHTS) == ["struct", "agentic", "chat"],
          f"occ/w = 1000/1000/1000 must tie in WEIGHT_QUEUES order, "
          f"got {qs.queue_pressure(occ, WEIGHTS)}")
    occ = {"struct": 400, "agentic": 400, "chat": 200}
    check(qs.queue_pressure(occ, WEIGHTS) == ["struct", "agentic", "chat"],
          f"got {qs.queue_pressure(occ, WEIGHTS)}")
    check(qs.queue_pressure({"struct": 0, "agentic": 10, "chat": 0}, WEIGHTS) == ["agentic"],
          "empty queues must not be candidates")
    # 权重是比例语义：整体缩放不改变顺序
    scaled = {q: w * 10 for q, w in WEIGHTS.items()}
    check(qs.queue_pressure(occ, scaled) == qs.queue_pressure(occ, WEIGHTS),
          "weights are ratios; scaling them must not change the order")


def test_struct_tier():
    """STRUCT 分档：share==share_min 先走，高于阈值最后；同档内 LRU。"""
    print("[2] struct tiers: share==min first, then share>min, LRU inside a tier")

    class Node:
        def __init__(self, share, ts):
            self.share = share
            self.last_access_time = ts

    share_of = lambda n: n.share  # noqa: E731
    just = Node(2, 100.0)
    older_high = Node(3, 10.0)
    newer_high = Node(9, 200.0)
    key = lambda n: qs.struct_tier(n, share_of, share_min=2)  # noqa: E731
    order = sorted([newer_high, older_high, just], key=key)
    check(order[:1] == [just], f"share==min must go first, got {order}")
    check(order[1:] == [older_high, newer_high],
          f"share>min must be LRU inside the tier, got {order}")


def test_accounting_insert_and_evict():
    """记账：插入累加、淘汰扣减；one-shot 两侧都不进账。"""
    print("[3] accounting: insert adds, evict subtracts, one-shot is excluded")
    st = _strategy()
    st.record_insert(_leaf(TAG["agentic"], 100))
    st.record_insert(_leaf(TAG["struct"], 40))
    check(st.occupancy.occ["agentic"] == 100,
          f"agentic={st.occupancy.occ['agentic']}, want 100")
    check(st.occupancy.occ["struct"] == 40,
          f"struct={st.occupancy.occ['struct']}, want 40")

    st.record_evict(_leaf(TAG["agentic"], 30))
    check(st.occupancy.occ["agentic"] == 70,
          f"after evict agentic={st.occupancy.occ['agentic']}, want 70")

    # TRANS 会话 + share < share_min -> one-shot，插入与淘汰两侧都不进账
    # （now=0.0：会话早已搁置，超过 ONE_SHOT_IDLE_S 时间豁免窗口）
    trans = _leaf(TAG["struct"], 50, ts=0.0, session_id="t1")
    st.session_registry.touch("t1", "r1", prompt_len=50, now=0.0)
    st.session_registry.get("t1").session_type = hu.SessionType.TRANS
    check(st.is_one_shot(trans), "precondition: a TRANS node with share 0 is one-shot")
    before = st.occupancy.occ["struct"]
    st.record_insert(trans)
    check(st.occupancy.occ["struct"] == before,
          f"one-shot insert must not be counted, {before} -> {st.occupancy.occ['struct']}")
    st.record_evict(trans)
    check(st.occupancy.occ["struct"] == before,
          f"one-shot evict must not be counted, {before} -> {st.occupancy.occ['struct']}")


def test_random_is_never_counted():
    """random 不参与权重：无会话证据的 QTRANS 节点也不能进账。"""
    print("[4] accounting: random-classified nodes never enter the ledger")
    st = _strategy()
    rnd = _leaf(hu.HeterCacheNodeTag.QTRANS, 70)
    st.record_insert(rnd)
    check(st.occupancy.occ.get("random") is None,
          f"random must not be a ledger key, got {st.occupancy.occ}")
    st.record_evict(rnd)  # 不得 KeyError
    st.record_insert(_leaf(TAG["chat"], 30))
    check(st.occupancy.occ["chat"] == 30,
          f"chat={st.occupancy.occ['chat']}, want 30")


def test_accounting_cascade_decrements_each_node():
    """级联删除逐点扣减：每个被删节点都要走 record_evict。"""
    print("[5] accounting: cascade eviction decrements every deleted node")
    st = _strategy()
    for _ in range(5):
        st.record_insert(_leaf(TAG["chat"], 40))
    check(st.occupancy.occ["chat"] == 200, f"chat={st.occupancy.occ['chat']}, want 200")
    for _ in range(4):
        st.record_evict(_leaf(TAG["chat"], 40))
    check(st.occupancy.occ["chat"] == 40, f"chat={st.occupancy.occ['chat']}, want 40")
    # 多扣也不会变负
    st.record_evict(_leaf(TAG["chat"], 999))
    check(st.occupancy.occ["chat"] == 0.0,
          f"occ must clamp at 0, got {st.occupancy.occ['chat']}")


def test_snapshot_heals_drift():
    """reconcile：全树快照覆盖增量漂移（含 one-shot 不计账）。"""
    print("[6] reconcile restores ground truth after tag moves and splits")
    st = _strategy()
    st.record_insert(_leaf(TAG["agentic"], 500))
    st.occupancy.occ["agentic"] += 300
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 600, "one_shot": 0},
                            "chat": {"total": 200, "one_shot": 200}}, capacity=1000)
    check(st.occupancy.occ["agentic"] == 600,
          f"agentic={st.occupancy.occ['agentic']}, want 600")
    check(st.occupancy.occ["chat"] == 0,
          f"chat={st.occupancy.occ['chat']}, want 0 (整行 one-shot)")


def test_stage0_one_shot_preempts_weights():
    """Stage 0：one-shot 带永远先清，即使别的队列压力更大。"""
    print("[7] stage 0: one-shot drains first even when another queue is heavier")
    st = _strategy()
    st.occupancy.reconcile({"struct": {"total": 900, "one_shot": 0},
                            "agentic": {"total": 0, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    struct_node = _leaf(TAG["struct"], 100, ts=1.0)      # share 0 -> 不老化，但受保护判据
    one = _leaf(TAG["struct"], 40, ts=99.0, session_id="s1")
    st.session_registry.touch("s1", "r1", prompt_len=40, now=0.0)
    st.session_registry.get("s1").session_type = hu.SessionType.TRANS
    check(st.is_one_shot(one), "precondition: TRANS node is one-shot")
    victim, source = qs.pick_victim(strategy=st, leaves=[struct_node, one])
    check(victim is one and source == "one_shot",
          f"one-shot must win, got source={source}")


def test_stage1_labels_and_skip():
    """Stage 1：按权重点名队列，出候选时标 weight:<queue>，无候选则跳队。"""
    print("[8] stage 1: weighted queues, labels and skip-on-empty")
    st = ep.HeterCacheStrategy(
        multi_agentic={"enabled": True}, weight={"weights": dict(WEIGHTS)})
    st.occupancy.reconcile({"struct": {"total": 900, "one_shot": 0},
                            "agentic": {"total": 600, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    rank = st.pressure_ranking()
    check(rank[0] == "struct",
          f"struct has the highest occ/w (4500 vs 1200), got {rank}")

    # 第一名 struct 无候选（一个 struct 叶子都没给）-> 跳到 agentic
    agentic_node = _leaf(TAG["agentic"], 100)
    victim, source = qs.pick_victim(strategy=st, leaves=[agentic_node])
    check(victim is agentic_node and source == "weight:agentic",
          f"must skip the empty struct queue, got {source}")

    victim, source = qs.pick_victim(strategy=st, leaves=[])
    check(victim is None and source is None,
          f"empty leaves must yield (None, None), got {(victim, source)}")


def test_fallback_is_seeded_random():
    """兜底：固定种子可复现，且不污染全局 random。"""
    print("[9] fallback: seeded RNG, reproducible, isolated")
    import random as _random

    now = time.monotonic()
    leaves = [_leaf(TAG["chat"], 10, ts=now - i) for i in range(30)]
    index = {id(n): i for i, n in enumerate(leaves)}

    def run(seed):
        st = ep.HeterCacheStrategy(weight={"weights": dict(WEIGHTS), "seed": seed})
        st.occupancy.reset()  # 全 0 -> 没有任何队列有压力 -> 直接兜底
        picks = []
        for _ in range(6):
            victim, source = qs.pick_victim(strategy=st, leaves=leaves)
            picks.append((index[id(victim)], source))
        return picks

    a, b, c = run(0), run(0), run(7)
    check(a == b, f"same seed must reproduce, got {a} vs {b}")
    check(a != c, "different seed should differ")
    check(all(src == "weight:fallback" for _, src in a),
          f"all picks must be labelled weight:fallback, got {a}")
    state_before = _random.getstate()
    run(0)
    check(_random.getstate() == state_before, "fallback must not touch global random")


def test_weight_params_validation():
    """权重必填三队、必须为正；比例等价。"""
    print("[10] weight params: all three queues, positive, ratio-invariant")
    check(hu.QueueWeightParams().weights == hu.DEFAULT_QUEUE_WEIGHTS,
          "missing config must fall back to DEFAULT_QUEUE_WEIGHTS")
    for bad in (
        {"struct": 0.2, "agentic": 0.5},                      # 缺 chat
        {"struct": 0.2, "agentic": 0.5, "chat": 0.3, "x": 1},  # 未知键
        {"struct": 0.0, "agentic": 0.5, "chat": 0.3},          # 非正
    ):
        try:
            hu.QueueWeightParams(weights=bad)
        except ValueError:
            pass
        else:
            check(False, f"invalid weights must be rejected: {bad}")
    # 比例等价：occ/w 在整体缩放下不变
    st = _strategy()
    st.occupancy.reconcile({"struct": {"total": 400, "one_shot": 0},
                            "agentic": {"total": 400, "one_shot": 0},
                            "chat": {"total": 200, "one_shot": 0}}, capacity=1000)
    scaled = ep.HeterCacheStrategy(
        weight={"weights": {q: w * 10 for q, w in WEIGHTS.items()}})
    scaled.occupancy.reconcile({"struct": {"total": 400, "one_shot": 0},
                                "agentic": {"total": 400, "one_shot": 0},
                                "chat": {"total": 200, "one_shot": 0}}, capacity=1000)
    check(st.pressure_ranking() == scaled.pressure_ranking(),
          "scaling all weights must not change the queue order")


def test_multi_agentic_orders_by_reuse_score():
    """MultiAgenticEvict：agentic 队内按复用分升序淘汰。"""
    print("[11] multi_agentic: agentic pool ordered by reuse score")
    st = _strategy(multi_agentic={"enabled": True})
    check(st.multi_agentic_evict is not None,
          "multi_agentic.enabled must build the picker")

    now = time.monotonic()
    expired = _leaf(TAG["agentic"], 100, ts=now - 200)  # 复用窗口已过期（未到老化 TTL）
    fresh = _leaf(TAG["agentic"], 400, ts=now - 1)      # 刚写入，仍可能被复用
    chat = _leaf(TAG["chat"], 10, ts=now - 3600)        # 不归本队
    check(st.multi_agentic_evict.pick([expired, fresh, chat]) is expired,
          "lowest reuse score (oldest elapsed) must be evicted first")

    small = _leaf(TAG["agentic"], 10, ts=now - 200)
    large = _leaf(TAG["agentic"], 90, ts=now - 200)
    check(st.multi_agentic_evict.pick([small, large]) is large,
          "same reuse score must prefer the larger segment")
    check(st.multi_agentic_evict.pick([]) is None, "empty pool must yield None")

    # 关闭时整个改写器不构造：agentic 无队内 picker
    off = _strategy()
    check(off.multi_agentic_evict is None, "multi_agentic must be opt-in")

    # force_agentic：所有标签都归 agentic，评分必须用 QAGENT 的分布
    fa = _strategy(force_agentic=True, multi_agentic={"enabled": True})
    struct_expired = _leaf(TAG["struct"], 100, ts=now - 3600)
    expected = hu.reuse_window_probability(
        3600, hu.DEFAULT_REUSE_WINDOW_S, *hu.HeterCacheNodeTag.QAGENT.reuse_config
    )
    check(abs(fa.multi_agentic_evict.reuse_score(struct_expired) - expected) < 1e-9,
          "force_agentic must score every node with the QAGENT distribution")
    check(fa.multi_agentic_evict.pick([struct_expired]) is struct_expired,
          "force_agentic must pool every tag into the agentic band")

    # 接线：权重点名 agentic 时确实由本改写器出候选
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 900, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    victim, source = qs.pick_victim(strategy=st, leaves=[expired, fresh])
    check(victim is expired and source == "weight:agentic",
          f"weight:agentic must use the reuse score, got {source}")


def test_queue_policy_assignment():
    """队内策略归属：chat 走 RR，agentic 只由 multi_agentic 出候选。"""
    print("[12] chat uses RR, agentic is served by multi_agentic only")
    st = ep.HeterCacheStrategy(
        rr={"fair_rr": True}, weight={"weights": dict(WEIGHTS)})
    check(st.rr_evict.queue is hu.HeterCacheNodeTag.QCHAT,
          f"RR must own the chat queue, got {st.rr_evict.queue}")

    now = time.monotonic()
    c1 = _leaf(TAG["chat"], 100, ts=now - 10, session_id="swap-c1")
    c2 = _leaf(TAG["chat"], 100, ts=now - 20, session_id="swap-c2")
    for sid in ("swap-c1", "swap-c2"):
        st.session_registry.touch(sid, f"r-{sid}", prompt_len=1000, now=now)
    first = st.queue_victim("chat", [c1, c2])
    second = st.queue_victim("chat", [c1, c2])
    check({id(first), id(second)} == {id(c1), id(c2)},
          "RR must rotate between the two chat sessions")

    off = ep.HeterCacheStrategy(weight={"weights": dict(WEIGHTS)})
    check(off.rr_evict is None and off.queue_victim("chat", [c1, c2]) is None,
          "without fair_rr the chat queue abstains")

    expired = _leaf(TAG["agentic"], 100, ts=now - 200, session_id="swap-a1")
    fresh = _leaf(TAG["agentic"], 100, ts=now - 1, session_id="swap-a2")
    check(st.multi_agentic_evict is None and
          st.queue_victim("agentic", [expired, fresh]) is None,
          "agentic must abstain when multi_agentic is off")
    ma = ep.HeterCacheStrategy(
        rr={"fair_rr": True}, multi_agentic={"enabled": True},
        weight={"weights": dict(WEIGHTS)})
    check(ma.queue_victim("agentic", [expired, fresh]) is expired,
          "multi_agentic must pick the node whose reuse window already expired")


def test_reuse_window_probability():
    """复用分 = P(reuse in (a, a+H] | not reused by a)，不是 1 - CDF(a)。"""
    print("[13] reuse score is the conditional window probability")
    mu, sigma = hu.HeterCacheNodeTag.QAGENT.reuse_config
    H = hu.DEFAULT_REUSE_WINDOW_S

    check(abs(hu.reuse_window_probability(0, H, mu, sigma)
              - hu._cached_lognorm_cdf(H, mu, sigma)) < 1e-9,
          "at age 0 the conditional probability must equal CDF(H)")

    p90 = hu.reuse_window_probability(10, H, mu, sigma)
    p180 = hu.reuse_window_probability(10, 2 * H, mu, sigma)
    check(p90 < p180, f"longer horizon must not lower the score: {p90} vs {p180}")

    p600 = hu.reuse_window_probability(600, H, mu, sigma)
    p3600 = hu.reuse_window_probability(3600, H, mu, sigma)
    check(0.0 < p3600 < p600 < 1.0,
          f"stale nodes must keep decreasing: p600={p600}, p3600={p3600}")
    check(hu.reuse_window_probability(10 ** 9, H, mu, sigma) == 0.0,
          "underflowed survival must score 0, not raise")

    st = _strategy(multi_agentic={"enabled": True})
    now = time.monotonic()
    young = _leaf(TAG["agentic"], 100, ts=now - 2)
    stale = _leaf(TAG["agentic"], 100, ts=now - 200)
    check(st.multi_agentic_evict.pick([young, stale]) is stale,
          "the lowest future-reuse probability must be evicted first")
    check(st.multi_agentic_evict.pick([young]) is young,
          "a single candidate must still be picked")


def test_per_type_ageing_ttl():
    """按类型老化：struct < chat < agentic，random 永不老化；带内最优先。"""
    print("[14] per-type ageing TTL: struct < chat < agentic, random off")
    ttl = hu.DEFAULT_AGEING_TTL_S
    check(ttl["struct"] < ttl["chat"] < ttl["agentic"],
          f"TTL must grow with re-prefill cost: {ttl}")
    check(ttl["random"] == 0, "random nodes are one-shot already; no ageing")
    st = _strategy()
    now = time.monotonic()

    def node(tag, age_s, share=0, tokens=100):
        n = _leaf(tag, tokens, ts=now - age_s)
        for i in range(share):
            st.slru_evict.on_node_hit(n, f"{id(n)}-{i}")
        n.hit_count = 1 if share else 0
        return n

    aged = node(TAG["struct"], ttl["struct"] + 1, share=st.slru_evict.share_min)
    fresh = node(TAG["struct"], ttl["struct"] - 1, share=st.slru_evict.share_min)
    check(st.is_aged(aged), "shared struct ages at the struct TTL")
    check(not st.is_aged(fresh), "idle<ttl must keep the template protected")
    check(st.is_one_shot(aged), "aged shared prefix must enter the one-shot band")
    check(not st.is_one_shot(fresh), "fresh shared prefix must stay in struct")
    check(not st.is_aged(node(TAG["struct"], ttl["struct"] + 1, share=0)),
          "unshared struct is left to the original 90s one-shot rule")

    check(not st.is_aged(node(TAG["chat"], ttl["struct"] + 1)),
          "chat must not age at the struct TTL")
    check(st.is_aged(node(TAG["chat"], ttl["chat"] + 1)), "chat ages at chat TTL")
    check(not st.is_aged(node(TAG["agentic"], ttl["chat"] + 1)),
          "agentic must not age at the chat TTL")
    check(st.is_aged(node(TAG["agentic"], ttl["agentic"] + 1)),
          "agentic ages at the agentic TTL")
    check(not st.is_aged(_leaf(hu.HeterCacheNodeTag.QTRANS, 100, ts=now - 10 ** 6)),
          "random class never ages")

    plain = _leaf(TAG["struct"], 100, ts=now - ttl["struct"] * 10)
    check(st.one_shot_key(aged) < st.one_shot_key(plain),
          "aged nodes must outrank plain one-shot inside the band")

    partial = ep.HeterCacheStrategy(
        slru={"ageing_ttl_s": {"struct": 10}}, weight={"weights": dict(WEIGHTS)})
    check(partial.slru.ageing_ttl_s["struct"] == 10 and
          partial.slru.ageing_ttl_s["chat"] == ttl["chat"],
          "a partial override must merge over the defaults")
    off = ep.HeterCacheStrategy(
        slru={"ageing_ttl_s": {"struct": 0}}, weight={"weights": dict(WEIGHTS)})
    check(not off.is_aged(aged), "0 disables ageing for that type only")
    for bad in ({"nope": 1}, {"struct": -1}):
        try:
            ep.HeterCacheStrategy(
                slru={"ageing_ttl_s": bad}, weight={"weights": dict(WEIGHTS)})
        except ValueError:
            pass
        else:
            check(False, f"invalid ageing_ttl_s must be rejected: {bad}")

    fa = _strategy(force_agentic=True)
    check(fa.is_aged(node(TAG["struct"], ttl["agentic"] + 1)),
          "force_agentic must score every node with the agentic TTL")

    before = st.slru_evict.aged_evicted
    st.record_evict(aged)
    check(st.slru_evict.aged_evicted == before + 100,
          f"aged evict must count {before + 100} tokens, "
          f"got {st.slru_evict.aged_evicted}")
    st.record_evict(fresh)
    check(st.slru_evict.aged_evicted == before + 100,
          "a non-aged evict must not be counted as aged")

    victim, source = qs.pick_victim(strategy=st, leaves=[aged, fresh])
    check(victim is aged and source == "one_shot",
          f"aged node must win Stage 0, got {source}")


def main():
    tests = [
        test_pressure_orders_by_weighted_occupancy,
        test_struct_tier,
        test_accounting_insert_and_evict,
        test_random_is_never_counted,
        test_accounting_cascade_decrements_each_node,
        test_snapshot_heals_drift,
        test_stage0_one_shot_preempts_weights,
        test_stage1_labels_and_skip,
        test_fallback_is_seeded_random,
        test_weight_params_validation,
        test_multi_agentic_orders_by_reuse_score,
        test_queue_policy_assignment,
        test_reuse_window_probability,
        test_per_type_ageing_ttl,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            FAILS.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)} problems)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
