"""按权重的跨队列淘汰：纯函数与策略接线单测（不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/heter_cache/selector.py`
  queue_order     —— 纯 VD 升序（占用只做同分兜底）
  pick_victim     —— Stage 0 one-shot → VD 仲裁 → 兜底
  struct_tier     —— STRUCT 两遍降级键

权重（weights）已整体删除：仲裁只看已兑现价值密度 VD。配额机制（caps /
reservations / target / deadzone / chunk）同样不在；账本记账仍在
QueueOccupancyTracker，VD 记账在 QueueValueTracker。
"""

import importlib.util
import os
import sys
import time
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CACHE_DIR = os.path.join(_ROOT, "python", "sglang", "srt", "mem_cache")
_HC_DIR = os.path.join(_CACHE_DIR, "heter_cache")


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


for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.heter_cache",
):
    _stub_package(_pkg)

hu = _load(
    "sglang.srt.mem_cache.heter_cache.utils", os.path.join(_HC_DIR, "utils.py")
)
# 占用账本已独立成模块，evict_policy 直接依赖它
_load(
    "sglang.srt.mem_cache.heter_cache.occupancy",
    os.path.join(_HC_DIR, "occupancy.py"),
)
_load("sglang.srt.mem_cache.heter_cache.value", os.path.join(_HC_DIR, "value.py"))
qs = _load(
    "sglang.srt.mem_cache.heter_cache.selector", os.path.join(_HC_DIR, "selector.py")
)
# 共享数账本已独立成模块（evict_policy 直接依赖它）
_load(
    "sglang.srt.mem_cache.heter_cache.node_share",
    os.path.join(_HC_DIR, "node_share.py"),
)
_load(
    "sglang.srt.mem_cache.heter_cache.override", os.path.join(_HC_DIR, "override.py")
)
_load(
    "sglang.srt.mem_cache.heter_cache.bench_bridge",
    os.path.join(_HC_DIR, "bench_bridge.py"),
)
ep = _load("sglang.srt.mem_cache.evict_policy", os.path.join(_CACHE_DIR, "evict_policy.py"))

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class _Leaf:
    """极简叶子替身：只带权重仲裁 / 记账 / 老化用到的字段。

    必须是普通类而不是 SimpleNamespace —— 共享数账本是
    WeakKeyDictionary，需要节点可弱引用。
    """

    _next_id = 0

    def __init__(self, tag, tokens, ts=None, session_id=None):
        _Leaf._next_id += 1
        self.id = _Leaf._next_id          # 平局判据（one_shot_key / struct_tier 末尾）
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
    return ep.HeterCacheStrategy(**kw)


def test_queue_order_by_value_density():
    """纯 VD 升序：密度最低的队列先削；占用不参与排序键，只做候选过滤与同分兜底。"""
    print("[1] order = ascending VD, occupancy never in the sort key")
    occ = {"struct": 200, "agentic": 500, "chat": 300}
    # 没有 VD 证据时三队同分，退化为按占用降序（一次决策腾出更多 token）
    check(qs.queue_order(occ, {}) == ["agentic", "chat", "struct"],
          f"got {qs.queue_order(occ, {})}")
    check(qs.queue_order({"struct": 0, "agentic": 10, "chat": 0}, {}) == ["agentic"],
          "empty queues must not be candidates")
    # 同占用下最小 VD 先削，高 VD 被保护
    even = {"struct": 400, "agentic": 400, "chat": 400}
    vd = {"struct": 1.0, "agentic": 0.0, "chat": 0.0}
    check(qs.queue_order(even, vd) == ["agentic", "chat", "struct"],
          f"highest VD must be protected, got {qs.queue_order(even, vd)}")
    # 占用完全不参与主序：小的队列只要 VD 更低就先削
    skew = {"struct": 10000, "agentic": 1000, "chat": 1000}
    vd = {"struct": 0.05, "agentic": 0.01, "chat": 0.30}
    check(qs.queue_order(skew, vd) == ["agentic", "struct", "chat"],
          f"a 10x smaller queue with lower VD must be cut first, "
          f"got {qs.queue_order(skew, vd)}")
    # 反例校验：同样的占用/密度代入已删除的 occ/(1+VD) 会得到相反结论
    legacy = sorted(hu.VALUE_QUEUES, key=lambda q: -skew[q] / (1.0 + vd[q]))
    check(legacy[0] == "struct",
          f"the removed occ/(1+VD) rule cut struct first, got {legacy}")
    # 同 VD 按占用降序，再按声明顺序（确定性）
    tie = {"struct": 800, "agentic": 400, "chat": 400}
    check(qs.queue_order(tie, {}) == ["struct", "agentic", "chat"],
          f"ties break by occupancy then declaration order, got {qs.queue_order(tie, {})}")


def test_struct_tier():
    """STRUCT 分档：share==share_min 先走，高于阈值最后；同档内 LRU。"""
    print("[2] struct tiers: share==min first, then share>min, LRU inside a tier")

    class Node:
        _next_id = 0

        def __init__(self, share, ts):
            Node._next_id += 1
            self.id = Node._next_id   # 平局判据：同档同 ts 时用它保持稳定
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


def test_stage0_one_shot_preempts_arbitration():
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
    """Stage 1：按 VD 排序点名队列，出候选时标 value:<queue>，无候选则跳队。"""
    print("[8] stage 1: VD-ordered queues, labels and skip-on-empty")
    st = ep.HeterCacheStrategy()
    st.occupancy.reconcile({"struct": {"total": 900, "one_shot": 0},
                            "agentic": {"total": 600, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    rank = st.queue_order()
    check(rank[0] == "struct",
          f"all VD=0 -> biggest queue first (tie-break), got {rank}")

    # 第一名 struct 无候选（一个 struct 叶子都没给）-> 跳到 agentic
    agentic_node = _leaf(TAG["agentic"], 100)
    victim, source = qs.pick_victim(strategy=st, leaves=[agentic_node])
    check(victim is agentic_node and source == "value:agentic",
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
        st = ep.HeterCacheStrategy(value={"seed": seed}, reuse={"enabled": False})
        st.occupancy.reset()  # 全 0 -> 没有任何队列有压力 -> 直接兜底
        picks = []
        for _ in range(6):
            victim, source = qs.pick_victim(strategy=st, leaves=leaves)
            picks.append((index[id(victim)], source))
        return picks

    a, b, c = run(0), run(0), run(7)
    check(a == b, f"same seed must reproduce, got {a} vs {b}")
    check(a != c, "different seed should differ")
    check(all(src == "value:fallback" for _, src in a),
          f"all picks must be labelled value:fallback, got {a}")
    state_before = _random.getstate()
    run(0)
    check(_random.getstate() == state_before, "fallback must not touch global random")


def test_value_params_validation():
    """VD 参数：决策视野/平滑倍数/统计下限必须为正，seed 必须是 int；未知键报错。"""
    print("[10] value params: horizon/smoothing/floor validated, unknown keys rejected")
    params = hu.QueueValueParams()
    check(params.horizon_s == hu.DEFAULT_VALUE_HORIZON_S,
          f"default horizon={params.horizon_s}, want {hu.DEFAULT_VALUE_HORIZON_S}")
    check(params.smoothing_k == hu.DEFAULT_VALUE_SMOOTHING_K,
          f"default k={params.smoothing_k}, want {hu.DEFAULT_VALUE_SMOOTHING_K}")
    check(params.floor_tokens == hu.DEFAULT_VALUE_FLOOR_TOKENS,
          f"default floor={params.floor_tokens}, want {hu.DEFAULT_VALUE_FLOOR_TOKENS}")
    for bad in (
        {"horizon_s": 0.0},
        {"horizon_s": -1.0},
        {"smoothing_k": 0.5},
        {"floor_tokens": 0.0},
        {"seed": True},
        {"nope": 1},
    ):
        try:
            hu.QueueValueParams.from_any(bad)
        except ValueError:
            pass
        else:
            check(False, f"invalid value params must be rejected: {bad}")


def test_reuse_picker_shared_by_chat_and_agentic():
    """ReuseScoreEvict：chat / agentic 共用一套复用分排序，只有 mu/sigma 不同。"""
    print("[11] reuse picker: chat and agentic share one ordering")
    st = _strategy()  # 默认即启用：复用分是策略口径，不是实验开关
    check(set(st.reuse_evicts) == {"chat", "agentic"},
          f"both queues must be served, got {sorted(st.reuse_evicts)}")
    chat_ev = st.reuse_evicts["chat"]
    agentic_ev = st.reuse_evicts["agentic"]
    check(chat_ev.params is st.reuse and agentic_ev.params is st.reuse,
          "both queues must share the same ReuseParams (single 口径)")
    check(chat_ev.queue is hu.HeterCacheNodeTag.QCHAT
          and agentic_ev.queue is hu.HeterCacheNodeTag.QAGENT,
          "each picker owns exactly one queue")

    now = time.monotonic()
    expired = _leaf(TAG["agentic"], 100, ts=now - 200)  # 复用窗口已过期（未到老化 TTL）
    fresh = _leaf(TAG["agentic"], 400, ts=now - 1)      # 刚写入，仍可能被复用
    chat = _leaf(TAG["chat"], 10, ts=now - 3600)        # 不归 agentic 队
    check(agentic_ev.pick([expired, fresh, chat]) is expired,
          "lowest reuse score (oldest elapsed) must be evicted first")
    check(chat_ev.pick([expired, fresh, chat]) is chat,
          "the chat picker must ignore foreign queues")

    small = _leaf(TAG["agentic"], 10, ts=now - 200)
    large = _leaf(TAG["agentic"], 90, ts=now - 200)
    check(agentic_ev.pick([small, large]) is large,
          "same reuse score must prefer the larger segment")
    check(agentic_ev.pick([]) is None, "empty pool must yield None")
    check(agentic_ev.stats()["abstains"] == 1,
          "abstains must be counted (picks=0 explains an idle queue in profile)")

    # 同一 elapsed 下两队列分数必须不同：唯一差异就是标签的 reuse_config。
    p_chat = chat_ev.reuse_score(chat, now)
    p_agentic = agentic_ev.reuse_score(expired, now)
    check(p_chat != p_agentic,
          f"per-queue mu/sigma must differ: chat={p_chat} agentic={p_agentic}")

    # 关闭时两队列都弃权（保留为消融臂）
    off = _strategy(reuse={"enabled": False})
    check(off.reuse_evicts == {}
          and off.queue_victim("chat", [chat]) is None
          and off.queue_victim("agentic", [expired]) is None,
          "reuse.enabled=False must make both queues abstain")

    # force_agentic：所有标签都归 agentic，评分必须用 QAGENT 的分布
    fa = _strategy(force_agentic=True)
    struct_expired = _leaf(TAG["struct"], 100, ts=now - 3600)
    expected = hu.reuse_window_probability(
        3600, hu.DEFAULT_REUSE_WINDOW_S, *hu.HeterCacheNodeTag.QAGENT.reuse_config
    )
    fa_ev = fa.reuse_evicts["agentic"]
    check(abs(fa_ev.reuse_score(struct_expired) - expected) < 1e-9,
          "force_agentic must score every node with the QAGENT distribution")
    check(fa_ev.pick([struct_expired]) is struct_expired,
          "force_agentic must pool every tag into the agentic band")

    # 接线：权重点名 agentic 时确实由本改写器出候选
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 900, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    victim, source = qs.pick_victim(strategy=st, leaves=[expired, fresh])
    check(victim is expired and source == "value:agentic",
          f"value:agentic must use the reuse score, got {source}")


def test_chat_uses_reuse_score_not_round_robin():
    """chat 队内与 agentic 同口径；RR 的轮询语义必须彻底消失。"""
    print("[12] chat queue: reuse score, RR removed")
    now = time.monotonic()
    st = _strategy()
    st.session_registry.clear()
    # 两条活着的 chat 会话：一条刚写过（1 秒前），一条静置 120 秒——超过 chat 的
    # 复用中位（T50≈118s）但还没到 180s 老化 TTL，所以仍在可调度带内、必须由
    # 复用分而不是 Stage 0 决定谁先走。
    fresh = _leaf(TAG["chat"], 100, ts=now - 1, session_id="reuse-c1")
    stale = _leaf(TAG["chat"], 100, ts=now - 120, session_id="reuse-c2")
    for sid in ("reuse-c1", "reuse-c2"):
        st.session_registry.touch(sid, f"r-{sid}", prompt_len=1000, now=now)
    picks = [st.queue_victim("chat", [fresh, stale]) for _ in range(4)]
    check(picks == [stale] * 4,
          "chat must keep picking the least reusable session, not rotate")

    # 会话轮询的排序键（最久未被削过）已不存在：同一调用序列必须幂等。
    check(picks[0] is picks[1],
          "stale reuse ordering must be idempotent (no per-call rotation state)")

    # 旧的 RR 参数组与改写器必须被删除，不能留下半开的后门。
    for legacy in ({"rr": {"fair_rr": True}},
                   {"mt": {"enabled": True}},
                   {"multi_agentic": {"enabled": True}}):
        try:
            ep.HeterCacheStrategy(**legacy)
        except TypeError:
            pass
        else:
            check(False, f"legacy config key must be rejected loudly: {legacy}")
    for name in ("RrParams", "MultiTurnParams", "MultiAgenticParams"):
        check(not hasattr(hu, name), f"{name} must be deleted from utils")
    for name in ("RrEvictPolicy", "MultiTurnEvictPolicy", "MultiAgenticEvict"):
        check(not hasattr(ep, name), f"{name} must be deleted from evict_policy")

    # chat 的接线：VD 点名 chat 时由复用分给出候选
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 0, "one_shot": 0},
                            "chat": {"total": 900, "one_shot": 0}}, capacity=1000)
    victim, source = qs.pick_victim(strategy=st, leaves=[fresh, stale])
    check(victim is stale and source == "value:chat",
          f"value:chat must use the reuse score, got {source}")


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

    st = _strategy()
    now = time.monotonic()
    young = _leaf(TAG["agentic"], 100, ts=now - 2)
    stale = _leaf(TAG["agentic"], 100, ts=now - 200)
    ev = st.reuse_evicts["agentic"]
    check(ev.pick([young, stale]) is stale,
          "the lowest future-reuse probability must be evicted first")
    check(ev.pick([young]) is young,
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
            st.node_share.on_hit(n, f"{id(n)}-{i}")
        n.hit_count = 1 if share else 0
        return n

    aged = node(TAG["struct"], ttl["struct"] + 1, share=st.node_share.share_min)
    fresh = node(TAG["struct"], ttl["struct"] - 1, share=st.node_share.share_min)
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

    partial = ep.HeterCacheStrategy(share={"ageing_ttl_s": {"struct": 10}})
    check(partial.share.ageing_ttl_s["struct"] == 10 and
          partial.share.ageing_ttl_s["chat"] == ttl["chat"],
          "a partial override must merge over the defaults")
    off = ep.HeterCacheStrategy(share={"ageing_ttl_s": {"struct": 0}})
    check(not off.is_aged(aged), "0 disables ageing for that type only")
    for bad in ({"nope": 1}, {"struct": -1}):
        try:
            ep.HeterCacheStrategy(share={"ageing_ttl_s": bad})
        except ValueError:
            pass
        else:
            check(False, f"invalid ageing_ttl_s must be rejected: {bad}")

    fa = _strategy(force_agentic=True)
    check(fa.is_aged(node(TAG["struct"], ttl["agentic"] + 1)),
          "force_agentic must score every node with the agentic TTL")

    before = st.node_share.aged_evicted
    st.record_evict(aged)
    check(st.node_share.aged_evicted == before + 100,
          f"aged evict must count {before + 100} tokens, "
          f"got {st.node_share.aged_evicted}")
    st.record_evict(fresh)
    check(st.node_share.aged_evicted == before + 100,
          "a non-aged evict must not be counted as aged")

    victim, source = qs.pick_victim(strategy=st, leaves=[aged, fresh])
    check(victim is aged and source == "one_shot",
          f"aged node must win Stage 0, got {source}")


def main():
    tests = [
        test_queue_order_by_value_density,
        test_struct_tier,
        test_accounting_insert_and_evict,
        test_random_is_never_counted,
        test_accounting_cascade_decrements_each_node,
        test_snapshot_heals_drift,
        test_stage0_one_shot_preempts_arbitration,
        test_stage1_labels_and_skip,
        test_fallback_is_seeded_random,
        test_value_params_validation,
        test_reuse_picker_shared_by_chat_and_agentic,
        test_chat_uses_reuse_score_not_round_robin,
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
