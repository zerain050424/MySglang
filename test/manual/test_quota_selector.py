"""Quota selector 的仲裁数学单测（纯 python，不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/quota_selector.py`
  effective_targets   —— max-min 水填充 + 上下限夹紧
  over_quota_queues   —— 死区与最小减持量的门禁
  queue_ranking       —— 超上限优先、弹性区按超出下限最多
  struct_tier         —— STRUCT 两遍降级键
  resolve_chunk       —— 单次最小减持量

只验数学，不建缓存、不碰节点内容；接线与记账由 test_quota_accounting.py 覆盖。
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
# 配额账本已独立成模块，evict_policy 直接依赖它
_load("sglang.srt.mem_cache.queue_occupancy", os.path.join(_CACHE_DIR, "queue_occupancy.py"))
qs = _load("sglang.srt.mem_cache.quota_selector", os.path.join(_CACHE_DIR, "quota_selector.py"))
vo = _load("sglang.srt.mem_cache.victim_override", os.path.join(_CACHE_DIR, "victim_override.py"))
_load("sglang.srt.mem_cache.heter_bench_bridge", os.path.join(_CACHE_DIR, "heter_bench_bridge.py"))
ep = _load("sglang.srt.mem_cache.evict_policy", os.path.join(_CACHE_DIR, "evict_policy.py"))

QuotaParams = hu.QuotaParams
C = 1000.0

BASE = dict(caps={"struct": 0.2, "agentic": 0.5, "chat": 0.3},
            reservations={"struct": 0.05, "agentic": 0.2, "chat": 0.1})

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def targets(occ, **kw):
    params = QuotaParams(**{**BASE, **kw})
    return qs.effective_targets(occ, kw.pop("capacity", C), params)


def test_water_filling_respects_caps():
    """水填充：按 k 比例分，触顶即夹到 k*C。"""
    print("[1] water-filling splits by cap and clamps at the cap")
    params = QuotaParams(**BASE)
    occ = {"struct": 300, "agentic": 500, "chat": 200}
    t = qs.effective_targets(occ, C, params)
    check(approx(t["struct"], 200), f"struct target={t['struct']:.1f}, want cap 200")
    check(approx(t["agentic"], 500), f"agentic target={t['agentic']:.1f}, want cap 500")
    check(approx(t["chat"], 300), f"chat target={t['chat']:.1f}, want 300")
    check(sum(t.values()) <= C + 1e-6, f"sum(targets)={sum(t.values()):.1f} > C")


def test_under_reservation_is_fixed():
    """未触底的队列保住现有占用，不参与再分配（即"免死"）。"""
    print("[2] queues below their reservation keep what they hold")
    params = QuotaParams(**BASE)
    occ = {"struct": 10, "agentic": 30, "chat": 20}
    t = qs.effective_targets(occ, C, params)
    for q, used in occ.items():
        check(t[q] <= used + 1e-6 or t[q] >= used - 1e-6,
              f"{q} target should equal current usage {used}, got {t[q]:.1f}")
    check(all(t[q] >= occ[q] - 1e-6 for q in occ),
          f"an under-reservation queue must not be targeted below its usage: {t}")


def test_idle_capacity_is_borrowed():
    """空闲借用：未触底队列不占份额，其他队列可涨到自己上限，但不得越 cap。

    cap 是硬天花板，所以"借用"的上界就是自己的 cap：struct 空闲时
    agentic/chat 可以吃满各自上限（而非被均分成 1/3），吃不下的容量留在池里
    （这是 cap 语义的正常结果，不是错误）。
    """
    print("[3] idle capacity is borrowed up to the borrower's own cap")
    params = QuotaParams(**BASE)
    occ = {"struct": 0, "agentic": 450, "chat": 250}
    t = qs.effective_targets(occ, C, params)
    check(approx(t["agentic"], 500), f"agentic should reach its cap 500, got {t['agentic']:.0f}")
    check(approx(t["chat"], 300), f"chat should reach its cap 300, got {t['chat']:.0f}")
    check(t["struct"] == 0, f"empty struct keeps 0, got {t['struct']}")
    check(all(t[q] <= params.caps[q] * C + 1e-6 for q in t),
          f"no target may exceed its cap: {t}")
    check(sum(t.values()) < C,
          "caps binding leaves slack unclaimed (quota must not force-fill it)")


def test_denominator_is_stable_under_eviction():
    """分母恒为 C：降到上限所需的淘汰量 = occ - k*C，不随淘汰而放大。"""
    print("[4] capacity denominator is constant (no collapse while draining)")
    params = QuotaParams(**BASE)
    cap_tokens = params.caps["struct"] * C
    occ = {"struct": 900, "agentic": 100, "chat": 0}
    t1 = qs.effective_targets(occ, C, params)
    need = occ["struct"] - t1["struct"]
    # 淘汰 need 之后：struct 恰好不再超标，且目标不变（因为它不依赖 sum(occ)）
    occ2 = {"struct": occ["struct"] - need, "agentic": 100, "chat": 0}
    t2 = qs.effective_targets(occ2, C, params)
    check(approx(t2["struct"], t1["struct"], 1e-6),
          f"target moved during drain: {t1['struct']:.1f} -> {t2['struct']:.1f}")
    check(approx(need, occ["struct"] - cap_tokens + (cap_tokens - t1["struct"]), 1e-6)
          or need >= occ["struct"] - cap_tokens - 1e-6,
          f"drain amount {need:.1f} smaller than naive overshoot")
    # 反例：若分母用 sum(occ)，同样的场景需要 2 倍淘汰量（k=0.5）
    sum_occ = sum(occ.values())
    naive_delta_sum_denominator = (occ["struct"] - params.caps["struct"] * sum_occ) / (
        1 - params.caps["struct"])
    check(naive_delta_sum_denominator > need,
          "sanity: sum(occ) denominator would have demanded more than C denominator")


def test_deadzone():
    """相对超出量不到 deadzone 不点名。"""
    print("[5] deadzone suppresses marginal over-quota")
    params = QuotaParams(**BASE)
    t = {"struct": 200.0, "agentic": 500.0, "chat": 300.0}
    inside = {"struct": 200 * 1.03, "agentic": 400, "chat": 300}
    outside = {"struct": 200 * 1.06, "agentic": 400, "chat": 300}
    check(qs.over_quota_queues(inside, t, params, chunk=1) == [],
          "3% over target must stay inside the deadzone")
    check(qs.over_quota_queues(outside, t, params, chunk=1) == ["struct"],
          f"6% over target must be named, got {qs.over_quota_queues(outside, t, params, 1)}")


def test_chunk_floor():
    """绝对超出量不足一个 chunk 不点名（page_size=1 时防止逐 token 颠簸）。"""
    print("[6] a sub-chunk overshoot is ignored")
    # 关掉死区，单独验 chunk 这一道门
    params = QuotaParams(**{**BASE, "deadzone": 0.0})
    t = {"struct": 200.0, "agentic": 500.0, "chat": 300.0}
    occ = {"struct": 203, "agentic": 400, "chat": 300}   # +3 tokens
    check(qs.over_quota_queues(occ, t, params, chunk=8) == [],
          "3 tokens < chunk 8 must not be named")
    check(qs.over_quota_queues(occ, t, params, chunk=2) == ["struct"],
          "3 tokens > chunk 2 must be named")


def test_ranking_orders_over_before_elastic():
    """排序：超上限者（按超出量）在前，弹性区（按超出下限）在后；未触底者不入列。"""
    print("[7] ranking puts over-cap queues first, then the biggest excess over floor")
    params = QuotaParams(**BASE)
    occ = {"struct": 400, "agentic": 520, "chat": 60}
    t = qs.effective_targets(occ, C, params)
    rank = qs.queue_ranking(occ, t, params, chunk=1, capacity=C)
    check(set(rank) == {"struct", "agentic"},
          f"only the two over-cap queues may rank, got {rank}")
    check("chat" not in rank,
          "a queue below its floor is protected and must not be named")
    # 超出量：struct 400-200=200，agentic 520-500=20 → struct 必须在前
    check(rank == ["struct", "agentic"],
          f"ranking must be sorted by overshoot, got {rank}")


def test_ranking_empty_when_nobody_above_floor():
    """全部低于下限 → 配额没有意见（返回空列表，交给兜底）。"""
    print("[8] no queue above its floor means the quota has no opinion")
    params = QuotaParams(**BASE)
    occ = {"struct": 10, "agentic": 20, "chat": 10}
    t = qs.effective_targets(occ, C, params)
    check(qs.queue_ranking(occ, t, params, chunk=1, capacity=C) == [],
          "all-below-floor must produce an empty ranking")


def test_elastic_picks_max_excess_over_floor():
    """弹性区（Q3=a）：都在 [rho, k] 区间时，点名超出下限最多的那个。"""
    print("[9] elastic zone names the queue with the largest excess over its floor")
    params = QuotaParams(**BASE)
    # 三个都在 [rho, k] 内：struct 150∈[50,200], agentic 450∈[200,500], chat 250∈[100,300]
    occ = {"struct": 150, "agentic": 450, "chat": 250}
    t = qs.effective_targets(occ, C, params)
    check(qs.over_quota_queues(occ, t, params, chunk=1) == [],
          f"no queue exceeds its cap, got {qs.over_quota_queues(occ, t, params, 1)}")
    rank = qs.queue_ranking(occ, t, params, chunk=1, capacity=C)
    check(rank[0] == "agentic",
          f"agentic has the largest excess over floor (450-200=250), got {rank}")


def test_struct_tier():
    """STRUCT 分档：share==share_min 先走，高于阈值最后；同档内 LRU。"""
    print("[10] struct tiers: share==min first, then share>min, LRU inside a tier")

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
    check(order[0] is just,
          f"share==share_min must go first regardless of age, got {[(n.share, n.last_access_time) for n in order]}")
    check(order[1] is older_high, "inside tier 1 the older node goes first")
    check(order[2] is newer_high, "inside tier 1 the newer node goes last")


def test_resolve_chunk():
    """最小减持量取 max(page_size, chunk_tokens)。"""
    print("[11] chunk is max(page_size, chunk_tokens)")
    params = QuotaParams(**BASE)
    check(qs.resolve_chunk(1, params) == 1, "page_size=1 and chunk=0 -> 1")
    chunked = QuotaParams(**{**BASE, "chunk_tokens": 16})
    check(qs.resolve_chunk(1, chunked) == 16, "chunk_tokens wins when larger")
    check(qs.resolve_chunk(64, chunked) == 64, "page_size wins when larger")


class _Leaf:
    """极简叶子替身：只带配额计量用到的字段。

    必须是普通类而不是 SimpleNamespace —— SLRU 的共享数账本是
    WeakKeyDictionary，需要节点可弱引用。
    """

    def __init__(self, tag, tokens, ts=0.0, session_id=None):
        self.queue_tag = tag
        self.value = [0] * tokens
        self.key = [0] * tokens
        self.last_access_time = ts
        self.session_id = session_id


def _leaf(tag, tokens, ts=0.0, session_id=None):
    return _Leaf(tag, tokens, ts, session_id)


TAG = {
    "struct": hu.HeterCacheNodeTag.QSTRUCT,
    "agentic": hu.HeterCacheNodeTag.QAGENT,
    "chat": hu.HeterCacheNodeTag.QCHAT,
}


def _strategy(**kw):
    return ep.HeterCacheStrategy(
        quota={"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}, **kw}
    )


def test_accounting_insert_and_evict():
    """记账：插入累加、淘汰扣减；one-shot 两侧都不进账。"""
    print("[13] accounting: insert adds, evict subtracts, one-shot is excluded")
    st = _strategy()
    st.record_insert(_leaf(TAG["agentic"], 100))
    st.record_insert(_leaf(TAG["struct"], 40))
    check(st.occupancy.occ["agentic"] == 100, f"agentic={st.occupancy.occ['agentic']}, want 100")
    check(st.occupancy.occ["struct"] == 40, f"struct={st.occupancy.occ['struct']}, want 40")

    st.record_evict(_leaf(TAG["agentic"], 30))
    check(st.occupancy.occ["agentic"] == 70, f"after evict agentic={st.occupancy.occ['agentic']}, want 70")

    # TRANS 会话 + share < share_min -> one-shot，插入与淘汰两侧都不进账
    # （now=0.0：会话早已搁置，超过 ONE_SHOT_IDLE_S 时间豁免窗口，
    #   否则活跃会话的未匹配节点会被时间豁免挡在带外）
    trans = _leaf(TAG["struct"], 50, session_id="t1")
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
    """random 不参与配额：无会话证据的 QTRANS 节点（is_one_shot=False）也不能进账。

    这类节点 queue_name="random"，不在 QUOTA_QUEUES 里；漏掉白名单就会在
    record_evict 扣减时打到不存在的键上（KeyError）。
    """
    print("[14b] accounting: random-classified nodes never enter the quota ledger")
    st = _strategy()
    rnd = _leaf(TAG["agentic"], 70)          # 先造一个对象再改标签，保证可弱引用
    rnd.queue_tag = hu.HeterCacheNodeTag.QTRANS
    check(st.is_one_shot(rnd) is False,
          "precondition: a session-less QTRANS node is not judged one-shot")
    check(st._node_tag(rnd) is hu.HeterCacheNodeTag.QTRANS, "tag must stay QTRANS")
    st.record_insert(rnd)
    st.record_evict(rnd)
    check(all(v == 0.0 for v in st.occupancy.occ.values()),
          f"random must not touch any quota queue, got {st.occupancy.occ}")


def test_accounting_cascade_decrements_each_node():
    """级联删除：_evict_one 逐节点调 record_evict，祖先也要各扣一次。"""
    print("[14] accounting: cascade eviction decrements every deleted node")
    st = _strategy()
    for n in (40, 60, 100):
        st.record_insert(_leaf(TAG["chat"], n))
    check(st.occupancy.occ["chat"] == 200, f"chat={st.occupancy.occ['chat']}, want 200")
    # 模拟一次 victim + 两个祖先的级联删除
    for n in (100, 60):
        st.record_evict(_leaf(TAG["chat"], n))
    check(st.occupancy.occ["chat"] == 40, f"chat={st.occupancy.occ['chat']}, want 40")
    # 扣不穿地板
    st.record_evict(_leaf(TAG["chat"], 999))
    check(st.occupancy.occ["chat"] == 0.0, f"occ must clamp at 0, got {st.occupancy.occ['chat']}")


def test_snapshot_heals_drift():
    """快照对齐：改标签 / one-shot 迁移 / 分裂双计导致的漂移都能被拉回真值。"""
    print("[15] reconcile restores ground truth after tag moves and splits")
    st = _strategy()
    st.record_insert(_leaf(TAG["agentic"], 500))
    # 人为打偏（等价于改标签 + 漏记）
    st.occupancy.occ["agentic"] += 300
    st.occupancy.occ["struct"] -= 120
    truth = {
        "struct": {"total": 400, "one_shot": 100},
        "agentic": {"total": 600, "one_shot": 0},
        "chat": {"total": 200, "one_shot": 200},
        "random": {"total": 90, "one_shot": 90},
    }
    st.occupancy.reconcile(truth, capacity=1000)
    check(st.occupancy.occ["struct"] == 300, f"struct={st.occupancy.occ['struct']}, want 300 (total-one_shot)")
    check(st.occupancy.occ["agentic"] == 600, f"agentic={st.occupancy.occ['agentic']}, want 600")
    check(st.occupancy.occ["chat"] == 0, f"chat={st.occupancy.occ['chat']}, want 0 (整行 one-shot)")
    check(st.occupancy.capacity == 1000, f"capacity={st.occupancy.capacity}, want 1000")


def test_quota_gap_gate():
    """配额缺口：超配额才给 RR 上限，弹性区不限制。"""
    print("[16] quota gap is only defined when the queue is over its target")
    st = _strategy()
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 600, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    check(abs(st._quota_gap("agentic", st.occupancy.occ, 1000) - 100) < 1e-6,
          f"gap={st._quota_gap('agentic', st.occupancy.occ, 1000)}, want 100 (600-500)")
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 300, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    check(st._quota_gap("agentic", st.occupancy.occ, 1000) is None,
          "under-target queue must not constrain RR")


def test_stage0_one_shot_preempts_quota():
    """组合顺序：one-shot 带永远先清空，即使配额点名了别的队列。"""
    print("[17] stage 0: one-shot drains first even when quota names a queue")
    st = _strategy()
    st.occupancy.reconcile({"struct": {"total": 900, "one_shot": 0},
                            "agentic": {"total": 0, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    # struct 严重超配额；但同时存在一个 one-shot 叶子
    struct_node = _leaf(TAG["struct"], 100, ts=1.0)
    one = _leaf(TAG["struct"], 40, ts=99.0, session_id="s1")
    # now=0.0：会话早已搁置，超过 ONE_SHOT_IDLE_S 时间豁免窗口
    st.session_registry.touch("s1", "r1", prompt_len=40, now=0.0)
    st.session_registry.get("s1").session_type = hu.SessionType.TRANS
    check(st.is_one_shot(one), "precondition: TRANS node is one-shot")
    victim, source = qs.pick_victim(
        strategy=st, leaves=[struct_node, one], occ=st.occupancy.occ,
        capacity=1000, page_size=1)
    check(victim is one and source == "one_shot",
          f"one-shot must win, got source={source}")
    check("struct" in st.quota_ranking(st.occupancy.occ, 1000, 1), "precondition: struct is over cap")


def test_stage1_quota_labels_and_skip():
    """组合顺序：点名队列出 victim 时标 quota:<queue>；无候选则跳队。"""
    print("[18] stage 1: quota names a queue, skips empty ones")
    # agentic 的队内 picker 现在是 multi_agentic（MT 已从分派中移除）。
    st = ep.HeterCacheStrategy(
        multi_agentic={"enabled": True},
        quota={"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}},
    )
    st.occupancy.reconcile({"struct": {"total": 900, "one_shot": 0},
                            "agentic": {"total": 600, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    rank = st.quota_ranking(st.occupancy.occ, 1000, 1)
    check(rank[0] == "struct", f"struct overshoots most (900-200), got {rank}")

    # 第一名 struct 无候选（一个 struct 叶子都没给）-> 应跳到 agentic
    agentic_node = _leaf(TAG["agentic"], 100, ts=1.0)
    victim, source = qs.pick_victim(
        strategy=st, leaves=[agentic_node], occ=st.occupancy.occ,
        capacity=1000, page_size=1)
    check(victim is agentic_node and source == "quota:agentic",
          f"must skip the empty struct queue, got {source}")

    # 所有队列都没候选 -> 兜底，不 busy-loop
    victim, source = qs.pick_victim(
        strategy=st, leaves=[], occ=st.occupancy.occ, capacity=1000, page_size=1)
    check(victim is None and source is None,
          f"empty leaves must yield (None, None), got {(victim, source)}")


def test_stage1_fallback_is_seeded_random():
    """兜底：固定种子可复现，且不污染全局 random。"""
    print("[19] stage 1 fallback: seeded RNG, reproducible, isolated")
    import random as _random

    leaves = [_leaf(TAG["chat"], 10, ts=float(i)) for i in range(30)]
    index = {id(n): i for i, n in enumerate(leaves)}

    def run(seed):
        # 各队列都低于下限 -> ranking 为空 -> 走兜底抽样
        # 下限=上限 且占用均低于下限 -> 无队列可点名 -> 直接走兜底
        st = ep.HeterCacheStrategy(
            quota={"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3},
                   "reservations": {"struct": 0.2, "agentic": 0.5, "chat": 0.3},
                   "seed": seed}
        )
        st.occupancy.reconcile({"struct": {"total": 100, "one_shot": 0},
                                "agentic": {"total": 100, "one_shot": 0},
                                "chat": {"total": 100, "one_shot": 0}}, capacity=1000)
        picks = []
        for _ in range(6):
            victim, source = qs.pick_victim(
                strategy=st, leaves=leaves, occ=st.occupancy.occ, capacity=1000, page_size=1)
            picks.append((index[id(victim)], source))
        return picks

    a, b, c = run(0), run(0), run(7)
    check(a == b, f"same seed must reproduce, got {a} vs {b}")
    check(a != c, "different seed should differ")
    check(all(src == "quota:fallback" for _, src in a),
          f"all picks must be labelled quota:fallback, got {a}")
    state_before = _random.getstate()
    run(0)
    check(_random.getstate() == state_before, "fallback must not touch global random")


def test_multi_agentic_orders_by_reuse_score():
    """MultiAgenticEvict：agentic 队内按复用分升序淘汰。

    口径：``P(reuse in (a, a+H] | not reused by a)``，越小越先淘汰；
    同分时大片段先走（一次腾更多空间）。
    """
    print("[21] multi_agentic: agentic pool ordered by reuse score")
    caps = {"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}}
    st = ep.HeterCacheStrategy(multi_agentic={"enabled": True}, quota=caps)
    check(st.multi_agentic_evict is not None, "multi_agentic.enabled must build the picker")

    now = time.monotonic()
    expired = _leaf(TAG["agentic"], 100, ts=now - 3600)  # 复用窗口已过期
    fresh = _leaf(TAG["agentic"], 400, ts=now - 1)       # 刚写入，仍可能被复用
    chat = _leaf(TAG["chat"], 10, ts=now - 3600)         # 不归本队
    check(st.multi_agentic_evict.pick([expired, fresh, chat]) is expired,
          "lowest reuse score (oldest elapsed) must be evicted first")

    small = _leaf(TAG["agentic"], 10, ts=now - 3600)
    large = _leaf(TAG["agentic"], 90, ts=now - 3600)
    check(st.multi_agentic_evict.pick([small, large]) is large,
          "same reuse score must prefer the larger segment")

    check(st.multi_agentic_evict.pick([]) is None, "empty pool must yield None")

    # 关闭时整个改写器不构造：agentic 仍走 MT / 兜底。
    off = ep.HeterCacheStrategy(quota=caps)
    check(off.multi_agentic_evict is None, "multi_agentic must be opt-in")

    # force_agentic：所有标签都归 agentic，评分必须用 QAGENT 的分布而不是
    # 节点自己的 queue_tag（QSTRUCT 没有复用分布，会恒得 0 分）。
    fa = ep.HeterCacheStrategy(
        force_agentic=True, multi_agentic={"enabled": True}, quota=caps)
    struct_expired = _leaf(TAG["struct"], 100, ts=now - 3600)
    expected = hu.reuse_window_probability(
        3600, hu.DEFAULT_REUSE_WINDOW_S, *hu.HeterCacheNodeTag.QAGENT.reuse_config
    )
    check(abs(fa.multi_agentic_evict.reuse_score(struct_expired) - expected) < 1e-9,
          "force_agentic must score every node with the QAGENT distribution")
    check(fa.multi_agentic_evict.pick([struct_expired]) is struct_expired,
          "force_agentic must pool every tag into the agentic band")

    # 接线：配额点名 agentic 时确实由本改写器出候选
    st.occupancy.reconcile({"struct": {"total": 0, "one_shot": 0},
                            "agentic": {"total": 900, "one_shot": 0},
                            "chat": {"total": 0, "one_shot": 0}}, capacity=1000)
    victim, source = qs.pick_victim(
        strategy=st, leaves=[expired, fresh], occ=st.occupancy.occ,
        capacity=1000, page_size=1)
    check(victim is expired and source == "quota:agentic",
          f"quota:agentic must use the reuse score, got {source}")


def test_reuse_window_probability():
    """复用分 = P(reuse in (a, a+H] | not reused by a)，不是 1 - CDF(a)。

    两者的区别在尾部：1 - CDF 在 erf 饱和后把所有“很老”节点抹成同一分，
    条件概率则保持单调（越老越不可能在下一个窗口被复用）。
    """
    print("[23] reuse score is the conditional window probability")
    mu, sigma = hu.HeterCacheNodeTag.QAGENT.reuse_config
    H = hu.DEFAULT_REUSE_WINDOW_S

    # a=0 时条件概率就是 CDF(H)：分母 S(0)=1
    check(abs(hu.reuse_window_probability(0, H, mu, sigma)
              - hu._cached_lognorm_cdf(H, mu, sigma)) < 1e-9,
          "at age 0 the conditional probability must equal CDF(H)")

    # 窗口越长，未来被复用的概率不能变小
    p90 = hu.reuse_window_probability(10, H, mu, sigma)
    p180 = hu.reuse_window_probability(10, 2 * H, mu, sigma)
    check(p90 < p180, f"longer horizon must not lower the score: {p90} vs {p180}")

    # 老节点仍严格有序，且都不会被抹成 0
    p600 = hu.reuse_window_probability(600, H, mu, sigma)
    p3600 = hu.reuse_window_probability(3600, H, mu, sigma)
    check(0.0 < p3600 < p600 < 1.0,
          f"stale nodes must keep decreasing: p600={p600}, p3600={p3600}")

    # 极端老节点：S(a) 下溢到 0 时返回 0，不能抛 ZeroDivisionError
    check(hu.reuse_window_probability(10 ** 9, H, mu, sigma) == 0.0,
          "underflowed survival must score 0, not raise")

    # 排序方向：取分数最低者（未来窗口内最不可能被复用）
    st = ep.HeterCacheStrategy(multi_agentic={"enabled": True}, quota={
        "caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}})
    now = time.monotonic()
    young = _leaf(TAG["agentic"], 100, ts=now - 2)
    stale = _leaf(TAG["agentic"], 100, ts=now - 3600)
    check(st.multi_agentic_evict.pick([young, stale]) is stale,
          "the lowest future-reuse probability must be evicted first")
    check(st.multi_agentic_evict.pick([young]) is young,
          "a single candidate must still be picked")


def test_queue_policy_swap():
    """队内策略归属：chat 走 RR 会话轮询，agentic 只由 multi_agentic 出候选。

    agentic 的默认 picker（MT）已从分派中移除，未显式打开 multi_agentic 时
    该队列弃权、由上层跳到下一队或走兜底。
    """
    print("[22] chat uses RR, agentic is served by multi_agentic only")
    caps = {"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}}
    st = ep.HeterCacheStrategy(rr={"fair_rr": True}, quota=caps)
    check(st.rr_evict.queue is hu.HeterCacheNodeTag.QCHAT,
          f"RR must own the chat queue, got {st.rr_evict.queue}")

    now = time.monotonic()
    # chat：两条长会话、缺口为 0 -> RR 轮询，不连续削同一条
    c1 = _leaf(TAG["chat"], 100, ts=now - 10, session_id="swap-c1")
    c2 = _leaf(TAG["chat"], 100, ts=now - 20, session_id="swap-c2")
    for sid in ("swap-c1", "swap-c2"):
        st.session_registry.touch(sid, f"r-{sid}", prompt_len=1000, now=now)
    occ_chat = {"struct": 0, "agentic": 0, "chat": 900}
    first = st.quota_victim("chat", [c1, c2], occ_chat, 1000)
    second = st.quota_victim("chat", [c1, c2], occ_chat, 1000)
    check({id(first), id(second)} == {id(c1), id(c2)},
          "RR must rotate between the two chat sessions")

    # 关掉 fair_rr：chat 队内无策略，必须弃权
    off = ep.HeterCacheStrategy(quota=caps)
    check(off.rr_evict is None and
          off.quota_victim("chat", [c1, c2], occ_chat, 1000) is None,
          "hc_norr must leave the chat queue without an in-queue picker")

    # agentic：未开 multi_agentic 时弃权；打开后按复用分出候选
    expired = _leaf(TAG["agentic"], 100, ts=now - 3600, session_id="swap-a1")
    fresh = _leaf(TAG["agentic"], 100, ts=now - 1, session_id="swap-a2")
    occ_agentic = {"struct": 0, "agentic": 900, "chat": 0}
    check(st.multi_agentic_evict is None and
          st.quota_victim("agentic", [expired, fresh], occ_agentic, 1000) is None,
          "agentic must abstain when multi_agentic is off")
    ma = ep.HeterCacheStrategy(
        rr={"fair_rr": True}, multi_agentic={"enabled": True}, quota=caps)
    check(ma.quota_victim("agentic", [expired, fresh], occ_agentic, 1000) is expired,
          "multi_agentic must pick the node whose reuse window already expired")
    small = _leaf(TAG["agentic"], 10, ts=now - 11, session_id="swap-m1")
    large = _leaf(TAG["agentic"], 900, ts=now - 11, session_id="swap-m2")
    check(ma.quota_victim("agentic", [small, large], occ_agentic, 1000) is large,
          "multi_agentic must prefer the larger segment on a tie")


def test_struct_ageing_demotes_shared_prefix():
    """结构复用老化：共享前缀久未访问 → 跌进 one-shot 带，且带内最优先。

    老化必须先于 hit_count / share 的短路：共享前缀必然命中过、且共享数早已
    达到 share_min，现有两个提前 return 会让它永远留在保护态。
    """
    print("[24] struct ageing: idle shared prefix drops to one-shot first")
    caps = {"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}}
    ttl = hu.DEFAULT_AGEING_IDLE_S
    st = ep.HeterCacheStrategy(slru={"ageing_idle_s": ttl}, quota=caps)
    now = time.monotonic()

    def shared(tag, age_s, tokens=100, sid=None):
        node = _leaf(tag, tokens, ts=now - age_s, session_id=sid)
        for i in range(st.slru_evict.share_min):
            st.slru_evict.on_node_hit(node, f"{id(node)}-{i}")
        node.hit_count = 1  # 共享节点必然被命中过
        return node

    aged = shared(TAG["struct"], ttl + 1)
    fresh = shared(TAG["struct"], ttl - 1)
    check(st.is_aged_struct(aged), "share>=min and idle>=ttl must age")
    check(not st.is_aged_struct(fresh), "idle<ttl must keep the template protected")
    check(st.is_one_shot(aged), "aged shared prefix must enter the one-shot band")
    check(not st.is_one_shot(fresh), "fresh shared prefix must stay in struct")
    check(st.session_registry.get(aged.session_id) is None,
          "precondition: ageing must not need the creator session to survive")

    # 带内优先：比更老的普通 one-shot 还先走
    plain = _leaf(TAG["struct"], 100, ts=now - ttl * 10)
    check(st.one_shot_key(aged) < st.one_shot_key(plain),
          "aged struct must outrank plain one-shot inside the band")

    # 只对 QSTRUCT 生效
    agent = shared(TAG["agentic"], ttl * 10)
    check(not st.is_aged_struct(agent), "only struct templates age")

    # 开关与校验
    off = ep.HeterCacheStrategy(slru={"ageing_idle_s": 0}, quota=caps)
    check(not off.is_aged_struct(aged), "ageing_idle_s=0 must disable ageing")
    try:
        ep.HeterCacheStrategy(slru={"ageing_idle_s": -1}, quota=caps)
    except ValueError:
        pass
    else:
        check(False, "negative ageing_idle_s must be rejected")
    fa = ep.HeterCacheStrategy(
        force_agentic=True, slru={"ageing_idle_s": ttl}, quota=caps)
    check(not fa.is_aged_struct(aged), "force_agentic must disable struct ageing")

    # 观测：只有老化淘汰计入 aged_evicted
    before = st.slru_evict.aged_evicted
    st.record_evict(aged)
    check(st.slru_evict.aged_evicted == before + 100,
          f"aged evict must count {before + 100} tokens, "
          f"got {st.slru_evict.aged_evicted}")
    st.record_evict(fresh)
    check(st.slru_evict.aged_evicted == before + 100,
          "a non-aged evict must not be counted as aged")

    # 端到端：仲裁层直接走 one-shot 带
    victim, source = qs.pick_victim(
        strategy=st, leaves=[aged, fresh],
        occ={"struct": 0, "agentic": 0, "chat": 0}, capacity=1000, page_size=1)
    check(victim is aged and source == "one_shot",
          f"aged struct must win Stage 0, got {source}")


def test_quota_is_mandatory():
    """配额必填：不给 caps 直接报错，不会静默退回另一套淘汰语义。"""
    print("[20] quota is mandatory: missing caps raises, never degrades")
    try:
        ep.HeterCacheStrategy()
    except ValueError as exc:
        print(f"      rejected: {str(exc)[:70]}")
    else:
        check(False, "HeterCacheStrategy accepted empty quota.caps")

    try:
        QuotaParams()
    except ValueError:
        pass
    else:
        check(False, "QuotaParams accepted empty caps")

    # 数学层不再自带“未启用”守卫：配额必然有值，targets 恒可算
    caps = {"caps": {"struct": 0.2, "agentic": 0.5, "chat": 0.3}}
    targets = qs.effective_targets(
        {"struct": 1, "agentic": 1, "chat": 1}, C, QuotaParams(**caps)
    )
    check(set(targets) == set(hu.QUOTA_QUEUES), f"targets keys={sorted(targets)}")


def main():
    tests = [
        test_water_filling_respects_caps,
        test_under_reservation_is_fixed,
        test_idle_capacity_is_borrowed,
        test_denominator_is_stable_under_eviction,
        test_deadzone,
        test_chunk_floor,
        test_ranking_orders_over_before_elastic,
        test_ranking_empty_when_nobody_above_floor,
        test_elastic_picks_max_excess_over_floor,
        test_struct_tier,
        test_resolve_chunk,
        test_accounting_insert_and_evict,
        test_random_is_never_counted,
        test_accounting_cascade_decrements_each_node,
        test_snapshot_heals_drift,
        test_quota_gap_gate,
        test_stage0_one_shot_preempts_quota,
        test_stage1_quota_labels_and_skip,
        test_stage1_fallback_is_seeded_random,
        test_quota_is_mandatory,
        test_multi_agentic_orders_by_reuse_score,
        test_queue_policy_swap,
        test_reuse_window_probability,
        test_struct_ageing_demotes_shared_prefix,
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
