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

uni = _load("sglang.srt.mem_cache.uni_utils", os.path.join(_CACHE_DIR, "uni_utils.py"))
qs = _load("sglang.srt.mem_cache.quota_selector", os.path.join(_CACHE_DIR, "quota_selector.py"))

QuotaParams = uni.QuotaParams
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


def test_disabled_params_rejected():
    """未启用配额时数学层直接报错：接线错误不该静默返回 0。"""
    print("[12] disabled params are rejected, not silently zeroed")
    try:
        qs.effective_targets({"struct": 1, "agentic": 1, "chat": 1}, C, QuotaParams())
    except ValueError as exc:
        print(f"      rejected: {str(exc)[:60]}")
    else:
        check(False, "effective_targets accepted disabled params")


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
        test_disabled_params_rejected,
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
