"""已兑现价值密度（VD）账本单测（不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/heter_cache/value.py`

口径（两个时间尺度，职责分离）::

    W_q  = max(horizon, k*T50_q)                  # 估计窗：只为平滑命中速率
    r_q  = H_q(W_q) / W_q                    # 稳态下 E[H] = r*W
    VD_q = r_q*horizon / max(occ_q, floor)         # 投影回共享视野 horizon

horizon 是政策（三队列共享，量纲一致）；W_q 只是统计（只影响方差，不影响尺度）。
本文件只测账本自身；仲裁顺序由 ``test_queue_selector.py`` 覆盖。
"""

import importlib.util
import math
import os
import sys
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CACHE_DIR = os.path.join(_ROOT, "python", "sglang", "srt", "mem_cache")
_HC_DIR = os.path.join(_CACHE_DIR, "heter_cache")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


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
qv = _load(
    "sglang.srt.mem_cache.heter_cache.value", os.path.join(_HC_DIR, "value.py")
)


class FakeClock:
    """可手动推进的单调时钟（虚拟秒）。"""

    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


HORIZON = 120.0
K = 4.0
T50_CHAT = hu.reuse_interval_median(hu.HeterCacheNodeTag.QCHAT)
T50_AGENTIC = hu.reuse_interval_median(hu.HeterCacheNodeTag.QAGENT)
W_CHAT = max(HORIZON, K * T50_CHAT)
W_AGENTIC = max(HORIZON, K * T50_AGENTIC)


def _tracker(horizon=HORIZON, k=K, floor=2048.0, start=0.0):
    clock = FakeClock(start)
    tracker = qv.QueueValueTracker(
        horizon_s=horizon, smoothing_k=k, floor_tokens=floor, now=clock
    )
    return tracker, clock


def test_density_is_projected_hits_over_occupancy():
    """VD = 投影命中量 / max(occ, floor)；占用低于统计下限时按 floor 计。"""
    print("[1] density = projected hits / max(occ, floor)")
    tracker, _ = _tracker()
    # agentic 的 W = horizon，投影系数 1：这里隔离出分母口径
    tracker.on_hit("agentic", 1000)
    tracker.materialize()
    dens = tracker.densities({"agentic": 10000, "chat": 0, "struct": 0})
    check(abs(dens["agentic"] - 0.1) < 1e-12, f"agentic={dens['agentic']}, want 0.1")
    small = tracker.densities({"agentic": 10})["agentic"]
    check(abs(small - 1000 / 2048.0) < 1e-12, f"small occ must use floor, got {small}")
    check(
        dens["chat"] == 0.0 and dens["struct"] == 0.0,
        f"queues without hits must be 0, got {dens}",
    )


def test_estimation_window_rule():
    """W_q = max(horizon, k*T50_q)；struct 无分布数据时回退 horizon，不给 magic number。"""
    print("[2] W_q = max(horizon, k*T50_q); struct falls back to horizon")
    windows = _tracker()[0].windows()
    check(abs(windows["chat"] - W_CHAT) < 1e-9,
          f"W_chat={windows['chat']}, want {W_CHAT}")
    check(windows["chat"] > HORIZON,
          f"chat 的 T50({T50_CHAT:.1f}s) 必须把窗口撑过 horizon，got {windows['chat']}")
    check(abs(windows["agentic"] - HORIZON) < 1e-9,
          f"W_agentic={windows['agentic']}, want horizon={HORIZON}")
    check(abs(windows["struct"] - HORIZON) < 1e-9,
          f"W_struct={windows['struct']}, want horizon={HORIZON} (no reuse distribution)")
    # k 只影响有分布数据的队列
    tight = _tracker(k=1.0)[0].windows()
    check(abs(tight["chat"] - max(HORIZON, T50_CHAT)) < 1e-9,
          f"k=1 -> W_chat={tight['chat']}, want max(horizon, T50)")
    check(abs(tight["struct"] - HORIZON) < 1e-9, "struct must stay at horizon for any k")


def test_projection_to_shared_horizon():
    """快照存 H*(horizon/W)，即"未来 horizon 内的预期命中"，不是原始命中质量。"""
    print("[3] snapshot projects hit mass onto the shared horizon")
    tracker, _ = _tracker()
    tracker.on_hit("chat", 1000)
    tracker.materialize()
    want = 1000.0 * HORIZON / W_CHAT
    check(abs(tracker.snapshot()["chat"] - want) < 1e-9,
          f"chat snapshot={tracker.snapshot()['chat']}, want {want}")
    # 同一次命中在 agentic（W=horizon）上不缩放，用来证明投影系数是按队列取的
    tracker.on_hit("agentic", 1000)
    tracker.materialize()
    check(abs(tracker.snapshot()["agentic"] - 1000.0) < 1e-9,
          f"agentic snapshot={tracker.snapshot()['agentic']}, want 1000")


def test_decay_uses_per_queue_window():
    """无命中时按各队列自己的 W_q 衰减：一个 W_q 后剩 1/e。"""
    print("[4] decay follows each queue's own W_q")
    tracker, clock = _tracker()
    tracker.on_hit("agentic", 1000)
    tracker.on_hit("chat", 1000)
    clock.advance(HORIZON)
    tracker.materialize()
    got_agentic = tracker.snapshot()["agentic"]
    check(abs(got_agentic - 1000 * math.exp(-HORIZON / W_AGENTIC)) < 1e-6,
          f"agentic H={got_agentic}, want 1000/e")
    got_chat = tracker.snapshot()["chat"]
    want_chat = 1000 * math.exp(-HORIZON / W_CHAT) * HORIZON / W_CHAT
    check(abs(got_chat - want_chat) < 1e-6,
          f"chat snapshot={got_chat}, want {want_chat} (W_chat={W_CHAT:.1f}s)")


def test_materialize_rebases_accumulator():
    """物化必须回写 _acc 且不重复投影：否则第二次物化会再乘一次 horizon/W。"""
    print("[5] materialize rebases the accumulator (no double projection)")
    tracker, clock = _tracker()
    tracker.on_hit("chat", 1000)
    clock.advance(HORIZON)
    tracker.materialize()
    first = tracker.snapshot()["chat"]
    tracker.on_hit("chat", 0)  # 同刻零命中：既不衰减也不加质量
    tracker.materialize()
    check(abs(tracker.snapshot()["chat"] - first) < 1e-9,
          f"second materialize drift: {tracker.snapshot()['chat']} vs {first}")
    check(abs(first - 1000 * math.exp(-HORIZON / W_CHAT) * HORIZON / W_CHAT) < 1e-9,
          "projection must be applied exactly once")


def test_steady_state_rate_is_window_invariant():
    """同一命中过程，不同 k（→不同 W_q）估出的 r 一致：W 只改方差，不改尺度。"""
    print("[6] r = H/W is invariant to the estimation window")
    period, per_hit = 10.0, 50

    def measure(k):
        tracker, clock = _tracker(k=k)
        for _ in range(200):
            clock.advance(period)
            tracker.on_hit("chat", per_hit)
        tracker.materialize()
        return tracker.snapshot()["chat"] / HORIZON  # r = 投影量 / horizon

    rate = per_hit / period
    tight, wide = measure(1.0), measure(K)
    check(abs(tight / rate - 1.0) < 0.06,
          f"k=1 r={tight:.3f}, want ~{rate:.3f}")
    check(abs(wide / rate - 1.0) < 0.06,
          f"k={K} r={wide:.3f}, want ~{rate:.3f}")
    check(abs(wide / tight - 1.0) < 0.05,
          f"窗口只该改方差：r(1)={tight:.3f} vs r({K})={wide:.3f}")


def test_long_gap_queue_does_not_zero_out():
    """chat 的间隙里不再读成 0：W=horizon 时 300s 后只剩 ~8%，W=4*T50 时保留更多。"""
    print("[7] long-gap queue keeps value through its gap")
    tracker, clock = _tracker()
    tracker.on_hit("chat", 1000)
    clock.advance(300.0)
    tracker.materialize()
    got = tracker.snapshot()["chat"]
    want = 1000 * math.exp(-300.0 / W_CHAT) * HORIZON / W_CHAT
    check(abs(got - want) < 1e-6, f"snapshot={got}, want {want}")
    old_scheme = 1000 * math.exp(-300.0 / HORIZON)  # 旧行为：W=horizon 且不投影
    check(got > old_scheme,
          f"gap 里必须保留更多价值：got {got:.1f} vs 旧口径 {old_scheme:.1f}")
    check(got > 100.0, f"300s 后不应接近 0，got {got:.1f}")


def test_random_and_unknown_queues_are_dropped():
    """random 与未知队列的命中直接丢弃：它们不参与仲裁。"""
    print("[8] random / unknown queues never enter the ledger")
    tracker, _ = _tracker()
    tracker.on_hit("random", 999)
    tracker.on_hit("nope", 999)
    tracker.materialize()
    check(
        tracker.snapshot() == {"struct": 0.0, "agentic": 0.0, "chat": 0.0},
        f"got {tracker.snapshot()}",
    )


def test_clock_injection_resets():
    """换时钟即清账：跨时钟的 elapsed 没有意义（虚拟时钟从 0 起算）。"""
    print("[9] use_clock resets to the new clock")
    tracker, _ = _tracker()
    tracker.on_hit("chat", 1000)
    clock = FakeClock(0.0)
    tracker.use_clock(clock)
    tracker.materialize()
    check(
        tracker.snapshot()["chat"] == 0.0,
        f"stale mass must be dropped on clock injection, got {tracker.snapshot()}",
    )
    tracker.on_hit("chat", 7)
    clock.advance(10_000 * HORIZON)
    tracker.materialize()
    check(
        tracker.snapshot()["chat"] == 0.0,
        f"hard underflow must clamp to 0, got {tracker.snapshot()['chat']}",
    )


def test_backwards_clock_does_not_decay():
    """时钟回退 / 同刻多事件：dt <= 0 不做衰减，保持累加语义。"""
    print("[10] dt <= 0 is not decayed")
    tracker, clock = _tracker()
    tracker.on_hit("chat", 10)
    clock.t = -5.0
    tracker.materialize()
    want = 10.0 * HORIZON / W_CHAT
    check(
        abs(tracker.snapshot()["chat"] - want) < 1e-12,
        f"backwards clock must not decay, got {tracker.snapshot()['chat']}",
    )


def test_reset_clears_everything():
    """reset 与缓存 flush 同源：质量与快照一起清空。"""
    print("[11] reset clears mass and snapshot")
    tracker, _ = _tracker()
    tracker.on_hit("struct", 500)
    tracker.materialize()
    tracker.reset()
    check(
        tracker.snapshot() == {"struct": 0.0, "agentic": 0.0, "chat": 0.0},
        f"got {tracker.snapshot()}",
    )
    check(
        tracker.densities({"struct": 0})["struct"] == 0.0,
        "reset must also drop the density",
    )


def test_determinism():
    """同一事件序列两次重放，逐位相同（无随机、无 set 迭代）。"""
    print("[12] same event sequence -> identical snapshot")

    def replay():
        tracker, clock = _tracker()
        for i in range(50):
            for queue in ("struct", "agentic", "chat"):
                tracker.on_hit(queue, 7 + i)
            clock.advance(3.0)
            if i % 5 == 0:
                tracker.materialize()
        return tracker.snapshot()

    check(replay() == replay(), "replay must be bit-identical")


def main():
    tests = [
        test_density_is_projected_hits_over_occupancy,
        test_estimation_window_rule,
        test_projection_to_shared_horizon,
        test_decay_uses_per_queue_window,
        test_materialize_rebases_accumulator,
        test_steady_state_rate_is_window_invariant,
        test_long_gap_queue_does_not_zero_out,
        test_random_and_unknown_queues_are_dropped,
        test_clock_injection_resets,
        test_backwards_clock_does_not_decay,
        test_reset_clears_everything,
        test_determinism,
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
