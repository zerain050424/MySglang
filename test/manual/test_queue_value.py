"""已兑现价值密度（VD）账本单测（不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/queue_value.py`

口径：``VD_q = H_q / max(occ_q, floor)``，``H_q`` 是时间衰减的命中质量。
本文件只测账本自身；仲裁顺序由 ``test_queue_selector.py`` 覆盖。
"""

import importlib.util
import os
import sys
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CACHE_DIR = os.path.join(_ROOT, "python", "sglang", "srt", "mem_cache")

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


for _pkg in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
    _stub_package(_pkg)

hu = _load(
    "sglang.srt.mem_cache.heter_utils", os.path.join(_CACHE_DIR, "heter_utils.py")
)
qv = _load(
    "sglang.srt.mem_cache.queue_value", os.path.join(_CACHE_DIR, "queue_value.py")
)


class FakeClock:
    """可手动推进的单调时钟（虚拟秒）。"""

    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _tracker(tau=120.0, floor=2048.0, start=0.0):
    clock = FakeClock(start)
    tracker = qv.QueueValueTracker(tau_s=tau, floor_tokens=floor, now=clock)
    return tracker, clock


def test_density_is_hit_mass_over_occupancy():
    """VD = H / max(occ, floor)；占用低于统计下限时按 floor 计。"""
    print("[1] density = H / max(occ, floor)")
    tracker, _ = _tracker(tau=120.0, floor=2048.0)
    tracker.on_hit("chat", 1000)
    tracker.materialize()
    dens = tracker.densities({"chat": 10000, "agentic": 0, "struct": 0})
    check(abs(dens["chat"] - 0.1) < 1e-12, f"chat={dens['chat']}, want 0.1")
    # 低于 floor：分母用 2048，而不是 10
    small = tracker.densities({"chat": 10})["chat"]
    check(abs(small - 1000 / 2048.0) < 1e-12, f"small occ must use floor, got {small}")
    check(
        dens["agentic"] == 0.0 and dens["struct"] == 0.0,
        f"queues without hits must be 0, got {dens}",
    )


def test_decay_half_life_and_tau():
    """无命中时质量按 exp(-dt/tau) 衰减：一个 tau 后剩 1/e，ln2 个 tau 后减半。"""
    print("[2] decay: 1/e per tau, half per tau*ln2")
    tau = 100.0
    tracker, clock = _tracker(tau=tau)
    tracker.on_hit("chat", 1000)
    clock.advance(tau)
    tracker.materialize()
    check(
        abs(tracker.snapshot()["chat"] - 1000 * 0.36787944117) < 1e-6,
        f"after tau H={tracker.snapshot()['chat']}, want 1000/e",
    )
    tracker.reset()
    tracker.on_hit("chat", 1000)
    clock.advance(tau * 0.69314718056)
    tracker.materialize()
    check(
        abs(tracker.snapshot()["chat"] - 500.0) < 1e-6,
        f"after tau*ln2 H={tracker.snapshot()['chat']}, want 500",
    )


def test_materialize_rebases_accumulator():
    """物化必须回写 _acc：否则下一次命中会从"未衰减质量 + 零间隔"起算，凭空多记。"""
    print("[3] materialize rebases the accumulator (no double counting)")
    tracker, clock = _tracker(tau=100.0)
    tracker.on_hit("chat", 1000)
    clock.advance(100.0)
    tracker.materialize()  # H = 1000/e
    tracker.on_hit("chat", 0)  # 同刻零命中：不应把质量拉回 1000
    tracker.materialize()
    check(
        abs(tracker.snapshot()["chat"] - 1000 / 2.718281828459045) < 1e-6,
        f"H={tracker.snapshot()['chat']}, want 1000/e",
    )


def test_steady_state_matches_rate_times_tau():
    """恒定速率 r 下 H 收敛到 r*tau（离散更新，误差 2% 内）。"""
    print("[4] steady state: H -> r*tau")
    tau = 100.0
    tracker, clock = _tracker(tau=tau)
    step = tau / 10.0
    hits_per_step = 50
    for _ in range(60):
        clock.advance(step)
        tracker.on_hit("agentic", hits_per_step)
    tracker.materialize()
    expected = hits_per_step / (1.0 - pow(2.718281828459045, -step / tau))
    got = tracker.snapshot()["agentic"]
    check(
        abs(got - expected) / expected < 0.02,
        f"H={got}, want ~{expected} (r*tau = {hits_per_step * tau})",
    )


def test_random_and_unknown_queues_are_dropped():
    """random 与未知队列的命中直接丢弃：它们不参与仲裁。"""
    print("[5] random / unknown queues never enter the ledger")
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
    print("[6] use_clock resets to the new clock")
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
    clock.advance(10_000 * 120.0)
    tracker.materialize()
    check(
        tracker.snapshot()["chat"] == 0.0,
        f"hard underflow must clamp to 0, got {tracker.snapshot()['chat']}",
    )


def test_backwards_clock_does_not_decay():
    """时钟回退 / 同刻多事件：dt <= 0 不做衰减，保持累加语义。"""
    print("[7] dt <= 0 is not decayed")
    tracker, clock = _tracker()
    tracker.on_hit("chat", 10)
    clock.t = -5.0
    tracker.materialize()
    check(
        tracker.snapshot()["chat"] == 10.0,
        f"backwards clock must not change mass, got {tracker.snapshot()['chat']}",
    )


def test_reset_clears_everything():
    """reset 与缓存 flush 同源：质量与快照一起清空。"""
    print("[8] reset clears mass and snapshot")
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
    print("[9] same event sequence -> identical snapshot")

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
        test_density_is_hit_mass_over_occupancy,
        test_decay_half_life_and_tau,
        test_materialize_rebases_accumulator,
        test_steady_state_matches_rate_times_tau,
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
