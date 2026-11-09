"""ResumeGate：burst 窗口结束后 simultaneous / batched 两种放行时序。

只测调度时序（纯 asyncio，不碰 HTTP），因此可以在没有 server 的情况下跑：
    /root/sl_old/bin/python test/manual/test_burst_pause_gate.py
"""

import asyncio
import importlib.util
import os
import sys
import time

BENCH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Heter-Bench", "bench_hetercache.py"
)
spec = importlib.util.spec_from_file_location("bench_hetercache", BENCH)
bench = importlib.util.module_from_spec(spec)
# dataclass 需要在 exec 期间能通过 sys.modules[cls.__module__] 找到模块
sys.modules["bench_hetercache"] = bench
spec.loader.exec_module(bench)
ResumeGate = bench.ResumeGate

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def test_wave_mapping():
    print("[1] victim -> wave mapping")
    sim = ResumeGate(6, mode="simultaneous", wave_size=2)
    check([sim.victim_wave(i) for i in range(6)] == [0] * 6,
          "simultaneous: every victim is wave 0")

    bat = ResumeGate(6, mode="batched", wave_size=2)
    check([bat.victim_wave(i) for i in range(6)] == [0, 0, 1, 1, 2, 2],
          f"batched wave_size=2: {[bat.victim_wave(i) for i in range(6)]}")

    bat3 = ResumeGate(7, mode="batched", wave_size=3)
    check([bat3.victim_wave(i) for i in range(7)] == [0, 0, 0, 1, 1, 1, 2],
          "batched wave_size=3: 7 victims -> [0,0,0,1,1,1,2]")


async def _run_waiters(gate, n):
    t_open_holder = {}

    async def waiter(i):
        await gate.wait(i)
        return i, time.perf_counter()

    tasks = [asyncio.create_task(waiter(i)) for i in range(n)]
    await asyncio.sleep(0.05)                     # 确认 open 之前全部阻塞
    blocked = [t for t in tasks if not t.done()]
    t_open = time.perf_counter()
    gate.open(t_open)
    t_open_holder["t"] = t_open
    done = await asyncio.gather(*tasks)
    return blocked, t_open, sorted(done, key=lambda x: x[0])


def test_simultaneous_release():
    print("[2] simultaneous: all victims resume within a tight window")
    gate = ResumeGate(6, mode="simultaneous", wave_size=2, wave_interval_s=0.2)
    blocked, t_open, done = asyncio.run(_run_waiters(gate, 6))
    check(len(blocked) == 6, f"all 6 waiters must block before open, got {len(blocked)}")
    offsets = [t - t_open for _, t in done]
    check(max(offsets) < 0.05,
          f"simultaneous: max offset {max(offsets):.3f}s must be < 0.05s")
    check(len(gate.resumed) == 6, f"resumed count {len(gate.resumed)}")


def test_batched_release():
    print("[3] batched: waves resume at t_open + k*interval")
    interval = 0.2
    gate = ResumeGate(6, mode="batched", wave_size=2, wave_interval_s=interval)
    blocked, t_open, done = asyncio.run(_run_waiters(gate, 6))
    check(len(blocked) == 6, "all waiters block before open")

    by_victim = dict(done)
    offsets = {i: by_victim[i] - t_open for i in range(6)}
    # 期望：victim 0,1 -> ~0；2,3 -> ~0.2；4,5 -> ~0.4
    for i, want in enumerate([0.0, 0.0, interval, interval, 2 * interval, 2 * interval]):
        got = offsets[i]
        check(abs(got - want) < 0.06,
              f"victim {i}: resumed at +{got:.3f}s, want +{want:.3f}s")

    spread = max(offsets.values()) - min(offsets.values())
    check(spread > 2 * interval * 0.8,
          f"batched spread {spread:.3f}s must approach {2 * interval:.3f}s")


def test_open_is_idempotent():
    print("[4] open() keeps the first timestamp")
    gate = ResumeGate(2, mode="batched", wave_size=1, wave_interval_s=0.1)
    t1 = time.perf_counter()
    gate.open(t1)
    gate.open(t1 + 1.0)
    check(gate.opened_at == t1, f"opened_at moved: {gate.opened_at} != {t1}")

    async def immediate():
        await gate.wait(0)
        return time.perf_counter()

    got = asyncio.run(immediate())
    check(got - t1 < 0.05, f"wave 0 must not wait, got +{got - t1:.3f}s")


def test_wait_blocks_until_open():
    print("[5] wait() blocks while the burst window is still open")

    async def scenario():
        gate = ResumeGate(1, mode="simultaneous")
        task = asyncio.create_task(gate.wait(0))
        await asyncio.sleep(0.1)
        still_blocked = not task.done()
        gate.open()
        await task
        return still_blocked

    check(asyncio.run(scenario()), "wait() must block before open()")


def main():
    tests = [
        test_wave_mapping,
        test_simultaneous_release,
        test_batched_release,
        test_open_is_idempotent,
        test_wait_blocks_until_open,
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
