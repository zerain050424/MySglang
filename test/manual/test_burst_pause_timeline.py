"""burst/pause 全流程时序测试：不需要 server。

用 monkeypatch 把 `one_request` 换成瞬时假响应，然后真跑 bench 的 main_async：
victim 同相位起跑 → 全部越过 pause 点 → burst 窗口 → resume（同时 / 分批）。

断言的是**时间线**与**相位标记**，不是命中率：
  1. phases.burst_start_s / burst_end_s 存在，且窗口长度 ≈ --burst-duration-s
  2. batched 的 victim 恢复时刻呈阶梯（wave_size / interval 对齐）
  3. simultaneous 的 victim 恢复时刻几乎同时
  4. 每个 victim 恰好有一行 resume_phase="resume"，且落在 pause_after_turn+1 轮
  5. 相位顺序：pre → pause → resume → post

Run: /root/sl_old/bin/python test/manual/test_burst_pause_timeline.py
"""

import asyncio
import importlib.util
import os
import statistics
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Uni-Bench"))
BENCH = os.path.join(BENCH_DIR, "bench_unicache.py")
BASE_YAML = os.path.join(BENCH_DIR, "RR_VS_TLRU", "configs", "burst_pause.yaml")

spec = importlib.util.spec_from_file_location("bench_unicache", BENCH)
bench = importlib.util.module_from_spec(spec)
sys.modules["bench_unicache"] = bench
spec.loader.exec_module(bench)

FAILS = []
PAUSE_AFTER = 3
BURST_S = 1.5
WAVE_SIZE = 4
WAVE_INTERVAL = 0.5
VICTIMS = 8
STIR = 24


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def make_tasks_config() -> str:
    """把真实配置的数据路径改成绝对路径，并缩小规模以便几秒内跑完。"""
    with open(BASE_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(BASE_YAML)
    for task in cfg["tasks"]:
        task["data_path"] = os.path.abspath(os.path.join(cfg_dir, task["data_path"]))
        if task["name"] == "agentic_idle":
            task["max_record"] = VICTIMS
            task["pause_after_turn"] = PAUSE_AFTER
        else:
            task["max_record"] = STIR
            task["lambda_rate"] = 20
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="bp_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def install_fake_transport():
    """瞬时假响应：不连 server，但保持 one_request 的返回结构。"""

    async def fake_one_request(http, url, payload, timeout_s):
        await asyncio.sleep(0.001)
        n = len(payload["input_ids"])
        return {
            "ttft": 0.001,
            "e2e": 0.002,
            "prompt_tokens": n,
            "cached_tokens": 0,
            "completion_tokens": 1,
            "queue_time_ms": 0.0,
            "prefill_ms": 0.5,
        }

    bench.one_request = fake_one_request


def run_case(tasks_config: str, resume_mode: str):
    argv = [
        "--tasks-config", tasks_config,
        "--burst-duration-s", str(BURST_S),
        "--resume-mode", resume_mode,
        "--resume-wave-size", str(WAVE_SIZE),
        "--resume-wave-interval-s", str(WAVE_INTERVAL),
        "--concurrency", "32",
        "--seed", "42",
        "--burst-gate-timeout-s", "60",
    ]
    old = sys.argv
    sys.argv = ["bench_unicache"] + argv
    try:
        args = bench.parse_args()
    finally:
        sys.argv = old
    return asyncio.run(bench.main_async(args))


def test_batched_timeline():
    print("[1] batched: burst window then staggered resume")
    cfg = make_tasks_config()
    summary, stats = run_case(cfg, "batched")
    ph = summary.get("phases")
    check(ph is not None, "phases must be recorded")
    if not ph:
        return
    check(ph["resume_mode"] == "batched", f"resume_mode={ph['resume_mode']}")
    check(ph["victims"] == VICTIMS, f"victims={ph['victims']}, want {VICTIMS}")

    start, end = ph.get("burst_start_s"), ph.get("burst_end_s")
    check(start is not None and end is not None, "burst_start/end must be recorded")
    if start is not None and end is not None:
        check(abs((end - start) - BURST_S) < 0.4,
              f"burst window {(end - start):.2f}s, want ~{BURST_S}s")

    resumed = ph.get("victim_resumed_s") or []
    check(len(resumed) == VICTIMS, f"resumed={len(resumed)}, want {VICTIMS}")
    spread = max(resumed) - min(resumed)
    want_spread = ((VICTIMS - 1) // WAVE_SIZE) * WAVE_INTERVAL
    check(abs(spread - want_spread) < 0.3,
          f"batched spread {spread:.2f}s, want ~{want_spread}s")
    check(min(resumed) >= (end or 0) - 0.05,
          "no victim may resume before the burst window closes")

    # 相位标记：每个 victim 恰好一行 resume，且轮次 = pause_after+1
    rows = stats.rows
    check(all("resume_phase" in r for r in rows),
          "every row must carry resume_phase when the gate is active")
    resume_rows = [r for r in rows if r.get("resume_phase") == "resume"]
    check(len(resume_rows) == VICTIMS,
          f"resume rows={len(resume_rows)}, want {VICTIMS} (one per victim)")
    check(all(r["turn"] == PAUSE_AFTER + 1 for r in resume_rows),
          f"resume rows must be turn {PAUSE_AFTER + 1}, got "
          f"{sorted({r['turn'] for r in resume_rows})}")
    check(all(r["task"] == "agentic_idle" for r in resume_rows),
          "only victims can have a resume phase")

    phases_of_victim = [r["resume_phase"] for r in rows if r["task"] == "agentic_idle"]
    order = [p for p in phases_of_victim if p in ("pre", "pause", "resume", "post")]
    check(order and order[0] == "pre" and "resume" in order and order[-1] == "post",
          f"phase order looks wrong: {order[:6]}...")

    os.unlink(cfg)


def test_simultaneous_timeline():
    print("[2] simultaneous: all victims resume at once")
    cfg = make_tasks_config()
    summary, stats = run_case(cfg, "simultaneous")
    ph = summary.get("phases")
    check(ph is not None, "phases must be recorded")
    if not ph:
        return
    check(ph["resume_mode"] == "simultaneous", f"resume_mode={ph['resume_mode']}")
    resumed = ph.get("victim_resumed_s") or []
    check(len(resumed) == VICTIMS, f"resumed={len(resumed)}, want {VICTIMS}")
    spread = max(resumed) - min(resumed)
    check(spread < 0.1, f"simultaneous spread {spread:.3f}s must be < 0.1s")

    resume_rows = [r for r in stats.rows if r.get("resume_phase") == "resume"]
    check(len(resume_rows) == VICTIMS,
          f"resume rows={len(resume_rows)}, want {VICTIMS}")
    os.unlink(cfg)


def main():
    install_fake_transport()
    for t in (test_batched_timeline, test_simultaneous_timeline):
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
