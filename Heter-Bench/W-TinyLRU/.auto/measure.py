#!/usr/bin/env python3
"""回放器保真度度量：逐 task 对比 CPU 回放与真机命中率。

输出 METRIC 行。主指标 max_dev_pp = 所有 (case, task) 里 |模拟 − 真机| 的最大值。
不放宽口径、不排除任何 task：任何一个点对不上都算没还原。
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

SIM = Path(__file__).resolve().parent.parent          # W-TinyLRU
BENCH = SIM.parent                                    # Heter-Bench
MIX = BENCH / "mixexp"
RESULT = BENCH / "result" / "mixexp"
REPO = BENCH.parent
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(
    [str(REPO / "python"), "/root/sl_old/lib/python3.12/site-packages"]))

# 与真机当轮参数一致：pool 一律取 server 日志里的 max_total_num_tokens
CASES = [
    # scale, arm, inflight_ms(0=用真机 e2e)
    ("full", "lru", None),
    ("full", "hc_full", None),
    ("smoke", "lru", None),
]


def arms() -> dict:
    out = {}
    for line in (MIX / "arms.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 2:
            continue
        cfg = p[2].strip() if len(p) > 2 else "-"
        out[p[0].strip()] = {"policy": p[1].strip(),
                             "config": None if cfg in ("", "-") else json.loads(cfg)}
    return out


def pool_of(scale: str, arm: str) -> int:
    log = RESULT / f"server_{scale}_{arm}.log"
    m = re.search(r"max_total_num_tokens=(\d+)", log.read_text(errors="replace")) if log.exists() else None
    return int(m.group(1)) if m else 0


def replay(scale: str, arm: str, spec: dict, inflight_ms: float) -> dict:
    real = json.loads((RESULT / f"{scale}_{arm}.json").read_text(encoding="utf-8"))
    pool = pool_of(scale, arm)
    conc = int(real.get("concurrency") or 32)
    out = Path("/tmp") / f"fid_{scale}_{arm}.json"
    cmd = [PY, "-m", "wtl.replay", "--arm", arm,
           "--policy", spec["policy"],
           "--tasks-config", str(MIX / "configs" / f"mix_{scale}.yaml"),
           "--num-sessions", "-1",
           "--pool-tokens", str(pool),
           "--concurrency", str(conc),
           "--wait-scale", str(real.get("wait_scale", 1.0)),
           "--arrival-scale", str(real.get("arrival_scale", 1.0)),
           "--inflight-ms", str(inflight_ms if inflight_ms is not None
                                else (real.get("e2e_mean_ms") or 0.0)),
           "--backend", "real", "--quiet", "--out", str(out)]
    if spec["config"] is not None:
        cmd += ["--policy-config", json.dumps(spec["config"])]
    t0 = time.time()
    subprocess.run(cmd, cwd=SIM, env=ENV, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sim = json.loads(out.read_text(encoding="utf-8"))
    rows = sim["rows"]
    acc = {}
    for r in rows:
        a = acc.setdefault(r["task"], [0, 0])
        a[0] += int(r["cached_tokens"]); a[1] += int(r["prompt_tokens"])
    sim_task = {t: c / p * 100 for t, (c, p) in acc.items() if p}
    return {"sim": sim_task, "real": {t: v["cache_hit_rate_pct"] for t, v in real["per_task"].items()},
            "sim_total": sim["summary"]["hit_pct"], "real_total": real["cache_hit_rate_pct"],
            "wall": time.time() - t0, "pool": pool, "conc": conc}


def main() -> int:
    specs = arms()
    rows, worst, worst_at, wall = [], -1.0, "", 0.0
    for scale, arm, ims in CASES:
        if not (RESULT / f"{scale}_{arm}.json").exists():
            print(f"[skip] 无真机结果：{scale}_{arm}", file=sys.stderr)
            continue
        r = replay(scale, arm, specs[arm], ims)
        wall += r["wall"]
        for task in sorted(set(r["sim"]) | set(r["real"])):
            s, t = r["sim"].get(task), r["real"].get(task)
            if s is None or t is None:
                continue
            d = s - t
            rows.append((scale, arm, task, s, t, d))
            if abs(d) > worst:
                worst, worst_at = abs(d), f"{scale}/{arm}/{task}"
        print(f"[{scale}/{arm}] sim={r['sim_total']:.2f}% real={r['real_total']:.2f}% "
              f"Δ={r['sim_total']-r['real_total']:+.2f}pp  {r['wall']:.0f}s "
              f"pool={r['pool']} conc={r['conc']}", file=sys.stderr)

    for scale, arm, task, s, t, d in rows:
        print(f"METRIC dev_{scale}_{arm}_{task}={d:+.2f}")
    by_arm = {}
    for scale, arm, task, s, t, d in rows:
        by_arm.setdefault(f"{scale}/{arm}", []).append(abs(d))
    for k, v in by_arm.items():
        print(f"METRIC dev_{k.replace('/', '_')}={max(v):.2f}")
    print(f"METRIC max_dev_pp={worst:.2f}")
    print(f"METRIC sim_wall_s={wall:.0f}")
    print(f"METRIC worst_case={worst_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
