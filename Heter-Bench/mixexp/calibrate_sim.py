#!/usr/bin/env python3
"""用真机结果标定 CPU 回放器（离线命中率 vs 在线命中率）。

为什么需要这一步
----------------
`wtl.replay --backend real` 在 CPU 上跑的是**同一份** `HeterCacheRadixCache` 与
同一套策略代码，所以复现的难点不在策略，而在**调度模型**：

* 在线端 `bench_hetercache` 用 `asyncio.Semaphore(concurrency)` 限制**并发会话数**；
* 在线端的 allocator 容量里有一部分被在途请求占着（prompt / decode / locked），
  不归缓存调度。

这两项如果不对齐，命中率能差一个数量级（full 档实测 7.5% vs 59.2%）。本脚本
负责把「真机已经跑过的档位」逐臂重放一遍，把偏差摆到桌面上：

    python mixexp/calibrate_sim.py --scale full --arms lru,hc_full

判定：每个 task 的命中率偏差绝对值 ≤ `--threshold`（默认 2pp）才算通过，
任一臂不通过则以非零码退出，方便串进 CI 或提交前的手工检查。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # Heter-Bench/mixexp
BENCH_DIR = HERE.parent                           # Heter-Bench
REPO_ROOT = BENCH_DIR.parent                      # sg-unicache
SIM_DIR = BENCH_DIR / "W-TinyLRU"
RESULT_DIR = BENCH_DIR / "result" / "mixexp"

# 与 run_mixexp.sh 的规模阶梯保持一致（真机被裁剪时以日志里的实际值为准）
SCALE_DEFAULTS = {
    "smoke": {"pool": 30000, "concurrency": 8},
    "quick": {"pool": 60000, "concurrency": 16},
    "mid": {"pool": 120000, "concurrency": 24},
    "full": {"pool": 200000, "concurrency": 32},
}

# 每个并发槽位在缓存之外占用的 token（full 档 32 并发拟合值 3000/32 ≈ 94）。
# 来源：在途请求在 prefill/decode 期间持有的 KV + lock_ref 住的前缀，不归缓存调度。
RESERVE_PER_SLOT = 94


def load_arms() -> dict[str, dict]:
    """读 arms.tsv：arm -> {policy, config, desc}。"""
    arms = {}
    for line in (HERE / "arms.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, policy = parts[0].strip(), parts[1].strip()
        raw_cfg = parts[2].strip() if len(parts) > 2 else "-"
        cfg = None
        if raw_cfg and raw_cfg != "-":
            try:
                cfg = json.loads(raw_cfg)
            except json.JSONDecodeError:
                cfg = None
        arms[name] = {"policy": policy, "config": cfg,
                      "desc": parts[3].strip() if len(parts) > 3 else ""}
    return arms


def detect_pool(scale: str, arm: str, fallback: int) -> int:
    """从真机 server 日志里取 allocator 实际容量（配置值会被显存裁剪）。"""
    log = RESULT_DIR / f"server_{scale}_{arm}.log"
    if log.exists():
        m = re.search(r"max_total_num_tokens=(\d+)", log.read_text(errors="replace"))
        if m:
            return int(m.group(1))
    return fallback


def sim_pythonpath() -> str:
    parts = [str(REPO_ROOT / "python")]
    # 实验机的 sglang 依赖装在 sl_old 的 site-packages 里
    sl = Path("/root/sl_old/lib/python3.12/site-packages")
    if sl.exists():
        parts.append(str(sl))
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)


def run_sim(scale: str, arm: str, *, pool: int, reserve: int, concurrency: int,
            wait_scale: float, arrival_scale: float, inflight_ms: float,
            python: str, extra: list[str]) -> dict:
    out = Path("/tmp") / f"calib_sim_{scale}_{arm}.json"
    cfg_path = HERE / "configs" / f"mix_{scale}.yaml"
    cmd = [
        python, "-m", "wtl.replay",
        "--arm", arm,
        "--tasks-config", str(cfg_path),
        "--num-sessions", "-1",
        "--pool-tokens", str(pool),
        "--inflight-reserve", str(reserve),
        "--concurrency", str(concurrency),
        # 轮间等待 / 到达节奏必须与真机那一轮完全一致：smoke 档真机用 0.1，
        # 差 10 倍的等待会让会话重叠度完全不同（实测 ShareGPT 差 33pp）。
        "--wait-scale", str(wait_scale),
        "--arrival-scale", str(arrival_scale),
        # 在途时长 = 真机的 e2e 均值：在途请求的前缀不可淘汰，这是配额/轮询
        # 能「把 agentic 压到目标以下」还是「压不下去」的分水岭。
        "--inflight-ms", str(inflight_ms),
        "--backend", "real",
        "--quiet",
        "--out", str(out),
        *extra,
    ]
    env = dict(os.environ, PYTHONPATH=sim_pythonpath())
    print(f"  $ {' '.join(cmd[1:])}", file=sys.stderr)
    subprocess.run(cmd, cwd=SIM_DIR, env=env, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def per_task(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[int]] = {}
    for r in rows:
        a = acc.setdefault(r["task"], [0, 0])
        a[0] += int(r["cached_tokens"])
        a[1] += int(r["prompt_tokens"])
    return {t: c / p * 100 for t, (c, p) in acc.items() if p}


def main() -> int:
    ap = argparse.ArgumentParser(description="用真机结果标定 CPU 回放器")
    ap.add_argument("--scale", default="full")
    ap.add_argument("--arms", default=None,
                    help="逗号分隔，缺省=所有存在真机结果的臂")
    ap.add_argument("--pool", type=int, default=None,
                    help="allocator 容量；缺省从 server 日志探测，再缺省用档位默认值")
    ap.add_argument("--reserve", type=int, default=None,
                    help="缓存之外的在途占用；缺省 = 并发数 × 94")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="并发会话数；缺省从真机结果读，再缺省用档位默认值")
    ap.add_argument("--threshold", type=float, default=2.0, help="单 task 允许偏差（pp）")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms_def = load_arms()
    real_files = {
        p.stem.split(f"{args.scale}_", 1)[1]: p
        for p in sorted(RESULT_DIR.glob(f"{args.scale}_*.json"))
        if re.match(rf"^{re.escape(args.scale)}_.+?(_r\d+)?$", p.stem)
    }
    wanted = [a.strip() for a in args.arms.split(",")] if args.arms else list(real_files)
    wanted = [a for a in wanted if a in real_files and a in arms_def]
    if not wanted:
        print(f"没有可标定的臂（scale={args.scale}，真机结果目录 {RESULT_DIR}）", file=sys.stderr)
        return 2

    defaults = SCALE_DEFAULTS.get(args.scale, SCALE_DEFAULTS["full"])
    report: dict[str, dict] = {}
    worst = 0.0
    failed: list[str] = []

    for arm in wanted:
        real = json.loads(real_files[arm].read_text(encoding="utf-8"))
        concurrency = args.concurrency or int(real.get("concurrency") or defaults["concurrency"])
        pool = args.pool or detect_pool(args.scale, arm, defaults["pool"])
        reserve = args.reserve if args.reserve is not None else concurrency * RESERVE_PER_SLOT

        print(f"\n=== {args.scale}/{arm}  pool={pool} reserve={reserve} conc={concurrency}",
              file=sys.stderr)
        # arms.tsv 里的 policy/config 与真机 server 启动参数同源，直接用
        extra = []
        if arms_def[arm]["policy"]:
            extra += ["--policy", arms_def[arm]["policy"]]
        if arms_def[arm]["config"] is not None:
            extra += ["--policy-config", json.dumps(arms_def[arm]["config"])]
        sim = run_sim(args.scale, arm, pool=pool, reserve=reserve,
                      concurrency=concurrency,
                      wait_scale=float(real.get("wait_scale", 1.0)),
                      arrival_scale=float(real.get("arrival_scale", 1.0)),
                      inflight_ms=float(real.get("e2e_mean_ms") or 0.0),
                      python=args.python, extra=extra)
        sim_task = per_task(sim["rows"])
        real_task = {t: v["cache_hit_rate_pct"] for t, v in (real.get("per_task") or {}).items()}
        sim_total = sim["summary"]["hit_pct"]
        real_total = real["cache_hit_rate_pct"]

        rows = []
        for task in sorted(set(sim_task) | set(real_task)):
            s, r = sim_task.get(task), real_task.get(task)
            d = None if (s is None or r is None) else s - r
            if d is not None:
                worst = max(worst, abs(d))
                if abs(d) > args.threshold:
                    failed.append(f"{arm}/{task}")
            rows.append({"task": task, "sim": s, "real": r, "delta": d})

        report[arm] = {
            "pool": pool, "reserve": reserve, "concurrency": concurrency,
            "wait_scale": float(real.get("wait_scale", 1.0)),
            "arrival_scale": float(real.get("arrival_scale", 1.0)),
            "inflight_ms": float(real.get("e2e_mean_ms") or 0.0),
            "sim_total": sim_total, "real_total": real_total,
            "delta_total": sim_total - real_total, "tasks": rows,
            "requests": len(sim["rows"]),
        }

        print(f"\n## {args.scale} / {arm}   模拟 {sim_total:.2f}%  真机 {real_total:.2f}%  "
              f"Δ {sim_total - real_total:+.2f}pp\n")
        print("| task | 模拟 % | 真机 % | Δ pp |")
        print("|---|---|---|---|")
        for row in rows:
            s = "—" if row["sim"] is None else f"{row['sim']:.2f}"
            r = "—" if row["real"] is None else f"{row['real']:.2f}"
            d = "—" if row["delta"] is None else f"{row['delta']:+.2f}"
            print(f"| {row['task']} | {s} | {r} | {d} |")
        print(f"| **合计** | {sim_total:.2f} | {real_total:.2f} | {sim_total - real_total:+.2f} |")

    print(f"\n最大单 task 偏差 {worst:.2f}pp（阈值 {args.threshold}pp）")
    if failed:
        print("未通过：" + ", ".join(failed))
    else:
        print("标定通过")

    out = Path(args.out) if args.out else RESULT_DIR / f"calib_sim_{args.scale}.json"
    out.write_text(json.dumps({
        "scale": args.scale, "threshold": args.threshold, "worst_pp": worst,
        "passed": not failed, "failed": failed, "arms": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
