#!/usr/bin/env python3
"""逐会话 TTFT 报表：把 H/W/C 组内每个对话的 TTFT 打出来。

数据源是 bench 产出的逐请求 CSV（每行一次 /generate）。

用法：
  python exp/session_ttft.py --csv result/cascade/R3.csv
  python exp/session_ttft.py --csv result/cascade/R3.csv --detail       # 逐轮明细
  python exp/session_ttft.py --csv result/cascade/R{1,2,3}.csv --out result/cascade/sess_ttft.json

列定义：
  chain      会话末轮 prompt 长度（链长）
  burst      第 1..20 轮的 TTFT p50 / max（全热，作该会话自身的基线）
  resume21.. 第 21..26 轮（pause 后）逐轮 TTFT
  rec@21     resume 首轮的重算 token 数（≈被清掉的链长）
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

GROUP_ORDER = ("hot", "warm", "cold")


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path, burst_turns):
    by = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            task = r["task"]
            if not task.startswith("AgentBank_alfred_"):
                continue
            grp = task.replace("AgentBank_alfred_", "")
            t = int(r["turn"])
            row = {
                "turn": t,
                "ttft": float(r["ttft_ms"]) if r["ttft_ms"] else None,
                "prefill": float(r["prefill_ms"]) if r.get("prefill_ms") else None,
                "queue": float(r["queue_time_ms"]) if r.get("queue_time_ms") else None,
                "prompt": int(r["prompt_tokens"]),
                "cached": int(r["cached_tokens"]),
                "rec": int(r["recompute_tokens"]),
            }
            by.setdefault((grp, r["session_id"]), []).append(row)
    for rs in by.values():
        rs.sort(key=lambda x: x["turn"])
    return by


def summarize(rows, burst_turns):
    burst = [r["ttft"] for r in rows if r["turn"] <= burst_turns and r["ttft"]]
    resume = [r for r in rows if r["turn"] > burst_turns]
    rt = [r["ttft"] for r in resume if r["ttft"]]
    first = resume[0] if resume else None
    return {
        "chain": rows[-1]["prompt"],
        "burst_p50": pctl(burst, 50),
        "burst_max": max(burst) if burst else None,
        "resume_ttft": rt,
        "resume_p50": pctl(rt, 50),
        "resume_max": max(rt) if rt else None,
        "rec_at_first_resume": first["rec"] if first else None,
        "cached_at_first_resume": first["cached"] if first else None,
        "prompt_at_first_resume": first["prompt"] if first else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--burst-turns", type=int, default=20)
    ap.add_argument("--detail", action="store_true", help="打印每个会话逐轮 TTFT")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dump = []
    for path in args.csv:
        by = load(path, args.burst_turns)
        print(f"\n########## {Path(path).name} ##########")
        groups = {}
        for (grp, sid), rows in by.items():
            groups.setdefault(grp, []).append((sid, rows))

        for grp in GROUP_ORDER + tuple(g for g in groups if g not in GROUP_ORDER):
            if grp not in groups:
                continue
            print(f"\n=== {grp}（{len(groups[grp])} 会话）===")
            print(f"{'session':22s} {'chain':>6s} {'burst p50/max':>16s} "
                  f"{'resume 逐轮 TTFT (ms)':<44s} {'p50':>7s} {'max':>7s} {'rec@21':>7s}")
            for sid, rows in sorted(groups[grp], key=lambda x: x[0]):
                s = summarize(rows, args.burst_turns)
                seq = " ".join(f"{v:7.1f}" if v else "      -"
                               for v in (s["resume_ttft"] + [None] * 6)[:6])
                print(f"{sid[:22]:22s} {s['chain']:>6d} "
                      f"{s['burst_p50']:>7.1f}/{s['burst_max']:<7.1f} {seq:<44s} "
                      f"{s['resume_p50']:>7.1f} {s['resume_max']:>7.1f} "
                      f"{s['rec_at_first_resume']:>7d}")
                if args.detail:
                    for r in rows:
                        print(f"    turn{r['turn']:>2}  ttft={r['ttft']:>7.1f}  "
                              f"prefill={r['prefill']:>7.1f}  queue={r['queue']:>5.1f}  "
                              f"cached={r['cached']:>5d}/{r['prompt']:<5d} rec={r['rec']:>5d}")
                dump.append({"csv": path, "group": grp, "session_id": sid, **s})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
