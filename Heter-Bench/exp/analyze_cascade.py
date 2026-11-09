#!/usr/bin/env python3
"""Cascade 实验分析：从逐请求 CSV 计算级联淘汰指标。

指标（详见 exp/README.md §2.7）：
  Wipe / WipeRate      cached_tokens <= --wipe-max 且 turn>1（整条链被抽干）
  HHI                  各会话 wipe 数的集中度；1/N=完全摊薄，1.0=全部集中在一个会话
  CliffRate            cached 单轮跌幅 > 50% 的比例
  LossQuantum          resume 首轮 recompute / 链长（1.0 = 整条链重算）
  SimultaneityIndex    每个 wipe 时刻其他存活会话的 cached/prompt 中位数
  p99_resume_ttft      仅 resume 段（turn > --burst-turns）的 TTFT p99

用法：
  python exp/analyze_cascade.py --csv result/cascade/R3.csv
  python exp/analyze_cascade.py --csv a.csv b.csv --out result/cascade/an.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

VICTIM_TASKS = ("AgentBank_alfred_hot", "AgentBank_alfred_warm", "AgentBank_alfred_cold")


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def hhi(weights):
    tot = sum(weights)
    if tot <= 0:
        return None
    return sum((w / tot) ** 2 for w in weights)


def spearman(x, y):
    if len(x) < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    rx, ry = rank(x), rank(y)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def analyze(path, burst_turns, wipe_max, system_prompt_guard):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("prompt_tokens"):
                continue
            r["turn"] = int(r["turn"])
            r["prompt_tokens"] = int(r["prompt_tokens"])
            r["cached_tokens"] = int(r["cached_tokens"])
            r["recompute_tokens"] = int(r["recompute_tokens"])
            r["ttft_ms"] = float(r["ttft_ms"]) if r.get("ttft_ms") else None
            r["ts"] = float(r["ts"]) if r.get("ts") else None
            rows.append(r)

    victims = [r for r in rows if r["task"] in VICTIM_TASKS]
    by_sess = defaultdict(list)
    for r in victims:
        by_sess[r["session_id"]].append(r)
    for rs in by_sess.values():
        rs.sort(key=lambda r: r["turn"])

    out = {"csv": str(path), "num_requests": len(rows), "num_victim_sessions": len(by_sess)}

    # ---- burst 段自检：这一段应当没有 wipe ----
    burst_wipe = sum(1 for r in victims
                     if r["turn"] > 1 and r["turn"] <= burst_turns
                     and r["cached_tokens"] <= wipe_max)

    # ---- 逐会话 ----
    per_sess = {}
    wipes = []
    cliff_num = cliff_den = 0
    for sid, rs in by_sess.items():
        chain = rs[-1]["prompt_tokens"]
        w = [r for r in rs if r["turn"] > 1 and r["cached_tokens"] <= wipe_max]
        for r in w:
            wipes.append(r)
        cliff_num += sum(1 for a, b in zip(rs, rs[1:])
                         if a["cached_tokens"] > 0 and b["cached_tokens"] < 0.5 * a["cached_tokens"])
        cliff_den += max(len(rs) - 1, 1)
        resume = [r for r in rs if r["turn"] > burst_turns]
        per_sess[sid] = {
            "group": rs[0]["task"],
            "turns": len(rs),
            "chain_tokens": chain,
            "wipes": len(w),
            "wipe_turns": [r["turn"] for r in w],
            "loss_quantum": (resume[0]["recompute_tokens"] / chain) if resume and chain else None,
            "resume_ttft_ms": [r["ttft_ms"] for r in resume if r["ttft_ms"] is not None],
            "resume_hit_ratio": (
                resume[0]["cached_tokens"] / resume[0]["prompt_tokens"]
                if resume and resume[0]["prompt_tokens"] else None),
        }

    # ---- 分组 ----
    groups = defaultdict(lambda: {"sessions": [], "wipes": 0, "resume_ttft": []})
    for sid, s in per_sess.items():
        g = groups[s["group"]]
        g["sessions"].append(sid)
        g["wipes"] += s["wipes"]
        g["resume_ttft"] += s["resume_ttft_ms"]
    for g, v in groups.items():
        v["wipe_rate"] = v["wipes"] / max(sum(per_sess[s]["turns"] - 1 for s in v["sessions"]), 1)
        v["resume_hit_ratio_median"] = st.median(
            [per_sess[s]["resume_hit_ratio"] for s in v["sessions"]
             if per_sess[s]["resume_hit_ratio"] is not None]) if v["sessions"] else None
        v["loss_quantum_median"] = st.median(
            [per_sess[s]["loss_quantum"] for s in v["sessions"]
             if per_sess[s]["loss_quantum"] is not None]) if v["sessions"] else None
        v["sessions"] = len(v["sessions"])

    # ---- 本次 run 的最终临界：所有 victim 请求都用 prompt_tokens-vs-chain 归一 ----
    losses = [r["recompute_tokens"] for r in victims if r["turn"] > burst_turns]
    out.update({
        "burst_turns": burst_turns,
        "wipe_max_cached": wipe_max,
        "burst_wipes": burst_wipe,                      # 期望 0
        "wipes_total": len(wipes),
        "wipe_rate": len(wipes) / max(len([r for r in victims if r["turn"] > 1]), 1),
        "hhi_sessions": hhi([per_sess[s]["wipes"] for s in per_sess]),   # 1/N=摊薄
        "hhi_groups": hhi([groups[g]["wipes"] for g in groups]),
        "cliff_rate": cliff_num / max(cliff_den, 1),
        "loss_quantum_median": st.median([s["loss_quantum"] for s in per_sess.values()
                                          if s["loss_quantum"] is not None])
        if per_sess else None,
        "max_single_session_loss": max(losses) if losses else None,
        "recompute_total": sum(losses),
        "ttft_p50_ms": pct([r["ttft_ms"] for r in rows if r["ttft_ms"]], 50),
        "ttft_p99_ms": pct([r["ttft_ms"] for r in rows if r["ttft_ms"]], 99),
        "resume_ttft_p99_ms": pct([r["ttft_ms"] for r in victims
                                   if r["turn"] > burst_turns and r["ttft_ms"]], 99),
        "groups": {g: {k: v2 for k, v2 in v.items() if k != "resume_ttft"}
                   for g, v in groups.items()},
        "per_session": per_sess,
    })

    # ---- 同时性：每次 wipe 时刻，其他会话最近一次请求的命中率 ----
    stamped = [r for r in victims if r["ts"]]
    sims = []
    for w in wipes:
        if not w["ts"]:
            continue
        others = []
        for sid, rs in by_sess.items():
            if sid == w["session_id"]:
                continue
            prior = [r for r in rs if r["ts"] and r["ts"] <= w["ts"]]
            if prior and prior[-1]["prompt_tokens"]:
                others.append(prior[-1]["cached_tokens"] / prior[-1]["prompt_tokens"])
        if others:
            sims.append(st.median(others))
    out["simultaneity_index_median"] = st.median(sims) if sims else None
    out["simultaneity_samples"] = len(sims)

    # ---- TTFT 分解：尾部是 prefill 还是排队 ----
    pf = [float(r["prefill_ms"]) for r in victims if r.get("prefill_ms")]
    qt = [float(r["queue_time_ms"]) for r in victims if r.get("queue_time_ms")]
    out["victim_prefill_p99_ms"] = pct(pf, 99)
    out["victim_queue_p99_ms"] = pct(qt, 99)

    # ---- H3'：run 级 p99 与最大单会话重算 ----
    out["spearman_resume_ttft_vs_loss"] = (
        spearman([s["loss_quantum"] or 0.0 for s in per_sess.values()],
                 [max(s["resume_ttft_ms"]) if s["resume_ttft_ms"] else 0.0
                  for s in per_sess.values()]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--burst-turns", type=int, default=20)
    ap.add_argument("--wipe-max", type=int, default=64, help="cached_tokens 低于此值算整链抽干")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = [analyze(p, args.burst_turns, args.wipe_max, True) for p in args.csv]
    for r in results:
        print(f"\n===== {Path(r['csv']).name} =====")
        print(f"  victim 会话数 {r['num_victim_sessions']}  请求 {r['num_requests']}")
        print(f"  burst 段 wipe = {r['burst_wipes']}   (期望 0)")
        print(f"  wipe 总数 {r['wipes_total']}  WipeRate {r['wipe_rate']:.3f}")
        hh, hg = r["hhi_sessions"], r["hhi_groups"]
        print(f"  HHI(会话) {hh if hh is None else round(hh, 3)}   "
              f"HHI(组) {hg if hg is None else round(hg, 3)}   (摊薄=1/N)")
        print(f"  CliffRate {r['cliff_rate']:.3f}   LossQuantum {r['loss_quantum_median']}")
        print(f"  SimultaneityIndex {r['simultaneity_index_median']}  (n={r['simultaneity_samples']})")
        print(f"  resume TTFT p99 {r['resume_ttft_p99_ms']} ms   全量 TTFT p50/p99 "
              f"{r['ttft_p50_ms']}/{r['ttft_p99_ms']} ms")
        print(f"  victim prefill p99 {r['victim_prefill_p99_ms']} ms   queue p99 {r['victim_queue_p99_ms']} ms")
        print(f"  Spearman(loss_quantum, max_resume_ttft) = {r['spearman_resume_ttft_vs_loss']}")
        print("  分组：")
        for g, v in sorted(r["groups"].items()):
            print(f"    {g:22s} wipes={v['wipes']:<3} wipe_rate={v['wipe_rate']:.3f} "
                  f"resume_hit={v['resume_hit_ratio_median']} loss_q={v['loss_quantum_median']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
