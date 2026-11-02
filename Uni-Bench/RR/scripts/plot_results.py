#!/usr/bin/env python3
"""RR 实验结果绘图（输出 RR/figs/*.png，图内标注英文以免缺 CJK 字体）。"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("RR/results")
FIG = Path("RR/figs")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})

# 格 → (显示名, 是否压力格, ΣL/pool)
CELLS = {
    "burst_s36b": ("seed1 n_a=3", True, 1.55),
    "burst_n4":   ("n_a=4", True, 1.60),
    "thrash_n8":  ("n_a=8 thrash", True, 1.85),
    "burst_s20251029": ("seed2 n_a=3", True, 1.59),
    "burst_s20251030": ("seed3 n_a=3", True, 1.50),
    "burst_n8":   ("n_a=8 pool73k", False, 0.91),
}


def load(arm: str) -> pd.DataFrame | None:
    p = RES / f"{arm}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d if "task" in d else None


def burst(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d.task == "agentic_idle") & (d.turn == 11)]


def stat(arm: str) -> dict | None:
    d = load(arm)
    if d is None:
        return None
    b = burst(d)
    if not len(b):
        return None
    a = d[d.task == "agentic_active"]
    return dict(
        arm=arm,
        hit=1 - d.recompute_tokens.sum() / d.prompt_tokens.sum(),
        b_hit=1 - b.recompute_tokens.sum() / b.prompt_tokens.sum(),
        b_tot=b.recompute_tokens.sum(), b_max=b.recompute_tokens.max(),
        p50=b.ttft_ms.quantile(.5), p99=b.ttft_ms.quantile(.99),
        qmax=b.queue_time_ms.max(),
        a_hit=(1 - a.recompute_tokens.sum() / a.prompt_tokens.sum()) if len(a) else np.nan,
    )


def main() -> None:
    rows = []
    for cell, (label, press, rho) in CELLS.items():
        for pol in ("base", "rr"):
            s = stat(f"{cell}_{pol}")
            if s:
                rows.append(dict(s, cell=cell, label=label, press=press, rho=rho, pol=pol))
    df = pd.DataFrame(rows)

    # ---------- fig1：分组柱状图 ----------
    press = df[df.press]
    labels = [CELLS[c][0] for c in dict.fromkeys(press.cell)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, key, title, unit in zip(
        axes, ("b_tot", "p99", "qmax"),
        ("Burst recompute tokens", "Burst TTFT p99 (ms)", "Max queue time (ms)"),
        ("tokens", "ms", "ms")):
        base = [press[(press.cell == c) & (press.pol == "base")][key].values[0] for c in dict.fromkeys(press.cell)]
        rr = [press[(press.cell == c) & (press.pol == "rr")][key].values[0] for c in dict.fromkeys(press.cell)]
        x = np.arange(len(labels)); w = .38
        ax.bar(x - w/2, base, w, label="base (LRU-like)", color="#8c8c8c")
        ax.bar(x + w/2, rr, w, label="RR (hard quota)", color="#1f77b4")
        for i, (b, r) in enumerate(zip(base, rr)):
            ax.text(i + w/2, r, f"{(r/b-1)*100:+.0f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(title); ax.set_ylabel(unit)
        if key == "b_tot":
            ax.legend(fontsize=7)
    fig.suptitle("Fig 1  RR vs base in the critical-concurrency regime (5 pressure cells)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig1_grouped.png"); plt.close(fig)

    # ---------- fig2：burst TTFT 的 ECDF（合并 4 个种子格） ----------
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for pol, color, name in (("base", "#8c8c8c", "base"), ("rr", "#1f77b4", "RR")):
        vals = []
        for cell in ("burst_s36b", "burst_n4", "burst_s20251029", "burst_s20251030"):
            d = load(f"{cell}_{pol}")
            if d is not None:
                vals += list(burst(d).ttft_ms.values)
        v = np.sort(np.asarray(vals))
        ax.plot(v, np.arange(1, len(v)+1)/len(v), lw=1.6, color=color,
                label=f"{name} (n={len(v)}, p99={np.quantile(v,.99):.0f} ms)")
    ax.set_xscale("log"); ax.set_xlabel("TTFT of the returning wave (ms, log)")
    ax.set_ylabel("CDF"); ax.legend(fontsize=7)
    ax.set_title("Fig 2  Burst TTFT distribution (4 pool seeds pooled)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_ttft_cdf.png"); plt.close(fig)

    # ---------- fig3：burst 窗口里每条会话保住了多少前缀 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for ax, cell, title in zip(axes, ("burst_s36b", "burst_n4"), ("n_a=3 (seed1)", "n_a=4")):
        for pol, color, name in (("base", "#8c8c8c", "base"), ("rr", "#1f77b4", "RR")):
            d = load(f"{cell}_{pol}")
            b = burst(d)
            ratio = np.sort((b.cached_tokens / b.prompt_tokens).values)
            ax.plot(np.arange(1, len(ratio)+1), ratio, "-o", ms=4, color=color, label=name)
        ax.axhline(.5, ls="--", lw=.8, color="k")
        ax.set_ylim(0, 1.03); ax.set_xlabel("session (sorted)")
        ax.set_ylabel("retained prefix / prompt"); ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Fig 3  Per-session retained prefix at the return wave", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig3_retained.png"); plt.close(fig)

    # ---------- fig4：idle 组逐轮命中（pause 前后） ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, cell, title in zip(axes, ("burst_s36b", "burst_n4"), ("n_a=3 (seed1)", "n_a=4")):
        for pol, color, name in (("base", "#8c8c8c", "base"), ("rr", "#1f77b4", "RR")):
            d = load(f"{cell}_{pol}")
            g = d[d.task == "agentic_idle"].groupby("turn").apply(
                lambda x: x.cached_tokens.sum()/x.prompt_tokens.sum(), include_groups=False)
            ax.plot(g.index, g.values, "-o", ms=3.5, color=color, label=name)
        ax.axvspan(10, 11, color="orange", alpha=.15)
        ax.set_xlabel("turn of idle-group session"); ax.set_ylabel("cache hit of prefix")
        ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle("Fig 4  Idle group hit-rate by turn (shaded = turn after the 60 s pause)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig4_timeline.png"); plt.close(fig)

    # ---------- fig5：κ 扫描（0.1–0.8，含配额违约数） ----------
    KS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    arms = {0.3: "kappa_0.3", 0.5: "kappa_0.5", 0.7: "kappa_0.7"}
    tot, p99, br = [], [], []
    for k in KS:
        arm = f"kappa_{k}"
        s = stat(arm)
        tot.append(s["b_tot"] if s else np.nan)
        p99.append(s["p99"] if s else np.nan)
        f = Path("result/burst") / f"{arm}.breach"
        br.append(int(f.read_text().split("=")[-1]) if f.exists() else np.nan)
    b = stat("burst_s36b_base")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    a1.plot(KS, tot, "-o", color="#1f77b4", label="burst recompute (tokens)")
    a1.axhline(b["b_tot"], ls="--", lw=.9, color="#1f77b4", alpha=.6, label="base")
    a1.set_xlabel("kappa (quota = kappa * L)"); a1.set_ylabel("burst recompute", color="#1f77b4")
    ax2 = a1.twinx()
    ax2.plot(KS, p99, "-s", color="#d62728", label="TTFT p99 (ms)")
    ax2.axhline(b["p99"], ls=":", lw=.9, color="#d62728", alpha=.6)
    ax2.set_ylabel("TTFT p99 (ms)", color="#d62728")
    a1.set_title("Effect vs kappa (noisy, 1 run/point)")
    a2.plot(KS, br, "-^", color="#2ca02c")
    a2.set_xlabel("kappa"); a2.set_ylabel("quota breaches (per 30 s)", color="#2ca02c")
    a2.set_title("Quota feasibility: breaches fall as kappa grows")
    fig.suptitle("Fig 5  kappa sweep 0.1–0.8", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig5_kappa.png"); plt.close(fig)

    # ---------- fig6：甜点区（溢出比 vs 收益） ----------
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for cell, (label, press, rho) in CELLS.items():
        b, r = stat(f"{cell}_base"), stat(f"{cell}_rr")
        if not b or not r:
            continue
        ax.scatter(rho, (r["hit"] - b["hit"]) * 100, s=55,
                   color="#1f77b4" if press else "#8c8c8c", zorder=3)
        ax.annotate(label, (rho, (r["hit"] - b["hit"]) * 100), fontsize=7,
                    textcoords="offset points", xytext=(5, -9))
    ax.axhline(0, color="k", lw=.8)
    ax.axvspan(1.3, 2.0, color="green", alpha=.06)
    ax.set_xlabel("overflow ratio  $\\Sigma L / pool$")
    ax.set_ylabel("hit-rate gain of RR (pt)")
    ax.set_title("Fig 6  Sweet spot: gain appears only when the pool overflows")
    fig.tight_layout(); fig.savefig(FIG / "fig6_sweetspot.png"); plt.close(fig)

    # ---------- fig7：两两轮转空闲（每对的回归命中） ----------
    roll = {"base": "roll3_base", "RR k=0.5": "roll3_rr_k5", "RR k=0.3": "roll3_rr_k3"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    colors = {"base": "#8c8c8c", "RR k=0.5": "#1f77b4", "RR k=0.3": "#2ca02c"}
    tot_rc = {}
    for name, arm in roll.items():
        d = load(arm)
        if d is None:
            raise FileNotFoundError(f"fig7 缺少 {arm}.csv（需先拷到 RR/results/）")
        d = d.copy(); d["pair"] = d.task.str.extract(r"pair(\d)").astype(int)
        g = d[d.turn == 11].groupby("pair").apply(
            lambda x: 1 - x.recompute_tokens.sum()/x.prompt_tokens.sum(), include_groups=False)
        a1.plot(g.index, g.values, "-o", ms=4, color=colors[name], label=name)
        tot_rc[name] = int(d[d.turn == 11].recompute_tokens.sum())
    a1.set_xlabel("rotation round (pair index)"); a1.set_ylabel("retained prefix at return")
    a1.set_ylim(0, 1); a1.legend(fontsize=7)
    a1.set_title("Per-round outcome: RR raises the floor")
    a2.bar(list(tot_rc), list(tot_rc.values()), color=[colors[k] for k in tot_rc])
    a2.set_ylabel("total recompute at returns"); a2.set_title("Aggregate (144 requests, 6 rounds)")
    fig.suptitle("Fig 7  Rolling two-by-two idle (12 sessions, 6 sequential idle windows of 15 s)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig7_rolling.png"); plt.close(fig)

    # ---------- fig8：TTFT 分解 + TTFT vs 重算量 ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))
    cells = [("burst_s36b", "seed1"), ("burst_n4", "n_a=4"), ("burst_s20251029", "seed2")]
    xs = np.arange(len(cells)); w = .38
    for j, pol in enumerate(("base", "rr")):
        qq, pp, oo = [], [], []
        for cell, _ in cells:
            d = load(f"{cell}_{pol}")
            win = d[(d.task == "agentic_idle") & (d.turn == 11)].copy()
            win["other"] = win.ttft_ms - win.queue_time_ms - win.prefill_ms.fillna(0)
            qq.append(win.queue_time_ms.mean()); pp.append(win.prefill_ms.mean()); oo.append(win.other.mean())
        off = (j - .5) * w
        a1.bar(xs + off, qq, w, label="queue" if j == 0 else None, color="#d62728")
        a1.bar(xs + off, pp, w, bottom=qq, label="prefill" if j == 0 else None, color="#1f77b4")
        a1.bar(xs + off, oo, w, bottom=np.array(qq) + np.array(pp),
               label="tokenize+decode" if j == 0 else None, color="#bbbbbb")
        for i, (q_, p_, o_) in enumerate(zip(qq, pp, oo)):
            a1.text(xs[i] + off, q_ + p_ + o_, f"{q_+p_+o_:.0f}", ha="center", fontsize=7)
    a1.set_xticks(xs)
    a1.set_xticklabels([c[1] for c in cells], fontsize=7)
    a1.set_ylabel("mean TTFT (ms)"); a1.legend(fontsize=7)
    a1.set_title("Mean TTFT decomposition at the return wave")
    d = load("burst_s36b_base")
    m = d[(d.recompute_tokens > 100) & d.prefill_ms.notna() & (d.ttft_ms < 3500)]
    a2.scatter(m.recompute_tokens, m.ttft_ms, s=6, alpha=.5, color="#8c8c8c")
    xx = np.linspace(0, m.recompute_tokens.max(), 50)
    a2.plot(xx, xx / 6.2, lw=1.2, color="#1f77b4", label="6.2k tok/s prefill")
    a2.set_xlabel("recompute tokens"); a2.set_ylabel("TTFT (ms)")
    a2.set_title("TTFT scales with recompute (r=0.87)"); a2.legend(fontsize=7)
    fig.suptitle("Fig 8  Where the TTFT spike comes from", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig8_ttft.png"); plt.close(fig)

    # ---------- fig9：串行服务 Gantt（burst 窗口 8 条回归） ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 4.6), sharex="col")
    for col, (cell, cname) in enumerate((("burst_s36b", "seed1"), ("burst_n4", "n_a=4"))):
        for row, pol in enumerate(("base", "rr")):
            ax = axes[row][col]
            d = load(f"{cell}_{pol}")
            w = d[(d.task == "agentic_idle") & (d.turn == 11)].sort_values("ts").copy()
            t0 = w.ts.min()
            w["a"] = (w.ts - t0) * 1000
            for i, (_, r) in enumerate(w.iterrows()):
                ax.barh(i, r.queue_time_ms, left=r.a, height=.62, color="#d62728")
                ax.barh(i, r.prefill_ms, left=r.a + r.queue_time_ms, height=.62, color="#1f77b4")
                ax.text(r.a + r.ttft_ms + 80, i, f"{r.ttft_ms:.0f}", va="center", fontsize=6.5)
            ax.set_yticks(range(len(w)))
            ax.set_yticklabels([f"#{i+1}" for i in range(len(w))], fontsize=6.5)
            ax.invert_yaxis()
            tot = w.prefill_ms.sum()
            ax.set_title(f"{cname} / {pol}: sum prefill = {tot:.0f} ms", fontsize=8)
            if row == 1:
                ax.set_xlabel("wall time since first return (ms)")
    axes[0][0].set_ylabel("request order")
    axes[1][0].set_ylabel("request order")
    fig.suptitle("Fig 9  Serial service during the return wave (red = queue, blue = prefill)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig9_serial.png"); plt.close(fig)

    # ---------- fig10：台阶式空闲（pause 15/30/45/60 s） ----------
    arms = {"base": "stair_base", "RR k=0.5": "stair_rr_k5", "RR k=0.3": "stair_rr_k3"}
    col = {"base": "#8c8c8c", "RR k=0.5": "#1f77b4", "RR k=0.3": "#2ca02c"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    pauses = [15, 30, 45, 60]
    for name, arm in arms.items():
        d = load(arm)
        if d is None:
            raise FileNotFoundError(f"fig10 缺少 {arm}.csv")
        d = d.copy(); d["pair"] = d.task.str.extract(r"pair(\d)").astype(int)
        r = d[d.turn == 11]
        ret = r.groupby("pair").apply(
            lambda x: 1 - x.recompute_tokens.sum()/x.prompt_tokens.sum(), include_groups=False)
        a1.plot(pauses, [ret[i] for i in range(4)], "-o", ms=5, color=col[name], label=name)
        a2.bar([p + (list(arms).index(name) - 1) * 3 for p in pauses],
               [r[r.pair == i].recompute_tokens.sum() for i in range(4)], width=3,
               color=col[name], label=name)
    a1.set_xlabel("pause duration (s) →  colder"); a1.set_ylabel("retained prefix at return")
    a1.set_ylim(0, 1); a1.legend(fontsize=7)
    a1.set_title("Cascade gradient: last pair loses 81% under LRU")
    a2.set_xlabel("pause duration (s)"); a2.set_ylabel("recompute tokens at return")
    a2.legend(fontsize=7); a2.set_title("Cost of each return")
    fig.suptitle("Fig 10  Staircase idle (8 sessions, pause 15/30/45/60 s)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig10_stair.png"); plt.close(fig)

    # ---------- fig11：κ=0.3 与基线对比（台阶负载） ----------
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
    pauses = [15, 30, 45, 60]
    series = {"base": ("stair_base", "#8c8c8c"), "RR κ=0.3": ("stair_rr_k3", "#2ca02c")}
    x = np.arange(4); w = .38
    for j, (name, (arm, color)) in enumerate(series.items()):
        d = load(arm)
        if d is None:
            raise FileNotFoundError(f"fig11 缺少 {arm}.csv")
        d = d.copy(); d["pair"] = d.task.str.extract(r"pair(\d)").astype(int)
        r = d[d.turn == 11]
        ret = r.groupby("pair").apply(
            lambda y: 1 - y.recompute_tokens.sum()/y.prompt_tokens.sum(), include_groups=False)
        rc = r.groupby("pair").recompute_tokens.sum()
        axes[0].bar(x + (j - .5) * w, [ret[i] for i in range(4)], w, color=color, label=name)
        axes[1].bar(x + (j - .5) * w, [rc[i] for i in range(4)], w, color=color, label=name)
        win = r.copy()
        win["other"] = win.ttft_ms - win.queue_time_ms - win.prefill_ms.fillna(0)
        qq, pp = win.queue_time_ms.mean(), win.prefill_ms.mean()
        axes[2].bar(j, qq, .55, color="#d62728", label="queue" if j == 0 else None)
        axes[2].bar(j, pp, .55, bottom=qq, color=color, label="prefill" if j == 0 else None)
        axes[2].text(j, qq + pp + 8, f"{qq+pp:.0f} ms", ha="center", fontsize=7)
    for ax, ttl, yl in ((axes[0], "Retained prefix at return", "cached / prompt"),
                        (axes[1], "Recompute tokens at return", "tokens"),
                        (axes[2], "Mean TTFT at return (queue vs prefill)", "ms")):
        ax.set_title(ttl, fontsize=9); ax.set_ylabel(yl)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{p}s" for p in pauses])
    axes[0].set_xlabel("pause duration"); axes[0].legend(fontsize=7)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{p}s" for p in pauses])
    axes[1].set_xlabel("pause duration")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["base", "RR κ=0.3"])
    axes[2].legend(fontsize=7)
    fig.suptitle("Fig 11  RR κ=0.3 vs baseline on the staircase workload (queue ≈ 0.6 ms)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig11_k3_vs_base.png"); plt.close(fig)

    df.to_csv(RES / "summary_by_cell.csv", index=False)
    print(f"→ {FIG}  图 11 张；汇总 {RES/'summary_by_cell.csv'}")
    print(df[["cell", "pol", "hit", "b_tot", "p99"]].to_string(index=False))


if __name__ == "__main__":
    main()
