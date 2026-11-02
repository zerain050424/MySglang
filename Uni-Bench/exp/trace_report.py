#!/usr/bin/env python3
"""统计 + 绘图：agent 会话的轮间间隔特征与工具调用特征。

输入 exp/data/rounds.parquet（由 exp/trace_features.py 生成）
输出 exp/agent_gap_stats.json、exp/figs/*.png
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "exp/data/rounds.parquet"
FIG = "exp/figs"
TD = "exp/tool_durations.json"
OUT = "exp/agent_gap_stats.json"

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def q(v, p):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if len(v) else float("nan")


def desc(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return {}
    return {"n": int(len(v)), "p10": q(v,.1), "p50": q(v,.5), "p75": q(v,.75),
            "p90": q(v,.9), "p99": q(v,.99), "p999": q(v,.999),
            "gt10": float((v>10).mean()), "gt60": float((v>60).mean()),
            "gt300": float((v>300).mean())}


def main():
    df = pd.read_parquet(DATA)
    g = df[df.gap.notna()].copy()
    S = {}

    # ---------- 1. 触发类型与间隔分桶 ----------
    mix = {}
    for src, d in g.groupby("source"):
        mix[src] = {"rounds": int(len(d)),
                    "sessions": int(d.session_id.nunique()),
                    "trigger": {k: round(v/len(d), 4) for k, v in
                                Counter(d.trigger).items()}}
    S["trigger_mix"] = mix

    buckets = {}
    for (src, trig), d in g.groupby(["source", "trigger"]):
        buckets[f"{src}/{trig}"] = desc(d.gap)
    S["gap_by_bucket"] = buckets

    # eps = gap − 上一轮工具耗时：模型答完到下次请求的真实空档
    eps_b = {}
    for (src, trig), d in g.groupby(["source", "trigger"]):
        eps_b[f"{src}/{trig}"] = desc(d.eps)
    S["eps_by_bucket"] = eps_b
    S["tool_share_of_gap"] = {
        src: {"n": int(len(d)),
              "median_tool_over_gap": float(np.median(d.tool_sum / d.gap))}
        for src, d in g[(g.n_tools > 0) & (g.gap > 0.01)].groupby("source")}

    # ---------- 2. 工具调用 ----------
    t = df[df.n_tools > 0]
    tool_cnt = Counter()
    for s, src in zip(t.tool_names, t.source):
        for nm in s.split("|"):
            if nm and nm != "?":
                tool_cnt[(src, nm)] += 1
    S["tool_top"] = {src: Counter({n: c for (s, n), c in tool_cnt.items() if s == src}).most_common(20)
                     for src in t.source.unique()}
    S["tools_per_round"] = {src: {str(k): int(v) for k, v in
                                  d.n_tools.clip(upper=10).value_counts().sort_index().items()}
                            for src, d in t.groupby("source")}

    # gap 与工具耗时的关系（仅两边都有值的轮）
    tt = t[(t.gap.notna()) & (t.tool_sum > 0)]
    corr = {}
    for src, d in tt.groupby("source"):
        if len(d) > 200:
            corr[src] = {"n": int(len(d)),
                         "pearson_log": float(np.corrcoef(np.log(d.gap), np.log(d.tool_sum))[0, 1]),
                         "share_tool_of_gap_med": float(np.median(d.tool_sum / d.gap))}
    S["gap_vs_toolsum"] = corr

    # ---------- 3. 会话内自相关 ----------
    acf = {}
    for src, d in g.groupby("source"):
        seqs = [np.log(x.gap.values) for _, x in
                d.sort_values("round_index").groupby("session_id") if len(x) >= 4]
        lags = range(1, 13)
        vals = []
        for k in lags:
            num = den = 0.0
            for a in seqs:
                if len(a) <= k:
                    continue
                m = a.mean()
                num += float(((a[:-k]-m) * (a[k:]-m)).sum()); den += float(((a-m)**2).sum())
            vals.append(num/den if den else float("nan"))
        acf[src] = {"n_sessions": len(seqs), "lag": list(lags), "acf": vals}
    S["acf"] = acf

    # ---------- 4. 会话间异质性 ----------
    per = (g.groupby(["source", "session_id"]).gap
             .agg(["count", "median", "mean", "std"]).reset_index())
    per = per[per["count"] >= 5]
    per["cv"] = per["std"] / per["median"].clip(lower=.05)
    S["per_session"] = {src: {"n": int(len(d)),
                              "med_median": desc(d["median"]), "cv": desc(d["cv"])}
                        for src, d in per.groupby("source")}

    # ---------- 5. 间隔随轮位置 ----------
    g["pos_bin"] = pd.cut(g["pos"], np.linspace(0, 1, 11), include_lowest=True,
                          labels=[f"{i*10}-{(i+1)*10}" for i in range(10)])
    pos_tab = {}
    for src, d in g.groupby("source"):
        pos_tab[src] = {str(k): {"n": int(len(x)), "p50": q(x.gap,.5), "p90": q(x.gap,.9)}
                        for k, x in d.groupby("pos_bin", observed=True)}
    S["gap_by_pos"] = pos_tab

    # ---------- 6. 会话时长与并发 ----------
    span = g.groupby(["source", "session_id"]).agg(t0=("t_start","min"), t1=("t_end","max"),
                                                   idle=("gap","sum"), nround=("gap","size")).reset_index()
    span["span"] = span.t1 - span.t0
    S["session_shape"] = {}
    for src, d in span.groupby("source"):
        S["session_shape"][src] = {
            "n": int(len(d)), "span": desc(d.span), "rounds": desc(d.nround),
            "idle_frac_med": float(np.median((d.idle/d.span.clip(lower=1)).clip(0,1)))}
    conc = {}
    for src, d in span.groupby("source"):
        conc[src] = d[["t0", "t1"]].values
    S["concurrency_raw"] = {k: v.tolist()[:0] for k, v in conc.items()}  # 占位，图内计算

    # ---------- 7. token 结构 ----------
    tok = {}
    for src, d in df[df.in_tok > 0].groupby("source"):
        tok[src] = {"n": int(len(d)),
                    "in_tok": desc(d.in_tok),
                    "prefix_ratio": desc((d.prefix_tok / d.in_tok).clip(0, 1)),
                    "new_tok": desc(d.new_tok)}
    S["tokens"] = tok

    open(OUT, "w").write(json.dumps(S, indent=1, ensure_ascii=False))
    print("统计 →", OUT)

    # =========================== 绘图 ===========================
    colors = {"claude": "#1f77b4", "codex": "#d62728", "cachewise": "#2ca02c"}
    srcs = [s for s in ["claude", "codex", "cachewise"] if s in set(g.source)]

    # fig01：间隔分布（按源 + 触发类型）
    fig, axes = plt.subplots(1, len(srcs), figsize=(4.2*len(srcs), 3.2), squeeze=False)
    for ax, src in zip(axes[0], srcs):
        d = g[g.source == src]
        bins = np.logspace(np.log10(0.05), np.log10(3600), 70)
        for trig, c in [("tool_result", "#1f77b4"), ("user_message", "#ff7f0e")]:
            x = d[d.trigger == trig].gap
            if len(x) > 30:
                ax.hist(x, bins=bins, alpha=.55, label=f"{trig} (n={len(x)})",
                        color=c, density=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("inter-turn gap (s)"); ax.set_title(f"{src}")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("density")
    fig.suptitle("Fig 1  Inter-turn gap distribution by trigger type", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig01_gap_dist.png"); plt.close(fig)

    # fig02：CDF
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for src in srcs:
        for trig, ls in [("tool_result", "-"), ("user_message", "--")]:
            x = np.sort(g[(g.source == src) & (g.trigger == trig)].gap.values)
            if len(x) < 50:
                continue
            ax.plot(x, np.arange(1, len(x)+1)/len(x), ls, color=colors.get(src, "k"),
                    lw=1.4, label=f"{src}/{trig}")
    ax.set_xscale("log"); ax.set_xlabel("inter-turn gap (s)"); ax.set_ylabel("CDF")
    ax.legend(fontsize=7); ax.set_title("Fig 2  Gap CDF by source and trigger")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig02_gap_cdf.png"); plt.close(fig)

    # fig03：触发类型占比 + 长间隔占比
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.0))
    srcs2 = list(mix)
    tr = ["tool_result", "user_message", "?"]
    bottom = np.zeros(len(srcs2))
    for k, c in zip(tr, ["#1f77b4", "#ff7f0e", "#999999"]):
        v = np.array([mix[s]["trigger"].get(k, 0) for s in srcs2])
        a1.bar(srcs2, v, bottom=bottom, label=k, color=c); bottom += v
    a1.set_ylabel("share of rounds"); a1.set_ylim(0, 1); a1.legend(fontsize=7)
    a1.set_title("Trigger mix")
    w = 0.35
    for i, src in enumerate(srcs2):
        d = g[g.source == src]
        a2.bar(i-w/2, (d.gap > 60).mean(), w, label=">60s" if i == 0 else None, color="#ff7f0e")
        a2.bar(i+w/2, (d.gap > 300).mean(), w, label=">300s" if i == 0 else None, color="#d62728")
    a2.set_xticks(range(len(srcs2))); a2.set_xticklabels(srcs2)
    a2.set_ylabel("share of gaps"); a2.legend(fontsize=7); a2.set_title("Long-idle share")
    fig.suptitle("Fig 3  Trigger composition and tail share", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig03_trigger_tail.png"); plt.close(fig)

    # fig04：工具出现频次 top20
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    top = Counter()
    for s in t.tool_names:
        for nm in s.split("|"):
            if nm and nm != "?": top[nm] += 1
    items = top.most_common(20)[::-1]
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4c72b0")
    ax.set_xscale("log"); ax.set_xlabel("tool calls (log)")
    ax.set_title("Fig 4  Tool call frequency (tracelab + cachewise)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig04_tool_freq.png"); plt.close(fig)

    # fig12b：eps 与人工间隔（时间预算分解）
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for src in srcs:
        x = np.sort(g[(g.source == src) & (g.trigger == "tool_result")].eps.values)
        if len(x) > 100:
            axes[0].plot(np.maximum(x, 1e-3), np.arange(1, len(x)+1)/len(x),
                         color=colors.get(src), lw=1.4, label=src)
    axes[0].set_xscale("log"); axes[0].set_xlabel("eps = gap − prior tool time (s)")
    axes[0].set_ylabel("CDF"); axes[0].legend(fontsize=7)
    axes[0].set_title("Self-driven rounds: residual inactivity")
    for src in srcs:
        x = np.sort(g[(g.source == src) & (g.trigger == "user_message")].gap.values)
        if len(x) > 100:
            axes[1].plot(x, np.arange(1, len(x)+1)/len(x), color=colors.get(src), lw=1.4, label=src)
    axes[1].set_xscale("log"); axes[1].set_xlabel("gap after user turn (s)")
    axes[1].set_ylabel("CDF"); axes[1].legend(fontsize=7)
    axes[1].set_title("Human turns: waiting for the user")
    fig.suptitle("Fig 13  Time budget: machine loop vs human loop", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig13_eps_human.png"); plt.close(fig)

    # fig05：工具耗时生存曲线
    td = json.load(open(TD))["tools"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for nm in ["Bash", "exec_command", "Read", "write_stdin", "Edit", "WebFetch", "Agent"]:
        if nm not in td:
            continue
        p, qq = np.array(td[nm]["p"])/100.0, np.array(td[nm]["q"])
        ax.plot(qq, 1-p, lw=1.4, label=f"{nm} (p50={td[nm]['p50']:.2f}s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tool duration (s)"); ax.set_ylabel("P(duration > x)")
    ax.legend(fontsize=7); ax.set_title("Fig 5  Tool duration survival (552,899 calls)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig05_tool_survival.png"); plt.close(fig)

    # fig06：每轮工具数
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for src in srcs:
        d = t[t.source == src]
        if not len(d):
            continue
        vc = d.n_tools.clip(upper=8).value_counts().sort_index()
        ax.bar(vc.index + (0.4 if src == "cachewise" else 0), vc.values/vc.sum(),
               width=.38, label=f"{src} (n={len(d)})", color=colors.get(src))
    ax.set_xlabel("tool calls per round (clipped at 8)"); ax.set_ylabel("share")
    ax.legend(fontsize=7); ax.set_title("Fig 6  Fan-out per round")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig06_tools_per_round.png"); plt.close(fig)

    # fig07：gap vs 工具总耗时
    sub = t[(t.gap.notna()) & (t.tool_sum > 0.01) & (t.gap > 0.01)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, src in zip(axes, [s for s in srcs if s in set(sub.source)][:2]):
        d = sub[sub.source == src]
        hb = ax.hexbin(np.log10(d.tool_sum), np.log10(d.gap), gridsize=45,
                       bins="log", cmap="viridis", mincnt=1)
        ax.set_xlabel("log10 tool sum (s)"); ax.set_ylabel("log10 gap (s)")
        c = np.corrcoef(np.log(d.gap), np.log(d.tool_sum))[0, 1]
        ax.set_title(f"{src}  r={c:.3f}  n={len(d)}", fontsize=9)
        fig.colorbar(hb, ax=ax, label="count (log)")
    fig.suptitle("Fig 7  Gap vs tool time", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig07_gap_vs_toolsum.png"); plt.close(fig)

    # fig08：间隔随轮位置
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for src in srcs:
        d = g[g.source == src]
        xs, ys, lo, hi = [], [], [], []
        for k, x in d.groupby("pos_bin", observed=True):
            if len(x) < 100:
                continue
            xs.append(float(str(k).split("-")[0]) + 5)
            ys.append(q(x.gap, .5)); lo.append(q(x.gap, .25)); hi.append(q(x.gap, .75))
        ax.plot(xs, ys, "-o", ms=3.5, color=colors.get(src), label=src)
        ax.fill_between(xs, lo, hi, color=colors.get(src), alpha=.15)
    ax.set_yscale("log"); ax.set_xlabel("round position in session (%)")
    ax.set_ylabel("gap (s), median + IQR"); ax.legend(fontsize=7)
    ax.set_title("Fig 8  Gap vs position in session")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig08_gap_by_pos.png"); plt.close(fig)

    # fig09：自相关
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for src, v in acf.items():
        ax.plot(v["lag"], v["acf"], "-o", ms=3.5, color=colors.get(src), label=f"{src} ({v['n_sessions']} sess)")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("lag (rounds)"); ax.set_ylabel("ACF of log gap")
    ax.legend(fontsize=7); ax.set_title("Fig 9  Within-session autocorrelation")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig09_acf.png"); plt.close(fig)

    # fig10：会话间异质性
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for src, d in per.groupby("source"):
        v = np.sort(d["median"].values)
        a1.plot(np.arange(len(v))/len(v), v, lw=1.4, color=colors.get(src), label=src)
        a2.hist(np.log10(d.cv.clip(lower=1e-2)), bins=40, alpha=.5,
                color=colors.get(src), label=f"{src} (med {np.median(d.cv):.2f})", density=True)
    a1.set_yscale("log"); a1.set_xlabel("session quantile"); a1.set_ylabel("session median gap (s)")
    a1.legend(fontsize=7); a1.set_title("Per-session median gap")
    a2.set_xlabel("log10 CV within session"); a2.set_ylabel("density")
    a2.legend(fontsize=7); a2.set_title("Within-session dispersion")
    fig.suptitle("Fig 10  Between- vs within-session heterogeneity", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig10_session_hetero.png"); plt.close(fig)

    # fig11：并发会话数
    fig, axes = plt.subplots(len(srcs), 1, figsize=(8.0, 2.2*len(srcs)), squeeze=False)
    for ax, src in zip(axes[:, 0], srcs):
        d = span[span.source == src]
        if not len(d):
            continue
        t0, t1 = d.t0.min(), d.t1.max()
        step = max(300.0, (t1-t0)/400)
        edges = np.arange(t0, t1+step, step)
        cnt = np.zeros(len(edges))
        for a, b in zip(d.t0.values, d.t1.values):
            i0 = int((a-t0)//step); i1 = min(int((b-t0)//step), len(edges)-1)
            cnt[i0] += 1
            if i1+1 < len(edges):
                cnt[i1+1] -= 1
        alive = np.cumsum(cnt)
        ax.plot(pd.to_datetime(edges, unit="s"), alive, lw=1.0, color=colors.get(src))
        ax.set_ylabel("live sessions"); ax.set_title(f"{src}: peak={int(alive.max())}, "
                                                     f"p99={int(np.quantile(alive,.99))}")
    axes[-1][0].set_xlabel("time")
    fig.suptitle("Fig 11  Concurrent sessions over time (session span = first..last round)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig11_concurrency.png"); plt.close(fig)

    # fig12：token 结构
    dd = df[(df.in_tok > 0) & (df.source == "claude")]
    if len(dd):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
        a1.hist(np.log10(dd.in_tok.clip(lower=1)), bins=60, color="#1f77b4")
        a1.set_xlabel("log10 input_tokens_total"); a1.set_ylabel("rounds")
        a1.axvline(np.log10(q(dd.in_tok,.5)), color="r", lw=1, ls="--",
                   label=f"p50={q(dd.in_tok,.5):.0f}")
        a1.legend(fontsize=7); a1.set_title("Context size per round")
        r = (dd.prefix_tok/dd.in_tok).clip(0,1)
        a2.hist(r, bins=50, color="#2ca02c")
        a2.set_xlabel("prefix_tokens / input_tokens"); a2.set_ylabel("rounds")
        a2.set_title(f"Prefix reuse (median {np.median(r):.3f})")
        fig.suptitle("Fig 12  Token profile (claude, tracelab)", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/fig12_tokens.png"); plt.close(fig)

    print("图 →", FIG)


if __name__ == "__main__":
    main()
