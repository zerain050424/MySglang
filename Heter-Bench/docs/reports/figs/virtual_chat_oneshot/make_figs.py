"""虚拟实验数据 + 配图：多轮 chat + one-shot 扫描流，LRU vs MultiTurn。

⚠️ **本文件里的数字全部是构造值（fabricated），不是任何一次实测的结果。**
用途只有两个：
1. 汇报前占位，把「要看什么、怎么比」先钉在图上；
2. 真机跑完后，把本文件的 ``DATA`` 换成 ``analyze.py`` 的真实输出再重跑，
   图与报告表格的格式不用动。

数字的量级锚定在本仓库既有实测（``docs/reports/lru_vs_hetercache.md``：
受害会话重算率 8.5% → 98.1%，HeterCache 稳定在 8.6%，TTFT p99 62 → 297 ms）。
也就是说：**趋势取自实测，具体数值是外推**。

用法：
    ~/miniconda3/envs/rain-llm/bin/python make_figs.py
产物：
    data.csv                 与报告表格同源的扁平数据
    fig1_hit_vs_scan.png     命中率 vs 扫描强度
    fig2_ttft_vs_scan.png    chat 轮 TTFT（p50 / p95）
    fig3_occupancy.png       池内驻留组成（chat vs one-shot）
    fig4_evict_mix.png       淘汰来源构成 + 每轮重算 token
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 0. 构造数据（唯一数据源）
# ---------------------------------------------------------------------------

SCAN_RATE = [0, 150, 400, 800, 1200, 1600]  # one-shot 扫描注入量（token/s）
CHAT_INJECT = 864.0                          # 多轮 chat 注入量（token/s，见报告 §2）
ARMS = ["lru", "multi_turn"]
ARM_LABEL = {
    "lru": "LRU (baseline)",
    "multi_turn": "MultiTurn",
}

# 多轮 chat 的 token 命中率 %：(mean, std over 3 seeds)
CHAT_HIT = {
    "lru": [(92.1, 0.6), (88.7, 0.9), (79.3, 1.3), (58.6, 1.8), (34.2, 2.1), (21.5, 1.9)],
    "multi_turn": [(93.0, 0.7), (92.2, 0.8), (91.1, 0.9), (89.6, 1.0), (88.2, 1.1), (87.0, 1.2)],
}

# chat 轮 TTFT（ms），p50 / p95
CHAT_TTFT_P50 = {
    "lru": [48, 62, 112, 232, 356, 430],
    "multi_turn": [46, 53, 66, 82, 94, 104],
}
CHAT_TTFT_P95 = {
    "lru": [104, 138, 262, 512, 690, 812],
    "multi_turn": [100, 112, 138, 166, 182, 196],
}

# 每轮重算 token（chat，均值，prompt 平均 3.2k）
RECOMPUTE = {
    "lru": [253, 362, 662, 1325, 2105, 2512],
    "multi_turn": [224, 250, 285, 333, 378, 416],
}

# one-shot 扫描请求的 TTFT p95（ms）：两者都冷 prefill，差距来自队列排队
SCAN_TTFT_P95 = {
    "lru": [1438, 1446, 1458, 1476, 1492, 1512],
    "multi_turn": [1440, 1450, 1468, 1492, 1524, 1550],
}

# 淘汰来源构成（占被淘汰 token 的 %），F=1200 与 F=0 两档
VICTIM_SRC = {
    0: {"one_shot": 0.8, "quota:chat": 97.5, "quota:fallback": 1.7},
    1200: {"one_shot": 61.2, "quota:chat": 37.4, "quota:fallback": 1.4},
}

# 池内驻留组成（token），F=1200，t = 0..600 s 每 10 s 一点
T = list(range(0, 601, 10))

def _chat_resident_lru(t):
    # 被扫描流冲刷：从 ~21k 指数塌到 ~3.5k，叠一点锯齿
    base = 3500 + 17500 * math.exp(-t / 140.0)
    return base + 900 * math.sin(t / 23.0) * math.exp(-t / 300.0)

def _chat_resident_mt(t):
    # 配额内稳在 ~24k，缓慢爬升到目标附近
    return 24200 + 900 * math.sin(t / 31.0) - 600 * math.exp(-t / 90.0)

OCCUPANCY = {
    "lru": {
        "chat": [max(0.0, _chat_resident_lru(t)) for t in T],
    },
    "multi_turn": {
        "chat": [max(0.0, _chat_resident_mt(t)) for t in T],
    },
}
POOL = 36000.0
for _arm, _d in OCCUPANCY.items():
    _d["one_shot"] = [max(0.0, POOL - c) for c in _d["chat"]]


# ---------------------------------------------------------------------------
# 1. 派生指标
# ---------------------------------------------------------------------------

def scan_share(rate: float) -> float:
    return rate / (rate + CHAT_INJECT)


def overall_hit(arm: str) -> list[float]:
    return [h * (1.0 - scan_share(f)) for (h, _), f in zip(CHAT_HIT[arm], SCAN_RATE)]


def write_csv() -> Path:
    path = HERE / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "scan_token_rate", "scan_token_share",
            "arm", "chat_hit_pct", "chat_hit_std", "overall_hit_pct",
            "chat_ttft_p50_ms", "chat_ttft_p95_ms", "scan_ttft_p95_ms",
            "recompute_tokens_per_turn",
        ])
        for arm in ARMS:
            for i, rate in enumerate(SCAN_RATE):
                hit, std = CHAT_HIT[arm][i]
                w.writerow([
                    rate, f"{scan_share(rate):.4f}", arm,
                    f"{hit:.1f}", f"{std:.1f}", f"{overall_hit(arm)[i]:.1f}",
                    CHAT_TTFT_P50[arm][i], CHAT_TTFT_P95[arm][i],
                    SCAN_TTFT_P95[arm][i],
                    RECOMPUTE[arm][i],
                ])
    return path


# ---------------------------------------------------------------------------
# 2. 图
# ---------------------------------------------------------------------------

def fig1(plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, key, ylabel in (
        (axes[0], "chat", "chat prefix hit rate (%)"),
        (axes[1], "overall", "overall token hit rate (%)"),
    ):
        for arm in ARMS:
            if key == "chat":
                ys = [h for h, _ in CHAT_HIT[arm]]
                err = [s for _, s in CHAT_HIT[arm]]
            else:
                ys = overall_hit(arm)
                err = None
            ax.errorbar(SCAN_RATE, ys, yerr=err, marker="o", ms=4, lw=1.6,
                        capsize=3, label=ARM_LABEL[arm])
        ax.set_xlabel("one-shot scan injection (token/s)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("multi-turn chat prefix reuse", fontsize=10)
    axes[1].set_title("all traffic (scan-token weighted)", fontsize=10)
    fig.tight_layout()
    out = HERE / "fig1_hit_vs_scan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2(plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, data, title in (
        (axes[0], CHAT_TTFT_P50, "chat turn TTFT p50"),
        (axes[1], CHAT_TTFT_P95, "chat turn TTFT p95"),
    ):
        for arm in ARMS:
            ax.plot(SCAN_RATE, data[arm], marker="o", ms=4, lw=1.8,
                    label=ARM_LABEL[arm])
        ax.set_xlabel("one-shot scan injection (token/s)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "fig2_ttft_vs_scan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3(plt):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.6), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        chat = OCCUPANCY[arm]["chat"]
        scan = OCCUPANCY[arm]["one_shot"]
        ax.stackplot(T, [chat, scan],
                     labels=["multi-turn chat (protected)", "one-shot scan (cold)"],
                     colors=["#3b7dd8", "#d0d0d0"], alpha=0.9)
        ax.set_ylabel("resident tokens")
        ax.set_title(f"{ARM_LABEL[arm]} — pool occupancy, scan load 1,200 token/s",
                     fontsize=9)
        ax.set_xlim(0, T[-1])
        ax.set_ylim(0, POOL)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    out = HERE / "fig3_occupancy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig4(plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    labels = ["scan 0 t/s", "scan 1,200 t/s"]
    keys = ["one_shot", "quota:chat", "quota:fallback"]
    colors = ["#d0d0d0", "#3b7dd8", "#e08a3c"]
    bottom = [0.0, 0.0]
    for k, c in zip(keys, colors):
        vals = [VICTIM_SRC[f][k] for f in (0, 1200)]
        ax.bar(labels, vals, bottom=bottom, label=k, color=c, width=0.55)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("share of evicted tokens (%)")
    ax.set_ylim(0, 100)
    ax.set_title("eviction source mix (MultiTurn)", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for arm in ARMS:
        ax.plot(SCAN_RATE, RECOMPUTE[arm], marker="s", ms=4, lw=1.8,
                label=ARM_LABEL[arm])
    ax.set_xlabel("one-shot scan injection (token/s)")
    ax.set_ylabel("recomputed tokens per chat turn")
    ax.set_title("prefix lost per chat turn", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "fig4_evict_mix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = write_csv()
    made = [fig1(plt), fig2(plt), fig3(plt), fig4(plt)]
    print(f"[virtual] csv  -> {csv_path}")
    for p in made:
        print(f"[virtual] fig  -> {p}")


if __name__ == "__main__":
    main()
