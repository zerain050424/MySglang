"""轮间间隔模型：把「会话离场时长」逐轮写入 session_data 的 gap_s。

四个模型（对应 exp/LargeScale.md §5）：

  ① fixed3 : 每条会话随机分配一个长 idle（10/30/90 s），落在随机的某一轮，
             其余轮 1.0 s。R3 的"3 组固定"在逐会话随机的版本。
  ② tool   : gap = Σ 工具耗时（按实测频次抽工具名 + 逆 CDF 抽时长）+ ε（经验分位）。
  ③ replay : gap 直接从实测「离场时长」池抽（away = tool_sum + gap，见下）。
  ④ claude : gap = Σ 工具耗时 + 0（claude 实测 ε=0），人工轮按 claude 人工分布。

人工轮：每条轮以 π_h 概率被视为"等用户"（默认 0.137 = claude 实测）。
  人工轮不执行工具，gap 从对应 harness 的人工等待分位表抽。

口径说明（重要）
  真实 trace 里，模型请求到下一次请求的时间 = 生成耗时 + 工具执行 + 调度空档。
  我们的 bench 生成是真实发生的，所以只需注入"离场时长"：
    ② ④ = 工具耗时 + ε        （工具在 bench 里不执行，用采样替代）
    ③   = tool_sum + gap       （parquet 中两列相加，即离场时长）
"""
from __future__ import annotations

import gzip
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 与 build_tool_durations.py 相同的非均匀分位网格（尾部加密到 99.99）
LEVELS = (
    [float(i) for i in range(100)]
    + [round(99.0 + 0.1 * i, 2) for i in range(1, 10)]
    + [round(99.9 + 0.01 * i, 2) for i in range(1, 10)]
    + [99.995, 100.0]
)


def quantile_table(v) -> dict:
    """样本 → {p, q} 分位表（线性插值）。"""
    v = np.sort(np.asarray(v, dtype=float))
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"p": [0.0, 100.0], "q": [0.0, 0.0], "n": 0}

    def at(p):
        x = (p / 100.0) * (n - 1)
        i = int(x)
        return float(v[i] if i >= n - 1 else v[i] + (v[i + 1] - v[i]) * (x - i))

    return {"p": LEVELS, "q": [at(p) for p in LEVELS], "n": n}


def sample_q(rng: random.Random, tab: dict) -> float:
    """按 (p, q) 分位表做逆 CDF 线性插值采样。"""
    p, q = tab["p"], tab["q"]
    u = rng.random() * 100.0
    i = int(np.searchsorted(p, u, side="right")) - 1
    if i < 0:
        return float(q[0])
    if i >= len(p) - 1:
        return float(q[-1])
    w = (u - p[i]) / (p[i + 1] - p[i])
    return float(q[i] + (q[i + 1] - q[i]) * w)


# --------------------------------------------------------------------------
# 表：工具时长 + ε + 人工等待
# --------------------------------------------------------------------------
@dataclass
class Tables:
    """逐工具分位表、ε 桶、以及按 harness 的人工等待分位表。"""

    tools: dict                       # name -> {p,q,n,class}
    eps: dict                         # 有工具轮的残余分位表
    human: dict                       # harness -> 人工轮 gap 分位表
    replay: dict                      # harness -> 离场时长(tool_sum+gap) 分位表
    freq: list                        # [(tool_name, weight)]，仅取有耗时的工具
    fanout: dict                      # {1: 0.895, 2: 0.071, ...} 每轮工具数占比

    @classmethod
    def load(
        cls,
        tool_json: str = "exp/tool_durations.json",
        rounds_parquet: str = "exp/data/rounds.parquet",
        fanout_source: str = "claude",
    ) -> "Tables":
        doc = json.load(open(tool_json, encoding="utf-8"))
        tools = doc["tools"]
        freq = [(n, d["n"]) for n, d in tools.items()
                if d.get("class") in ("fast", "medium", "slow", "interactive") and d["n"] >= 20]
        freq.sort(key=lambda x: -x[1])

        human, replay = {}, {}
        df = pd.read_parquet(rounds_parquet, columns=["source", "trigger", "gap", "tool_sum", "eps"])
        for src in df.source.unique():
            d = df[df.source == src]
            human[src] = quantile_table(d[d.trigger == "user_message"].gap.dropna())
            dd = d[d.trigger == "tool_result"]
            away = (dd.tool_sum.fillna(0) + dd.gap.fillna(0))
            away = away[(away > 0) & (away < 3600)]
            replay[src] = quantile_table(away)

        fan = df[df.source == fanout_source].copy()
        fan = fan[fan.n_tools.fillna(0) >= 0] if "n_tools" in fan else fan
        return cls(tools=tools, eps=doc["buckets"]["eps"], human=human,
                   replay=replay, freq=freq, fanout={1: .895, 2: .071, 3: .022, 4: .008, 5: .004})

    def tool_dur(self, rng: random.Random, name: str) -> float:
        d = self.tools.get(name)
        return sample_q(rng, d) if d else 1.0

    def draw_tools(self, rng: random.Random) -> list[str]:
        """按实测频次抽本轮工具（1~5 个，按 fan-out 分布）。"""
        u, acc = rng.random(), 0.0
        k = 1
        for kk, w in self.fanout.items():
            acc += w
            if u <= acc:
                k = kk
                break
        total = sum(w for _, w in self.freq)
        out = []
        for _ in range(k):
            x, c = rng.random() * total, 0.0
            for nm, w in self.freq:
                c += w
                if x <= c:
                    out.append(nm)
                    break
        return out


# --------------------------------------------------------------------------
# 间隔模型
# --------------------------------------------------------------------------
@dataclass
class IdleModel:
    """生成一条会话逐轮的 gap_s。"""

    kind: str                 # fixed3 | tool | replay | claude
    pi_human: float = 0.137   # 人工轮概率
    harness: str = "claude"   # 人工/重放取哪个 harness 的表
    eps: bool = True          # tool 模型是否加 ε
    pauses: tuple = (10.0, 30.0, 90.0)
    tables: Tables | None = field(default=None, repr=False)

    def gaps(self, n_turns: int, rng: random.Random) -> list[float]:
        if self.kind == "fixed3":
            base = 1.0
            pause_at = rng.randrange(1, max(2, n_turns))
            long_gap = rng.choice(self.pauses)
            return [long_gap if i == pause_at else base for i in range(n_turns)]

        out = []
        for _ in range(n_turns):
            if rng.random() < self.pi_human:
                out.append(sample_q(rng, self.tables.human[self.harness]))
                continue
            if self.kind == "replay":
                out.append(sample_q(rng, self.tables.replay[self.harness]))
                continue
            names = self.tables.draw_tools(rng)
            g = sum(self.tables.tool_dur(rng, n) for n in names)
            if self.eps:
                g += sample_q(rng, self.tables.eps)
            out.append(g)
        return out


def main() -> None:
    """自检：打印四种模型生成的间隔分位点。"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=6000)
    ap.add_argument("--pi-human", type=float, default=0.137)
    args = ap.parse_args()

    T = Tables.load()
    rng = random.Random(0)
    n_sess = max(1, args.turns // 35)          # 用真实会话长度（alfred 轮数中位 35）
    print(f"{'模型':10s}{'p10':>8s}{'p50':>9s}{'p90':>9s}{'p99':>10s}{'>300s':>8s}")
    for kind in ("fixed3", "tool", "replay", "claude"):
        m = IdleModel(kind=kind, pi_human=args.pi_human, eps=(kind != "claude"), tables=T)
        g = np.array([x for _ in range(n_sess)
                      for x in m.gaps(rng.randint(6, 161), rng)])
        q = lambda p: float(np.quantile(g, p))
        print(f"{kind:10s}{q(.1):>8.2f}{q(.5):>9.2f}{q(.9):>9.1f}{q(.99):>10.1f}"
              f"{(g > 300).mean()*100:>7.2f}%")


if __name__ == "__main__":
    main()
