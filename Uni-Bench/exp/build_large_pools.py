#!/usr/bin/env python3
"""构造大规模实验的会话池：按目标占用比 ρ 抽样，并逐轮写入 gap_s。

与 Cascade 的区别
  Cascade 用「burst(固定 1 s) + BurstGate 屏障 + 统一 pause」，所有会话同节奏；
  本脚本给每条会话**独立采样**逐轮 gap，出错峰、无屏障，冷会话由分布自然产生。

用法
  /root/sl_old/bin/python exp/build_large_pools.py \
      --source alfred --pool 73120 --rho 1.2 --model claude --seed 1

产出
  TaskData/large/<name>.jsonl          已写入 turn_N.gap_s 的会话数据
  TaskData/large/<name>.manifest.json  抽样参数、链长分布、间隔分布、KS 检验
"""
from __future__ import annotations

import argparse
import gzip
import os
import json
import random
from pathlib import Path

import numpy as np

from idle_model import IdleModel, Tables, quantile_table

ALFRED = Path("TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl")
APIGEN = Path("/root/data_processed/apigen-mt/qwen2.5-7b/APIGenMT_session_data.jsonl.gz")
OUT = Path("TaskData/large")


# --------------------------------------------------------------------------
# 索引：只读链长/轮数，不保留正文
# --------------------------------------------------------------------------
ALFRED_INDEX = Path("exp/data/alfred_index.json")


def index_alfred(path: Path, cache: bool = True) -> list[dict]:
    """逐行读 alfred（3 GB），记下每会话的字节偏移，供第二阶段 seek。

    索引缓存在 exp/data/alfred_index.json：30 MB 的 JSON 比重新解析 3 GB 快 20×。
    cache: True 时优先读缓存。
    """
    if cache and ALFRED_INDEX.exists():
        return json.loads(ALFRED_INDEX.read_text())
    idx, off = [], 0
    with open(path, "rb") as f:
        for line in f:
            r = json.loads(line)
            sid, s = next(iter(r.items()))
            n = s["num_turns"]
            last = s[f"turn_{n}"]
            idx.append(dict(sid=sid, n=n, L=len(last["input_ids"]) + len(last["response_ids"]),
                            off=off, size=len(line)))
            off += len(line)
    if cache:
        ALFRED_INDEX.parent.mkdir(parents=True, exist_ok=True)
        ALFRED_INDEX.write_text(json.dumps(idx))
    return idx


def index_apigen(path: Path) -> list[dict]:
    """读 gzip，记行号（压缩流不能 seek）。"""
    idx, i = [], 0
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            sid, s = next(iter(r.items()))
            n = s["num_turns"]
            last = s[f"turn_{n}"]
            idx.append(dict(sid=sid, n=n, L=len(last["input_ids"]) + len(last["response_ids"]), line=i))
            i += 1
    return idx


def take(rows: list[dict], budget: int, rng: random.Random,
         max_turns: int = 0) -> list[dict]:
    """随机打乱后贪心取会话，直到总量 ≈ budget（±3% 收敛）。

    max_turns: >0 时把每条会话截断到前 max_turns 轮（L 按截断后的末轮重算）。
               用于控制单次运行时长——见 exp/LargeScale.md 的时间尺度一节。
    """
    rows = rows[:]
    rng.shuffle(rows)
    pick, tot = [], 0
    for r in rows:
        if tot >= budget * 0.97:
            break
        if max_turns and r["n"] > max_turns:
            r = dict(r, n=max_turns, L=r["L"] * max_turns / r["n"])
        pick.append(r)
        tot += int(r["L"])
    return pick


# --------------------------------------------------------------------------
# 采样：目标 ρ = ΣL / pool
# --------------------------------------------------------------------------
def sample_pool(source: str, pool: int, rho: float, seed: int,
                max_turns: int = 0) -> tuple[list[dict], list[dict], dict]:
    """返回 (选中的会话, 总体索引, 各源抽样统计)。"""
    rng = random.Random(seed)
    budget = int(pool * rho)
    parts, idx_all, stat = [], {}, {}

    if source in ("alfred", "mix"):
        a = index_alfred(ALFRED)
        idx_all["alfred"] = a
        stat["alfred_total"] = {"n": len(a), "tokens": int(sum(r["L"] for r in a))}

    if source in ("apigen", "mix"):
        g = index_apigen(APIGEN)
        idx_all["apigen"] = g
        stat["apigen_total"] = {"n": len(g), "tokens": int(sum(r["L"] for r in g))}

    if source == "mix":
        # 对半分的 token 预算
        for key, budget_i in (("alfred", budget // 2), ("apigen", budget - budget // 2)):
            pick = take(idx_all[key], budget_i, rng, max_turns)
            pick = [dict(r, source=key) for r in pick]
            parts.extend(pick)
            stat[key] = {"n": len(pick), "tokens": int(sum(r["L"] for r in pick))}
    else:
        pick = take(idx_all[source], budget, rng, max_turns)
        pick = [dict(r, source=source) for r in pick]
        parts.extend(pick)
        stat[source] = {"n": len(pick), "tokens": int(sum(r["L"] for r in pick))}
    return parts, idx_all, stat


def _cap(gaps: list[float], max_gap: float) -> list[float]:
    """间隔上限：超过池周转时间的空闲对缓存状态等效，可直接截断。"""
    return [min(g, max_gap) for g in gaps] if max_gap > 0 else gaps


def ks_two_sample(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return float(d)


def emit(picks: list[dict], idx_all: dict, model: IdleModel, out: Path, seed: int,
         max_gap: float = 0.0) -> dict:
    """第二阶段：按偏移读出会话、写入 gap_s、落盘。"""
    rng = random.Random(seed + 7919)
    by_src: dict[str, list[dict]] = {}
    for p in picks:
        by_src.setdefault(p["source"], []).append(p)

    gaps_all = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fo:
        for src, rows in by_src.items():
            if src == "alfred":
                f = open(ALFRED, "rb")
                for r in rows:
                    f.seek(r["off"])
                    sess = next(iter(json.loads(f.readline()).items()))
                    n = r["n"]
                    gaps = _cap(model.gaps(n, rng), max_gap)
                    for t in range(1, n + 1):
                        sess[1][f"turn_{t}"]["gap_s"] = round(gaps[t - 1], 4)
                    sess[1]["num_turns"] = n
                    fo.write(json.dumps({sess[0]: sess[1]}, ensure_ascii=False) + "\n")
                    gaps_all.extend(gaps)
                f.close()
            else:
                want = {r["line"]: r for r in rows}
                with gzip.open(APIGEN, "rt") as f:
                    for i, line in enumerate(f):
                        if i not in want:
                            continue
                        sess = next(iter(json.loads(line).items()))
                        n = want[i]["n"]
                        gaps = _cap(model.gaps(n, rng), max_gap)
                        for t in range(1, n + 1):
                            sess[1][f"turn_{t}"]["gap_s"] = round(gaps[t - 1], 4)
                        sess[1]["num_turns"] = n
                        fo.write(json.dumps({sess[0]: sess[1]}, ensure_ascii=False) + "\n")
                        gaps_all.extend(gaps)
    return {"gaps": gaps_all}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["alfred", "apigen", "mix"], default="alfred")
    ap.add_argument("--pool", type=int, default=73120, help="服务端池大小（token）")
    ap.add_argument("--rho", type=float, default=1.2, help="占用比 ΣL/pool")
    ap.add_argument("--model", choices=["fixed3", "tool", "replay", "claude"], default="claude")
    ap.add_argument("--pi-human", type=float, default=0.137, help="人工轮概率（claude 实测 0.137）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=0,
                    help="每条会话截断到前 N 轮（0=不截断）；用于控制单次运行时长")
    ap.add_argument("--max-gap", type=float, default=0.0,
                    help="间隔上限（秒）；0=不截断。超过池周转时间的空闲对缓存状态等效")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    name = args.name or f"{args.source}_p{args.pool}_r{args.rho}_{args.model}_h{args.pi_human}_s{args.seed}"
    out = OUT / f"{name}.jsonl"

    picks, idx_all, stat = sample_pool(args.source, args.pool, args.rho, args.seed,
                                      args.max_turns)
    T = Tables.load()
    model = IdleModel(kind=args.model, pi_human=args.pi_human,
                      eps=(args.model != "claude"), tables=T)
    info = emit(picks, idx_all, model, out, args.seed, args.max_gap)

    # 池周转时间：池里内容被替换一遍所需的时间。间隔超过几倍周转时间后，
    # 会话必然已被完全清出，再长的空闲对缓存状态没有区别 → 可以安全截断。
    n_turns = int(sum(r["n"] for r in picks))
    tok_per_turn = max(sum(r["L"] for r in picks) / max(n_turns, 1), 1.0)
    turn_s = 0.15 + 0.02 * float(np.mean(info["gaps"]))  # s=0.02 的典型单轮墙钟
    rate = len(picks) * tok_per_turn / max(turn_s, 1e-6)
    turnover = args.pool / max(rate, 1e-6)

    L = [p["L"] for p in picks]
    pop_L = [r["L"] for src in idx_all for r in idx_all[src]]
    g = np.asarray(info["gaps"], float)
    man = {
        "name": name, "source": args.source, "model": args.model,
        "pi_human": args.pi_human, "pool": args.pool, "rho_target": args.rho,
        "rho_realized": round(sum(L) / args.pool, 4), "seed": args.seed,
        "max_turns": args.max_turns, "max_gap": args.max_gap,
        "sessions": len(picks), "turns": int(len(g)),
        "L": {"p10": float(np.quantile(L, .1)), "p50": float(np.quantile(L, .5)),
              "p90": float(np.quantile(L, .9)), "max": int(max(L)), "sum": int(sum(L))},
        "L_ks_vs_population": round(ks_two_sample(L, pop_L), 4),
        "gap": {"p10": float(np.quantile(g, .1)), "p50": float(np.quantile(g, .5)),
                "p90": float(np.quantile(g, .9)), "p99": float(np.quantile(g, .99)),
                "gt300": float((g > 300).mean()), "max": float(g.max())},
        "by_source": stat,
        "turnover_s": round(turnover, 2),
        "cap_over_turnover": round(args.max_gap / turnover, 2) if args.max_gap else None,
    }
    (OUT / f"{name}.manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))

    # 同步生成 tasks 配置：无 pause_after_turn / 无 BurstGate，逐轮 gap 来自数据
    yml = Path("tailshot/large") / f"{name}.yaml"
    yml.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(out, yml.parent)
    yml.write_text(
        f"# 自动生成（exp/build_large_pools.py）：{name}\n"
        f"# model={args.model} rho={args.rho} pi_human={args.pi_human} seed={args.seed}\n"
        f"# 逐轮 gap_s 已写入数据文件；此格不使用 BurstGate 与统一 pause。\n"
        f"tasks:\n"
        f"- name: agentic_large\n"
        f"  task_id: 3\n"
        f"  data_path: {rel}\n"
        f"  lambda_rate: 0\n"
        f"  seed: 42\n"
        f"  max_record: -1\n"
        f"  shuffle: false\n"
        f"  arrival: {{mode: fixed, offset_s: 0.0, step_s: 0.05}}\n"
        f"  gap_mode: preassigned\n",
        encoding="utf-8",
    )
    print(f"  → {yml}")

    print(f"→ {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  会话 {man['sessions']}  轮 {man['turns']}  ΣL {man['L']['sum']}  "
          f"ρ={man['rho_realized']}（目标 {args.rho}）")
    print(f"  L p50={man['L']['p50']:.0f} p90={man['L']['p90']:.0f}  KS(总体)={man['L_ks_vs_population']}")
    print(f"  gap p50={man['gap']['p50']:.2f} p90={man['gap']['p90']:.1f} "
          f"p99={man['gap']['p99']:.1f} >300s={man['gap']['gt300']*100:.1f}%")
    print(f"  池周转 ≈ {turnover:.1f}s；间隔上限 {args.max_gap or '-'}s = "
          f"{man['cap_over_turnover'] or '-'}× 周转")
    if args.max_gap and man["cap_over_turnover"] < 3:
        print("  ⚠ 上限 < 3× 周转，可能截掉了仍会造成额外损失的空闲")


if __name__ == "__main__":
    main()
