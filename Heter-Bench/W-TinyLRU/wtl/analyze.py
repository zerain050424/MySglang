"""汇总、配对检验与出图（结果表 / results.csv / summary.md / 三张图）。

    python -m wtl.analyze --offline result/offline --online result/online \\
        --prefix-stats result/prefix_stats.json --struct result/struct_A3.jsonl \\
        --out result/analysis

产出
----
result/analysis/results_offline.csv   每行 = 臂 × 负载 × 压力 × seed
result/analysis/results_online.csv    每行 = 臂 × 负载 × 池大小 × seed
result/analysis/paired.csv            逐配置的配对 Δ（含 bootstrap 95% CI）
result/analysis/summary.md            主表 + 预注册判据 PASS/FAIL
result/analysis/fig1_hit_vs_pressure.png
result/analysis/fig2_retention_vs_depth.png
result/analysis/fig3_ttft.png
result/analysis/struct_timeseries.csv

判定口径（与设计文档 §1 一致）
------------------------------
M1 R_prefix(depth>=2k) 提升 >= +10pp；M2 hit +5pp；M3 TTFT_p50 <= -3%；
M5 ShareGPT per_task 回退 <= 1pp；M4-gate victim_src.qstruct >= 5%。
CI 用请求级配对 bootstrap（10k 次，不假设正态）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LAYER_ORDER = (256, 512, 1024, 2048, 4096, 8192)


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def load_offline(pattern: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(pattern, "*.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("schema") != "wtl.replay/1":
            continue
        d["_path"] = path
        d["workload"] = _workload_of(d)
        d["ratio"] = d.get("pool_ratio")
        d["seed"] = d.get("seed", 42)
        out.append(d)
    return out


def _workload_of(d: dict) -> str:
    meta = d.get("meta") or {}
    ds = d.get("dataset") or {}
    wl = meta.get("workload") or ds.get("workload")
    if wl:
        return str(wl)
    stem = Path(d.get("_path", "x")).stem
    parts = stem.split("_")
    return parts[2] if len(parts) > 3 else stem


def load_online(pattern: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(pattern, "*.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if "ttft_p50_ms" not in d and "cache_hit_rate_pct" not in d:
            continue
        stem = Path(path).stem
        d["_path"] = path
        d["arm"] = d.get("arm") or _parse_arm(stem)
        d["workload"] = d.get("workload") or _parse_workload(stem)
        d["pool_tokens"] = d.get("pool_tokens") or _parse_tokens(stem)
        d["seed"] = d.get("seed") or _parse_seed(stem)
        out.append(d)
    return out


def _parse_arm(stem: str) -> str:
    for part in stem.split("_"):
        if part.startswith("A") and part[1:].isdigit():
            return part
    return "?"


def _parse_workload(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) > 2 else stem


def _parse_tokens(stem: str) -> Optional[int]:
    for part in stem.split("_"):
        if part.endswith("k") and part[:-1].isdigit():
            return int(part[:-1]) * 1000
        if part.isdigit() and len(part) >= 4:
            return int(part)
    return None


def _parse_seed(stem: str) -> int:
    for part in stem.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return 42


def load_online_rows(pattern: str) -> Dict[str, List[dict]]:
    """读取 bench_hetercache 的逐请求 CSV（文件名去掉 .json 加 .csv）。"""
    out: Dict[str, List[dict]] = {}
    for path in sorted(glob.glob(os.path.join(pattern, "*.csv"))):
        rows = []
        try:
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
        except Exception:
            continue
        out[path] = rows
    return out


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def bootstrap_ci(xs: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    n = len(xs)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return (mean, lo, hi)


def paired_delta(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    value_fn,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """按 (task, sid, turn) 配对后求 mean(a-b) 与 bootstrap CI。"""
    ka = {(r["task"], r["sid"], r["turn"]): r for r in rows_a}
    kb = {(r["task"], r["sid"], r["turn"]): r for r in rows_b}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    deltas = [value_fn(ka[k]) - value_fn(kb[k]) for k in keys]
    mean, lo, hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    return {"n": len(keys), "mean": mean, "lo": lo, "hi": hi}


def paired_ratio_delta(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    *,
    task: Optional[str] = None,
    kind: Optional[str] = None,
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """配对（task,sid,turn）+ 请求级重采样的**汇总命中率**差（单位 pp）。

    统计量 = Σcached/Σprompt（token 级），而不是逐请求比率的平均：
    后者方差极大（长 prompt 与短 prompt 等权），会让 CI 宽到没有判别力。
    """

    def key(r):
        return (r["task"], r["sid"], r["turn"])

    def keep(r):
        if task is not None and r["task"] != task:
            return False
        if kind is not None and _kind(r.get("sid", "")) != kind:
            return False
        return True

    ka = {key(r): r for r in rows_a if keep(r)}
    kb = {key(r): r for r in rows_b if keep(r)}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    ca = [float(ka[k]["cached_tokens"]) for k in keys]
    pa = [float(ka[k]["prompt_tokens"]) for k in keys]
    cb = [float(kb[k]["cached_tokens"]) for k in keys]
    pb = [float(kb[k]["prompt_tokens"]) for k in keys]

    def delta(idx) -> float:
        sa = sum(ca[i] for i in idx)
        spa = sum(pa[i] for i in idx)
        sb = sum(cb[i] for i in idx)
        spb = sum(pb[i] for i in idx)
        ra = 100.0 * sa / spa if spa else 0.0
        rb = 100.0 * sb / spb if spb else 0.0
        return ra - rb

    idx_all = list(range(len(keys)))
    mean = delta(idx_all)
    rng = random.Random(seed)
    boots = sorted(
        delta([rng.randrange(len(keys)) for _ in range(len(keys))])
        for _ in range(n_boot)
    )
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"n": len(keys), "mean": mean, "lo": lo, "hi": hi}


def attach_hist_flag(rows: List[dict]) -> List[dict]:
    """从逐请求明细里**推导**「上一轮 prompt 是否仍在树上」。

    不依赖 replay 写的 ``hist_full`` 字段：上一轮的 prompt_tokens 就在同一份
    rows 里（同一 session 的 turn-1 行），推导口径对离线/在线完全一致。
    """
    prev = {}
    for r in sorted(rows, key=lambda x: (x["task"], x["sid"], x["turn"])):
        key = (r["task"], r["sid"])
        try:
            pt = float(r["prompt_tokens"])
            turn = int(r["turn"])
        except (TypeError, ValueError):
            continue
        if turn > 1 and key in prev:
            r["hist_full"] = 1 if float(r["cached_tokens"]) >= prev[key] else 0
        else:
            r["hist_full"] = None
        prev[key] = pt
    return rows


def _hist_full(r: dict) -> float:
    """逐请求：上一轮 prompt 是否仍完整命中（行里由 replay 打标记）。"""
    return 1.0 if r.get("hist_full") else 0.0


def _norm_online_row(r: dict) -> dict:
    """把 bench_hetercache 的 CSV 行规范化成与离线 rows 同构（并转数值）。"""

    def num(key):
        v = r.get(key)
        if v in (None, "", "None", "nan"):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "task": r.get("task") or "?",
        "sid": r.get("session_id") or "?",
        "turn": r.get("turn") or "?",
        "prompt_tokens": num("prompt_tokens") or 0.0,
        "cached_tokens": num("cached_tokens") or 0.0,
        "ttft_ms": num("ttft_ms"),
        "prefill_ms": num("prefill_ms"),
        "queue_time_ms": num("queue_time_ms"),
    }


def retained(L: int):
    def f(r: dict) -> float:
        return 1.0 if int(r["cached_tokens"]) >= L else 0.0

    return f


def cover(L: int):
    def f(r: dict) -> float:
        return min(int(r["cached_tokens"]), L) / float(L)

    return f


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------
def _kind(sid: str) -> str:
    """按 session_id 前缀把请求分成两个成分（与 wtl.synth 的命名约定一致）。"""
    return "burst" if "noise" in (sid or "") else "struct"


def _split_hit(rows, kind=None) -> dict:
    """成分内命中率：Σcached/Σprompt。"""
    sel = [r for r in rows if kind is None or _kind(r.get("sid", "")) == kind]
    prompt = sum(float(r["prompt_tokens"]) for r in sel)
    cached = sum(float(r["cached_tokens"]) for r in sel)
    return {
        "n": len(sel),
        "prompt": prompt,
        "cached": cached,
        "hit_pct": round(100.0 * cached / prompt, 3) if prompt else 0.0,
    }


def _ret_from_rows(rows, layers, kind=None) -> Dict[int, dict]:
    """按成分计算前缀保留度：I = 1[cached>=L]，cover = min(cached,L)/L。

    直接用逐请求 rows 算，不需要 replay 参与——这样 burst 请求不会稀释
    结构复用的保留度（它们是「本就没有前缀」的干扰项）。
    """
    out: Dict[int, dict] = {}
    for L in layers or []:
        sel = [
            r for r in rows
            if (kind is None or _kind(r.get("sid", "")) == kind)
        ]
        if not sel:
            continue
        full = sum(1 for r in sel if float(r["cached_tokens"]) >= L)
        cover = sum(min(float(r["cached_tokens"]), L) / L for r in sel)
        out[int(L)] = {
            "n": len(sel),
            "full_rate": round(full / len(sel), 4),
            "cover": round(cover / len(sel), 4),
        }
    return out


def _rho(d: dict) -> Optional[float]:
    """干扰强度 ρ = **两次模板复用之间的 burst token** / 池容量。

    burst 在时间上均匀铺开时，一个模板的两次使用之间会插入
    ``noise_sessions / U`` 个一次性请求（U = 模板总使用次数 = 变体数×每模板会话×轮数），
    它们合计 ``(noise/U)×doc_len`` 个 token —— 这才是「干扰能否冲掉前缀」的判据。
    ρ ≳ 1 表示 burst 有能力在模板复用之间把池子刷一遍。
    """
    meta = d.get("meta") or {}
    variants = meta.get("variants") or []
    breadth = int(meta.get("breadth") or 0)
    turns = int(meta.get("turns") or 0)
    noise = int(meta.get("noise_sessions") or 0)
    doc_len = int(meta.get("doc_len") or 0)
    pool = int(d.get("pool_tokens") or 0)
    uses = max(len(variants), 1) * max(breadth, 1) * max(turns, 1)
    if noise <= 0 or doc_len <= 0 or pool <= 0 or uses <= 0:
        return None
    return round((noise / uses) * doc_len / pool, 3)


def offline_table(runs: Sequence[dict]) -> List[dict]:
    rows = []
    for d in runs:
        s = d.get("summary") or {}
        ret = s.get("retention") or {}
        regret = s.get("regret") or {}
        vsrc = s.get("victim_src") or {}
        tot_ev = sum(vsrc.values()) or 0.0
        row = {
            "backend": d.get("backend"),
            "arm": d.get("arm"),
            "label": d.get("label"),
            "workload": d.get("workload"),
            "ratio": d.get("ratio"),
            "ladder": (
                f"pool={d.get('pool_tokens')}" if d.get("ratio") is None
                else f"r={d.get('ratio')}"
            ),
            "pool_tokens": d.get("pool_tokens"),
            "calib_peak": d.get("calib_peak"),
            "seed": d.get("seed"),
            "num_requests": s.get("num_requests"),
            "hit_pct": s.get("hit_pct"),
            "oom": s.get("oom"),
            "evictions": s.get("evictions"),
            "regret_rate": regret.get("regret_rate"),
            "protected_waste_rate": regret.get("protected_waste_rate"),
            "one_shot_pct": round(100.0 * vsrc.get("one_shot", 0.0) / tot_ev, 3) if tot_ev else 0.0,
            "qstruct_pct": round(100.0 * vsrc.get("qstruct", 0.0) / tot_ev, 3) if tot_ev else 0.0,
            # 机制证据：共享数账本 + SLRU 实际给出的候选数
            "slru_picks": (s.get("slru") or {}).get("slru_picks"),
            "share_counted": (s.get("slru") or {}).get("share_counted"),
            "share_dedup": (s.get("slru") or {}).get("share_dedup"),
            # 唯一旋钮：越大保护越难触发（A2 用不可达值做基线）
            "share_min": (d.get("config") or {}).get("slru", {}).get("share_min"),
        }
        for task, layers in (ret or {}).items():
            for L, v in layers.items():
                row[f"ret_{task}@{L}_full"] = v.get("full_rate")
                row[f"ret_{task}@{L}_cover"] = v.get("cover")
        rows_ = d.get("rows") or []
        if rows_:
            for kind in ("struct", "burst"):
                st_ = _split_hit(rows_, kind)
                row[f"hit_{kind}"] = st_["hit_pct"]
                row[f"n_{kind}"] = st_["n"]
        row["rho"] = _rho(d)
        layers = []
        for v in (d.get("layers_by_task") or {}).values():
            layers = sorted(set(layers) | set(v or []))
        for kind in ("struct", "burst"):
            for L, v in _ret_from_rows(rows_, layers, kind).items():
                row[f"ret_{kind}@{L}_full"] = v["full_rate"]
                row[f"ret_{kind}@{L}_cover"] = v["cover"]
        for k, v in (s.get("per_task") or {}).items():
            row[f"hit_{k}"] = v.get("hit_pct")
        for k, v in (s.get("history_retention") or {}).items():
            row[f"hist_{k}_full"] = v.get("full_rate")
            row[f"hist_{k}_cover"] = v.get("cover")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown_table(rows: Sequence[dict], cols: Sequence[str], headers: Sequence[str]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------
def criterion(passed: Optional[bool], name: str, detail: str) -> str:
    tag = "PASS" if passed else ("FAIL" if passed is False else "N/A")
    return f"| {name} | {tag} | {detail} |"


def judge(offline: Sequence[dict], paired: Sequence[dict], online: Sequence[dict]) -> List[str]:
    """H1–H6 预注册判定（见 EXPERIMENT.md §1）。

    ``paired`` 里是请求级配对差（单位 pp / ms）；``offline`` 是逐 run 汇总（用于 H4/H6）。
    """
    lines = ["| 判据 | 结论 | 依据 |", "|---|---|---|"]
    MAIN_POOL = 20000

    def best(metric: str, arm: str, base: str, prefer_pool: int = MAIN_POOL):
        cands = [p for p in paired if p["metric"] == metric and p["arm"] == arm and p["base"] == base]
        if not cands:
            return None
        cands.sort(key=lambda p: (abs((p.get("pool") or 0) - prefer_pool), -p["n"]))
        return cands[0]

    def fmt(p):
        return (f"{p['arm']} vs {p['base']} @ {p['workload']} pool={p.get('pool')}: "
                f"{p['mean']:+.2f}pp [{p['lo']:+.2f},{p['hi']:+.2f}]")

    # ---- H1 结构复用保留度 ----
    ret = [p for p in paired if p["metric"].startswith("ret_struct@")
           and p["arm"] == "A3" and p["base"] == "A2"]
    if ret:
        # 取最深的层（= 模板整段前缀）
        depth = max(int(p["metric"].split("@")[1]) for p in ret)
        h1 = best(f"ret_struct@{depth}", "A3", "A2")
        ok = h1["mean"] >= 5.0 and h1["lo"] > 0
        lines.append(criterion(ok, f"H1 保留度 ret@{depth} ≥ +5pp", fmt(h1)))
    else:
        lines.append(criterion(None, "H1 保留度 ≥ +5pp", "无 ret_struct@ 数据（检查 synth meta 的 layers）"))

    # ---- H2 命中率（成分拆分）----
    h2s = best("hit_struct", "A3", "A2")
    h2b = best("hit_burst", "A3", "A2")
    if h2s and h2b:
        ok = h2s["mean"] >= 3.0 and h2s["lo"] > 0 and h2b["mean"] >= -1.0
        lines.append(criterion(
            ok, "H2 结构复用子集 hit ≥ +3pp 且 burst 回退 ≤1pp",
            f"struct {h2s['mean']:+.2f}pp [{h2s['lo']:+.2f},{h2s['hi']:+.2f}]；"
            f"burst {h2b['mean']:+.2f}pp [{h2b['lo']:+.2f},{h2b['hi']:+.2f}]",
        ))
    else:
        lines.append(criterion(None, "H2 命中率（成分拆分）", "无成分配对数据"))

    # ---- H3 one-shot 消融 ----
    h3 = best("hit_struct", "A4", "A3") or best("hit_pct", "A4", "A3")
    if h3:
        gain = -h3["mean"]  # A3 - A4
        ok = gain >= 3.0 and h3["hi"] < 0
        lines.append(criterion(ok, "H3 A3−A4 ≥ +3pp（会话内复用不进 one-shot 带）",
                               f"A3−A4 = {gain:+.2f}pp（A4−A3 = {h3['mean']:+.2f}pp）"))
    else:
        lines.append(criterion(None, "H3 one-shot 消融", "未跑 A4（S4）"))

    # ---- H4 无压力趋同（最大池 = 压力阶梯的最小端）----
    pools = sorted({int(d.get("pool_tokens") or 0) for d in offline if d.get("pool_tokens")})
    big = pools[-1] if pools else None
    at1 = [d for d in offline if big and int(d.get("pool_tokens") or 0) == big]
    hits = {}
    for d in at1:
        if d.get("arm") in ("A0", "A2", "A3"):
            hits[d["arm"]] = (d.get("summary") or {}).get("hit_pct")
    if len(hits) >= 2:
        spread = max(hits.values()) - min(hits.values())
        lines.append(criterion(spread < 0.5, f"H4 无压力（pool={big}）四臂趋同 <0.5pp",
                               " ".join(f"{k}={v:.2f}" for k, v in sorted(hits.items()))
                               + f" spread={spread:.3f}pp"))
    else:
        lines.append(criterion(None, "H4 无压力趋同", "无池容量阶梯数据"))

    # ---- H5 在线 TTFT ----
    tt = [p for p in paired if p["metric"] == "ttft_p50_ms"]
    if tt:
        h5 = min(tt, key=lambda p: p["mean"])
        base = next((o for o in online if o.get("arm") == h5["base"]), None)
        rel = (h5["mean"] / base["ttft_p50_ms"] * 100.0) if base and base.get("ttft_p50_ms") else 0.0
        lines.append(criterion(rel <= -3.0 and h5["hi"] < 0, "H5 在线 TTFT_p50 ≤ −3%",
                               f"{h5['arm']} vs {h5['base']}: {h5['mean']:+.2f}ms ({rel:+.1f}%)"))
    else:
        lines.append(criterion(None, "H5 在线 TTFT_p50 ≤ −3%", "无在线数据（S6 需 GPU）"))

    # ---- H6 收益随干扰强度 ρ 单调 ----
    by_rho = {}
    for d in offline:
        if d.get("arm") != "A3":
            continue
        rho = _rho(d)
        if rho is None:
            continue
        by_rho[(d.get("workload"), rho)] = (d.get("summary") or {}).get("hit_pct")
    base_rho = {}
    for d in offline:
        if d.get("arm") != "A2":
            continue
        rho = _rho(d)
        if rho is None:
            continue
        base_rho[(d.get("workload"), rho)] = (d.get("summary") or {}).get("hit_pct")
    gains = sorted(
        (key[1], by_rho[key] - base_rho[key])
        for key in by_rho
        if key in base_rho and by_rho[key] is not None and base_rho[key] is not None
    )
    if len(gains) >= 3:
        mono = all(gains[i][1] <= gains[i + 1][1] + 1e-9 for i in range(len(gains) - 1))
        lines.append(criterion(mono, "H6 收益随 ρ 单调上升",
                               " ".join(f"ρ={r}:{g:+.2f}pp" for r, g in gains)))
    else:
        lines.append(criterion(None, "H6 收益随 ρ 单调", f"ρ 数据点不足（{len(gains)} 个）"))
    return lines


def _src_share_pct(victim_src: dict, key: str) -> float:
    """某个淘汰来源（one_shot / qstruct / global）占被淘汰 token 的比例。"""
    total = sum(victim_src.values()) or 0.0
    if not total:
        return 0.0
    return round(100.0 * (victim_src.get(key, 0.0) or 0.0) / total, 3)


def _qstruct_pct(d: dict) -> float:
    vsrc = (d.get("summary") or {}).get("victim_src") or {}
    tot = sum(vsrc.values()) or 0.0
    return 100.0 * vsrc.get("qstruct", 0.0) / tot if tot else 0.0


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def make_figures(out: Path, offline_rows: Sequence[dict], online: Sequence[dict],
                 online_rows: Dict[str, List[dict]], runs: Sequence[dict]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made = []

    # fig1: hit% vs pool ratio
    by = defaultdict(list)
    for r in offline_rows:
        if r["backend"] != "real":
            continue
        key = (r["workload"], r["arm"])
        if r["hit_pct"] is not None and r["ratio"] is not None:
            by[key].append((float(r["ratio"]), float(r["hit_pct"])))
    if by:
        fig, axes = plt.subplots(1, max(1, len({k[0] for k in by})),
                                 figsize=(4.2 * max(1, len({k[0] for k in by})), 3.4),
                                 squeeze=False)
        workloads = sorted({k[0] for k in by})
        for ax, wl in zip(axes[0], workloads):
            for (w, arm), pts in sorted(by.items()):
                if w != wl:
                    continue
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=arm)
            ax.set_title(wl)
            ax.set_xlabel("pool / working_set")
            ax.set_ylabel("token hit %")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        f1 = out / "fig1_hit_vs_pressure.png"
        fig.savefig(f1, dpi=130)
        plt.close(fig)
        made.append(str(f1))

    # fig2: retention cover vs depth
    curves = defaultdict(dict)
    for d in runs:
        if d.get("backend") != "real":
            continue
        arm = d.get("arm")
        for task, layers in ((d.get("summary") or {}).get("retention") or {}).items():
            for L, v in layers.items():
                curves[arm].setdefault(int(L), []).append(float(v["cover"]))
    if curves:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for arm, pts in sorted(curves.items()):
            xs = sorted(pts)
            ys = [statistics.mean(pts[x]) for x in xs]
            ax.plot(xs, ys, marker="s", label=arm)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("shared prefix depth (tokens)")
        ax.set_ylabel("retention cover")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        f2 = out / "fig2_retention_vs_depth.png"
        fig.savefig(f2, dpi=130)
        plt.close(fig)
        made.append(str(f2))

    # fig3: TTFT CDF（在线逐请求 CSV）
    if online_rows:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for path, rows in sorted(online_rows.items()):
            xs = sorted(float(r["ttft_ms"]) for r in rows if r.get("ttft_ms"))
            if not xs:
                continue
            ys = [(i + 1) / len(xs) for i in range(len(xs))]
            ax.plot(xs, ys, label=Path(path).stem.replace("bench_", ""))
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        f3 = out / "fig3_ttft.png"
        fig.savefig(f3, dpi=130)
        plt.close(fig)
        made.append(str(f3))
    return made


# ---------------------------------------------------------------------------
# struct 时间序列
# ---------------------------------------------------------------------------
def online_retention(rows: Sequence[dict], layers_by_task: Dict[str, List[int]]) -> Dict[str, dict]:
    """在线逐请求 CSV 的保留度（与离线同一套判据：cached >= L / 覆盖度）。"""
    acc: Dict[str, Dict[int, dict]] = {}
    for r in rows:
        task = r.get("task")
        cached = r.get("cached_tokens")
        if not task or cached is None:
            continue
        try:
            cached = float(cached)
        except (TypeError, ValueError):
            continue
        for L in layers_by_task.get(task) or []:
            b = acc.setdefault(task, {}).setdefault(L, {"n": 0, "full": 0, "cover": 0.0})
            b["n"] += 1
            b["full"] += int(cached >= L)
            b["cover"] += min(cached, L) / float(L)
    return {
        task: {
            str(L): {
                "n": b["n"],
                "full_rate": round(b["full"] / b["n"], 4) if b["n"] else 0.0,
                "cover": round(b["cover"] / b["n"], 4) if b["n"] else 0.0,
            }
            for L, b in sorted(buckets.items())
        }
        for task, buckets in acc.items()
    }


def struct_timeseries(paths: Sequence[str]) -> List[dict]:
    rows = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                snap = d.get("slru") or {}
                vsrc = d.get("victim_src") or {}
                tot = sum(vsrc.values()) or 0.0
                rows.append(
                    {
                        "slru": d.get("slru"),
                        "ts": d.get("ts"),
                        "evict_calls": d.get("evict_calls"),
                        "evict_us_p50": d.get("evict_us_p50"),
                        "evict_us_p99": d.get("evict_us_p99"),
                        "available": d.get("available"),
                        "evictable_size": d.get("evictable_size"),
                        "slru_picks": snap.get("slru_picks"),
                        "share_counted": snap.get("share_counted"),
                        "share_dedup": snap.get("share_dedup"),
                        "share_min": d.get("slru_share_min"),
                        "one_shot_evicted_pct": round(
                            100.0 * (vsrc.get("one_shot", 0.0) or 0.0) / tot, 2
                        ) if tot else 0.0,
                        "qstruct_evicted_pct": round(
                            100.0 * (vsrc.get("qstruct", 0.0) or 0.0) / tot, 2
                        ) if tot else 0.0,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="W-TinyLRU 实验分析")
    p.add_argument("--offline", default="result/offline")
    p.add_argument("--online", default="result/online")
    p.add_argument("--struct", action="append", default=[])
    p.add_argument("--out", default="result/analysis")
    p.add_argument("--prefix-stats", default=None,
                   help="prefix_stats.json；给了就算在线逐请求 CSV 的保留度")
    p.add_argument("--baselines", default="A0,A2,A3",
                   help="配对基线；A3 用于 H3 的 A3−A4 消融")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--ratio-focus", type=float, default=None,
                   help="配对检验只用这个压力点（缺省=全部）")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    baselines = [x.strip() for x in a.baselines.split(",") if x.strip()]

    runs = load_offline(a.offline)
    online = load_online(a.online)
    online_rows = load_online_rows(a.online)
    if not runs and not online:
        print("[analyze] 没找到结果：先跑 wtl.run_matrix / scripts/run_online.sh")
        return 2

    off_rows = offline_table(runs)
    write_csv(out / "results_offline.csv", off_rows)

    # 在线保留度（可选，需要 prefix_stats.json）
    ps_layers: Dict[str, List[int]] = {}
    if a.prefix_stats and Path(a.prefix_stats).exists():
        ps = json.loads(Path(a.prefix_stats).read_text(encoding="utf-8"))
        ps_layers = {
            task: [int(x) for x in (st.get("layers") or [])]
            for task, st in (ps.get("tasks") or {}).items()
        }
        print(f"[analyze] prefix_stats: {len(ps_layers)} 个 task 的分层深度 {ps_layers}")

    ret_by_stem: Dict[str, dict] = {}
    if ps_layers and online_rows:
        for path, rows in online_rows.items():
            norm = [_norm_online_row(r) for r in rows]
            ret_by_stem[Path(path).stem] = online_retention(norm, ps_layers)
        ret_rows = [
            {"tag": stem, "task": task, "layer": int(L), **vals}
            for stem, per_task in sorted(ret_by_stem.items())
            for task, layers in per_task.items()
            for L, vals in layers.items()
        ]
        if ret_rows:
            write_csv(out / "retention_online.csv", ret_rows)

    on_rows = [
        {
            "arm": d.get("arm"), "workload": d.get("workload"),
            "pool_tokens": d.get("pool_tokens"), "seed": d.get("seed"),
            "hit_pct": d.get("cache_hit_rate_pct"),
            "ttft_p50_ms": d.get("ttft_p50_ms"),
            "ttft_p90_ms": d.get("ttft_p90_ms"),
            "ttft_p99_ms": d.get("ttft_p99_ms"),
            "e2e_p99_ms": d.get("e2e_p99_ms"),
            "num_requests": d.get("num_requests"),
            "num_errors": d.get("num_errors"),
            "throughput": d.get("request_throughput"),
            **{
                f"hit_{name}": v.get("cache_hit_rate_pct")
                for name, v in (d.get("per_task") or {}).items()
            },
            **{
                f"ret_{task}@{L}_{k}": vals[k]
                for task, layers in (ret_by_stem.get(Path(d.get("_path", "")).stem) or {}).items()
                for L, vals in layers.items()
                for k in ("full_rate", "cover")
            },
        }
        for d in online
    ]
    write_csv(out / "results_online.csv", on_rows)

    # ---- 配对检验（离线逐请求 + 在线逐请求）----
    paired: List[dict] = []
    fake_runs = [d for d in runs if d.get("backend") != "real"]
    real_runs = [d for d in runs if d.get("backend") == "real"]
    if fake_runs:
        print(f"[analyze] 忽略 {len(fake_runs)} 个 fake 后端回放（仅用于链路验证）")

    for d in real_runs:
        if d.get("rows"):
            attach_hist_flag(d["rows"])
    by_key = defaultdict(dict)
    for d in real_runs:
        if a.ratio_focus is not None and d.get("ratio") != a.ratio_focus:
            continue
        by_key[
            (d["workload"], d.get("ratio"), d.get("pool_tokens"), d.get("seed"))
        ][d["arm"]] = d

    for (wl, ratio, pool, seed), arms in sorted(by_key.items(), key=lambda x: str(x[0])):
        for base in baselines:
            if base not in arms:
                continue
            for arm, d in arms.items():
                if arm == base:
                    continue
                pa = d.get("rows") or []
                pb = arms[base].get("rows") or []
                if not pa or not pb:
                    continue
                max_layer = max(
                    (max(v) for v in (d.get("layers_by_task") or {}).values() if v),
                    default=0,
                )
                specs = [("hit_pct", None), ("hit_struct", "struct"), ("hit_burst", "burst")]
                for task in sorted({r["task"] for r in pa}):
                    if any(r["task"] == task for r in pb):
                        specs.append((f"hit_{task}", task))
                if d.get("summary", {}).get("history_retention"):
                    specs.append(("hist_full", "hist"))
                for L in sorted({int(x) for v in (d.get("layers_by_task") or {}).values() for x in (v or [])}):
                    if L <= max_layer:
                        specs.append((f"ret_struct@{L}", L))
                if 2048 <= max_layer:
                    specs.append(("cover@2048", 2048))

                for metric, arg in specs:
                    if metric == "hist_full":
                        # 会话内历史保留：逐请求指示函数（上一轮 prompt 是否仍在树上）
                        fn = _hist_full
                        pa_h = [r for r in pa if r.get("hist_full") is not None]
                        pb_h = [r for r in pb if r.get("hist_full") is not None]
                        st = paired_delta(pa_h, pb_h, fn, n_boot=a.n_boot, seed=seed)
                        st = {**st, "mean": st["mean"] * 100,
                              "lo": st["lo"] * 100, "hi": st["hi"] * 100}
                        unit = "pp"
                    elif metric in ("hit_pct", "hit_struct", "hit_burst"):
                        st = paired_ratio_delta(
                            pa, pb,
                            kind=None if metric == "hit_pct" else metric.split("_", 1)[1],
                            n_boot=a.n_boot, seed=seed,
                        )
                        unit = "pp"
                    elif metric.startswith("hit_"):
                        st = paired_ratio_delta(
                            pa, pb, task=arg, n_boot=a.n_boot, seed=seed
                        )
                        unit = "pp"
                    else:
                        L = int(arg)
                        fn = retained(L) if metric.startswith("ret_") else cover(L)
                        pa_k = [r for r in pa if _kind(r.get("sid", "")) == "struct"]
                        pb_k = [r for r in pb if _kind(r.get("sid", "")) == "struct"]
                        st = paired_delta(pa_k, pb_k, fn, n_boot=a.n_boot, seed=seed)
                        st = {**st, "mean": st["mean"] * 100,
                              "lo": st["lo"] * 100, "hi": st["hi"] * 100}
                        unit = "pp"
                    if st["n"] == 0:
                        continue
                    paired.append(
                        {
                            "workload": wl, "ratio": ratio, "pool": pool, "seed": seed,
                            "arm": arm, "base": base, "metric": metric, "unit": unit,
                            "n": st["n"], "mean": round(st["mean"], 4),
                            "lo": round(st["lo"], 4), "hi": round(st["hi"], 4),
                        }
                    )
    write_csv(out / "paired.csv", paired)

    # 在线：按 arm 汇总 TTFT 与配对差（若有逐请求 CSV）
    on_paired: List[dict] = []
    if online_rows:
        per_arm = defaultdict(list)
        for path, rows in online_rows.items():
            arm = _parse_arm(Path(path).stem)
            wl = _parse_workload(Path(path).stem)
            per_arm[(wl, arm)] = attach_hist_flag([_norm_online_row(r) for r in rows])
        for (wl, arm), rows in sorted(per_arm.items()):
            for base in baselines:
                br = per_arm.get((wl, base))
                if not br or arm == base:
                    continue
                ka = {(r["task"], r["sid"], r["turn"]): r for r in rows}
                kb = {(r["task"], r["sid"], r["turn"]): r for r in br}
                keys = sorted(set(ka) & set(kb))
                xs = [
                    ka[k]["ttft_ms"] - kb[k]["ttft_ms"]
                    for k in keys
                    if ka[k].get("ttft_ms") is not None and kb[k].get("ttft_ms") is not None
                ]
                if xs:
                    mean, lo, hi = bootstrap_ci(xs, n_boot=a.n_boot)
                    on_paired.append(
                        {"workload": wl, "arm": arm, "base": base,
                         "metric": "ttft_p50_ms", "unit": "ms", "n": len(xs),
                         "mean": round(mean, 3), "lo": round(lo, 3), "hi": round(hi, 3)}
                    )
                # 逐 task 的命中率差（M5 护栏：ShareGPT 不得回退 >1pp）
                st = paired_ratio_delta(rows, br, n_boot=a.n_boot)
                if st["n"]:
                    on_paired.append(
                        {"workload": wl, "arm": arm, "base": base, "metric": "hit_pct",
                         "unit": "pp", "n": st["n"], "mean": round(st["mean"], 3),
                         "lo": round(st["lo"], 3), "hi": round(st["hi"], 3)}
                    )
                for task in sorted({r["task"] for r in rows}):
                    st = paired_ratio_delta(rows, br, task=task, n_boot=a.n_boot)
                    if st["n"]:
                        on_paired.append(
                            {"workload": wl, "arm": arm, "base": base,
                             "metric": f"hit_{task}", "unit": "pp", "n": st["n"],
                             "mean": round(st["mean"], 3), "lo": round(st["lo"], 3),
                             "hi": round(st["hi"], 3)}
                        )
    write_csv(out / "paired_online.csv", on_paired)

    # ---- struct 时间序列 ----
    st_rows = struct_timeseries(a.struct or [])
    if st_rows:
        write_csv(out / "struct_timeseries.csv", st_rows)

    # ---- summary.md ----
    lines: List[str] = []
    lines.append("# W-TinyLRU 实验结果\n")
    lines.append(f"- 离线回放：{len(real_runs)} 条（fake {len(fake_runs)} 条已排除）")
    lines.append(f"- 在线 runs：{len(online)} 条；逐请求 CSV：{len(online_rows)} 份")
    lines.append(f"- 基线：{', '.join(baselines)}\n")

    lines.append("## 1. 预注册判据\n")
    lines.extend(judge(real_runs, paired + on_paired, online))
    lines.append("")

    if off_rows:
        lines.append("## 2. 离线主表（真实后端）\n")
        rows = [r for r in off_rows if r["backend"] == "real"]
        cols = ["arm", "workload", "ladder", "hit_pct", "regret_rate",
                "protected_waste_rate", "one_shot_pct", "qstruct_evicted_pct",
                "share_counted", "slru_picks", "evictions"]
        heads = ["arm", "workload", "压力轴", "hit%", "regret", "prot_waste",
                 "one_shot_evict%", "qstruct_evicted%", "share_counted", "picks",
                 "evicts"]
        extra = sorted({
            k for r in rows for k in r
            if (k.startswith("ret_struct@") or k.startswith("hist_")) and k.endswith("_full")
        })
        for e in extra[:6]:
            cols.append(e)
            heads.append(e.replace("ret_", "").replace("hist_", "hist:")
                            .replace("_full", ""))
        lines.append(markdown_table(rows, cols, heads))
        lines.append("")

    if paired:
        lines.append("## 3. 配对差（请求级，bootstrap 95% CI）\n")
        rows = sorted(paired, key=lambda x: -abs(x["mean"]))[:25]
        lines.append(
            markdown_table(
                rows,
                ["arm", "base", "workload", "ratio", "metric", "n", "mean", "lo", "hi"],
                ["arm", "vs", "workload", "pool", "metric", "n", "Δmean", "CI lo", "CI hi"],
            )
        )
        lines.append("")

    if on_rows:
        lines.append("## 4. 在线 TTFT / 命中率\n")
        lines.append(
            markdown_table(
                on_rows,
                ["arm", "workload", "pool_tokens", "seed", "hit_pct", "ttft_p50_ms",
                 "ttft_p90_ms", "ttft_p99_ms", "num_requests", "num_errors"],
                ["arm", "workload", "pool", "seed", "hit%", "ttft50", "ttft90", "ttft99", "reqs", "err"],
            )
        )
        lines.append("")

    if st_rows:
        latest = defaultdict(dict)
        for r in st_rows:
            latest[r["tag"]] = r
        lines.append("## 5. struct 机制观测（最后一次快照）\n")
        rows = [
            {"tag": tag, **{k: v for k, v in rec.items() if k != "tag"}}
            for tag, rec in sorted(latest.items())
        ]
        lines.append(
            markdown_table(
                rows,
                ["tag", "evict_calls", "evict_us_p50", "evict_us_p99",
                 "promote", "counted", "dedup", "qstruct_evicted_pct"],
                ["tag", "evicts", "us_p50", "us_p99",
                 "promote", "counted", "dedup", "qstruct_evicted%"],
            )
        )
        lines.append("")

    figs = make_figures(out, off_rows, online, online_rows, real_runs)
    if figs:
        lines.append("## 6. 图\n")
        for f in figs:
            lines.append(f"![{Path(f).stem}]({Path(f).name})\n")

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[analyze] -> {out}/summary.md")
    print(f"[analyze] offline={len(off_rows)} online={len(on_rows)} paired={len(paired)} "
          f"paired_online={len(on_paired)} struct={len(st_rows)} figs={len(figs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
