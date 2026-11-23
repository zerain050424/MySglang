"""尾部 TTFT 压测负载：专门暴露「LRU 整链淘汰 → P90 TTFT 劣化」。

## 为什么要这条负载

真实 chat 负载里，LRU 与 heter 的 P90 TTFT 只差几 ms，因为"整链丢失"占比不高、
且与排队混在一起。要观测（或复现）"淘汰粒度导致的大 miss"这类缺陷，需要一条
**把机制单独放大**的负载。设计原则是三条互相独立的选择题：

1. **让 recency 与复用价值背离**。LRU 只有"最近被访问"一个信号。若让低价值流量
   在 recency 上永远比高价值流量更新，LRU 就会系统性地（而不是偶发地）挑错人。
       `hot` 类会话轮间只有 ~3s，持续刷新 recency；
       `warm` 类会话轮间 ~45s，recency 立刻变旧，但它必然回来。
   容量周转尺度被刻意放在两者之间（由 hot 的写入速率 + cold 洪峰决定）。
2. **让代价是"整链"而不是"尾巴"**。`warm` 的 prompt 逐轮增长到 ~12k，且**完全私有**
   （会话之间不共享任何前缀）。于是被淘汰就是 hit=0、need≈12k ⇒ 单次 TTFT ≈ 1.4s。
   如果共享前缀很长，命中会被共享部分兜住，缺陷就被掩盖了。
3. **让缺陷落在 P90 而不是 P99 上**。`warm` 请求占比刻意做到 ~19%（>10%），
   这样 P90 落在 warm 区间内；否则只是尾部几个离群点。

另外用 `cold` 提供容量压力，但让它的 TTFT 保持便宜：每个 cold 请求 8k token 中
7k 是**共享模板**，命中它几乎免费 ⇒ cold 的 p90 只有 200-300ms。这样 P90 的抬升
就只可能来自 warm 的整链重算，而不是"没有复用机会的流量"或排队积压。

## 自检三条（换参数时用）

* `cold` 的 p90 TTFT ≤ 300ms   ⇒ 没有排队积压，P90 不是被拥塞顶上去的；
* `warm` 的整链丢失率 ≥ 50%    ⇒ 缺陷确实被触发；
* LRU 的 P90 / OPT 的 P90 ≥ 3  ⇒ 有足够的可回收空间，值得当作 benchmark。

## 用法

    python build_tail_stress.py --cap 198000 --arms lru,heter_cache,opt
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import pathlib
import random
import sys

REPO = pathlib.Path(os.environ.get("SIM_REPO", "/root/sg-unicache"))
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from sim.core.config import SimConfig  # noqa: E402
from sim.drivers.engine import EngineConfig  # noqa: E402
from sim.opt_map import use_map_from_run, write_use_map  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.arrival import TaskSpec  # noqa: E402
from sim.workload.service import ServiceModel  # noqa: E402
from sim.workload.trace import Session, Turn  # noqa: E402

PAGE, RATE = 8, 8319.6
T_END = 1200.0


def _tk(rng, n):
    return array.array("q", [rng.randrange(1000, 60000) for _ in range(n)])


def build_sessions(n_hot, n_warm, n_cold, tpl, tail):
    """三类会话：hot（快回来、短链）、warm（慢回来、私有长链）、cold（一次性、共享模板）。"""
    rng = random.Random(7)
    out = []
    for i in range(n_hot):
        cur = _tk(rng, 800)
        turns = []
        for _k in range(8):
            resp = _tk(rng, 200)
            turns.append(Turn(t=0.0, prompt=array.array("q", cur), out=len(resp), response=resp))
            cur = array.array("q", list(cur) + list(resp) + list(_tk(rng, 400)))
        out.append(Session(session_id=f"hot-{i:03d}", turns=tuple(turns), task_name="hot"))
    for i in range(n_warm):
        cur = _tk(rng, 1500)
        turns = []
        for _k in range(8):
            resp = _tk(rng, 200)
            turns.append(Turn(t=0.0, prompt=array.array("q", cur), out=len(resp), response=resp))
            cur = array.array("q", list(cur) + list(resp) + list(_tk(rng, 1300)))
        out.append(Session(session_id=f"warm-{i:03d}", turns=tuple(turns), task_name="warm"))
    tmpl = _tk(random.Random(99), tpl)
    for i in range(n_cold):
        prompt = array.array("q", list(tmpl) + list(_tk(rng, tail)))
        out.append(Session(session_id=f"cold-{i:04d}",
                           turns=(Turn(t=0.0, prompt=prompt, out=100),), task_name="cold"))
    return out


def build_specs(a):
    # 轮间空窗由 lognormal 的 mu 控制；sigma 取小值让间隔接近确定值。
    return [
        TaskSpec(name="hot", lambda_rate=a.n_hot / 600.0, mu=math.log(a.hot_gap),
                 sigma=0.05, seed=42, task_id=0),
        TaskSpec(name="warm", lambda_rate=a.n_warm / 600.0, mu=math.log(a.warm_gap),
                 sigma=0.05, seed=43, task_id=1),
        TaskSpec(name="cold", lambda_rate=a.n_cold / T_END, mu=0.0, sigma=0.0,
                 seed=44, task_id=2),
    ]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))]


def run_arm(sessions, specs, policy, cap, use_map=None, policy_config=None,
            chain_policy=None, regret_gate=None):
    sim = Simulator(SimConfig(capacity=cap, page_size=PAGE, eviction_policy=policy,
                              opt_use_map=use_map, policy_config=policy_config,
                              chain_policy=chain_policy, regret_gate=regret_gate,
                              seed=0, oversize="skip"))
    svc = ServiceModel(prefill_base_s=0.0, prefill_tokens_per_s=RATE, decode_tokens_per_s=40.0)
    res = sim.run_engine(sessions, specs, scheduler=EngineConfig(schedule_policy="fcfs"),
                         service=svc)
    st = sim.cache.eviction_strategy
    run_arm.last_gate_stats = (
        st.regret_gate.stats() if getattr(st, "regret_gate", None) else None
    )
    rows = []
    for r in res.records:
        if r.skipped or r.turn < 0:
            continue
        due = r.arrival_t
        queue = max(0.0, r.t - due) if due is not None else 0.0
        miss = r.prompt_len - r.hit
        rows.append({"task": r.task_name, "plen": r.prompt_len, "hit": r.hit,
                     "miss": miss, "ttft": queue + miss / RATE, "whole": r.hit == 0})
    return res, rows


def report(tag, res, rows):
    tt = [x["ttft"] for x in rows]
    out = {"hit": round(res.hit_rate, 4),
           "ttft_ms": {"p50": round(pct(tt, 50) * 1000, 1),
                       "p90": round(pct(tt, 90) * 1000, 1),
                       "p95": round(pct(tt, 95) * 1000, 1),
                       "p99": round(pct(tt, 99) * 1000, 1)}}
    print(f"\n=== {tag}  hit={out['hit']:.4f}  requests={len(rows)}  "
          f"TTFT p50 {out['ttft_ms']['p50']:.0f}  "
          f"**p90 {out['ttft_ms']['p90']:.0f}**  p95 {out['ttft_ms']['p95']:.0f}  "
          f"p99 {out['ttft_ms']['p99']:.0f} (ms)")
    for task in ("hot", "warm", "cold"):
        sel = [x for x in rows if x["task"] == task]
        if not sel:
            continue
        hit = sum(x["hit"] for x in sel) / max(1, sum(x["plen"] for x in sel))
        tt2 = [x["ttft"] for x in sel]
        cell = {"share_pct": round(100 * len(sel) / len(rows), 1), "hit": round(hit, 3),
                "whole_loss_pct": round(100 * sum(1 for x in sel if x["whole"]) / len(sel), 1),
                "p50_ms": round(pct(tt2, 50) * 1000), "p90_ms": round(pct(tt2, 90) * 1000),
                "mean_miss_tok": round(sum(x["miss"] for x in sel) / len(sel))}
        out[task] = cell
        print(f"  {task:5s} {cell['share_pct']:5.1f}%  hit={hit:5.3f}  "
              f"整链丢失 {cell['whole_loss_pct']:5.1f}%  TTFT p50 {cell['p50_ms']:5.0f} "
              f"p90 {cell['p90_ms']:6.0f} ms  mean miss {cell['mean_miss_tok']:6.0f} tok")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=198_000)
    ap.add_argument("--n-hot", type=int, default=60)
    ap.add_argument("--n-warm", type=int, default=40)
    ap.add_argument("--n-cold", type=int, default=900)
    ap.add_argument("--tpl", type=int, default=7000, help="cold 的共享模板长度")
    ap.add_argument("--tail", type=int, default=1000, help="cold 的私有尾巴长度")
    ap.add_argument("--hot-gap", type=float, default=3.0)
    ap.add_argument("--warm-gap", type=float, default=45.0)
    ap.add_argument("--arms", default="lru,heter_cache,opt")
    ap.add_argument("--floor", type=int, default=4096,
                    help="chain_lru：浅段保护下限（token）")
    ap.add_argument("--weight", type=float, default=0.01,
                    help="chain_lru：缺口罚斜率（秒/token）")
    ap.add_argument("--rg-window", type=float, default=120.0,
                    help="heter_rgate：后悔滑窗 W（秒）")
    ap.add_argument("--rg-k", type=float, default=1.0,
                    help="heter_rgate：弃权阈值 = k × 已采样中位数")
    ap.add_argument("--rg-rmin", type=float, default=0.5,
                    help="heter_rgate：R 绝对下限")
    ap.add_argument("--sla-chat", type=float, default=1.0,
                    help="chain_lru：chat 锚罚乘数（类别公平旋钮）")
    ap.add_argument("--sla-agentic", type=float, default=1.0)
    ap.add_argument("--sla-struct", type=float, default=1.0)
    ap.add_argument("--pf", action="store_true",
                    help="heter_rgate：档 B——全弃权兑底换全局叶子定价")
    ap.add_argument("--pf-floor", type=int, default=8192, help="档 B：锚下限先验")
    ap.add_argument("--pf-weight", type=float, default=0.02, help="档 B：罚斜率")
    ap.add_argument("--pf-beta", type=float, default=1.0, help="档 B：R 归因加权系数")
    ap.add_argument("--gate", default="rerode",
                    choices=["none", "rerode", "multiturn"],
                    help="chain_lru：浅段保护证据门（rerode=重读过；multiturn=再加会话回来过）")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    sessions = build_sessions(a.n_hot, a.n_warm, a.n_cold, a.tpl, a.tail)
    specs = build_specs(a)
    print(f"[tail-stress] cap={a.cap} page={PAGE} | hot={a.n_hot}×8 轮(gap {a.hot_gap}s) "
          f"warm={a.n_warm}×8 轮(gap {a.warm_gap}s, 私有 12k) "
          f"cold={a.n_cold}×1 ({a.tpl} 共享 + {a.tail} 私有)")
    out = {}
    chain_cfg = {
        "floor_tokens": a.floor, "weight": a.weight, "gate": a.gate,
        "sla": {"chat": a.sla_chat, "agentic": a.sla_agentic,
                "struct": a.sla_struct},
    }
    heter_chain_cfg = {
        "enabled": True, "floor_tokens": a.floor, "weight": a.weight,
        "min_gaps": 1,
    }
    rgate_cfg = {
        "enabled": True, "window_s": a.rg_window, "k": a.rg_k, "r_min": a.rg_rmin,
        "priced_fallback": a.pf, "floor_tokens": a.pf_floor,
        "weight": a.pf_weight, "beta": a.pf_beta,
    }
    for arm in [x for x in a.arms.split(",") if x]:
        if arm != "opt":
            cfg = chain_cfg if arm == "chain_lru" else None
            cp = heter_chain_cfg if arm in ("heter_chain", "heter_full") else None
            rgc = rgate_cfg if arm in ("heter_rgate", "heter_full") else None
            real = "heter_cache" if arm in ("heter_chain", "heter_rgate", "heter_full") else arm
            res, rows = run_arm(sessions, specs, real, a.cap, policy_config=cfg,
                                chain_policy=cp, regret_gate=rgc)
            if arm == "heter_rgate" and run_arm.last_gate_stats:
                print("  [rgate]", run_arm.last_gate_stats, flush=True)
            out[arm] = report(arm, res, rows)
            if arm == "lru":
                base_lru = res
            continue
        src, live = base_lru, [s for s in sessions
                               if (s.session_id, 0) in base_lru.dispatched]
        for it in range(1, 5):
            write_use_map("/tmp/tail_stress_use_map.json",
                          use_map_from_run(src, live, page_size=PAGE))
            res, rows = run_arm(sessions, specs, "opt", a.cap,
                                use_map="/tmp/tail_stress_use_map.json")
            common = set(src.dispatched) & set(res.dispatched)
            ndiff = sum(1 for k in common
                        if abs(src.dispatched[k] - res.dispatched[k]) > 1e-9)
            if ndiff == 0:
                break
            src = res
        print(f"  [opt] 时间轴固定点迭代 {it} 轮，残差 {ndiff}/{len(common)}")
        out["opt"] = report("opt (Belady 上界)", res, rows)
        out["opt"]["time_axis_residual"] = ndiff

    lru_p90 = out["lru"]["ttft_ms"]["p90"]
    opt_p90 = out.get("opt", {}).get("ttft_ms", {}).get("p90")
    cold_p90 = out["lru"]["cold"]["p90_ms"]
    warm_whole = out["lru"]["warm"]["whole_loss_pct"]
    print("\n[自检] cold p90 "
          f"{cold_p90:.0f}ms（≤300 ⇒ 无排队积压）｜warm 整链丢失 {warm_whole:.1f}%（≥50 ⇒ 缺陷触发）"
          + (f"｜LRU/OPT P90 = {lru_p90 / opt_p90:.1f}×（≥3 ⇒ 有可回收空间）" if opt_p90 else ""))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
