"""drift3（构成漂移）驱动 + gate 检查。

用法::

    # gate：只跑 seed=42 / bg=10，检查 G1~G6
    python mixexp/scripts/sim/drift3_run.py --plan TaskData/drift3/bg_10.json --gate

    # 正式矩阵（gate 通过后再跑）
    python mixexp/scripts/sim/drift3_run.py --plan TaskData/drift3/bg_10.json --arms lru,heter_cache,opt
"""
import argparse, collections, dataclasses, json, os, pathlib, statistics, sys, time

_HERE = pathlib.Path(__file__).resolve()            # Heter-Bench/mixexp/scripts/sim/
REPO = pathlib.Path(os.environ.get("SIM_REPO", _HERE.parents[4]))  # 仓库根
os.chdir(REPO)
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))

from sim.core.config import SimConfig
from sim.drivers.engine import EngineConfig
from sim.analysis.opt_map import use_map_from_run, write_use_map
from sim.simulator import Simulator
from sim.workload.service import ServiceModel
from sim.workload.arrival import TaskSpec
from sim.workload.session_data import load_tasks

PAGE = int(os.environ.get("PAGE", "8"))
CAP = int(os.environ.get("CAP", "198000"))


REVERSE_WINDOW = {"warmup": "cooldown", "cooldown": "warmup", "p1": "p3", "p3": "p1", "p2": "p2"}


def build_tasks(plan_path, *, seed=42, reverse=False, only_slices=None, fixed_lam=None):
    """把 plan 展开成 sim 的 tasks 列表。

    Args:
        seed: 会话加载/到达的种子。
        reverse: 相位反转对照——数据切片与其 λ 一起搬到反相的窗口（窗口等长，
            构成目标不变），用来检验"先 struct 还是先 agentic"的路径依赖。
        only_slices: 只保留这些切片的（stationary 对照用，单切片铺满整条时间轴）。
        fixed_lam: ``{cls: lam}``，给 stationary 对照用常数 λ 覆盖 plan 的 λ。
    """
    plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
    total = plan["total_s"]
    tasks = []
    for cls, meta in plan["classes"].items():
        for sl, spec in meta["per_slice"].items():
            if only_slices is not None and sl not in only_slices:
                continue
            win = REVERSE_WINDOW[sl] if reverse else sl
            start = next(s["start_s"] for s in plan["slices"] if s["name"] == win)
            end = next(s["end_s"] for s in plan["slices"] if s["name"] == win)
            lam = fixed_lam[cls] if fixed_lam else spec["lambda_rate"]
            segs = []
            if only_slices is not None:
                # stationary：只用这一片数据、从 0 铺到窗口末端，构成与目标相位
                # 一致但**不切换**（用来把"漂移代价"从"构成效应"里剥出来）
                secs = next(s["seconds"] for s in plan["slices"] if s["name"] == sl)
                start = 0.0
                end = secs
                segs.append({"until_s": end, "lambda_rate": lam})
            else:
                if start > 0:
                    segs.append({"until_s": start, "lambda_rate": 0.0})
                segs.append({"until_s": end, "lambda_rate": lam})
            # 末段静默 ⇒ 窗口外的会话不发射（sim/schedule.py 的截尾语义）。
            # cooldown 的 end 恰好等于时间轴末端，也要补（否则 spill 到 4200s 之后）。
            segs.append({"until_s": max(total, end) + 1.0, "lambda_rate": 0.0})
            tasks.append({
                "name": f"{cls}@{sl}", "task_id": len(tasks),
                "data_path": spec["file"].replace("TaskData/", "TaskData/", 1),
                "max_record": -1, "shuffle": False, "seed": seed,
                "mu": meta["mu"], "sigma": meta["sigma"],
                "arrival": {"mode": "segments", "segments": segs},
            })
    return plan, tasks


def service_model():
    calib = json.loads((REPO / "Uni-Bench/result/tailshot/calib_lru.json").read_text())
    return ServiceModel(
        prefill_base_s=max(0.0, float(calib.get("ttft_intercept_ms", 0.0)) / 1000.0),
        prefill_tokens_per_s=float(calib["prefill_rate_tok_per_s"]),
        decode_tokens_per_s=40.0,
    )


def run_once(sessions, specs, policy, use_map=None, engine=None, service=None,
             regret_gate=None, chain_policy=None, records_path=None):
    sim = Simulator(SimConfig(capacity=CAP, page_size=PAGE, eviction_policy=policy,
                              opt_use_map=use_map, seed=42, oversize="skip",
                              regret_gate=regret_gate, chain_policy=chain_policy))
    t0 = time.time()
    res = sim.run_engine(sessions, specs, scheduler=engine, service=service,
                         arrival_scale=1.0, wait_scale=1.0)
    st = sim.cache.eviction_strategy
    emit = (res.summary or {}).get("emit") or {}
    info = {"hit_rate": res.hit_rate, "requests": len(res.records),
            "map_lookups": getattr(st, "_lookups", None),
            "map_misses": getattr(st, "_misses", None),
            "steps": emit.get("steps"), "prefill_steps": emit.get("prefill_steps"),
            "send_lag_p99_ms": emit.get("send_lag_p99_ms"),
            "wall_s": round(time.time() - t0, 1)}
    if getattr(st, "regret_gate", None) is not None:
        info["regret_gate_stats"] = st.regret_gate.stats()
    if records_path:
        _dump_records(res, pathlib.Path(records_path))
    return res, info


def _dump_records(res, out_path: pathlib.Path) -> None:
    """逐请求记录落盘（发送端统计组件的输入，零探针：只读引擎已有 metric）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in res.records:
            d = dataclasses.asdict(r)
            d["dispatch_t"] = res.dispatched.get((r.session_id, r.turn))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


_HG_FLOOR = 8192   # 锚下限先验 / pf floor（与 tail-stress 运行器同口径）
_HG_WEIGHT = 0.02  # 缺口罚斜率


def _rg_cfg(a) -> dict:
    """后悔门 + PricedFallback（best_configs.md 档 A/B 推荐口径）。"""
    return {
        "enabled": True, "k": a.rg_k, "r_min": a.rg_rmin, "window_s": 120.0,
        "priced_fallback": True, "floor_tokens": _HG_FLOOR,
        "weight": _HG_WEIGHT, "beta": a.pf_beta,
    }


def _heter_chain_cfg() -> dict:
    """heter 内队内 ChainEvict（ChainParams 生产默认）。"""
    return {
        "enabled": True, "floor_tokens": _HG_FLOOR, "weight": _HG_WEIGHT,
        "min_gaps": 1, "tail_first": True, "expire_anchors": True,
    }


def slice_of(plan, t):
    for s in plan["slices"]:
        if s["start_s"] <= t < s["end_s"]:
            return s["name"]
    return None


def gate(plan, tasks, sessions, specs, engine, service):
    fails, rows = [], []

    def check(name, ok, detail):
        rows.append((name, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(f"{name}: {detail}")

    # ---- 一次 heter 全量 run（G1/G3/G6 的载体）----
    het, het_info = run_once(sessions, specs, "heter_cache", engine=engine, service=service)
    by_slice_class = collections.defaultdict(lambda: collections.defaultdict(int))
    evicted = collections.defaultdict(int)
    used_peak = collections.defaultdict(int)
    types = collections.defaultdict(collections.Counter)
    for r in het.records:
        sl = slice_of(plan, r.t)
        if sl is None:
            continue
        cls = r.task_name.split("@")[0]
        by_slice_class[sl][cls] += r.prompt_len
        evicted[sl] += r.evicted
        used_peak[sl] = max(used_peak[sl], r.used)
        types[cls][r.session_type] += 1

    # G1 构成：按**会话 cohort**（会话第一轮的派发时刻）归相位。
    # 之所以不用「请求自己的时刻」：多轮会话必然跨相位（chat 单会话 ≈500s、
    # agentic ≈200s），到达侧的构成会被 carry-over 拉平——那是要**测量**的现象，
    # 不是控制失败。控制判据看 cohort，arrival 侧只作诊断打印。
    cohort = collections.defaultdict(collections.Counter)
    nominal = collections.defaultdict(dict)
    for cls, meta in plan["classes"].items():
        for sl, spec in meta["per_slice"].items():
            nominal[sl][cls] = spec["target_share"]
    n_sess = collections.defaultdict(collections.Counter)
    for s_ in sessions:
        cls, task_sl = s_.task_name.split("@")
        t0 = het.dispatched.get((s_.session_id, 0))
        if t0 is None:
            continue
        sl = slice_of(plan, t0) or task_sl      # 真正落在哪个相位
        cohort[sl][cls] += sum(len(t.prompt) for t in s_.turns)
        n_sess[sl][cls] += 1
    # cooldown 只用于观察「退出漂移后的回落」，300s 窗口里 agentic 期望只有 ~4 个
    # 会话，构成份额本身就有 ±8pp 的抽样噪声 → 控制判据只管 warmup + 三个主导相位。
    gate_slices = {"warmup", "p1", "p2", "p3"}
    worst, detail = 0.0, []
    for sl, per in cohort.items():
        if sl not in gate_slices:
            diag = " ".join(f"{c}={v/(sum(per.values()) or 1):.0%}" for c, v in sorted(per.items()))
            detail.append(f"[diag] {sl}: {diag}")
            continue
        tot = sum(per.values()) or 1
        denom = sum(nominal[sl].values()) or 1
        for cls, tok in sorted(per.items()):
            got = tok / tot
            want = nominal[sl][cls] / denom
            # 泊松抽样噪声：会话数少时份额本身就有 ±share/sqrt(n) 的抖动
            sigma = want / max(1.0, (n_sess[sl][cls] ** 0.5))
            tol = max(0.03, 2.0 * sigma)
            dev = abs(got - want)
            worst = max(worst, (dev - tol) * 100)
            detail.append(f"{sl}/{cls}:{got:.0%}vs{want:.0%}(n={n_sess[sl][cls]},tol={tol*100:.0f}pp)")
    check("G1 构成(cohort)", worst <= 0.0,
          f"超差 {worst:.1f}pp | " + " ".join(detail))

    # G2 注入速率（每 slice 每类的首轮派发数 / 秒数 vs λ，±5%）
    rate_bad = []
    # 用会话表统计：每个 (cls, sl) 的首轮时刻
    s_by_name = {s.session_id: s for s in sessions}
    counts = collections.Counter()
    for (sid, idx), t in het.dispatched.items():
        if idx != 0:
            continue
        sess = s_by_name.get(sid)
        if sess is None:
            continue
        cls, sl = sess.task_name.split("@")
        if slice_of(plan, t) == sl:
            counts[(cls, sl)] += 1
    detail2 = []
    for cls, meta in plan["classes"].items():
        for sl, spec in meta["per_slice"].items():
            secs = next(s["seconds"] for s in plan["slices"] if s["name"] == sl)
            want = spec["lambda_rate"] * secs
            got = counts[(cls, sl)]
            # 泊松抽样：允许 3σ = 3*sqrt(want) 的绝对偏差（大 λ 时≈±5%，
            # 小 λ 如 agentic 每相位 3~28 个会话时必然更宽）
            sigma = max(1.0, want ** 0.5)
            dev = abs(got - want)
            detail2.append(f"{cls}@{sl}:{got}/{want:.0f}({dev/sigma:.1f}σ)")
            if dev > 3.0 * sigma:
                rate_bad.append(f"{cls}@{sl} {dev:.0f}vs{want:.0f} ({dev/sigma:.1f}σ)")
    check("G2 注入速率(3σ)", not rate_bad, "; ".join(rate_bad) or "全部在 3σ 内")

    # G3 池饱和（每 slice 有淘汰 + 峰值到 capacity）
    g3_bad = [f"{sl}(ev={evicted[sl]},peak={used_peak[sl]})" for sl in by_slice_class
              if evicted[sl] <= 0 or used_peak[sl] < CAP * 0.99]
    check("G3 池饱和", not g3_bad, "; ".join(g3_bad) or "各 slice 均有淘汰且到容量")

    # G4 确定性（lru 跑两次 fingerprint 相同）
    a, _ = run_once(sessions, specs, "lru", engine=engine, service=service)
    b, _ = run_once(sessions, specs, "lru", engine=engine, service=service)
    check("G4 确定性", a.fingerprint() == b.fingerprint(),
          f"{a.fingerprint()} vs {b.fingerprint()}")

    # G5 OPT 时间轴
    # 只对**真的被发射过**的会话建表：末段静默会截掉余量会话（文件按 2× 背景档
    # 留了 1.2 倍余量），它们没有记录、也不该进 use map。
    live = [s for s in sessions if (s.session_id, 0) in a.dispatched]
    it_bad = []
    src = a
    for it in range(1, 4):
        p = f"/tmp/drift3_gate_use_map_{it}.json"
        write_use_map(p, use_map_from_run(src, live, page_size=PAGE))
        opt, info = run_once(sessions, specs, "opt", use_map=p, engine=engine, service=service)
        common = set(src.dispatched) & set(opt.dispatched)
        ndiff = sum(1 for k in common if abs(src.dispatched[k] - opt.dispatched[k]) > 1e-9)
        if ndiff == 0:
            break
        src = opt
    check("G5 OPT 时间轴", ndiff == 0 and info["map_misses"] == 0,
          f"iter{it} 残差 {ndiff}/{len(common)}，map_miss={info['map_misses']}/{info['map_lookups']}")

    # G6 分类有效性（目标队列占比 ≥85%）
    want = {"trans": "trans", "struct": "struct", "chat": "chat", "agentic": "agentic"}
    multi_turn = {"chat", "agentic"}      # 这些类的首轮请求按定义还没被分类
    g6 = []
    for cls, target in want.items():
        c = types[cls]
        if cls in multi_turn:
            c = collections.Counter()
            for r in het.records:
                if r.task_name.split("@")[0] == cls and r.turn >= 1:
                    c[r.session_type] += 1
        tot = sum(c.values()) or 1
        share = c.get(target, 0) / tot
        g6.append(f"{cls}->{target}:{share:.0%}")
        if share < 0.85:
            fails.append(f"G6 {cls} 只有 {share:.0%} 落在 {target}")
    check("G6 分类有效性(≥85%)", not any("G6 " in f for f in fails), " ".join(g6))

    print(f"\n=== gate (bg={plan['bg']}×, cap={CAP}, page={PAGE}) ===")
    for name, status, detail in rows:
        print(f"  [{status}] {name:22s} {detail}")
    print(f"  heter hit={het_info['hit_rate']:.4f} reqs={het_info['requests']} "
          f"steps={het_info['steps']} prefill={het_info['prefill_steps']} wall={het_info['wall_s']}s")
    return 0 if not fails else 1




# ---------------------------------------------------------------------------
# 正式矩阵
# ---------------------------------------------------------------------------
def windows(plan):
    return [(s["name"], s["start_s"], s["end_s"]) for s in plan["slices"]]


def phase_of(plan, t):
    for name, lo, hi in windows(plan):
        if lo <= t < hi:
            return name
    return None


def summarize_phases(plan, res):
    """按相位聚合：分类型/分类命中、淘汰量、稳态与切换窗口、T90 恢复时间。"""

    out = {}
    recs = res.records
    for name, lo, hi in windows(plan):
        sel = [r for r in recs if lo <= r.t < hi]
        if not sel:
            continue
        tot_prompt = sum(r.prompt_len for r in sel)
        tot_hit = sum(r.hit for r in sel)
        per_cls = {}
        for r in sel:
            cls = r.task_name.split("@")[0]
            d = per_cls.setdefault(cls, [0, 0])
            d[0] += r.hit
            d[1] += r.prompt_len
        per_type = {}
        for r in sel:
            d = per_type.setdefault(r.session_type, [0, 0])
            d[0] += r.hit
            d[1] += r.prompt_len
        head = [r for r in sel if r.t < lo + 300.0]
        tail = [r for r in sel if r.t >= hi - 300.0]
        hr = lambda rows: (sum(x.hit for x in rows) / max(1, sum(x.prompt_len for x in rows)))
        steady = hr(tail)
        # 主导类（份额最大）的 T90：60s 滚动窗口首次进入 steady-2pp 且连续 3 窗
        dom = max(per_cls.items(), key=lambda kv: kv[1][1])[0]
        dom_bins, n_bins = {}, max(1, int((hi - lo) // 60))
        for r in sel:
            if r.task_name.split("@")[0] != dom:
                continue
            b = int((r.t - lo) // 60)
            d = dom_bins.setdefault(b, [0, 0])
            d[0] += r.hit
            d[1] += r.prompt_len
        dom_steady_rows = [r for r in sel
                           if r.task_name.split("@")[0] == dom and r.t >= hi - 300.0]
        dom_steady = hr(dom_steady_rows) if dom_steady_rows else 0.0
        bins = [dom_bins.get(b, [0, 0]) for b in range(n_bins)]
        rates = [h / p if p else None for h, p in bins]
        t90 = None
        for i in range(max(0, n_bins - 2)):
            seq = rates[i:i + 3]
            if all(x is not None and x >= dom_steady - 0.02 for x in seq):
                t90 = i * 60
                break
        out[name] = {
            "prompt_tokens": tot_prompt,
            "hit_tokens": tot_hit,
            "hit_rate": round(tot_hit / max(1, tot_prompt), 4),
            "per_class": {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(per_cls.items())},
            "per_class_tokens": {k: v[1] for k, v in sorted(per_cls.items())},
            "per_type": {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(per_type.items())},
            "evicted_tokens": sum(r.evicted for r in sel),
            "steady_hit_rate": round(steady, 4),
            "switch_cost_pp": round((hr(head) - steady) * 100, 2) if head else None,
            "dominant_class": dom,
            "t90_s": t90,
            # 300s 分箱命中率：相位内是否还在爬升（判"瞬态 vs 永久损失"的直接证据）
            "bins_300s": [
                [int(b * 300), round(sum(x.hit for x in sel
                                         if int((x.t - lo) // 300) == b)
                                     / max(1, sum(x.prompt_len for x in sel
                                                  if int((x.t - lo) // 300) == b)), 4)]
                for b in range(int((hi - lo) // 300))
            ],
        }
    return out


def plan_of_phase(plan, phase):
    return {cls: meta["per_slice"][phase]["lambda_rate"]
            for cls, meta in plan["classes"].items()}


def matrix(a, engine, service):
    plan_paths = a.plan if isinstance(a.plan, list) else [a.plan]
    out_root = pathlib.Path(a.out or "/root/sim_runs/drift3")
    out_root.mkdir(parents=True, exist_ok=True)
    result = {}

    for plan_path in plan_paths:
        plan, tasks = build_tasks(plan_path, seed=a.seed)
        sessions = load_tasks(tasks, base_dir=str(REPO / "Heter-Bench"))
        specs = [TaskSpec.from_config(t) for t in tasks]
        bg = plan["bg"]
        print(f"\n[drift3] bg={bg}× seed={a.seed} sessions={len(sessions)} "
              f"requests={sum(s.n_turns for s in sessions)}", flush=True)

        arm_rows, base_res = {}, None
        live = None
        for arm in a.arms.split(","):
            if arm != "opt":
                # 臂 → (真实策略, regret_gate, chain_policy) 映射；未列出的臂
                # （lru/lfu/fifo/slru/heter_cache）按原名直通，不带额外配置。
                policy, rg, cp = arm, None, None
                if arm == "heter_rgate":
                    # 后悔门 + ReuseScoreEvict（reuse 默认开）
                    policy, rg = "heter_cache", _rg_cfg(a)
                elif arm == "heter_rgate_chain":
                    # 后悔门 + 队内 ChainEvict
                    policy, rg, cp = "heter_cache", _rg_cfg(a), _heter_chain_cfg()
                rec_path = ""
                if a.records_dir:
                    rec_path = str(pathlib.Path(a.records_dir)
                                   / f"bg_{bg:g}_s{a.seed}" / f"{arm}.records.jsonl")
                res, info = run_once(sessions, specs, policy, engine=engine,
                                     service=service, regret_gate=rg,
                                     chain_policy=cp, records_path=rec_path)
                if base_res is None:
                    base_res, live = res, [s for s in sessions if (s.session_id, 0) in res.dispatched]
                arm_rows[arm] = {"info": info, "phases": summarize_phases(plan, res)}
                print(f"  {arm:12s} hit={info['hit_rate']:.4f} "
                      f"phases=" + " ".join(f"{k}={v['hit_rate']:.3f}"
                                            for k, v in arm_rows[arm]["phases"].items()), flush=True)
                continue
            # OPT：时间轴取自 lru（时间轴对策略敏感 → 迭代到固定点）
            src = base_res
            for it in range(1, 4):
                p = str(out_root / f"use_map_{bg:g}x_s{a.seed}_it{it}.json")
                write_use_map(p, use_map_from_run(src, live, page_size=PAGE))
                res, info = run_once(sessions, specs, "opt", use_map=p,
                                     engine=engine, service=service)
                common = set(src.dispatched) & set(res.dispatched)
                ndiff = sum(1 for k in common
                            if abs(src.dispatched[k] - res.dispatched[k]) > 1e-9)
                if ndiff == 0:
                    break
                src = res
            if a.records_dir:
                _dump_records(
                    res,
                    pathlib.Path(a.records_dir) / f"bg_{bg:g}_s{a.seed}" / "opt.records.jsonl",
                )
            arm_rows["opt"] = {"info": dict(info, time_axis_iter=it, residual=ndiff),
                               "phases": summarize_phases(plan, res)}
            print(f"  {'opt':12s} hit={info['hit_rate']:.4f} iter={it} 残差={ndiff} "
                  f"map_miss={info['map_misses']}/{info['map_lookups']}", flush=True)

        cmp_ = {}
        for ph in arm_rows.get("lru", {}).get("phases", {}):
            cmp_[ph] = {
                arm: arm_rows[arm]["phases"].get(ph, {}).get("hit_rate")
                for arm in arm_rows
            }
            if "opt" in arm_rows and arm_rows[arm]["phases"].get(ph):
                cmp_[ph]["opt_minus_heter"] = round(
                    (arm_rows["opt"]["phases"][ph]["hit_rate"]
                     - arm_rows["heter_cache"]["phases"][ph]["hit_rate"]) * 100, 2)

        entry = {"bg": bg, "seed": a.seed, "arms": arm_rows, "phase_hit": cmp_}

        # ---- stationary 对照：每个主导相位单独一条恒定构成 run ----
        if a.stationary:
            entry["stationary"] = {}
            for ph in ("p1", "p2", "p3"):
                st_tasks = build_tasks(plan_path, seed=a.seed, only_slices=(ph,),
                                       fixed_lam=plan_of_phase(plan, ph))[1]
                st_sessions = load_tasks(st_tasks, base_dir=str(REPO / "Heter-Bench"))
                st_specs = [TaskSpec.from_config(t) for t in st_tasks]
                row = {}
                for arm in ("lru", "heter_cache"):
                    r2, i2 = run_once(st_sessions, st_specs, arm, engine=engine, service=service)
                    row[arm] = round(i2["hit_rate"], 4)
                entry["stationary"][ph] = row
                print(f"  stationary {ph}: {row}", flush=True)

        # ---- 逆序对照（agentic → chat → struct）----
        if a.reverse:
            rv_plan, rv_tasks = build_tasks(plan_path, seed=a.seed, reverse=True)
            rv_sessions = load_tasks(rv_tasks, base_dir=str(REPO / "Heter-Bench"))
            rv_specs = [TaskSpec.from_config(t) for t in rv_tasks]
            entry["reverse"] = {}
            for arm in ("lru", "heter_cache"):
                r3, i3 = run_once(rv_sessions, rv_specs, arm, engine=engine, service=service)
                entry["reverse"][arm] = {"hit_rate": round(i3["hit_rate"], 4),
                                         "phases": summarize_phases(rv_plan, r3)}
                print(f"  reverse {arm}: hit={i3['hit_rate']:.4f}", flush=True)

        result[f"bg_{bg:g}x_s{a.seed}"] = entry
        (out_root / f"bg_{bg:g}x_s{a.seed}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_root / f"all_s{a.seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[drift3] done -> {out_root}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, nargs="+")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--arms", default="lru,heter_cache,opt")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stationary", action="store_true", help="跑稳态构成对照")
    ap.add_argument("--reverse", action="store_true", help="跑逆序路径依赖对照")
    ap.add_argument("--rg-k", type=float, default=0.6)
    ap.add_argument("--rg-rmin", type=float, default=0.3)
    ap.add_argument("--pf-beta", type=float, default=1.0)
    ap.add_argument("--records-dir", default=str(REPO / "Heter-Bench" / "mixexp" / "result" / "sim_drift3_records"),
                    help="逐请求 records.jsonl 落盘目录（空字符串关闭）")
    a = ap.parse_args(argv)

    engine = EngineConfig(schedule_policy="fcfs")
    service = service_model()
    if not a.gate:
        return matrix(a, engine, service)

    plan, tasks = build_tasks(a.plan[0], seed=a.seed)
    sessions = load_tasks(tasks, base_dir=str(REPO / "Heter-Bench"))
    specs = [TaskSpec.from_config(t) for t in tasks]
    print(f"[drift3] bg={plan['bg']}× sessions={len(sessions)} "
          f"requests={sum(s.n_turns for s in sessions)} tasks={len(tasks)}")
    return gate(plan, tasks, sessions, specs, engine, service)


if __name__ == "__main__":
    raise SystemExit(main())
