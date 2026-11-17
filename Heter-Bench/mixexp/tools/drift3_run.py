"""drift3（构成漂移）驱动 + gate 检查。

用法::

    # gate：只跑 seed=42 / bg=10，检查 G1~G6
    python mixexp/tools/drift3_run.py --plan TaskData/drift3/bg_10.json --gate

    # 正式矩阵（gate 通过后再跑）
    python mixexp/tools/drift3_run.py --plan TaskData/drift3/bg_10.json --arms lru,heter_cache,opt
"""
import argparse, collections, json, os, pathlib, statistics, sys, time

_HERE = pathlib.Path(__file__).resolve()            # Heter-Bench/mixexp/tools/
REPO = pathlib.Path(os.environ.get("SIM_REPO", _HERE.parents[3]))  # 仓库根
os.chdir(REPO)
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "python"))

from sim.runner import SimConfig, Simulator
from sim.session_data import load_tasks
from sim.schedule import ServiceModel, TaskSpec
from sim.engine import EngineConfig
from sim.opt_map import use_map_from_run, write_use_map

PAGE = int(os.environ.get("PAGE", "8"))
CAP = int(os.environ.get("CAP", "198000"))


def build_tasks(plan_path):
    plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
    total = plan["total_s"]
    tasks = []
    for cls, meta in plan["classes"].items():
        for sl, spec in meta["per_slice"].items():
            start = next(s["start_s"] for s in plan["slices"] if s["name"] == sl)
            end = next(s["end_s"] for s in plan["slices"] if s["name"] == sl)
            segs = []
            if start > 0:
                segs.append({"until_s": start, "lambda_rate": 0.0})
            segs.append({"until_s": end, "lambda_rate": spec["lambda_rate"]})
            # 末段静默 ⇒ 窗口外的会话不发射（sim/schedule.py 的截尾语义）。
            # cooldown 的 end 恰好等于时间轴末端，也要补（否则 spill 到 4200s 之后）。
            segs.append({"until_s": max(total, end) + 1.0, "lambda_rate": 0.0})
            tasks.append({
                "name": f"{cls}@{sl}", "task_id": len(tasks),
                "data_path": spec["file"].replace("TaskData/", "TaskData/", 1),
                "max_record": -1, "shuffle": False, "seed": 42,
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


def run_once(sessions, specs, policy, use_map=None, engine=None, service=None):
    sim = Simulator(SimConfig(capacity=CAP, page_size=PAGE, eviction_policy=policy,
                              opt_use_map=use_map, seed=42, oversize="skip"))
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
    return res, info


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--arms", default="lru,heter_cache,opt")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    plan, tasks = build_tasks(a.plan)
    sessions = load_tasks(tasks, base_dir=str(REPO / "Heter-Bench"))
    specs = [TaskSpec.from_config(t) for t in tasks]
    engine = EngineConfig(schedule_policy="fcfs")
    service = service_model()
    print(f"[drift3] bg={plan['bg']}× sessions={len(sessions)} "
          f"requests={sum(s.n_turns for s in sessions)} tasks={len(tasks)}")
    if a.gate:
        return gate(plan, tasks, sessions, specs, engine, service)
    raise SystemExit("正式矩阵模式未实现（先过 gate）")


if __name__ == "__main__":
    raise SystemExit(main())
