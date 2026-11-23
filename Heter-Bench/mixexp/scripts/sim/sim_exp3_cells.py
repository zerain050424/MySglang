#!/usr/bin/env python3
"""exp3 四 cell（scan / struct / chat / agentic）引擎仿真对比驱动。

基于 ``Simulator.run_engine``（FCFS 成批 prefill），复用 ``TaskData/exp3/<cell>/tasks.yaml``
运行多策略横向对比，涵盖 LRU、HeterCache、chain_lru 全局版、档 B (RegretGate +
PricedFallback) 以及固定点迭代的 OPT 上界。

用法::

    # 运行 chat cell 的推荐对比组合
    PYTHONPATH=.:python python3 Heter-Bench/mixexp/scripts/sim/sim_exp3_cells.py \\
        --cells chat --arms lru,heter_cache,chain_lru,heter_rgate

    # 运行全四 cell（含 OPT 上界）并落盘 JSON
    PYTHONPATH=.:python python3 Heter-Bench/mixexp/scripts/sim/sim_exp3_cells.py \\
        --cells scan,struct,chat,agentic \\
        --arms lru,heter_cache,chain_lru,heter_rgate,opt \\
        --out Heter-Bench/mixexp/result/sim_exp3_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve()
REPO = pathlib.Path(os.environ.get("SIM_REPO", _HERE.parents[4]))
BENCH_DIR = REPO / "Heter-Bench"
os.chdir(REPO)
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim.analysis.opt_map import use_map_from_run, write_use_map  # noqa: E402
from sim.core.config import SimConfig  # noqa: E402
from sim.drivers.engine import EngineConfig  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.service import ServiceModel  # noqa: E402
from sim.workload.session_data import load_tasks  # noqa: E402


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]


def _service_model() -> ServiceModel:
    calib_path = BENCH_DIR / "result" / "tailshot" / "calib_lru.json"
    if calib_path.is_file():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        return ServiceModel(
            prefill_base_s=calib.get("prefill_base_s", 0.0),
            prefill_tokens_per_s=calib.get("prefill_tokens_per_s", 8319.6),
            decode_tokens_per_s=calib.get("decode_tokens_per_s", 40.0),
        )
    return ServiceModel(prefill_base_s=0.0, prefill_tokens_per_s=8319.6, decode_tokens_per_s=40.0)


def _run_once(
    sessions,
    specs,
    *,
    cap: int,
    page: int,
    seed: int,
    svc: ServiceModel,
    eviction_policy: str,
    policy_config=None,
    chain_policy=None,
    regret_gate=None,
    opt_use_map: str | None = None,
):
    cfg = SimConfig(
        capacity=cap,
        page_size=page,
        eviction_policy=eviction_policy,
        policy_config=policy_config,
        chain_policy=chain_policy,
        regret_gate=regret_gate,
        opt_use_map=opt_use_map,
        seed=seed,
        oversize="skip",
    )
    sim = Simulator(cfg)
    return sim.run_engine(
        sessions,
        specs,
        scheduler=EngineConfig(schedule_policy="fcfs", max_prefill_tokens=16384),
        service=svc,
    )


def _summarize(res) -> dict:
    ttfts = [
        (r.finish_t - r.decode_tokens / 40.0 - r.arrival_t) * 1000.0
        for r in res.records
        if r.arrival_t is not None and r.finish_t is not None
    ]
    return {
        "hit_rate_pct": round(res.hit_rate * 100.0, 4),
        "total_requests": len(res.records),
        "prompt_tokens": res.total_prompt_tokens,
        "cached_tokens": res.total_cached_tokens,
        "ttft_p50_ms": round(_pct(ttfts, 50), 2),
        "ttft_p90_ms": round(_pct(ttfts, 90), 2),
        "ttft_p99_ms": round(_pct(ttfts, 99), 2),
    }


def run_cell(cell: str, arms: list[str], args) -> dict:
    tasks_yaml = BENCH_DIR / "TaskData" / "exp3" / cell / "tasks.yaml"
    if not tasks_yaml.is_file():
        raise FileNotFoundError(
            f"未找到负载配置: {tasks_yaml}\n"
            f"请先运行: python Heter-Bench/mixexp/tools/build_exp3_data.py --mix all --total-tokens 25e6"
        )
    sessions, specs = load_tasks(str(tasks_yaml))
    svc = _service_model()
    cell_res: dict[str, dict] = {}
    lru_res = None

    print(f"\n=== exp3 cell: {cell} (cap={args.cap}, page={args.page}) ===")
    for arm in arms:
        if arm == "lru":
            res = _run_once(
                sessions, specs, cap=args.cap, page=args.page, seed=args.seed,
                svc=svc, eviction_policy="lru",
            )
            lru_res = res
        elif arm == "heter_cache":
            res = _run_once(
                sessions, specs, cap=args.cap, page=args.page, seed=args.seed,
                svc=svc, eviction_policy="heter_cache",
            )
        elif arm == "chain_lru":
            chain_cfg = {
                "floor_tokens": args.floor,
                "weight": args.weight,
                "gate": args.gate,
                "tail_first": True,
            }
            res = _run_once(
                sessions, specs, cap=args.cap, page=args.page, seed=args.seed,
                svc=svc, eviction_policy="chain_lru", policy_config=chain_cfg,
            )
        elif arm == "heter_rgate":
            rg_cfg = {
                "enabled": True,
                "k": args.rg_k,
                "r_min": args.rg_rmin,
                "window_s": 120.0,
                "priced_fallback": True,
                "floor_tokens": args.floor,
                "weight": args.weight,
                "beta": args.pf_beta,
            }
            res = _run_once(
                sessions, specs, cap=args.cap, page=args.page, seed=args.seed,
                svc=svc, eviction_policy="heter_cache", regret_gate=rg_cfg,
            )
        elif arm == "opt":
            if lru_res is None:
                lru_res = _run_once(
                    sessions, specs, cap=args.cap, page=args.page, seed=args.seed,
                    svc=svc, eviction_policy="lru",
                )
            with tempfile.TemporaryDirectory() as tmpdir:
                map_path = str(pathlib.Path(tmpdir) / "use_map.json")
                prev_res = lru_res
                for _it in range(3):
                    umap = use_map_from_run(prev_res.records, sessions, page_size=args.page)
                    write_use_map(umap, map_path, page_size=args.page)
                    prev_res = _run_once(
                        sessions, specs, cap=args.cap, page=args.page, seed=args.seed,
                        svc=svc, eviction_policy="opt", opt_use_map=map_path,
                    )
                res = prev_res
        else:
            raise ValueError(f"未知实验臂: {arm}")

        stats = _summarize(res)
        cell_res[arm] = stats
        print(
            f"  [{arm:12s}] hit={stats['hit_rate_pct']:6.2f}%  "
            f"ttft(p50/p90/p99)={stats['ttft_p50_ms']:6.1f}/{stats['ttft_p90_ms']:6.1f}/{stats['ttft_p99_ms']:7.1f} ms"
        )
    return cell_res


def main() -> None:
    p = argparse.ArgumentParser(description="exp3 四 cell 引擎仿真对比")
    p.add_argument("--cells", default="scan,struct,chat,agentic", help="逗号分隔的 cell 列表")
    p.add_argument("--arms", default="lru,heter_cache,chain_lru,heter_rgate", help="逗号分隔的策略臂")
    p.add_argument("--cap", type=int, default=198000, help="token 池容量")
    p.add_argument("--page", type=int, default=8, help="page size")
    p.add_argument("--seed", type=int, default=0, help="仿真随机种子")
    # chain_lru / 档 B 推荐参数（默认对齐 best_configs.md）
    p.add_argument("--floor", type=int, default=8192)
    p.add_argument("--weight", type=float, default=0.02)
    p.add_argument("--gate", default="multiturn")
    p.add_argument("--rg-k", type=float, default=0.6)
    p.add_argument("--rg-rmin", type=float, default=0.3)
    p.add_argument("--pf-beta", type=float, default=1.0)
    p.add_argument("--out", default=None, help="可选结果 JSON 输出路径")
    args = p.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    summary = {}
    for c in cells:
        summary[c] = run_cell(c, arms, args)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ 汇总结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
