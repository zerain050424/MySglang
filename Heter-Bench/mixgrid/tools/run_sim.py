#!/usr/bin/env python3
"""mixgrid · 仿真器后端：用一份 tasks.yaml 跑 ``sim`` 并落盘可比指标。

真机（``scripts/serve.sh`` + ``scripts/bench_heap.py``）与仿真器吃**同一份**
tasks.yaml、同一个 capacity / page_size；本脚本只是把 ``sim.run_engine`` 的产物
折成与 bench 结果同名同义的字段，便于两侧并排看。

用法::

    python mixgrid/tools/run_sim.py \\
      --tasks-config mixgrid/configs/mix/i_2525.yaml \\
      --capacity 198000 --page-size 8 --policy heter_cache \\
      --out mixgrid/result/sim/i_2525_cap198k_heter.json

    # LRU 基线 / OPT（OPT 需要先备 use map）
    python mixgrid/tools/run_sim.py ... --policy lru
    python mixgrid/tools/run_sim.py ... --policy opt --use-map TaskData/exp3/chat/use_map_p8.json

口径说明
--------
* ``capacity`` 必须等于真机的 ``--max-total-tokens``（用 server 启动日志校准）；
* ``page_size`` 必须等于真机的 ``--page-size``（默认 8，与 OPT 的 block_size 一致）；
* 服务时长模型读 ``tools/calibrate.py`` 的产物；不给就用 ``sim.schedule`` 的缺省；
* 输出的 ``realized`` 段是**落地校验**：目标份额（tasks.yaml 的 λ）是否真的
  变成了 token 份额 / 队列占用份额。份额跑偏时先修数据，别急着解释命中率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parents[2]   # Heter-Bench
REPO_ROOT = BENCH_DIR.parent                      # MySglang（sim 的包根）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.runner import SimConfig, Simulator
from sim.schedule import ServiceModel, TaskSpec
from sim.session_data import load_tasks


def shares(counter: dict) -> dict:
    """把 {名字: 量} 折成百分比（保持降序，便于肉眼对配比）。"""

    total = sum(float(v) for v in counter.values())
    if total <= 0:
        return {}
    ordered = sorted(counter.items(), key=lambda kv: -float(kv[1]))
    return {k: round(float(v) / total * 100, 2) for k, v in ordered}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-config", required=True)
    ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument("--page-size", type=int, default=8)
    ap.add_argument("--policy", default="heter_cache",
                    choices=["heter_cache", "lru", "lfu", "slru", "tlru", "opt"])
    ap.add_argument("--horizon-s", type=float, default=120.0)
    ap.add_argument("--smoothing-k", type=float, default=4.0)
    ap.add_argument("--floor-tokens", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-map", default=None, help="policy=opt 时必填")
    ap.add_argument("--calib", default=None, help="tools/calibrate.py 的 JSON 产物")
    ap.add_argument("--decode-tok-s", type=float, default=40.0)
    ap.add_argument("--arrival-scale", type=float, default=1.0)
    ap.add_argument("--wait-scale", type=float, default=1.0)
    ap.add_argument("--schedule", default="fcfs", choices=["fcfs", "lpm"])
    ap.add_argument("--max-prefill-tokens", type=int, default=16384)
    ap.add_argument("--oversize", default="skip", choices=["skip", "error"])
    ap.add_argument(
        "--warmup-s",
        type=float,
        default=600.0,
        help="统计窗起点（秒，虚拟时间）：丢掉 t<warmup 的记录，对齐 bench 的 --warmup-s",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default=None, help="逐请求明细落盘路径（可选）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.tasks_config).read_text(encoding="utf-8"))
    sessions = load_tasks(cfg["tasks"], base_dir=str(BENCH_DIR))
    specs = [TaskSpec.from_config(t) for t in cfg["tasks"]]

    if args.calib:
        service = ServiceModel.from_calibration(
            str(Path(args.calib) if Path(args.calib).is_absolute() else BENCH_DIR / args.calib),
            decode_tokens_per_s=args.decode_tok_s,
        )
    else:
        service = ServiceModel(decode_tokens_per_s=args.decode_tok_s)

    from sim.engine import EngineConfig

    sim = Simulator(
        SimConfig(
            capacity=args.capacity,
            page_size=args.page_size,
            eviction_policy=args.policy,
            value_horizon_s=args.horizon_s,
            value_smoothing_k=args.smoothing_k,
            value_floor_tokens=args.floor_tokens,
            opt_use_map=args.use_map,
            seed=args.seed,
            oversize=args.oversize,
        )
    )
    result = sim.run_engine(
        sessions,
        specs,
        scheduler=EngineConfig(
            schedule_policy=args.schedule,
            max_prefill_tokens=args.max_prefill_tokens,
        ),
        service=service,
        arrival_scale=args.arrival_scale,
        wait_scale=args.wait_scale,
    )

    summary = result.summary
    # 对齐 bench_heap 的 --warmup-s：bench 的统计窗从 warmup 之后开始，
    # sim 默认全量计入。两侧要并排比，就必须同一统计窗。
    kept = [r for r in result.records if r.t >= args.warmup_s]
    window = {
        "warmup_s": args.warmup_s,
        "requests": len(kept),
        "prompt_tokens": sum(r.prompt_len for r in kept),
        "hit_tokens": sum(r.hit for r in kept),
        "hit_rate": round(
            (sum(r.hit for r in kept) / sum(r.prompt_len for r in kept) * 100)
            if kept and sum(r.prompt_len for r in kept)
            else 0.0,
            2,
        ),
        "evicted_tokens": sum(r.evicted for r in kept),
        "per_task": {},
    }
    for task in {r.task_name for r in kept}:
        rows = [r for r in kept if r.task_name == task]
        prompt = sum(r.prompt_len for r in rows)
        hit = sum(r.hit for r in rows)
        window["per_task"][task] = {
            "requests": len(rows),
            "prompt_tokens": prompt,
            "hit_tokens": hit,
            "hit_rate": round(hit / prompt * 100, 2) if prompt else 0.0,
        }

    realized = {
        "task_token_share_pct": shares(
            {k: v["prompt_tokens"] for k, v in summary.get("per_task", {}).items()}
        ),
        "session_type_token_share_pct": shares(
            {k: v["prompt_tokens"] for k, v in summary.get("per_type", {}).items()}
        ),
        "queue_occupancy_share_pct": shares(summary.get("sched_occ", {})),
        "queue_hit_pct": shares(summary.get("queue_hit", {})),
    }

    payload = {
        "backend": "sim",
        "tasks_config": args.tasks_config,
        "policy": args.policy,
        "capacity": args.capacity,
        "page_size": args.page_size,
        "arrival_scale": args.arrival_scale,
        "wait_scale": args.wait_scale,
        "schedule": args.schedule,
        "seed": args.seed,
        "hit_rate": window["hit_rate"],
        "window": window,
        "summary": summary,
        "realized": realized,
        "fingerprint": result.fingerprint(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.jsonl:
        result.to_jsonl(args.jsonl)

    print(result.render())
    print(
        f"\n  hit_rate(warmup {args.warmup_s:g}s)={window['hit_rate']:.2f}%  "
        f"requests={window['requests']}  → {out}"
    )
    for name, share in realized["task_token_share_pct"].items():
        print(f"    realized token share  {name:10s} {share:6.2f}%")
    for name, share in realized["queue_occupancy_share_pct"].items():
        print(f"    realized queue occ    {name:10s} {share:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
