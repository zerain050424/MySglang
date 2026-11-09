"""离线矩阵并发执行器（L1 全消融）。

    python -m wtl.run_matrix --arms all --workloads sharegpt,alfred,codeparrot \\
        --ratios 0.3,0.6,1.0 --seeds 42,43,44 --jobs 12

    # 合成前缀阶梯（场景 A）
    python -m wtl.run_matrix --mode synth --arms A0,A2,A3 \\
        --synth-depths 512,2048,8192 --synth-breadths 3,10 --jobs 8

特性
----
* **可续跑**：输出文件已存在且 ``--resume`` 时跳过（默认开启）；
* **校准串行**：工作集先串行算好写进 ``result/calib.json``，避免并发写坏；
* **进程隔离**：每个回放在独立进程里跑（SimClock 的 patch 不会串味）；
* **无依赖友好**：``--backend fake`` 可在没有 torch/sglang 的机器上验证调度链路。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .arms import select as select_arms
from .synth import build as build_synth
from .synth import write as write_synth
from .trace import WORKLOADS

HERE = Path(__file__).resolve().parents[1]  # Heter-Bench/W-TinyLRU


@dataclass
class Job:
    run_id: str
    argv: List[str]
    out: Path


def _name(arm: str, workload: str, ratio: Optional[float], seed: int, extra: str = "") -> str:
    r = f"r{ratio:.2f}" if ratio is not None else "rabs"
    tail = f"_{extra}" if extra else ""
    return f"offline_{arm}_{workload}_{r}_s{seed}{tail}"


def build_trace_jobs(a, outdir: Path) -> List[Job]:
    jobs: List[Job] = []
    for workload in a.workloads:
        for seed in a.seeds:
            for arm in select_arms(a.arms):
                for ratio in a.ratios:
                    run_id = _name(arm.id, workload, ratio, seed)
                    argv = [
                        "-m", "wtl.replay",
                        "--arm", arm.id,
                        "--workload", workload,
                        "--num-sessions", str(a.num_sessions),
                        "--read-limit", str(a.read_limit),
                        "--seed", str(seed),
                        "--wait-scale", str(a.wait_scale),
                        "--arrival-scale", str(a.arrival_scale),
                        "--service-ms-per-1k", str(a.service_ms_per_1k),
                        "--pool-ratio", str(ratio),
                        "--calib-file", str(outdir.parent / "calib.json"),
                        "--backend", a.backend,
                        "--regret-window", str(a.regret_window),
                        "--out", str(outdir / f"{run_id}.json"),
                    ]
                    if a.prefix_stats:
                        argv += ["--prefix-stats", a.prefix_stats]
                    if a.no_outputs:
                        argv += ["--no-outputs"]
                    jobs.append(Job(run_id, argv, outdir / f"{run_id}.json"))
    return jobs


def build_synth_jobs(a, outdir: Path, datadir: Path) -> List[Job]:
    jobs: List[Job] = []
    for depth in a.synth_depths:
        for breadth in a.synth_breadths:
            tag = f"p{depth}_b{breadth}"
            ddir = datadir / f"synth_{tag}"
            if not (ddir / "session_data.jsonl").exists():
                built = build_synth(
                    prefix_depth=depth,
                    breadth=breadth,
                    variants=a.synth_variants,
                    turns=a.synth_turns,
                    body_mu=a.synth_body_mu,
                    body_sigma=a.synth_body_sigma,
                    noise_frac=a.synth_noise_frac,
                    doc_len=a.synth_doc_len,
                    seed=42,
                )
                write_synth(ddir, built)
                print(f"[matrix] synth {tag}: {built['meta']['num_sessions']} sessions")
            for arm in select_arms(a.arms):
                for ratio in a.ratios:
                    run_id = _name(arm.id, f"synth_{tag}", ratio, 42)
                    argv = [
                        "-m", "wtl.replay",
                        "--arm", arm.id,
                        "--data-path", str(ddir / "session_data.jsonl"),
                        "--synth-meta", str(ddir / "meta.json"),
                        "--num-sessions", "-1",
                        "--seed", "42",
                        "--wait-scale", str(a.wait_scale),
                        "--arrival-scale", str(a.arrival_scale),
                        "--pool-ratio", str(ratio),
                        "--calib-file", str(outdir.parent / "calib.json"),
                        "--backend", a.backend,
                        "--regret-window", str(a.regret_window),
                        "--out", str(outdir / f"{run_id}.json"),
                    ]
                    if a.no_outputs:
                        argv += ["--no-outputs"]
                    if a.no_shuffle:
                        argv.append("--no-shuffle")
                    jobs.append(Job(run_id, argv, outdir / f"{run_id}.json"))
    return jobs


def build_datadir_jobs(a, outdir: Path) -> List[Job]:
    """回放一个已生成的 synth 目录（正式实验的 D1/D2/D3 走这条路径）。"""
    ddir = Path(a.data_dir)
    if not ddir.is_absolute():
        ddir = (HERE / ddir).resolve()
    data = ddir / "session_data.jsonl"
    meta = ddir / "meta.json"
    if not data.exists():
        raise SystemExit(f"[matrix] 找不到 {data}")
    tag = ddir.name
    jobs: List[Job] = []
    pools = a.pool_tokens or [None]
    ratios = [None] if a.pool_tokens else a.ratios   # 绝对池优先，避免无谓重复
    for arm in select_arms(a.arms):
        for ratio in ratios:
          for pool in pools:
            for seed in a.seeds:
                run_id = _name(arm.id, f"exp_{tag}",
                               (pool / 1000.0) if pool else ratio, seed)
                argv = [
                    "-m", "wtl.replay",
                    "--arm", arm.id,
                    "--data-path", str(data),
                    "--synth-meta", str(meta) if meta.exists() else "",
                    "--num-sessions", "-1",
                    "--seed", str(seed),
                    "--wait-scale", str(a.wait_scale),
                    "--arrival-scale", str(a.arrival_scale),
                    "--service-ms-per-1k", str(a.service_ms_per_1k),
                    "--calib-file", str(outdir.parent / "calib.json"),
                    "--backend", a.backend,
                    "--regret-window", str(a.regret_window),
                    "--out", str(outdir / f"{run_id}.json"),
                ]
                if pool:
                    argv += ["--pool-tokens", str(pool)]
                else:
                    argv += ["--pool-ratio", str(ratio)]
                if not (meta.exists()):
                    argv = [x for x in argv if x != ""]
                    i = argv.index("--synth-meta")
                    del argv[i:i + 2]
                if a.no_shuffle:
                    argv.append("--no-shuffle")
                jobs.append(Job(run_id, argv, outdir / f"{run_id}.json"))
    return jobs


def run_one(job: Job, env: Dict[str, str], quiet: bool) -> tuple:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *job.argv],
        cwd=str(HERE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0 and job.out.exists()
    if not ok:
        return job.run_id, False, dt, proc.stdout[-2000:]
    if not quiet:
        last = [ln for ln in proc.stdout.splitlines() if ln.startswith("[")]
        tail = last[-1] if last else ""
        return job.run_id, True, dt, tail
    return job.run_id, True, dt, ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="离线实验矩阵")
    p.add_argument("--mode", choices=["trace", "synth"], default="trace")
    p.add_argument("--arms", default="A0,A1,A2,A3")
    p.add_argument("--workloads", default="sharegpt,alfred")
    p.add_argument("--ratios", default="0.3,0.6,1.0")
    p.add_argument("--pool-tokens", default="",
                   help="绝对池容量（token，逗号分隔）；给了就覆盖 --ratios（显存压力阶梯用）")
    p.add_argument("--seeds", default="42")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--num-sessions", type=int, default=300)
    p.add_argument("--read-limit", type=int, default=0,
                   help="只读 jsonl 前 N 行（大文件加速；建议 >= num_sessions*10）")
    p.add_argument("--wait-scale", type=float, default=1.0)
    p.add_argument("--arrival-scale", type=float, default=1.0)
    p.add_argument("--service-ms-per-1k", type=float, default=0.0)
    p.add_argument("--regret-window", type=float, default=60.0)
    p.add_argument("--backend", default="auto", choices=["auto", "real", "fake"])
    p.add_argument("--prefix-stats", default=None)
    p.add_argument("--no-outputs", action="store_true")
    p.add_argument("--outdir", default="result/offline")
    p.add_argument("--datadir", default="data/synth")
    p.add_argument("--data-dir", default=None,
                   help="已生成的 synth 目录（含 session_data.jsonl + meta.json）："
                        "直接回放，不重新生成（正式实验用 D1/D2/D3）")
    p.add_argument("--no-shuffle", action="store_true",
                   help="保持 jsonl 原始顺序（phased 负载必须；默认打乱）")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    # synth 参数
    p.add_argument("--synth-depths", default="512,2048,8192")
    p.add_argument("--synth-breadths", default="3,10")
    p.add_argument("--synth-variants", type=int, default=4)
    p.add_argument("--synth-turns", type=int, default=3)
    p.add_argument("--synth-body-mu", type=float, default=6.5)
    p.add_argument("--synth-body-sigma", type=float, default=0.8)
    p.add_argument("--synth-noise-frac", type=float, default=0.3)
    p.add_argument("--synth-doc-len", type=int, default=2000)
    a = p.parse_args(argv)

    a.workloads = [w.strip() for w in a.workloads.split(",") if w.strip()]
    a.ratios = [float(x) for x in a.ratios.split(",") if x.strip()]
    a.pool_tokens = [int(x) for x in str(a.pool_tokens).split(",") if x.strip()]
    a.seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    a.synth_depths = [int(x) for x in a.synth_depths.split(",") if x.strip()]
    a.synth_breadths = [int(x) for x in a.synth_breadths.split(",") if x.strip()]

    for w in a.workloads:
        if w not in WORKLOADS:
            print(f"[matrix] 未知 workload {w}; 可用 {sorted(WORKLOADS)}", file=sys.stderr)
            return 2

    outdir = (HERE / a.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = (HERE / a.datadir).resolve()

    if a.data_dir:
        jobs = build_datadir_jobs(a, outdir)
    elif a.mode == "synth":
        jobs = build_synth_jobs(a, outdir, datadir)
    else:
        jobs = build_trace_jobs(a, outdir)

    if a.resume:
        todo = [j for j in jobs if not j.out.exists()]
    else:
        todo = jobs
    skipped = len(jobs) - len(todo)
    print(
        f"[matrix] mode={a.mode} jobs={len(jobs)} todo={len(todo)} "
        f"skip={skipped} parallel={a.jobs} outdir={outdir}"
    )
    if a.dry_run:
        for j in todo[:20]:
            print("  ", j.run_id, " ".join(j.argv))
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["WTL_BACKEND"] = a.backend
    # 让子进程能 import wtl（cwd 已设，双保险）
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    fails = []
    done = 0
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(run_one, j, env, a.quiet): j for j in todo}
        for fut in cf.as_completed(futs):
            job = futs[fut]
            run_id, ok, dt, tail = fut.result()
            done += 1
            status = "ok " if ok else "FAIL"
            print(f"[{done}/{len(todo)}] {status} {run_id} {dt:6.1f}s {tail}".rstrip())
            if not ok:
                fails.append((run_id, tail))

    wall = time.perf_counter() - t0
    print(f"[matrix] 完成 {done - len(fails)}/{len(todo)}，失败 {len(fails)}，"
          f"总耗时 {wall / 60:.1f} min")
    for run_id, tail in fails[:10]:
        print(f"  !! {run_id}\n{tail}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
