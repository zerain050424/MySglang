"""离线回放驱动（L1，本方案主力）。

用法
----
    python -m wtl.replay --arm A3 --workload alfred --pool-ratio 0.3 \\
        --num-sessions 300 --seed 42 --out result/offline_A3_alfred_r0.3_s42.json

    # 混合四负载（与在线实验同构）
    python -m wtl.replay --arm A3 --tasks-config ../hybrid_tasks_config.yaml \\
        --pool-ratio 0.6 --out result/offline_A3_hybrid_r0.6_s42.json

    # 合成前缀阶梯（场景 A）
    python -m wtl.synth --out-dir data/synth_p2048_b10 --prefix-depth 2048 --breadth 10
    python -m wtl.replay --arm A3 --data-path data/synth_p2048_b10/session_data.jsonl \\
        --synth-meta data/synth_p2048_b10/meta.json --pool-ratio 0.3 --out ...

产出
----
一份 JSON：``summary``（命中率 / 保留度 / 后悔率 / struct 状态 / victim_src）
+ ``rows``（逐请求明细，可与在线 CSV 逐条配对）。
``schema = wtl.replay/1``；``backend`` 字段区分 real/fake，fake 结果不得进主表。

工作集与压力点
--------------
``--pool-ratio r``：池容量 = r × 工作集，工作集 = **不淘汰时 radix 树的 token 占用**
（用镜像 LRU 树以无限池跑一遍得到，缓存在 ``--calib-file``）。
这正是设计文档 §5.3 想要的「pool / 工作集」口径，换模型/换数据集仍然可比。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import SCHEMA_REPLAY
from .compat import apply_all as apply_compat
from .compat import install_stable_ties
from .arms import Arm, get as get_arm
from .backends import FakeBackend, make_backend
from .metrics import RegretTracker, RetentionTracker, merge_task_stats
from .simclock import SimClock, install as install_clock, uninstall as uninstall_clock
from .trace import (
    Request,
    WORKLOADS,
    load_tasks_config,
    schedule_hybrid,
    schedule_single_workload,
    trace_fingerprint,
)

logger = logging.getLogger("wtl.replay")

DEFAULT_LAYERS = (256, 512, 1024, 2048, 4096, 8192)


# ---------------------------------------------------------------------------
# 工作集校准
# ---------------------------------------------------------------------------
def calibrate_working_set(
    requests: Sequence[Request],
    *,
    max_requests: int = -1,
    include_outputs: bool = True,
) -> int:
    """不淘汰时的树占用（token）。用镜像 radix 树 + 无限池求得，与策略无关。"""
    n = len(requests) if max_requests <= 0 else min(len(requests), max_requests)
    total = sum(
        (r.total_len if include_outputs else r.prompt_len) for r in requests[:n]
    )
    fake = FakeBackend(pool_tokens=total + 1)
    saved_out = [(r.prompt, r.output) for r in requests[:n]]
    for r in requests[:n]:
        if not include_outputs:
            r.output = []
        fake.acquire(r)
    for r, (p, o) in zip(requests[:n], saved_out):
        r.prompt, r.output = p, o
    return int(fake.used)


def _calib_key(workload: str, seed: int, num_sessions: int, include_outputs: bool,
               wait_scale: float, arrival_scale: float, tasks_config: Optional[str],
               data_path: Optional[str]) -> str:
    return "|".join(
        str(x) for x in (
            workload, seed, num_sessions, int(include_outputs),
            round(wait_scale, 3), round(arrival_scale, 3),
            os.path.abspath(tasks_config) if tasks_config else "",
            os.path.abspath(data_path) if data_path else "",
        )
    )


def load_or_build_calib(
    path: str,
    key: str,
    builder,
) -> int:
    data: Dict[str, int] = {}
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if key in data:
        return int(data[key])
    t0 = time.perf_counter()
    value = int(builder())
    data[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[calib] 工作集=%s tokens (%.1fs) -> %s", value, time.perf_counter() - t0, p)
    return value


# ---------------------------------------------------------------------------
# 分层前缀长度
# ---------------------------------------------------------------------------
def layers_from_requests(
    requests: Sequence[Request],
    *,
    layers: Sequence[int] = DEFAULT_LAYERS,
    max_pairs: int = 100,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """从 trace 自身估计每个 task 的共享前缀分层长度（无 prefix_stats 时的回退）。"""
    import random

    from .prefix_stats import lcp_len

    by_task: Dict[str, List[List[int]]] = {}
    for r in requests:
        if r.turn != 1:
            continue
        by_task.setdefault(r.task, []).append(r.prompt)
    out: Dict[str, List[int]] = {}
    for task, prompts in by_task.items():
        rng = random.Random(seed)
        pairs = []
        n = len(prompts)
        for _ in range(min(max_pairs, max(1, n * (n - 1) // 2))):
            if n < 2:
                break
            i, j = rng.sample(range(n), 2)
            pairs.append(lcp_len(prompts[i], prompts[j]))
        if not pairs:
            out[task] = []
            continue
        max_lcp = max(pairs)
        out[task] = [L for L in layers if L <= max_lcp]
    return out


def layers_from_prefix_stats(path: str, tasks: Optional[Sequence[str]] = None) -> Dict[str, List[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, List[int]] = {}
    for task, st in (data.get("tasks") or {}).items():
        if tasks and task not in tasks:
            continue
        out[task] = [int(x) for x in st.get("layers") or []]
    return out


# ---------------------------------------------------------------------------
# 主回放
# ---------------------------------------------------------------------------
def run_replay(
    requests: Sequence[Request],
    *,
    arm: Arm,
    pool_tokens: int,
    backend_kind: str = "auto",
    include_outputs: bool = True,
    layers_by_task: Optional[Dict[str, List[int]]] = None,
    regret_window: float = 60.0,
    collect_rows: bool = True,
    pool_ratio: Optional[float] = None,
    calib_peak: Optional[int] = None,
    meta: Optional[dict] = None,
    quiet: bool = True,
    compat: bool = True,
    stable_ties: bool = True,
) -> dict:
    t0 = time.perf_counter()
    compat_fixes = apply_compat(compat)
    clock = SimClock()
    install_clock(clock)

    evictions = {"n": 0}

    def on_evict(ts, depth_end, ph, tokens, region, is_victim):
        evictions["n"] += 1
        if regret is not None:
            regret.on_evict(ts, depth_end, ph, tokens, region, is_victim)

    regret = RegretTracker(window_s=regret_window) if regret_window > 0 else None
    retention = RetentionTracker(layers_by_task or {})

    kwargs = dict(pool_tokens=int(pool_tokens), on_evict=on_evict)
    if backend_kind in ("real", "auto"):
        kwargs.update(policy=arm.policy, config=arm.config)

    backend = make_backend(backend_kind, **kwargs)

    actual_backend = "fake" if isinstance(backend, FakeBackend) else "real"
    tied = False
    if actual_backend == "real" and stable_ties:
        # 仿真时钟下平票极多；给排序键加确定性后缀，否则同配置两次回放会分叉。
        tied = install_stable_ties(backend.cache.eviction_strategy)

    rows: List[dict] = []
    prev_prompt: Dict[str, int] = {}
    prompt_tokens = cached_tokens = total_tokens = 0
    oom = 0
    peak_used = 0

    try:
        for req in requests:
            clock.set(req.ts)
            if not include_outputs and req.output:
                req = Request(
                    idx=req.idx, ts=req.ts, sid=req.sid, task=req.task,
                    turn=req.turn, prompt=req.prompt, output=[],
                )
            res = backend.acquire(req)
            prompt_tokens += res.prompt_len
            cached_tokens += res.cached_len
            total_tokens += res.total_len
            oom += int(res.oom)
            peak_used = max(peak_used, backend.used_tokens())

            if regret is not None:
                regret.on_request(req.ts, req.prompt, min(res.cached_len, req.prompt_len))
            cached_on_prompt = min(res.cached_len, req.prompt_len)
            prev_len = prev_prompt.get(req.sid, 0)   # 必须在覆盖之前取
            ret = retention.add(req.task, cached_on_prompt)
            retention.add_history(req.task, cached_on_prompt, prev_len)
            prev_prompt[req.sid] = req.prompt_len
            if collect_rows:
                rows.append(
                    {
                        "ts": round(req.ts, 4),
                        "task": req.task,
                        "sid": req.sid,
                        "turn": req.turn,
                        "prompt_tokens": res.prompt_len,
                        "cached_tokens": res.cached_len,
                        "total_tokens": res.total_len,
                        "evicted": res.evicted_tokens,
                        "oom": int(res.oom),
                        # 会话内历史保留（首轮为 None，配对时会被过滤）
                        "hist_full": (
                            int(res.cached_len >= prev_len) if prev_len > 0 else None
                        ),
                    }
                )
        stats = backend.stats()
    finally:
        uninstall_clock()

    hit_pct = round(100.0 * cached_tokens / prompt_tokens, 3) if prompt_tokens else 0.0
    summary = {
        "num_requests": len(requests),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "hit_pct": hit_pct,
        "oom": oom,
        "evictions": evictions["n"],
        "peak_used": peak_used,
        "per_task": merge_task_stats(rows) if rows else {},
        "retention": retention.summary(),
        "history_retention": retention.history_summary(),
        "regret": regret.summary() if regret is not None else None,
        "struct": stats.get("struct"),
        "slru": stats.get("slru") or {},
        "slru_share_min": stats.get("slru_share_min"),
        "victim_src": stats.get("victim_src") or {},
        "wall_s": round(time.perf_counter() - t0, 3),
    }

    result = {
        "schema": SCHEMA_REPLAY,
        "compat": compat_fixes,
        "stable_ties": tied,
        "backend": actual_backend,
        "arm": arm.id,
        "label": arm.label,
        "policy": arm.policy,
        "config": arm.config,
        "pool_tokens": int(pool_tokens),
        "pool_ratio": pool_ratio,
        "calib_peak": calib_peak,
        "include_outputs": bool(include_outputs),
        "regret_window_s": regret_window,
        "trace_fp": trace_fingerprint(requests),
        "layers_by_task": layers_by_task or {},
        "summary": summary,
        "rows": rows,
        "meta": meta or {},
    }
    if not quiet:
        _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    s = result["summary"]
    print(
        f"[{result['arm']} {result['label']}] backend={result['backend']} "
        f"hit={s['hit_pct']:.2f}% reqs={s['num_requests']} "
        f"evict={s['evictions']} oom={s['oom']} "
        f"regret={s['regret']['regret_rate'] if s['regret'] else '-'} "
        f"struct={s['struct']}"
    )
    for task, h in (s.get("history_retention") or {}).items():
        print(f"    history[{task}] n={h['n']} full={h['full_rate']:.3f} cover={h['cover']:.3f}")
    for task, layers in (s["retention"] or {}).items():
        parts = [f"{L}:{v['full_rate']:.3f}/{v['cover']:.3f}" for L, v in layers.items()]
        print(f"    retention[{task}] " + " ".join(parts) if parts else f"    retention[{task}] -")
    if s["victim_src"]:
        print("    victim_src " + " ".join(f"{k}={v:.0f}" for k, v in s["victim_src"].items()))


# ---------------------------------------------------------------------------
# 调度构建（CLI 共用）
# ---------------------------------------------------------------------------
def build_requests(args) -> tuple[List[Request], dict]:
    wait_scale = args.wait_scale
    service = args.service_ms_per_1k

    if args.data_path:
        dpath = Path(args.data_path)
        # task 标签优先用 --task-name，其次父目录名（synth_p2048_b10 / 数据集目录）
        task_name = args.task_name or (
            "synth" if args.synth_meta else (dpath.parent.name or dpath.stem)
        )
        requests = schedule_single_workload(
            "synth" if args.synth_meta or args.workload not in WORKLOADS else args.workload,
            num_sessions=args.num_sessions,
            seed=args.seed,
            wait_scale=wait_scale,
            arrival_scale=args.arrival_scale,
            service_ms_per_1k=service,
            data_path=args.data_path,
            lambda_rate=args.lambda_rate,
            mu=args.mu,
            sigma=args.sigma,
            task_name=task_name,
            read_limit=args.read_limit,
            shuffle=not args.no_shuffle,
        )
        label = dpath.stem
        meta = {"source": str(dpath.resolve()), "workload": label}
        if args.synth_meta:
            synth_meta = json.loads(Path(args.synth_meta).read_text(encoding="utf-8"))
            # 合成 trace：workload 用目录名（synth_p2048_b10），便于跨臂分组
            meta.update(synth_meta)
            meta["source"] = str(dpath.resolve())
            meta["workload"] = dpath.parent.name or label
            meta["synth"] = synth_meta
        return requests, meta

    if args.tasks_config:
        requests = schedule_hybrid(
            tasks_config=args.tasks_config,
            seed=args.seed,
            wait_scale=wait_scale,
            arrival_scale=args.arrival_scale,
            service_ms_per_1k=service,
            num_sessions_override=args.num_sessions,
        )
        cfg = load_tasks_config(args.tasks_config)
        return requests, {
            "source": str(Path(args.tasks_config).resolve()),
            "workload": "hybrid",
            "tasks": [{"name": t["name"], "lambda_rate": t.get("lambda_rate", 1)}
                      for t in cfg.get("tasks", [])],
        }

    requests = schedule_single_workload(
        args.workload,
        num_sessions=args.num_sessions,
        seed=args.seed,
        wait_scale=wait_scale,
        arrival_scale=args.arrival_scale,
        service_ms_per_1k=service,
        lambda_rate=args.lambda_rate,
        mu=args.mu,
        sigma=args.sigma,
        read_limit=args.read_limit,
        shuffle=not args.no_shuffle,
    )
    return requests, {"source": str(WORKLOADS.get(args.workload, {}).get("path", "")),
                      "workload": args.workload}


def resolve_pool(args, requests, key_builder_kwargs) -> tuple[int, Optional[int], Optional[float]]:
    if args.pool_tokens:
        return int(args.pool_tokens), None, None
    include_outputs = not args.no_outputs
    key = _calib_key(include_outputs=include_outputs, **key_builder_kwargs)
    peak = load_or_build_calib(
        args.calib_file,
        key,
        lambda: calibrate_working_set(
            requests, max_requests=args.calib_max_requests,
            include_outputs=include_outputs,
        ),
    )
    ratio = float(args.pool_ratio)
    return max(1, int(round(peak * ratio))), peak, ratio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_common_args(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("trace")
    src.add_argument("--workload", default="sharegpt", choices=sorted(WORKLOADS))
    src.add_argument("--data-path", default=None, help="直接指定 session_data.jsonl")
    src.add_argument("--tasks-config", default=None, help="混合负载 yaml")
    src.add_argument("--synth-meta", default=None, help="wtl.synth 生成的 meta.json")
    src.add_argument("--task-name", default=None,
                     help="自定义 trace 的 task 标签（默认取父目录名；需与 prefix_stats 的 --task 一致）")
    src.add_argument("--num-sessions", type=int, default=300, help="-1=全部")
    src.add_argument("--read-limit", type=int, default=0,
                     help="只读 jsonl 前 N 行再打乱（大文件加速，0=全量）")
    src.add_argument("--no-shuffle", action="store_true",
                     help="保持 jsonl 原始 session 顺序（phased 合成负载的块顺序必须保留）")
    src.add_argument("--seed", type=int, default=42)
    src.add_argument("--wait-scale", type=float, default=1.0,
                     help="轮间等待缩放（0=不等，1=真实 lognormal）")
    src.add_argument("--arrival-scale", type=float, default=1.0,
                     help="到达节奏缩放（1=Poisson，0=全部立即）")
    src.add_argument("--service-ms-per-1k", type=float, default=0.0,
                     help="每 1k prompt token 的近似服务时间（补上在线端的轮间间隔）")
    src.add_argument("--lambda-rate", type=float, default=1.0)
    src.add_argument("--mu", type=float, default=None)
    src.add_argument("--sigma", type=float, default=None)

    pool = p.add_argument_group("capacity")
    pool.add_argument("--pool-ratio", type=float, default=0.3,
                      help="池容量 / 工作集（0.3/0.6/1.0/2.0）")
    pool.add_argument("--pool-tokens", type=int, default=0, help="直接指定 token 池（覆盖 ratio）")
    pool.add_argument("--calib-file", default="result/calib.json")
    pool.add_argument("--calib-max-requests", type=int, default=-1)
    pool.add_argument("--no-outputs", action="store_true",
                      help="不把 response token 计入 KV（与纯 prompt 口径对照）")

    misc = p.add_argument_group("run")
    misc.add_argument("--backend", default="auto", choices=["auto", "real", "fake"])
    misc.add_argument("--prefix-stats", default=None, help="prefix_stats.json")
    misc.add_argument("--layers", default="256,512,1024,2048,4096,8192")
    misc.add_argument("--regret-window", type=float, default=60.0,
                      help="后悔率的复用窗口（秒，0=关闭）")
    misc.add_argument("--no-rows", action="store_true", help="不输出逐请求明细")
    misc.add_argument("--out", default=None)
    misc.add_argument("--no-compat", action="store_true",
                      help="不安装框架缺陷兼容补丁（见 wtl/compat.py）")
    misc.add_argument("--unstable-ties", action="store_true",
                      help="不安装确定性平票后缀（默认安装；见 wtl/compat.py）")
    misc.add_argument("--quiet", action="store_true")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="W-TinyLRU 离线回放（L1）")
    p.add_argument("--arm", default="A3", help="实验臂 A0..A4（A3=两带 SLRU）")
    add_common_args(p)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    arm = get_arm(a.arm)

    requests, meta = build_requests(a)
    if not requests:
        print("[wtl] 空 trace：检查 --data-path / --num-sessions / --tasks-config", file=sys.stderr)
        return 2

    if a.prefix_stats:
        layers = layers_from_prefix_stats(a.prefix_stats)
    elif a.synth_meta:
        # 合成 trace 的共享前缀长度是构造出来的，直接用 meta 里的 layers 更准
        sm = json.loads(Path(a.synth_meta).read_text(encoding="utf-8"))
        merged = [int(x) for x in (sm.get("layers") or [])]
        tasks = sorted({r.task for r in requests})
        if merged:
            layers = {t: merged for t in tasks}
        else:
            layers = layers_from_requests(
                requests, layers=[int(x) for x in a.layers.split(",") if x.strip()], seed=a.seed)
    else:
        layers = layers_from_requests(
            requests, layers=[int(x) for x in a.layers.split(",") if x.strip()],
            seed=a.seed,
        )

    # 让「前缀层深度」覆盖 trace 里出现的每个 task：
    #   - prefix_stats 的 task 名与 trace 的 task 名一致时直接用；
    #   - 不一致时回退到所有已知层的并集（并打日志，避免静默算错）。
    tasks = {r.task for r in requests}
    if layers:
        missing = tasks - set(layers)
        if missing:
            merged = sorted({L for v in layers.values() for L in v})
            logger.warning(
                "[wtl] prefix layers 未覆盖 task=%s，回退到 %s（检查 --prefix-stats 的 --task 命名）",
                sorted(missing), merged,
            )
            for t in missing:
                layers[t] = merged

    key_kwargs = dict(
        workload=meta.get("workload", a.workload),
        seed=a.seed,
        num_sessions=a.num_sessions,
        wait_scale=a.wait_scale,
        arrival_scale=a.arrival_scale,
        tasks_config=a.tasks_config,
        data_path=a.data_path,
    )
    pool_tokens, calib_peak, ratio = resolve_pool(a, requests, key_kwargs)

    result = run_replay(
        requests,
        arm=arm,
        pool_tokens=pool_tokens,
        backend_kind=a.backend,
        include_outputs=not a.no_outputs,
        layers_by_task=layers,
        regret_window=a.regret_window,
        collect_rows=not a.no_rows,
        pool_ratio=ratio,
        calib_peak=calib_peak,
        meta=meta,
        quiet=a.quiet,
        compat=not a.no_compat,
        stable_ties=not a.unstable_ties,
    )
    result["dataset"] = meta
    result["seed"] = a.seed
    result["num_sessions"] = a.num_sessions
    result["read_limit"] = a.read_limit
    result["shuffle"] = not a.no_shuffle
    result["wait_scale"] = a.wait_scale
    result["arrival_scale"] = a.arrival_scale

    if a.out:
        outp = Path(a.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not a.quiet:
            print(f"[wtl] -> {outp}  ({outp.stat().st_size / 1e6:.2f} MB)")
    if a.quiet:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
