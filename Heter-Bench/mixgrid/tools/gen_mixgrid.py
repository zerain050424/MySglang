#!/usr/bin/env python3
"""mixgrid：把「四类负载的 token 份额」翻成可直接跑的 tasks.yaml。

与 ``mixexp/tools/build_exp3_data.py`` 同一套算术，只把「单类主导 + 配角均分」
推广成任意份额，并且**不复制数据**——四类各一份池子文件，mix 只改
``max_record`` 与 ``lambda_rate``，因此 20 个 mix 共用同一批 token。

口径
----
* 份额按 **prompt token** 计（不是会话数）：会话粒度差两个数量级
  （apps ≈0.4k、agentic ≈89k），按会话配比会让 25% 的会话变成 95% 的 token。
* ``mean_tokens`` = 每条会话 Σ_轮 len(input_ids)，与命中率分母同口径。
* 单 run 预算 ``B = min(budget_cap, min_i pool_tokens_i / share_i)``：
  份额优先于规模——任何一类都不会跑到没数据。``pool_tokens_i / share_i``
  正是「第 i 类占 share_i 时池子允许的最大总预算」。
* 时长 ``T = B / token_rate``：token 到达率恒定，各 mix 压力可比，只有时长不同。
* ``n_i = share_i · B / mean_tokens_i``，``lambda_rate_i = n_i / T``，
  ``max_record_i = n_i``（池子刚好用满，不会跑到一半没会话）。

产物
----
``<out-dir>/<mix_id>.yaml``  每次 run 的 tasks_config（sim 与真机共用同一份）
``<out-dir>/manifest.csv``   mix / B / T / 各类 n 与 λ / 低样本告警
``<out-dir>/pool_stats.json`` 池子统计缓存（``--refresh`` 重算）

用法
----
    python mixgrid/tools/gen_mixgrid.py                      # 用 configs/mixgrid.yaml
    python mixgrid/tools/gen_mixgrid.py --dry-run            # 只打印计划
    python mixgrid/tools/gen_mixgrid.py --refresh            # 重测池子统计
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

BENCH_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GRID = BENCH_DIR / "mixgrid" / "configs" / "mixgrid.yaml"
DEFAULT_OUT = BENCH_DIR / "mixgrid" / "configs" / "mix"
# 单文件超过这个大小就只抽样估 mean_tokens（agentic 池可达数百 MB）
FULL_SCAN_LIMIT_BYTES = 400 * 1024 * 1024
SAMPLE_SESSIONS = 200
# 低于这个会话数，该 mix 的这一类方差会明显变大（写进 manifest 告警）
LOW_SESSION_WARN = 15


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BENCH_DIR / p


def measure(path: Path, sample: int, full_scan: bool) -> dict[str, Any]:
    """测一个池子：会话数 + 每条会话 prompt token 数的均值。"""

    if not path.exists():
        raise SystemExit(f"缺少池子文件：{path}（先跑 mixexp/prepare_data.sh 等准备步骤）")

    size = path.stat().st_size
    use_sample = (not full_scan) and size > FULL_SCAN_LIMIT_BYTES
    total_sessions = 0
    seen = 0
    token_sum = 0

    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            total_sessions += 1
            if use_sample and seen >= sample:
                continue
            payload = json.loads(line)
            session = next(iter(payload.values()))
            n_turns = session.get("num_turns") or sum(
                1 for k in session if k.startswith("turn_")
            )
            tok = 0
            for idx in range(1, int(n_turns) + 1):
                row = session.get(f"turn_{idx}") or {}
                tok += len(row.get("input_ids") or [])
            token_sum += tok
            seen += 1

    if seen == 0:
        raise SystemExit(f"{path} 里没有可用会话")
    mean_tokens = token_sum / seen
    pool_tokens = mean_tokens * total_sessions
    return {
        "path": str(path),
        "sessions": total_sessions,
        "planning_sessions": seen,
        "mean_tokens": mean_tokens,
        "pool_tokens": pool_tokens,
        "sampled": use_sample,
    }


def plan_mix(
    mix: dict[str, Any],
    classes: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    token_rate: float,
    budget_cap: float,
) -> dict[str, Any]:
    share = {k: float(v) for k, v in mix["share"].items()}
    if abs(sum(share.values()) - 100.0) > 1e-6:
        raise SystemExit(f"{mix['id']}: share 之和 = {sum(share.values())}，必须 100")
    present = [k for k in classes if share.get(k, 0) > 0]
    if not present:
        raise SystemExit(f"{mix['id']}: 没有任何类")

    # 份额优先于规模：取「每类允许的最大总预算」的最小值
    budget = min(
        budget_cap,
        min(stats[k]["pool_tokens"] / (share[k] / 100.0) for k in present),
    )
    duration = budget / token_rate

    counts, lambdas = {}, {}
    for k in present:
        n = round(share[k] / 100.0 * budget / stats[k]["mean_tokens"])
        n = max(1, min(n, stats[k]["sessions"]))
        counts[k] = n
        lambdas[k] = n / duration

    return {
        "id": mix["id"],
        "group": mix.get("group", ""),
        "share": share,
        "present": present,
        # 落地份额（截断/取整后的真实配比）：与目标份额对不上就告警，
        # 不要拿「名义配比」去解释命中率。
        "realized": {
            k: counts[k] * stats[k]["mean_tokens"] for k in present
        },
        "budget": budget,
        "duration": duration,
        "counts": counts,
        "lambdas": lambdas,
    }


def render_yaml(plan: dict[str, Any], classes: dict[str, dict[str, Any]], args) -> str:
    shares = "/".join(
        f"{k}{plan['share'].get(k, 0):g}" for k in ("T", "S", "C", "A")
    )
    lines = [
        "# =============================================================================",
        f"# mixgrid: {plan['id']}  (group={plan['group']})",
        f"# 目标 token 份额（T/S/C/A）= {shares}",
        (
            f"# 总 prompt token B = {plan['budget']:,.0f}，"
            f"时长 T = {plan['duration']:,.0f}s（token_rate={args.token_rate:g} tok/s）"
        ),
        "# 由 mixgrid/tools/gen_mixgrid.py 生成，勿手改（改 configs/mixgrid.yaml）",
        "# =============================================================================",
        "tasks:",
    ]
    for key in ("T", "S", "C", "A"):
        if key not in plan["present"]:
            continue
        cls = classes[key]
        lines += [
            f"  # --- {key}: {cls['name']} 目标 share={plan['share'][key]:g}% ---",
            f"  - name: {cls['name']}",
            f"    task_id: {cls['task_id']}",
            f"    data_path: {cls['data_path']}",
            f"    lambda_rate: {plan['lambdas'][key]:.6f}",
            f"    mu: {cls.get('mu', 0)}",
            f"    sigma: {cls.get('sigma', 0)}",
            "    max_trace_duration: -1",
            f"    seed: {args.seed}",
            f"    max_record: {plan['counts'][key]}",
            "    shuffle: true",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default=str(DEFAULT_GRID), help="网格定义 yaml")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="tasks.yaml 输出目录")
    ap.add_argument("--bench-dir", default=str(BENCH_DIR), help="相对路径的基准目录")
    ap.add_argument("--token-rate", type=float, default=None, help="覆盖 token_rate")
    ap.add_argument("--budget-cap", type=float, default=None, help="覆盖 budget_cap")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 seed")
    ap.add_argument("--refresh", action="store_true", help="重测池子统计（默认读缓存）")
    ap.add_argument("--full-scan", action="store_true", help="大文件也全量扫描")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不落盘")
    args = ap.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    args.token_rate = args.token_rate or float(grid["token_rate"])
    args.budget_cap = args.budget_cap or float(grid["budget_cap"])
    args.seed = args.seed if args.seed is not None else int(grid.get("seed", 42))
    classes: dict[str, dict[str, Any]] = grid["classes"]

    out_dir = Path(args.out_dir)
    stats_path = out_dir / "pool_stats.json"
    cache: dict[str, Any] = {}
    if stats_path.exists() and not args.refresh:
        cache = json.loads(stats_path.read_text(encoding="utf-8"))

    stats: dict[str, dict[str, Any]] = {}
    for key, cls in classes.items():
        path = resolve(cls["data_path"])
        if key in cache and not args.refresh:
            stats[key] = cache[key]
        else:
            print(f"[measure] {key} ({cls['name']}) ← {path}", file=sys.stderr)
            stats[key] = measure(path, SAMPLE_SESSIONS, args.full_scan)
        s = stats[key]
        print(
            f"  {key}/{cls['name']:8s} sessions={s['sessions']:6d} "
            f"mean_tokens={s['mean_tokens']:8.1f} "
            f"pool_tokens={s['pool_tokens']:>12,.0f}"
            + ("  (sampled)" if s["sampled"] else ""),
            file=sys.stderr,
        )

    rows: list[dict[str, Any]] = []
    for mix in grid["grid"]:
        plan = plan_mix(mix, classes, stats, args.token_rate, args.budget_cap)
        warns = [
            f"{k}={plan['counts'][k]}会话"
            for k in plan["present"]
            if plan["counts"][k] < LOW_SESSION_WARN
        ]
        tot = sum(plan["realized"].values())
        drift = {
            k: share - plan["realized"][k] / tot * 100
            for k, share in plan["share"].items()
            if k in plan["present"]
        }
        if any(abs(v) > 2.0 for v in drift.values()):
            warns.append(
                "份额漂移 " + " ".join(f"{k}{v:+.1f}pp" for k, v in drift.items() if abs(v) > 2.0)
            )
        print(
            f"{plan['id']:10s} {plan['group']:8s} "
            f"B={plan['budget']:>10,.0f} T={plan['duration']:6.0f}s "
            + " ".join(f"{k}:n={plan['counts'][k]}({plan['lambdas'][k]:.4f}/s)" for k in plan["present"])
            + (f"  ! 低样本 {' '.join(warns)}" if warns else "")
        )
        rows.append(
            {
                "mix": plan["id"],
                "group": plan["group"],
                "share_T": plan["share"].get("T", 0),
                "share_S": plan["share"].get("S", 0),
                "share_C": plan["share"].get("C", 0),
                "share_A": plan["share"].get("A", 0),
                "budget_tokens": int(plan["budget"]),
                "duration_s": round(plan["duration"], 1),
                "n_T": plan["counts"].get("T", 0),
                "n_S": plan["counts"].get("S", 0),
                "n_C": plan["counts"].get("C", 0),
                "n_A": plan["counts"].get("A", 0),
                "low_sample": " ".join(warns),
            }
        )
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{plan['id']}.yaml").write_text(
                render_yaml(plan, classes, args), encoding="utf-8"
            )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n→ {out_dir}/<mix>.yaml 与 manifest.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
