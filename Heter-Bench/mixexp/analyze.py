#!/usr/bin/env python3
"""汇总 mixexp 的实验结果：命中率 / 最弱类 / 时延，并按判据给出结论。

读取 ``--dir`` 下形如 ``<scale>_<arm>.json`` 的结果文件（``_rN`` 后缀视为重复run，
按臂取均值）。只依赖标准库，可独立运行。

判据（对应 README 的设计意图）：
  1. 主判据  hc_full 的总命中率严格高于所有 baseline；
  2. 公平判据 hc_full 的「最弱类命中率」不低于最好的 baseline——
     异构负载下靠饿死一类换总命中率不算赢；
  3. 时延判据 hc_full 的 ttft_p99 不劣于 baseline 的 1.10 倍；
  4. 消融判据 hc_full 不弱于任一 hc_* 消融臂（说明每个部件都在起作用）；
  5. 有效性 总 prompt_tokens 是否真的超过池容量——否则没发生淘汰，结论无效。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASELINES = ("lru", "lfu", "slru", "tlru")
FULL_ARM = "hc_full"


def _as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def load_scale(directory: Path, scale: str) -> Dict[str, Dict[str, Any]]:
    """按臂聚合（重复 run 取均值）。"""
    pattern = re.compile(rf"^{re.escape(scale)}_(?P<arm>.+?)(?:_r\d+)?$")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(directory.glob(f"{scale}_*.json")):
        match = pattern.match(path.stem)
        if not match:
            continue
        try:
            grouped.setdefault(match.group("arm"), []).append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)

    arms: Dict[str, Dict[str, Any]] = {}
    for arm, runs in grouped.items():
        hits = [_as_float(r.get("cache_hit_rate_pct")) for r in runs]
        hits = [h for h in hits if h is not None]
        ttft50 = [_as_float(r.get("ttft_p50_ms")) for r in runs]
        ttft99 = [_as_float(r.get("ttft_p99_ms")) for r in runs]
        e2e99 = [_as_float(r.get("e2e_p99_ms")) for r in runs]
        per_task_acc: Dict[str, List[float]] = {}
        for run in runs:
            for task, stats in (run.get("per_task") or {}).items():
                value = _as_float(stats.get("cache_hit_rate_pct"))
                if value is not None:
                    per_task_acc.setdefault(task, []).append(value)
        per_task = {k: sum(v) / len(v) for k, v in per_task_acc.items()}
        arms[arm] = {
            "runs": len(runs),
            "hit": sum(hits) / len(hits) if hits else None,
            "ttft50": _mean(ttft50),
            "ttft99": _mean(ttft99),
            "e2e99": _mean(e2e99),
            "per_task": per_task,
            "prompt_tokens": sum(_as_float(r.get("prompt_tokens")) or 0.0 for r in runs),
            "errors": sum(int(r.get("num_errors") or 0) for r in runs),
        }
    return arms


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def render(scale: str, arms: Dict[str, Dict[str, Any]], tokens: Optional[float]) -> Optional[bool]:
    best_baseline = max(
        (arms[a] for a in BASELINES if a in arms and arms[a]["hit"] is not None),
        key=lambda x: x["hit"],
        default=None,
    )

    print(f"\n## {scale} 规模 · 结果\n")
    print("| arm | 命中率 % | Δ vs best baseline | 最弱类 % | 类间差距 | ttft p50 ms | ttft p99 ms | e2e p99 ms | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in sorted(arms, key=lambda a: -(arms[a]["hit"] or -1)):
        row = arms[arm]
        hit = row["hit"]
        delta = (
            f"{hit - best_baseline['hit']:+.2f}"
            if (hit is not None and best_baseline and best_baseline["hit"] is not None)
            else "—"
        )
        per_task = row["per_task"]
        weakest = min(per_task.values()) if per_task else None
        spread = (max(per_task.values()) - weakest) if per_task else None
        print(
            f"| `{arm}` | {_fmt(hit)} | {delta} | {_fmt(weakest)} | {_fmt(spread)} "
            f"| {_fmt(row['ttft50'])} | {_fmt(row['ttft99'])} | {_fmt(row['e2e99'])} | {row['errors']} |"
        )

    print("\n### 判据\n")
    full = arms.get(FULL_ARM)
    if full is None or full["hit"] is None:
        print(f"- ✗ 缺少 `{FULL_ARM}` 的结果，无法裁决")
        return None

    checks = []
    if best_baseline and best_baseline["hit"] is not None:
        checks.append((
            "主判据：总命中率高于所有 baseline",
            full["hit"] > best_baseline["hit"],
            f"{full['hit']:.2f}% vs {best_baseline['hit']:.2f}%",
        ))
        full_weak = min(full["per_task"].values()) if full["per_task"] else None
        base_weak = min(best_baseline["per_task"].values()) if best_baseline["per_task"] else None
        if full_weak is not None and base_weak is not None:
            checks.append((
                "公平判据：最弱类命中率不低于最好的 baseline",
                full_weak >= base_weak - 0.05,
                f"{full_weak:.2f}% vs {base_weak:.2f}%",
            ))
        if full["ttft99"] is not None and best_baseline["ttft99"] is not None:
            checks.append((
                "时延判据：ttft_p99 不劣于 baseline 的 1.10 倍",
                full["ttft99"] <= best_baseline["ttft99"] * 1.10,
                f"{full['ttft99']:.1f} ms vs {best_baseline['ttft99']:.1f} ms",
            ))

    for arm, row in sorted(arms.items()):
        if not arm.startswith("hc_") or arm == FULL_ARM or row["hit"] is None:
            continue
        checks.append((
            f"消融判据：`{FULL_ARM}` 不弱于 `{arm}`",
            full["hit"] >= row["hit"] - 0.2,
            f"{full['hit']:.2f}% vs {row['hit']:.2f}%",
        ))

    if tokens:
        total = full["prompt_tokens"]
        checks.append((
            "有效性：总 prompt_tokens 超过池容量（确实发生了淘汰）",
            total > tokens,
            f"{total:,.0f} vs {tokens:,.0f}",
        ))

    for name, passed, detail in checks:
        print(f"- {'✓' if passed else '✗'} {name}　（{detail}）")

    hard = [c for c in checks if c[0].startswith(("主判据", "有效性"))]
    print(f"\n结论：{'通过' if all(c[1] for c in hard) else '未通过'}"
          f"（{sum(c[1] for c in checks)}/{len(checks)} 项满足）")
    if tokens and full["prompt_tokens"] <= tokens:
        print("注意：池子没被填满，淘汰可能没发生——先调小 --tokens 或放大 --scale。")
    return all(c[1] for c in hard)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 mixexp 结果")
    parser.add_argument("--dir", default="result/mixexp", help="结果目录")
    parser.add_argument("--scale", action="append", help="规模；可重复，缺省为目录下全部")
    parser.add_argument("--tokens", type=float, default=None, help="本次运行的池容量（判定有效性）")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"目录不存在：{directory}", file=sys.stderr)
        return 1

    scales = args.scale or sorted(
        {p.stem.split("_", 1)[0] for p in directory.glob("*.json") if "_" in p.stem}
    )
    if not scales:
        print(f"{directory} 下没有结果文件", file=sys.stderr)
        return 1

    verdicts = []
    for scale in scales:
        arms = load_scale(directory, scale)
        if not arms:
            continue
        verdict = render(scale, arms, args.tokens)
        if verdict is not None:
            verdicts.append(verdict)

    # 硬判据未过时以非零退出，便于脚本串接；报告本身仍然完整打印。
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
