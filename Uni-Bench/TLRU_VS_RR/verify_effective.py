#!/usr/bin/env python3
"""检查实验是否「真的在比插件」——读 profile_<seed>.txt。

server 端每 alpha.update_interval（默认 30s）打一行：

    [SimpleHybrid profile]
      workload ...
      pool=... schedulable=... one_shot_evicted=...
      victim_src: one_shot=... qagent=...
      tlru: tel_safe=... lru=... fallback=... hist_raises=...

三个判定（README §6）：

  1. a2 的 tel_safe > 0      —— 否则阈值配错，TLRU 退化成 LRU；
  2. a2 的 fallback 远小于淘汰次数 —— 否则插件没被询问（没有 QAGENT 叶子）；
  3. victim_src 里 one_shot 占比低 —— 否则 one-shot 带抢走了决策权。

用法：python verify_effective.py [--root result] [--one-shot-warn 0.5]
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

VICTIM_RE = re.compile(r"victim_src:\s*(.+)")
TLRU_RE = re.compile(
    r"tlru:\s*tel_safe=(\d+)\s+lru=(\d+)\s+fallback=(\d+)\s+hist_raises=(\d+)"
)
PAIR_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?)")


def parse_profile(path: Path) -> Dict[str, float]:
    """把同一份 profile 里的所有行累加（多次打印取最后一次计数即可，计数器是累计值）。"""
    victim: Dict[str, float] = {}
    tlru: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = VICTIM_RE.search(line)
        if m:
            # 计数器是累计的，最后一次打印即终值
            victim = {k: float(v) for k, v in PAIR_RE.findall(m.group(1))}
        m = TLRU_RE.search(line)
        if m:
            tel, lru, fb, hr = (float(x) for x in m.groups())
            tlru = {
                "tlru_tel_safe_picks": tel,
                "tlru_lru_picks": lru,
                "tlru_fallbacks": fb,
                "tlru_history_raises": hr,
            }
    return {**victim, **tlru}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result")
    ap.add_argument("--one-shot-warn", type=float, default=0.5,
                    help="one_shot 占淘汰 token 的比例超过此值时告警")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*/*/profile_*.txt"))
    if not files:
        print(f"[verify] no profile_*.txt under {root}")
        print("        每次运行至少要比 update_interval（默认 30s）长，否则不打印 profile。")
        return 2

    rows: List[Tuple[str, str, str, Dict[str, float]]] = []
    for path in files:
        arm, budget = path.parts[-3], path.parts[-2]
        seed = path.stem.replace("profile_", "")
        stats = parse_profile(path)
        if stats:
            rows.append((arm, budget, seed, stats))

    print("| 臂 | 预算 | seed | one_shot 占比 | qagent 占比 | tel_safe | lru | fallback | hist_raises |")
    print("|---|---|---|---|---|---|---|---|---|")

    warn_one_shot = []
    warn_telru = []
    warn_degenerate = []
    agg: Dict[str, List[float]] = defaultdict(list)

    for arm, budget, seed, s in rows:
        one_shot = s.get("one_shot", 0.0)
        qagent = s.get("qagent", 0.0)
        total = one_shot + qagent + s.get("global", 0.0)
        os_share = one_shot / total if total else 0.0
        qa_share = qagent / total if total else 0.0
        agg[f"{arm}:one_shot"].append(os_share)
        agg[f"{arm}:tel_safe"].append(s.get("tlru_tel_safe_picks", 0.0))
        agg[f"{arm}:fallback"].append(s.get("tlru_fallbacks", 0.0))
        print(f"| {arm} | {budget} | {seed} | {os_share:.1%} | {qa_share:.1%} | "
              f"{s.get('tlru_tel_safe_picks', 0):.0f} | {s.get('tlru_lru_picks', 0):.0f} | "
              f"{s.get('tlru_fallbacks', 0):.0f} | {s.get('tlru_history_raises', 0):.0f} |")
        if os_share > args.one_shot_warn:
            warn_one_shot.append((arm, budget, seed, os_share))
        if arm == "a2_tlru" and s.get("tlru_tel_safe_picks", 0) == 0:
            warn_telru.append((arm, budget, seed))
            continue
        if arm == "a2_tlru":
            tel = s.get("tlru_tel_safe_picks", 0.0)
            lru = s.get("tlru_lru_picks", 0.0)
            picks = tel + lru
            if picks > 0 and tel / picks >= 0.98:
                warn_degenerate.append((arm, budget, seed, tel / picks))

    print()
    problems = 0
    if warn_one_shot:
        problems += 1
        print(f"⚠ one_shot 占比 > {args.one_shot_warn:.0%} 的运行（插件决策权被削弱）：")
        for arm, budget, seed, share in warn_one_shot[:10]:
            print(f"   - {arm}/{budget}/seed={seed}: one_shot={share:.1%}")
        print("   => 结论只能表述为「可调度带残余部分」，除非能压低该比例。")
    if warn_telru:
        problems += 1
        print("⚠ a2_tlru 的 tel_safe 计数为 0（阈值配错，TLRU 实际退化成 LRU）：")
        for arm, budget, seed in warn_telru[:10]:
            print(f"   - {arm}/{budget}/seed={seed}")
        print("   => 用 calibrate_tlru.py 重新取阈值，确认 threshold > next_prompt_estimate。")
    if warn_degenerate:
        problems += 1
        print("⚠ a2_tlru 几乎所有候选都是 tel-safe（threshold 过大 → budget=0）：")
        for arm, budget, seed, share in warn_degenerate[:10]:
            print(f"   - {arm}/{budget}/seed={seed}: tel_safe 占比={share:.1%}")
        print("   => 此时所有节点同档，排序退化为 last_access_time = 纯 LRU，")
        print("      实验会得出「TLRU 与 LRU 无差异」的假象；请降 threshold（参考 calibrate_tlru.py）。")
    if not problems:
        print("✅ 有效性检查通过：插件被询问、tel-safe 命中非零、one-shot 未主导。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
