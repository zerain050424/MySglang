#!/usr/bin/env python3
"""drift3（构成漂移）数据切片 + tasks.yaml 生成器。

设计见 ``Heter-Bench/mixexp/drift3_design.md``。做完两件事：

1. **按会话切片**成互斥文件：``TaskData/drift3/<slice>/<class>.jsonl``。
   同一份 jsonl 不能给两个 slice 复用——``session_id`` 重复会让 ``Emitter`` 直接
   报错（实测：``ValueError: session_id 重复：apps-synth-15376``）。
2. **按实测 token/会话 反算 λ**，写出 ``TaskData/drift3/bg_{05,10,20}.yaml``：
   λ = 目标 token 份额 × 每相位 token 预算 / (相位秒数 × 该类实测 token/会话)。
   背景（trans）在 1× 档占 25% token；0.5×/2× 只改 trans 的 λ，前景 λ 不变
   （背景是"纯加注"，不是"挤占"）。

用法::

    python mixexp/tools/build_drift3_data.py            # 生成到 TaskData/drift3
    python mixexp/tools/build_drift3_data.py --dry-run  # 只打印规划
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]      # Heter-Bench/


def _multi_turn_min_from_source() -> int:
    """从 heter_cache/utils.py 源码读 MULTI_TURN_MIN（与 drop_short_sessions 同口径）。"""
    import re
    txt = (HERE.parent / "python" / "sglang" / "srt" / "mem_cache" /
           "heter_cache" / "utils.py").read_text(encoding="utf-8")
    m = re.search(r"^\s*MULTI_TURN_MIN\s*=\s*(\d+)", txt, re.M)
    return int(m.group(1)) if m else 5


MULTI_TURN_MIN = _multi_turn_min_from_source()
ROOT = HERE.parent                               # repo root

# 时间轴（虚拟秒）：warmup 300 + 3×phase + cooldown 300。phase 长度由 --phase-s 给
# （1200 = 原设计；2400 用来分辨"瞬态 vs 永久损失"，见 drift3_report.md §4.7）。
EDGE_S = 300.0
PHASE_S = 1200.0


def make_slices(phase_s: float = PHASE_S, edge_s: float = EDGE_S):
    return (
        ("warmup", edge_s, 0.0),
        ("p1", phase_s, edge_s),
        ("p2", phase_s, edge_s + phase_s),
        ("p3", phase_s, edge_s + 2 * phase_s),
        ("cooldown", edge_s, edge_s + 3 * phase_s),
    )


SLICES = make_slices()
TOTAL_S = SLICES[-1][2] + SLICES[-1][1]

# 每相位 token 预算（exp3 标定 ≈2100 tok/s）
TOK_PER_S = 2100.0

# 构成目标（处于 1× 背景档时）：主导 55%，另两类前景各 10%，trans 25%
TARGET = {
    "p1": {"struct": 0.55, "chat": 0.10, "agentic": 0.10, "trans": 0.25},
    "p2": {"chat": 0.55, "struct": 0.10, "agentic": 0.10, "trans": 0.25},
    "p3": {"agentic": 0.55, "struct": 0.10, "chat": 0.10, "trans": 0.25},
    "warmup": {k: 0.25 for k in ("struct", "chat", "agentic", "trans")},
    "cooldown": {k: 0.25 for k in ("struct", "chat", "agentic", "trans")},
}
DOMINANT = {"p1": "struct", "p2": "chat", "p3": "agentic",
            "warmup": "struct", "cooldown": "agentic"}

# 类 -> 源文件 / 输出 task 名 / 等待参数 / 轮数过滤
CLASSES = {
    "trans": dict(
        task="AgentBank_apps",
        src="TaskData/exp3/scan/apps_exp3.nosys.jsonl",
        mu=0.0, sigma=0.0, min_turns=None, max_turns=None,
    ),
    "struct": dict(
        task="Struct",
        src="TaskData/exp3/struct/struct_exp3.nosys.jsonl",
        mu=0.0, sigma=0.0, min_turns=None, max_turns=None,
    ),
    "chat": dict(
        task="ShareGPT",
        src="TaskData/exp3/chat/chat_exp3.nosys.jsonl",
        mu=4.15, sigma=0.971, min_turns=5, max_turns=None,
    ),
    "agentic": dict(
        task="AgentBank_alfred",
        # mt<N> 动态取值：W1 的 MULTI_TURN_MIN=2（旧上游是 5，写死会找不到文件）
        src="TaskData/AgentBankData/Qwen2.5-7B-Instruct/"
            f"AgentBank_alfred_scale_5000_session_data.nosys.mt{MULTI_TURN_MIN}.jsonl",
        mu=1.81, sigma=1.092, min_turns=20, max_turns=40,
    ),
}
BG_LEVELS = {"bg_05": 0.5, "bg_10": 1.0, "bg_20": 2.0}
MARGIN = 1.2            # 会话数余量（λ 反算后可能要多要几条）
SEED = 42


def _num_turns(sess: dict) -> int:
    n = sess.get("num_turns")
    if n is not None:
        return int(n)
    n = 0
    while sess.get(f"turn_{n + 1}") is not None:
        n += 1
    return n


def _prompt_tokens(sess: dict, n_turns: int) -> int:
    total = 0
    for i in range(1, n_turns + 1):
        total += len(((sess.get(f"turn_{i}") or {}).get("input_ids")) or [])
    return total


def load_class(cls: str, limit: int) -> list:
    """读源文件，按轮数过滤，确定性打乱后取 ``limit`` 条。"""
    spec = CLASSES[cls]
    path = HERE / spec["src"]
    if not path.exists():
        raise SystemExit(f"源文件不存在：{path}")
    picked = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 数据格式是「一行一个 {"<sid>": {id, num_turns, turn_1..}}」，
            # 与 sim/session_data.load_task 读的一致；这里剥掉外层键。
            if len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
                obj = next(iter(obj.values()))
            sess = obj
            n = _num_turns(sess)
            if spec["min_turns"] and n < spec["min_turns"]:
                continue
            if spec["max_turns"] and n > spec["max_turns"]:
                continue
            picked.append((line, sess, n))
            if len(picked) >= limit:
                break
    if len(picked) < limit:
        raise SystemExit(
            f"{cls}: 源里只找到 {len(picked)} 条满足轮数过滤的会话（需要 {limit}）"
        )
    random.Random(f"{SEED}-{cls}").shuffle(picked)
    return picked[:limit]


def plan_counts() -> dict:
    """每 slice 每类的会话数上限（按 2× 背景档 + 余量）。"""
    plan = {}
    for slice_name, seconds, _start in SLICES:
        budget = TOK_PER_S * seconds
        row = {}
        for cls, share in TARGET[slice_name].items():
            if cls == "trans":
                share = share * max(BG_LEVELS.values())      # 2× 档最费
            row[cls] = share * budget
        plan[slice_name] = row
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="drift3 数据切片 + tasks.yaml")
    ap.add_argument("--out", default="TaskData/drift3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖已有输出目录")
    ap.add_argument("--phase-s", type=float, default=PHASE_S,
                    help="三个主导相位各自的长度（秒）；1200=原设计，2400=瞬态检验")
    args = ap.parse_args(argv)

    global SLICES, TOTAL_S
    SLICES = make_slices(args.phase_s)
    TOTAL_S = SLICES[-1][2] + SLICES[-1][1]

    out_root = HERE / args.out
    if out_root.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"{out_root} 已存在；加 --force 覆盖")
    out_root = ROOT / out_root if not out_root.is_absolute() else out_root
    out_root = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)

    # ---- 1) 规划需要的会话数（按类求总量，再均分到 slice 的需求上）----
    need = {cls: {} for cls in CLASSES}
    for slice_name, seconds, _start in SLICES:
        for cls, tokens in plan_counts()[slice_name].items():
            need[cls][slice_name] = tokens
    # 先按 tok/会话 估值（用样本算），再定会话数
    sample = {cls: load_class(cls, 200) for cls in CLASSES}
    tok_per_sess = {
        cls: (sum(_prompt_tokens(s, n) for _l, s, n in rows) / max(1, len(rows)))
        for cls, rows in sample.items()
    }
    print(f"[plan] 实测 token/会话： " + " ".join(
        f"{c}={v:.0f}" for c, v in tok_per_sess.items()
    ))

    sess_need = {}
    for cls in CLASSES:
        sess_need[cls] = {
            sl: int(round(need[cls][sl] * MARGIN / max(1.0, tok_per_sess[cls])))
            for sl in need[cls]
        }
    total_need = {cls: sum(v.values()) for cls, v in sess_need.items()}
    print("[plan] 各 slice 需要的会话数：")
    for sl, _sec, _st in SLICES:
        print(f"  {sl:9s} " + " ".join(
            f"{c}={sess_need[c][sl]}" for c in CLASSES
        ))
    print("[plan] 合计： " + " ".join(f"{c}={n}" for c, n in total_need.items()))
    if args.dry_run:
        return 0

    # ---- 2) 取会话 + 切片落盘 ----
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pools = {cls: load_class(cls, total_need[cls]) for cls in CLASSES}
    cursor = {cls: 0 for cls in CLASSES}
    id_seen: dict[str, str] = {}
    manifest = {"slices": {}, "sessions": {}, "tokens": {}, "tok_per_sess": tok_per_sess}

    for sl, seconds, start in SLICES:
        d = out_root / sl
        d.mkdir(parents=True, exist_ok=True)
        manifest["slices"][sl] = {"seconds": seconds, "start_s": start}
        manifest["sessions"][sl] = {}
        manifest["tokens"][sl] = {}
        for cls in CLASSES:
            k = sess_need[cls][sl]
            rows = pools[cls][cursor[cls]: cursor[cls] + k]
            if len(rows) < k:
                raise SystemExit(
                    f"{cls}/{sl}: 需要 {k} 条但只剩 {len(rows)} 条（加大源或降低目标）"
                )
            cursor[cls] += k
            with (d / f"{cls}.jsonl").open("w", encoding="utf-8") as f:
                for line, sess, _n in rows:
                    sid = str(sess.get("id"))
                    if sid in id_seen:
                        raise SystemExit(f"session_id 重复：{sid}（{id_seen[sid]} vs {cls}/{sl}）")
                    id_seen[sid] = f"{cls}/{sl}"
                    f.write(line if line.endswith("\n") else line + "\n")
            manifest["sessions"][sl][cls] = len(rows)
            manifest["tokens"][sl][cls] = sum(_prompt_tokens(s, n) for _l, s, n in rows)

    # ---- 3) 反算 λ：写 yaml（人读）+ plan.json（驱动读）----
    for tag, bg in BG_LEVELS.items():
        plan = {
            "bg": bg,
            "total_s": TOTAL_S,
            "tok_per_s": TOK_PER_S,
            "slices": [
                {"name": sl, "start_s": start, "end_s": start + seconds,
                 "seconds": seconds}
                for sl, seconds, start in SLICES
            ],
            "classes": {},
        }

        lines = [
            f"# drift3 · 背景（trans）注入速率 = {bg:g}×（1× 时 trans 占 25% token）",
            f"# 时间轴：warmup {EDGE_S:g}s → p1/p2/p3 各 {args.phase_s:g}s → cooldown {EDGE_S:g}s",
            f"# 生成器：mixexp/tools/build_drift3_data.py（勿手改）",
            "tasks:",
        ]
        for cls, meta in CLASSES.items():
            segs = []
            for sl, seconds, start in SLICES:
                budget = TOK_PER_S * seconds
                share = TARGET[sl][cls] * (bg if cls == "trans" else 1.0)
                tokens = share * budget
                per_sess = max(1.0, manifest["tokens"][sl][cls] / max(1, manifest["sessions"][sl][cls]))
                lam = tokens / (seconds * per_sess)
                segs.append((start + seconds, lam))
            per_slice = {}
            for sl, seconds, start in SLICES:
                budget = TOK_PER_S * seconds
                share = TARGET[sl][cls] * (bg if cls == "trans" else 1.0)
                tokens = share * budget
                per_sess = max(1.0, manifest["tokens"][sl][cls] / max(1, manifest["sessions"][sl][cls]))
                lam = tokens / (seconds * per_sess)
                per_slice[sl] = {
                    "file": f"{args.out}/{sl}/{cls}.jsonl",
                    "lambda_rate": round(lam, 6),
                    "sessions": manifest["sessions"][sl][cls],
                    "tokens": manifest["tokens"][sl][cls],
                    "target_share": round(share, 4),
                }
            plan["classes"][cls] = {
                "task": meta["task"], "mu": meta["mu"], "sigma": meta["sigma"],
                "per_slice": per_slice,
            }
            lines.append(f"  - name: {meta['task']}")
            lines.append(f"    task_id: {list(CLASSES).index(cls)}")
            lines.append(f"    data_path: {args.out}/{SLICES[0][0]}/{cls}.jsonl")
            lines.append("    # 每个 slice 用独立文件；下面 data_path 只是占位，"
                         "drift3 驱动按 slice 重新拼 tasks")
            lines.append(f"    mu: {meta['mu']}")
            lines.append(f"    sigma: {meta['sigma']}")
            lines.append("    seed: 42")
            lines.append("    arrival:")
            lines.append("      mode: segments")
            lines.append("      segments:")
            for until, lam in segs:
                lines.append(f"        - {{until_s: {until:g}, lambda_rate: {lam:.6f}}}")
        (out_root / f"{tag}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_root / f"{tag}.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] -> {out_root}（{len(id_seen)} 条会话，{len(BG_LEVELS)} 份 yaml）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
