"""会话级指标：准确率、混淆矩阵、P/R/F1、收敛轮、可达上限、逐轮精度曲线。

口径约定（与 `evaluate_node_labels.py` 对齐）：

- 单元是「会话」：一条会话最终只贡献一个判定，即它最后一轮结束时的 `session_type`。
- 类别四桶：struct / random / chat / agentic，其中 `trans` 映射到 random。
- `acc_at_turn[k]` 是「只重放前 k 轮」时的准确率（只统计轮数 >= k 的会话），
  用来回答分类器「多早才能定型」，而不是拿短会话充数。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from session_data import GT_LABELS

LABELS = GT_LABELS


def confusion_matrix(results: Sequence) -> List[List[int]]:
    idx = {lab: i for i, lab in enumerate(LABELS)}
    cm = [[0] * len(LABELS) for _ in LABELS]
    for res in results:
        cm[idx[res.gt]][idx[res.final]] += 1
    return cm


def per_class_prf(cm: Sequence[Sequence[int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        support = sum(cm[i])
        pred = sum(row[i] for row in cm)
        precision = tp / pred if pred else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return out


def accuracy(results: Sequence) -> float:
    if not results:
        return 0.0
    return sum(res.gt == res.final for res in results) / len(results)


def macro_f1(prf: Dict[str, Dict[str, float]]) -> float:
    return round(statistics.fmean(v["f1"] for v in prf.values()), 4)


def accuracy_curve(results: Sequence) -> List[Dict[str, float]]:
    """acc_at_turn：截断到前 k 轮时的会话级准确率。"""
    max_turns = max((res.n_turns for res in results), default=0)
    curve = []
    for k in range(1, max_turns + 1):
        subset = [r for r in results if r.n_turns >= k]
        if not subset:
            continue
        hit = sum(r.preds[k - 1] == r.gt for r in subset)
        curve.append({"turn": k, "n": len(subset), "accuracy": round(hit / len(subset), 4)})
    return curve


def converged_turn(res) -> int:
    """判定第一次等于最终判定、且此后不再改变的轮次（1-based）。"""
    final = res.final
    turn = len(res.preds)
    for k in range(len(res.preds), 0, -1):
        if res.preds[k - 1] != final:
            break
        turn = k
    return turn


def convergence_stats(results: Sequence) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for lab in LABELS:
        turns = [converged_turn(r) for r in results if r.gt == lab]
        if turns:
            out[lab] = {
                "median_turn": statistics.median(turns),
                "p90_turn": sorted(turns)[min(len(turns) - 1, int(0.9 * len(turns)))],
                "n": len(turns),
            }
    return out


def ceiling(results: Sequence, multi_turn_min: int) -> Dict[str, object]:
    """规则可达上限。

    `>4 轮` 门槛是不可绕过的结构性限制：chat / agentic 只有在轮数 >= multi_turn_min
    时才可能被判出来；struct / random 不受轮数限制（其可达性取决于有没有结构证据）。
    因此上限 = 可达会话占比，`accuracy/ceiling` 才是「规则范围内的达成率」。
    """
    reachable = []
    unreachable = {"chat": 0, "agentic": 0, "struct": 0, "random": 0}
    for res in results:
        ok = True
        if res.gt in ("chat", "agentic") and res.n_turns < multi_turn_min:
            ok = False
        if not ok:
            unreachable[res.gt] += 1
        reachable.append(ok)
    frac = sum(reachable) / len(results) if results else 0.0
    return {
        "ceiling": round(frac, 4),
        "unreachable_by_class": unreachable,
        "missed_owner": {
            lab: (unreachable[lab] / sum(1 for r in results if r.gt == lab))
            if sum(1 for r in results if r.gt == lab)
            else 0.0
            for lab in LABELS
        },
    }


@dataclass
class Report:
    target: str
    config: Dict[str, object]
    n_sessions: int
    by_class: Dict[str, int]
    accuracy: float
    macro_f1: float
    prf: Dict[str, Dict[str, float]]
    confusion: List[List[int]]
    curve: List[Dict[str, float]]
    convergence: Dict[str, Dict[str, float]]
    ceiling: Dict[str, object]
    structure_stats: Dict[str, object]
    results: Optional[Sequence] = None


def summarize(results: Sequence, cfg, target: str) -> Report:
    cm = confusion_matrix(results)
    prf = per_class_prf(cm)
    ceil = ceiling(results, cfg.multi_turn_min)
    by_class = {lab: sum(1 for r in results if r.gt == lab) for lab in LABELS}
    shared = [r.shared_len for r in results]
    structure_stats = {
        "sessions_with_shared_prefix": sum(1 for r in results if r.shared_len > 0),
        "shared_prefix_median": statistics.median(shared) if shared else 0,
        "shared_prefix_max": max(shared) if shared else 0,
        "min_share_tokens": cfg.min_share_tokens,
        "structure": cfg.structure,
    }
    acc = accuracy(results)
    return Report(
        target=target,
        config={
            "multi_turn_min": cfg.multi_turn_min,
            "gap_split_s": cfg.gap_split_s,
            "growth_tokens": cfg.growth_tokens,
            "structure": cfg.structure,
            "structure_turns": cfg.structure_turns,
            "block": cfg.block,
            "min_share_tokens": cfg.min_share_tokens,
            "tool_call_mode": cfg.tool_call_mode,
        },
        n_sessions=len(results),
        by_class=by_class,
        accuracy=round(acc, 4),
        macro_f1=macro_f1(prf),
        prf=prf,
        confusion=cm,
        curve=accuracy_curve(results),
        convergence=convergence_stats(results),
        ceiling={**ceil, "achievement": round(acc / ceil["ceiling"], 4) if ceil["ceiling"] else 0.0},
        structure_stats=structure_stats,
        results=results,
    )


def render_markdown(rep: Report) -> str:
    cfg = rep.config
    lines = [
        f"# 会话级分类准确率 — {rep.target}",
        "",
        f"- 会话数：**{rep.n_sessions}**（" + "，".join(f"{k}={v}" for k, v in rep.by_class.items()) + "）",
        f"- 配置：`{cfg}`",
        "",
        "## 总体",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 会话级准确率 | **{rep.accuracy:.4f}** |",
        f"| macro-F1 | {rep.macro_f1:.4f} |",
        f"| 规则可达上限 | {rep.ceiling['ceiling']:.4f} |",
        f"| 达成率（acc / 上限） | **{rep.ceiling['achievement']:.4f}** |",
        f"| 观测到跨会话共享前缀的会话 | {rep.structure_stats['sessions_with_shared_prefix']}"
        f"（中位共享 {rep.structure_stats['shared_prefix_median']} tok，"
        f"max {rep.structure_stats['shared_prefix_max']} tok） |",
        "",
        "## 混淆矩阵（行=GT，列=pred）",
        "",
        "| GT \\ pred | " + " | ".join(LABELS) + " | 合计 |",
        "| --- | " + " | ".join("---" for _ in LABELS) + " | --- |",
    ]
    for i, lab in enumerate(LABELS):
        row = rep.confusion[i]
        lines.append(f"| {lab} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")
    lines += [
        "",
        "## 分类指标",
        "",
        "| 类别 | precision | recall | F1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lab in LABELS:
        m = rep.prf[lab]
        lines.append(
            f"| {lab} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
        )
    lines += [
        "",
        "## 逐轮精度曲线（只统计轮数 >= k 的会话）",
        "",
        "| k | 会话数 | 准确率 |",
        "| --- | --- | --- |",
    ]
    for point in rep.curve:
        lines.append(f"| {point['turn']} | {point['n']} | {point['accuracy']:.4f} |")
    lines += [
        "",
        "## 判定收敛轮次",
        "",
        "| 类别 | 中位轮次 | p90 | 会话数 |",
        "| --- | --- | --- | --- |",
    ]
    for lab in LABELS:
        c = rep.convergence.get(lab)
        if c:
            lines.append(f"| {lab} | {c['median_turn']} | {c['p90_turn']} | {c['n']} |")
    lines += [
        "",
        "## 规则不可达会话（>4 轮门槛）",
        "",
        "| 类别 | 不可达会话数 | 占该类比例 |",
        "| --- | --- | --- |",
    ]
    for lab in LABELS:
        n = rep.ceiling["unreachable_by_class"][lab]
        lines.append(f"| {lab} | {n} | {rep.ceiling['missed_owner'][lab]:.3f} |")
    return "\n".join(lines) + "\n"


def write_sessions_csv(path, results: Sequence) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["session_id", "gt", "pred", "correct", "n_turns", "converged_turn", "shared_len", "preds"]
        )
        for r in results:
            w.writerow(
                [
                    r.session_id,
                    r.gt,
                    r.final,
                    int(r.gt == r.final),
                    r.n_turns,
                    converged_turn(r),
                    r.shared_len,
                    "/".join(r.preds),
                ]
            )
