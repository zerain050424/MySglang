#!/usr/bin/env python3
"""节点级标签评估：**节点等权**口径（每个节点算 1 票，不按 token 加权）。

`evaluate_node_labels.py` 是 token 加权口径——它回答「池子里多少比例的 KV 被
正确归属」，巨型 alfred 节点会主导结果，小节点判错几乎不影响总分。
本脚本换成分母是「节点个数」的口径，回答「有多少个节点被判对了」，
两者差距可达 25pt 以上，必须分开看。

三种取样视图（`--views` 可选子集）：

    last        最后一个 tick 的整树快照——唯一一份无重复的静态普查
    cumulative  所有 tick 累加，每个 tick 的节点都计一次（含跨 tick 重复）
    dedup       按 (sid, abs, tok) 唯一化，每棵树上真实存在的节点只算一次，
                取其**最终**标签。跨 tick 稳定，是推荐的汇总口径。

⚠️ cumulative 会随 dump 持续增长而漂移：实验结束后树静止，后续 tick 只是重复
快照同一棵树，会把均值往静止期的准确率拉。要报「这次实验的节点召回」请用 dedup。

真值（GT）仍由节点插入时携带的 session_id 前缀决定：
    Code-… -> struct   apps_… -> random   ShareGPT… -> chat   alfred_… -> agentic

预测（pred）仍由状态机的 queue_tag 决定：
    qstruct -> struct   qagent -> agentic   qchat -> chat
    qtrans  -> pending（未判定，不计入 settled 分母）

用法：
    python eval/evaluate_node_count.py --dump result/node_labels_phased.jsonl \
        --views last,cumulative,dedup --output result/node_count_accuracy.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from evaluate_node_labels import (
    LABELS,
    PENDING,
    TAG_TO_LABEL,
    gt_of,
    load_dump,
    pick_policy,
)

VIEWS = ("last", "cumulative", "dedup")
METRICS = LABELS + (PENDING,)


# ---------------------------------------------------------------- helpers


def _empty_confusion() -> Dict[str, Dict[str, int]]:
    return {g: {p: 0 for p in METRICS} for g in LABELS}


def _pred_of(node: Dict[str, Any]) -> str:
    return TAG_TO_LABEL.get(node.get("tag"), PENDING)


def _node_key(node: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """物理节点标识：同一个节点在多个 tick 里 (sid, abs, tok) 不变。

    abs 是节点在序列中的绝对起始偏移；sid+abs+tok 三元组足以区分不同的 radix 节点。
    注意节点被淘汰后重建会复用这个键，此时「标签迁移」可能是两条不同节点的寿命，
    见 dedup 视图里的 migrations 统计。
    """
    return (node.get("sid"), node.get("abs"), node.get("tok"))


def _score(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """把节点混淆矩阵折算成准确率 / 基线 / 分类别指标。"""
    gt_n = {g: sum(confusion[g].values()) for g in LABELS}
    pred_n = {p: sum(confusion[g][p] for g in LABELS) for p in METRICS}
    total = sum(gt_n.values())
    settled = sum(pred_n[p] for p in LABELS)
    correct = sum(confusion[g][g] for g in LABELS)

    per_class: Dict[str, Dict[str, Any]] = {}
    for g in LABELS:
        row_settled = sum(confusion[g][p] for p in LABELS)
        tp = confusion[g][g]
        recall = tp / row_settled if row_settled else None
        precision = tp / pred_n[g] if pred_n[g] else None
        f1 = (2 * recall * precision / (recall + precision)
              if recall and precision else 0.0)
        per_class[g] = {
            "gt_nodes": gt_n[g],
            "settled_nodes": row_settled,
            "correct_nodes": tp,
            "pending_nodes": confusion[g][PENDING],
            "pending_ratio": (confusion[g][PENDING] / gt_n[g]) if gt_n[g] else None,
            "recall": recall,
            "pred_nodes": pred_n[g],
            "precision": precision,
            "f1": f1,
        }

    # 多数类基线：对所有 settled 节点恒判「settled 里最多的那个类」
    majority = max(LABELS, key=lambda g: sum(confusion[g][p] for p in LABELS))
    baseline = sum(confusion[majority][p] for p in LABELS)

    # macro 只在「该 tick/view 里真的出现过」的类上平均；分母类数单独返回，
    # 否则最后一个 tick（struct/random 已全部被驱逐）会和全时段的口径混为一谈。
    macro_recall = [c["recall"] for c in per_class.values() if c["recall"] is not None]
    return {
        "nodes_total": total,
        "nodes_settled": settled,
        "nodes_pending": pred_n[PENDING],
        "settled_ratio": settled / total if total else None,
        "accuracy_settled": correct / settled if settled else None,
        "baseline_class": majority,
        "baseline_accuracy": baseline / settled if settled else None,
        "delta_vs_baseline": (correct - baseline) / settled if settled else None,
        "macro_recall": sum(macro_recall) / len(macro_recall) if macro_recall else None,
        "macro_recall_classes": len(macro_recall),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(LABELS),
        "confusion_nodes": confusion,
        "pred_nodes": pred_n,
        "per_class": per_class,
    }


# ---------------------------------------------------------------- views


def view_last(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    confusion = _empty_confusion()
    skipped = 0
    for node in records[-1].get("nodes", []):
        gt = gt_of(node.get("sid"))
        if gt is None:
            skipped += 1
            continue
        confusion[gt][_pred_of(node)] += 1
    out = _score(confusion)
    out["tick"] = records[-1].get("tick")
    out["ts"] = records[-1].get("ts")
    out["nodes_skipped_unknown_sid"] = skipped
    return out


def view_cumulative(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    confusion = _empty_confusion()
    skipped = 0
    for record in records:
        for node in record.get("nodes", []):
            gt = gt_of(node.get("sid"))
            if gt is None:
                skipped += 1
                continue
            confusion[gt][_pred_of(node)] += 1
    out = _score(confusion)
    out["ticks"] = len(records)
    out["nodes_skipped_unknown_sid"] = skipped
    return out


def view_dedup(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按物理节点去重，取最后一次观察到的标签作为最终判定。"""
    first: Dict[Tuple[Any, Any, Any], str] = {}
    last: Dict[Tuple[Any, Any, Any], str] = {}
    for record in records:
        for node in record.get("nodes", []):
            if gt_of(node.get("sid")) is None:
                continue
            key = _node_key(node)
            pred = _pred_of(node)
            first.setdefault(key, pred)
            last[key] = pred

    confusion = _empty_confusion()
    migrations: Dict[str, int] = {}
    for key, pred in last.items():
        confusion[gt_of(key[0])][pred] += 1
        if first[key] != pred:
            migrations[f"{first[key]}->{pred}"] = migrations.get(
                f"{first[key]}->{pred}", 0) + 1

    out = _score(confusion)
    out["unique_nodes"] = len(last)
    out["first_tag_dist"] = dict(sorted(
        ((t, sum(1 for v in first.values() if v == t)) for t in set(first.values())),
        key=lambda kv: -kv[1]))
    out["final_tag_dist"] = dict(sorted(
        ((t, sum(1 for v in last.values() if v == t)) for t in set(last.values())),
        key=lambda kv: -kv[1]))
    out["migrated_nodes"] = sum(migrations.values())
    out["migrated_ratio"] = (sum(migrations.values()) / len(last)) if last else None
    out["migrations"] = dict(sorted(migrations.items(), key=lambda kv: -kv[1]))
    return out


def per_tick(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        confusion = _empty_confusion()
        for node in record.get("nodes", []):
            gt = gt_of(node.get("sid"))
            if gt is None:
                continue
            confusion[gt][_pred_of(node)] += 1
        score = _score(confusion)
        rows.append({
            "tick": record.get("tick"),
            "nodes_settled": score["nodes_settled"],
            "accuracy_settled": score["accuracy_settled"],
            "baseline_accuracy": score["baseline_accuracy"],
        })
    return rows


# ---------------------------------------------------------------- report


def _pct(x: Optional[float]) -> str:
    return "   n/a" if x is None else f"{100 * x:5.2f}%"


def print_view(name: str, s: Dict[str, Any]) -> None:
    labels = {
        "last": f"最后一个 tick（tick={s.get('tick')}）",
        "cumulative": f"全 {s.get('ticks')} tick 节点记录累加",
        "dedup": f"去重物理节点（唯一 {s.get('unique_nodes')} 个）",
    }
    print(f"\n===== {name} —— {labels[name]} =====")
    print(f"  节点总数={s['nodes_total']}  settled={s['nodes_settled']}  "
          f"pending={s['nodes_pending']}  settled 占比={_pct(s['settled_ratio'])}")
    print(f"  节点准确率(settled) = {_pct(s['accuracy_settled'])}   "
          f"基线(恒判 {s['baseline_class']}) = {_pct(s['baseline_accuracy'])}   "
          f"差 = {100 * s['delta_vs_baseline']:+.2f}pt")
    print(f"  macro recall = {_pct(s['macro_recall'])} "
          f"(仅 {s['macro_recall_classes']}/4 类有样本)   "
          f"macro F1 = {_pct(s['macro_f1'])} (4 类)")
    print(f"  {'GT':<9}{'GT_n':>7}{'settled':>9}{'correct':>9}{'recall':>9}"
          f"{'pred_n':>8}{'precision':>11}{'F1':>9}")
    for g in LABELS:
        c = s["per_class"][g]
        print(f"  {g:<9}{c['gt_nodes']:>7}{c['settled_nodes']:>9}{c['correct_nodes']:>9}"
              f"{_pct(c['recall']):>9}{c['pred_nodes']:>8}{_pct(c['precision']):>11}"
              f"{_pct(c['f1']):>9}")
    print(f"  {'':9}" + "".join(f"{p:>9}" for p in METRICS) + f"{'GT合计':>9}")
    for g in LABELS:
        row = s["confusion_nodes"][g]
        print(f"  {g:<9}" + "".join(f"{row[p]:>9}" for p in METRICS)
              + f"{sum(row.values()):>9}")
    print(f"  {'pred合计':<9}" + "".join(f"{s['pred_nodes'][p]:>9}" for p in METRICS))


def print_report(result: Dict[str, Any]) -> None:
    for name in VIEWS:
        if name in result["views"]:
            print_view(name, result["views"][name])

    dedup = result["views"].get("dedup")
    if dedup:
        print("\n===== 去重视图：标签迁移 =====")
        print(f"  首次标签分布 = {dedup['first_tag_dist']}")
        print(f"  最终标签分布 = {dedup['final_tag_dist']}")
        print(f"  发生迁移的节点 = {dedup['migrated_nodes']} "
              f"({_pct(dedup['migrated_ratio'])})  方向 = {dedup['migrations']}")

    rows = result["per_tick"]
    vals = [r["accuracy_settled"] for r in rows if r["accuracy_settled"] is not None]
    if vals:
        vals_sorted = sorted(vals)
        print("\n===== 逐 tick 节点准确率 =====")
        print(f"  tick 数={len(vals)}  均值={_pct(sum(vals) / len(vals))}  "
              f"中位={_pct(vals_sorted[len(vals_sorted) // 2])}  "
              f"最小={_pct(vals_sorted[0])}  最大={_pct(vals_sorted[-1])}")
        below = [r for r in rows
                 if r["accuracy_settled"] is not None
                 and r["baseline_accuracy"] is not None
                 and r["accuracy_settled"] < r["baseline_accuracy"]]
        print(f"  低于当 tick 基线的 tick 数 = {len(below)}/{len(rows)}")
        step = max(1, len(rows) // 10)
        for r in rows[::step]:
            print(f"    tick={r['tick']:>4} settled={r['nodes_settled']:>5} "
                  f"acc={_pct(r['accuracy_settled'])} baseline={_pct(r['baseline_accuracy'])}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="SGLANG_NODE_LABEL_DUMP 落盘的 JSONL")
    p.add_argument("--output", required=True, help="结果 JSON 输出路径")
    p.add_argument("--views", default="last,cumulative,dedup",
                   help=f"逗号分隔，可选 {','.join(VIEWS)}（默认全开）")
    p.add_argument("--tail-ticks", type=int, default=0, help="只统计最后 N 个 tick（0=全部）")
    p.add_argument("--policy", default=None,
                   help="只统计该 policy 的快照（默认自动选 token 最多的那棵主树）")
    args = p.parse_args()

    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = set(wanted) - set(VIEWS)
    if unknown:
        raise SystemExit(f"未知 view: {sorted(unknown)}；可选 {list(VIEWS)}")

    records = load_dump(args.dump)
    if not records:
        raise SystemExit(f"dump 为空: {args.dump}")
    policy, totals = pick_policy(records, args.policy)
    if policy is None:
        raise SystemExit(f"dump 里没有有效节点: {args.dump}")
    records = [r for r in records if (r.get("policy") or "?") == policy]
    print(f"[tree] 选中 policy={policy}；各树 token 总量={totals}")
    if args.tail_ticks > 0:
        records = records[-args.tail_ticks:]

    builders = {"last": view_last, "cumulative": view_cumulative, "dedup": view_dedup}
    result = {
        "dump": args.dump,
        "policy": policy,
        "ticks": len(records),
        "metric": "node_count",
        "views": {name: builders[name](records) for name in wanted},
        "per_tick": per_tick(records),
    }

    print_report(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
