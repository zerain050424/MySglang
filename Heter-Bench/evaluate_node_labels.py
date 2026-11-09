#!/usr/bin/env python3
"""节点级标签准确率评估：状态机给 radix 节点的队列标签，能否正确反映其真实归属。

数据来源：`SGLANG_NODE_LABEL_DUMP` 让 server 每 N 秒把整棵树逐节点落盘为 JSONL
（见 `radix_cache.py:_dump_node_labels`）。

真值（GT）：节点插入时携带的 `session_id` 前缀 → 负载类别
    Code-…               -> struct   (Codeparrot)
    apps_…               -> random   (AgentBank apps)
    ShareGPT-multi-turn… -> chat     (ShareGPT)
    alfred_…             -> agentic  (AgentBank alfred)

预测（pred）：状态机的 `queue_tag`
    qstruct -> struct    qagent -> agentic    qchat -> chat
    qtrans  -> 「未判定」（不是 random！非共享内容/尚未复用都停在这里）

用法：
    python evaluate_node_labels.py --dump result/node_labels_phased.jsonl \
        --output result/node_label_accuracy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LABELS = ("struct", "random", "chat", "agentic")
PENDING = "pending"

# session_id 前缀 -> 真值类别（顺序敏感：apps_ 与 alfred_ 都以 a 开头）
SID_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("apps_", "random"),
    ("alfred_", "agentic"),
    ("ShareGPT", "chat"),
    ("Code-", "struct"),
)

# queue_tag -> 预测类别；qtrans 单独作为「未判定」
TAG_TO_LABEL = {
    "qstruct": "struct",
    "qagent": "agentic",
    "qchat": "chat",
}


def gt_of(sid: Optional[str]) -> Optional[str]:
    if not sid:
        return None
    for prefix, label in SID_PREFIXES:
        if sid.startswith(prefix):
            return label
    return None


def load_dump(path: str) -> List[Dict[str, Any]]:
    ticks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("nodes"):  # 空快照跳过
                ticks.append(record)
    return ticks


def pick_policy(records: List[Dict[str, Any]], override: Optional[str]) -> Tuple[Optional[str], Dict[str, float]]:
    """同一进程里可能不止一棵树（例如额外创建的空 RadixCache）。

    这里按 token 总量选出「主树」对应的 policy，避免把两棵树混在一起统计。
    """
    totals: Dict[str, float] = {}
    for r in records:
        pol = r.get("policy") or "?"
        totals[pol] = totals.get(pol, 0.0) + sum(
            float(n.get("tok", 0)) for n in r.get("nodes", [])
        )
    if override:
        return override, totals
    if not totals:
        return None, totals
    return max(totals.items(), key=lambda kv: kv[1])[0], totals


def _empty_confusion() -> Dict[str, Dict[str, float]]:
    return {g: {p: 0.0 for p in LABELS + (PENDING,)} for g in LABELS}


def score_tick(record: Dict[str, Any]) -> Dict[str, Any]:
    """对单个 tick 的节点快照做统计。"""
    confusion = _empty_confusion()
    nodes_by = {g: {p: 0 for p in LABELS + (PENDING,)} for g in LABELS}
    gt_tokens = {g: 0.0 for g in LABELS}
    gt_nodes = {g: 0 for g in LABELS}
    tag_tokens: Dict[str, float] = {}
    skipped_unknown_sid = 0

    for node in record.get("nodes", []):
        gt = gt_of(node.get("sid"))
        tok = float(node.get("tok", 0))
        tag = node.get("tag", "qtrans")
        tag_tokens[tag] = tag_tokens.get(tag, 0.0) + tok
        if gt is None:
            skipped_unknown_sid += 1
            continue
        pred = TAG_TO_LABEL.get(tag, PENDING)
        confusion[gt][pred] += tok
        nodes_by[gt][pred] += 1
        gt_tokens[gt] += tok
        gt_nodes[gt] += 1

    total = sum(gt_tokens.values())
    settled = sum(confusion[g][p] for g in LABELS for p in LABELS)
    correct = sum(confusion[g][g] for g in LABELS)

    per_class = {}
    for g in LABELS:
        if not gt_tokens[g]:
            continue
        row_settled = sum(confusion[g][p] for p in LABELS)
        per_class[g] = {
            "gt_tokens": gt_tokens[g],
            "gt_nodes": gt_nodes[g],
            "pending_tokens": confusion[g][PENDING],
            "pending_ratio": confusion[g][PENDING] / gt_tokens[g] if gt_tokens[g] else None,
            "settled_tokens": row_settled,
            "correct_tokens": confusion[g][g],
            "recall": confusion[g][g] / row_settled if row_settled else None,
        }
    precision = {}
    for p in LABELS:
        pred_total = sum(confusion[g][p] for g in LABELS)
        precision[p] = {
            "pred_tokens": pred_total,
            "precision": confusion[p][p] / pred_total if pred_total else None,
        }

    return {
        "tick": record.get("tick"),
        "ts": record.get("ts"),
        "nodes_total": sum(gt_nodes.values()),
        "nodes_skipped_unknown_sid": skipped_unknown_sid,
        "tokens_total": total,
        "tokens_settled": settled,
        "tokens_pending": sum(confusion[g][PENDING] for g in LABELS),
        "settled_ratio": settled / total if total else None,
        "accuracy_settled": correct / settled if settled else None,
        "tag_tokens": tag_tokens,
        "confusion_tokens": confusion,
        "confusion_nodes": nodes_by,
        "per_class": per_class,
        "precision": precision,
    }


def merge_ticks(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把所有 tick 的混淆矩阵累加（等价于按 tick 时间加权）。"""
    confusion = _empty_confusion()
    for t in ticks:
        for g in LABELS:
            for p in LABELS + (PENDING,):
                confusion[g][p] += t["confusion_tokens"][g][p]
    total = sum(confusion[g][p] for g in LABELS for p in LABELS + (PENDING,))
    settled = sum(confusion[g][p] for g in LABELS for p in LABELS)
    correct = sum(confusion[g][g] for g in LABELS)
    return {
        "tokens_total": total,
        "tokens_settled": settled,
        "settled_ratio": settled / total if total else None,
        "accuracy_settled": correct / settled if settled else None,
        "confusion_tokens": confusion,
    }


def fmt_pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def print_report(ticks: List[Dict[str, Any]], last: Dict[str, Any]) -> None:
    agg = merge_ticks(ticks)
    print(f"ticks={len(ticks)}  last_tick={last['tick']}  ts={last['ts']}")

    print("\n== 最后一个 tick 的总体 ==")
    print(f"  统计节点数(有 session_id)={last['nodes_total']}  "
          f"跳过(session_id 缺失/未知)={last['nodes_skipped_unknown_sid']}")
    print(f"  token 总数={last['tokens_total']:.0f}  "
          f"已判定(settled)={last['tokens_settled']:.0f} "
          f"({fmt_pct(last['settled_ratio'])})  "
          f"未判定(qtrans)={last['tokens_pending']:.0f}")
    print(f"  settled 准确率 = {fmt_pct(last['accuracy_settled'])}")
    print(f"  全时段(tick 累加) settled 准确率 = {fmt_pct(agg['accuracy_settled'])}  "
          f"settled 占比 {fmt_pct(agg['settled_ratio'])}")

    print("\n== 队列标签 token 分布（最后一个 tick）==")
    for tag, tok in sorted(last["tag_tokens"].items(), key=lambda x: -x[1]):
        share = tok / last["tokens_total"] if last["tokens_total"] else 0
        print(f"  {tag:8s} {tok:9.0f} tok  {fmt_pct(share)}")

    print("\n== 混淆矩阵（token 加权，只含 settled 节点；行=真值 列=预测）==")
    print("  " + " " * 10 + "".join(f"{p:>10s}" for p in LABELS) + f"{'pending':>10s}{'GT合计':>10s}")
    for g in LABELS:
        row = last["confusion_tokens"][g]
        print(f"  {g:10s}" + "".join(f"{row[p]:10.0f}" for p in LABELS)
              + f"{row[PENDING]:10.0f}{sum(row.values()):10.0f}")

    print("\n== 分类别：未判定比例 / settled 召回率 ==")
    for g in LABELS:
        c = last["per_class"].get(g)
        if not c:
            continue
        print(f"  GT={g:8s} tok={c['gt_tokens']:8.0f}  "
              f"仍为 qtrans={fmt_pct(c['pending_ratio'])}  "
              f"settled 召回={fmt_pct(c['recall'])}")

    print("\n== 各类预测的纯度（precision）==")
    for p in LABELS:
        c = last["precision"][p]
        print(f"  pred={p:8s} tok={c['pred_tokens']:8.0f}  precision={fmt_pct(c['precision'])}")

    print("\n== 逐 tick 轨迹 ==")
    step = max(1, len(ticks) // 12)
    for t in ticks[::step]:
        print(f"  tick={t['tick']:>3} nodes={t['nodes_total']:>5} "
              f"acc={fmt_pct(t['accuracy_settled'])} "
              f"settled={fmt_pct(t['settled_ratio'])}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="SGLANG_NODE_LABEL_DUMP 落盘的 JSONL")
    p.add_argument("--output", required=True, help="结果 JSON 输出路径")
    p.add_argument("--tail-ticks", type=int, default=0, help="只统计最后 N 个 tick（0=全部）")
    p.add_argument("--policy", default=None,
                   help="只统计该 policy 的快照（默认自动选 token 最多的那棵主树）")
    args = p.parse_args()

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

    ticks = [score_tick(r) for r in records]
    last = ticks[-1]
    print_report(ticks, last)

    result = {
        "dump": args.dump,
        "policy": policy,
        "ticks": len(ticks),
        "final": last,
        "aggregate": merge_ticks(ticks),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
