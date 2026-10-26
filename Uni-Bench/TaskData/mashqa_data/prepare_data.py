#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MASHQA（webmd_squad_v2）原始 JSON → 单轮对话 JSON（服务器版）。

【逻辑复刻（对齐 UniCache TaskData/mashqa_data/prepare_data.py）】
  1. 读 train/test/val 三个 _full.json，每个 "data" 元素是一个文档（paragraph）。
  2. 每个文档取 context 作为长上下文，随机取 ~gauss(5.74, 1.0)（clamp 到 [1, len(qas)]）个问题。
  3. 每个问题生成一条单轮对话：
       prompt = "Please answer the question based on the texts below.\\n{doc}\\nQuestion: {q}\\nAnswer: "
     —— 同一文档的多个问题共享整段长 context，构成"结构化复用"的长共享前缀。
  4. id 前缀 "QA-" → sglang SimpleHybrid 分类为 SINGLE_TURN（结构化单轮队列）。

【用法】
  /root/sl_old/bin/python prepare_data.py \
      --input-dir /root/autodl-tmp/TaskData/mashqa_data \
      --output mashqa.json [--max-docs 200]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as stats
from pathlib import Path

FULL_FILES = [
    "train_webmd_squad_v2_full.json",
    "test_webmd_squad_v2_full.json",
    "val_webmd_squad_v2_full.json",
]


def build_raw_sample(item: dict) -> list:
    """一个文档（item）→ 多条单轮对话（每条一个 QA）。"""
    prar = item["paragraphs"][0]
    doc = prar["context"]
    res = []
    constraint = int(random.gauss(mu=5.74, sigma=1.0))
    constraint = max(1, min(constraint, len(prar["qas"])))
    for qa in prar["qas"][:constraint]:
        question = qa["question"]
        answers = qa["answers"]
        if not answers:
            continue
        answer_text = answers[0]["text"]
        sample = {
            "id": "QA-" + qa["id"],
            "conversations": [
                {
                    "from": "user",
                    "value": (
                        "Please answer the question based on the texts below.\n"
                        f"{doc}\n"
                        f"Question: {question}\n"
                        "Answer: "
                    ),
                },
                {"from": "assistant", "value": answer_text},
            ],
        }
        res.append(sample)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="MASHQA 原始 JSON → 单轮对话 JSON")
    p.add_argument("--input-dir", default="/root/autodl-tmp/TaskData/mashqa_data")
    p.add_argument("--output", default="mashqa.json")
    p.add_argument("--max-docs", type=int, default=-1, help="最多处理文档数（-1=全部）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    indir = Path(args.input_dir)

    all_docs = []
    for fn in FULL_FILES:
        fp = indir / fn
        if not fp.exists():
            print(f"[MASHQA] skip missing {fp}")
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)["data"]
        all_docs.extend(data)
        print(f"[MASHQA] {fn}: {len(data)} docs")
    print(f"[MASHQA] total docs: {len(all_docs)}")

    if args.max_docs > 0:
        all_docs = all_docs[: args.max_docs]
        print(f"[MASHQA] only first {args.max_docs} docs")

    raw_data = []
    per_doc_counts = []
    for item in all_docs:
        samples = build_raw_sample(item)
        per_doc_counts.append(len(samples))
        raw_data.extend(samples)

    print(f"[MASHQA] generated turns: {len(raw_data)}")
    if per_doc_counts:
        print(f"  Mean={stats.mean(per_doc_counts):.2f} "
              f"Median={stats.median(per_doc_counts):.1f} "
              f"Min={min(per_doc_counts)} Max={max(per_doc_counts)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"[MASHQA] done -> {out}")


if __name__ == "__main__":
    main()
