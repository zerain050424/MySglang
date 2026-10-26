#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Codeparrot(APPS) 原始数据 → 单轮对话 JSON（服务器版，直接读本地 jsonl）。

【与原版差异】
  原版用 `datasets.load_dataset("codeparrot/apps", split="test", difficulties=["introductory"])`
  从 HF 拉数据；服务器已用 hf-mirror aria2 下载好 test.jsonl（1.29GB），
  这里改为直接读本地 jsonl，避免重新下载。字段与 apps 数据集一致：
  {id, question, solutions(str-JSON), input_output(str-JSON), difficulty, url, starter_code}

【逻辑复刻（对齐 UniCache TaskData/Codeparrot/prepare_data.py）】
  1. 只保留 difficulty == "introductory" 的题目（与 load_dataset 过滤一致）。
  2. 每道题取第一个 input/output 样例拼进 prompt：
       "Question: <question> <Use Standard/Call-Based format> <io> \n\nAnswer:\n"
  3. 随机选 1..N 个 solution（N ~ gauss(6.5, 1.5)，clamp 到 [1, len(sols)]），
     每个 solution 生成一条单轮对话 {id: "<prefix>-<uuid>-<idx>", conversations: [user, assistant]}。

【分类标签（重要）】
  id 前缀决定 SimpleHybrid 的 task_type 分类（见 uni_utils.classify_simple_hybrid_task_type）：
    - 默认 "Code-"  → SINGLE_TURN（与原版 UniCache 一致）
    - 改 "--id-prefix apps_" → SINGLE_TURN_RANDOM（apps 数据集专属队列，论文里 Codeparrot 归此类）
  发送时 bench_unicache.py 会把 id 原样作为 session_id，故前缀直接影响分类。

【用法】
  /root/sl_old/bin/python prepare_data.py \
      --input  /root/autodl-tmp/TaskData/Codeparrot/test.jsonl \
      --output Code.json [--id-prefix Code-] [--max-samples 1000]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import statistics as stats
import uuid


# 共享长前缀：所有请求共用的 system prompt
CODING_SYSTEM_PROMPT = """You are an expert competitive programmer and Python engineer. Solve the given programming problem by writing a complete, correct, and efficient Python program.

Follow these instructions carefully:
1. Read the problem statement, input specification, output specification, and the provided examples thoroughly before writing any code.
2. Reason about the algorithm and its time and memory complexity before implementing. Choose an approach that runs within the given constraints.
3. Write a single, self-contained Python program that reads from standard input and writes to standard output. Do not read from files or use hard-coded test cases.
4. Include a concise explanation of your approach and its complexity as a comment block at the top of the program.
5. Handle edge cases, including empty input, large inputs, and boundary values.
6. Use only Python's standard library unless the problem explicitly requires otherwise.
7. Follow PEP 8 style: meaningful variable names, consistent indentation, and no dead code.
8. Verify your solution against the provided examples before producing the final answer.
9. Pay attention to integer overflow and floating-point precision issues when applicable.
10. Prefer clear and readable code over clever tricks; correctness always comes first.
11. For large inputs, read efficiently using sys.stdin and avoid repeated expensive operations inside loops.
12. Keep the program complete and runnable, with no missing functions or placeholders.

Format your answer as follows:
[Explanation]
A concise explanation of the algorithm and its time/space complexity.

[Code]
```python
<your complete program>
```"""


# --------------------------------------------------------------------------
# 字段解析（原版同款，容错单引号/格式问题）
# --------------------------------------------------------------------------
def parse_io(io_raw):
    """input_output 字段 → {"inputs": [...], "outputs": [...]}"""
    if isinstance(io_raw, dict):
        return io_raw
    if io_raw is None:
        return {"inputs": [], "outputs": []}
    s = str(io_raw).strip()
    if not s:
        return {"inputs": [], "outputs": []}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {"inputs": [], "outputs": []}


def parse_solutions(sol_raw):
    """solutions 字段 → 字符串列表"""
    if isinstance(sol_raw, list):
        return sol_raw
    if sol_raw is None:
        return []
    s = str(sol_raw).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else [s]
    except Exception:
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else [s]
        except Exception:
            return [s]


# --------------------------------------------------------------------------
# 单题 → 多条单轮对话（对齐原版 build_raw_sample）
# --------------------------------------------------------------------------
def build_raw_sample(sample, id_prefix):
    question = sample.get("question", "")
    io = parse_io(sample.get("input_output"))
    inputs = (io.get("inputs", []) or [])[:1]   # 只取第一个 IO 样例
    outputs = (io.get("outputs", []) or [])[:1]

    input_output_strings = "".join(str(x) + "\n" for x in inputs)
    input_output_strings += "".join(str(y) + "\n" for y in outputs)
    # APPS 数据里 fn_name 存在 → 函数调用格式，否则标准输入格式
    input_format = "\nUse Standard Input format" if not sample.get("fn_name") else "\nUse Call-Based format"
    sample_prompt = f"Question: {question} {input_format} {input_output_strings} \n\nAnswer:\n"

    sols = parse_solutions(sample.get("solutions"))
    sample_id = id_prefix + str(uuid.uuid4())

    mu, sigma = 6.5, 1.5
    if len(sols) == 0:
        selected_indices = []
    else:
        constraint = int(random.gauss(mu, sigma))
        constraint = max(1, min(constraint, len(sols)))
        selected_indices = random.sample(range(len(sols)), constraint)

    result = []
    for idx in selected_indices:
        item = {
            "id": f"{sample_id}-{idx}",
            "conversations": [
                {"from": "system", "value": CODING_SYSTEM_PROMPT},
                {"from": "user", "value": sample_prompt},
                {"from": "assistant", "value": str(sols[idx])},
            ],
        }
        result.append(item)
    return result


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Codeparrot(APPS) 原始 jsonl → 单轮对话 JSON")
    p.add_argument("--input", default="/root/autodl-tmp/TaskData/Codeparrot/test.jsonl",
                   help="apps test.jsonl 路径")
    p.add_argument("--output", default="Code.json", help="输出对话 JSON")
    p.add_argument("--id-prefix", default="Code-",
                   help="id 前缀（默认 Code-→SINGLE_TURN；apps_→SINGLE_TURN_RANDOM）")
    p.add_argument("--max-samples", type=int, default=-1, help="最多处理题目数（-1=全部）")
    args = p.parse_args()

    print(f"[Codeparrot] 读取 {args.input}")
    rows = []
    for i, line in enumerate(open(args.input, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        # 只保留 introductory（与 load_dataset(difficulties=["introductory"]) 一致）
        if r.get("difficulty") != "introductory":
            continue
        rows.append(r)
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break
    print(f"[Codeparrot] introductory 题目数: {len(rows)}")

    raw_data = []
    per_sample_counts = []
    for s in rows:
        res = build_raw_sample(s, args.id_prefix)
        per_sample_counts.append(len(res))
        raw_data += res
    print(f"[Codeparrot] 生成对话数: {len(raw_data)}")
    if per_sample_counts:
        print(f"  Mean={stats.mean(per_sample_counts):.2f} Median={stats.median(per_sample_counts):.1f} "
              f"Stdev={stats.stdev(per_sample_counts):.2f} Min={min(per_sample_counts)} Max={max(per_sample_counts)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"[Codeparrot] ✅ 完成 → {args.output}")


if __name__ == "__main__":
    main()
