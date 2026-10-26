#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniCache 数据准备：raw 对话 JSON → session_data.jsonl（独立版，无 vllm 依赖）。

【背景】
  本地 UniCache 用 vllm 的 vllm.simulator.data_parser.prepare_simulator_data_streaming_jsonl
  把"多轮对话原始数据"转成逐 session、逐 turn 的 token 序列。服务器上没有 vllm，
  这里用纯 transformers 复刻同样的逻辑（输出格式逐字节一致），方便后续
  bench_unicache.py 直接复用。

【输出格式】（与 UniCache 完全一致，一行一个 session）
  {"<session_id>": {
      "id": "<session_id>",
      "num_turns": N,
      "turn_1": {"input_text": ..., "input_ids": [...], "response_text": ..., "response_ids": [...]},
      ...
  }}

【关键点】
  1. tokenizer 必须与 sglang server 加载的模型一致（本项目用 Qwen2.5-7B-Instruct）。
     server 直发 input_ids，tokenizer 不一致会全部前缀 miss。
  2. input_ids 带 BOS（add_special_tokens=True）；response_ids 不带 BOS（对齐 UniCache）。
  3. apply_chat_template 的 date_string="13 Sep 2025" 是 Llama 3 模板的固定日期；
     Qwen2.5 模板不引用它（自动忽略），保留参数仅为对齐 UniCache 的 prompt 构造。

【用法】
  /root/sl_old/bin/python prepare_session_data.py \
      --raw_data_path   TaskData/Codeparrot/Code.json \
      --save_data_path  TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl \
      --tokenizer_path  /root/autodl-tmp/model/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from transformers import AutoTokenizer


# --------------------------------------------------------------------------
# 单个 prompt 的文本构造（复刻 UniCache get_prompt_text）
# --------------------------------------------------------------------------
def get_prompt_text(tokenizer, conversations, add_generation_prompt: bool = True) -> str:
    """把 conversations 按 chat template 拼成 prompt 文本。

    - conversations: [{"role": "user"/"assistant", "content": ...}, ...]
    - add_generation_prompt=True: 末尾追加"等待 assistant 回复"的占位
      （Qwen 是 <|im_start|>assistant\n；Llama 3 是 <|start_header_id|>assistant<|end_header_id|>）
    - date_string="13 Sep 2025": 对齐 UniCache 的固定日期（Qwen 模板忽略该 kwarg）
    """
    chat = [{"role": t["from"], "content": t["value"].strip()} for t in conversations]
    # 校验：生成前最后一轮必须是 user
    ends_with_user = chat[-1].get("role", "").lower() == "user"
    if not ends_with_user and add_generation_prompt:
        raise ValueError("The last turn must be from 'user' when add_generation_prompt=True.")
    if ends_with_user and not add_generation_prompt:
        raise ValueError("The last turn must be from 'assistant' when add_generation_prompt=False.")
    try:
        # 对齐 UniCache 的固定日期（Llama 模板需要；Qwen 模板未引用则自动忽略）
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt,
            date_string="13 Sep 2025")
    except TypeError:
        # 兼容性回退：某些老 tokenizer 的模板不接受 date_string
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt)


# --------------------------------------------------------------------------
# 单 session 的 turn 切分（复刻 UniCache prepare_session_data_*）
# --------------------------------------------------------------------------
def prepare_session_data_with_system_prompt(entry: Dict[str, Any], tokenizer):
    """conversations 以 system 开头（system, user, assistant, ...）：
       turn i 的输入 = conversations[:2i]（system 之前的不算 turn），响应 = conversations[2i]。"""
    sid = entry["id"]
    conversations = entry["conversations"]
    assert conversations[0]["from"] == "system"
    num_turns = (len(conversations) - 1) // 2
    result: Dict[str, Any] = {"id": sid, "num_turns": num_turns}
    for i in range(1, num_turns + 1):
        key = f"turn_{i}"
        result[key] = dict()
        input_text = get_prompt_text(tokenizer, conversations[:2 * i], add_generation_prompt=True)
        # input ids 带 BOS
        result[key]["input_text"] = input_text
        result[key]["input_ids"] = tokenizer.encode(input_text, return_tensors=None, add_special_tokens=True)
        try:
            response_text = conversations[2 * i]["value"] + tokenizer.decode([tokenizer.eos_token_id])
            # 响应不带 BOS
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = tokenizer.encode(response_text, return_tensors=None, add_special_tokens=False)
        except Exception:
            response_text = tokenizer.decode([tokenizer.eos_token_id])
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = [tokenizer.eos_token_id]
    return result


def prepare_session_data(entry: Dict[str, Any], tokenizer):
    """conversations 以 user 开头（user, assistant, user, ...）：
       turn i 的输入 = conversations[:2i-1]，响应 = conversations[2i-1]（0 起始 = 第 2i-1 个）。"""
    sid = entry["id"]
    conversations = entry["conversations"]
    if conversations[0]["from"] == "system":
        return prepare_session_data_with_system_prompt(entry, tokenizer)
    num_turns = len(conversations) // 2
    result: Dict[str, Any] = {"id": sid, "num_turns": num_turns}
    for i in range(1, num_turns + 1):
        key = f"turn_{i}"
        result[key] = dict()
        input_text = get_prompt_text(tokenizer, conversations[:2 * i - 1], add_generation_prompt=True)
        # input ids 带 BOS
        result[key]["input_text"] = input_text
        result[key]["input_ids"] = tokenizer.encode(input_text, return_tensors=None, add_special_tokens=True)
        try:
            response_text = conversations[2 * i - 1]["value"] + tokenizer.decode([tokenizer.eos_token_id])
            # 响应不带 BOS
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = tokenizer.encode(response_text, return_tensors=None, add_special_tokens=False)
        except Exception:
            response_text = tokenizer.decode([tokenizer.eos_token_id])
            result[key]["response_text"] = response_text
            result[key]["response_ids"] = [tokenizer.eos_token_id]
    return result


# --------------------------------------------------------------------------
# 主流程：流式写 JSONL
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="UniCache 数据准备（独立版，无 vllm 依赖）")
    p.add_argument("--raw_data_path", required=True, help="raw 对话 JSON（[{id, conversations:[{from,value}]}]）")
    p.add_argument("--save_data_path", required=True, help="输出 session_data.jsonl（一行一个 session）")
    p.add_argument("--tokenizer_path", default="/root/autodl-tmp/model/Qwen2.5-7B-Instruct",
                   help="tokenizer/模型路径（必须与 server 模型一致）")
    p.add_argument("--max-records", type=int, default=-1, help="最多处理几个 session（-1=全部，冒烟用）")
    args = p.parse_args()

    print(f"[prepare_session_data] 加载 tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"[prepare_session_data] 读取 raw 数据: {args.raw_data_path}")
    # 服务器内存 754GB，ShareGPT(642MB)/Code.json 全量 json.load 无压力（无需 ijson）
    with open(args.raw_data_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[prepare_session_data] raw sessions: {len(data)}")

    if args.max_records > 0:
        data = data[: args.max_records]

    out = args.save_data_path
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(data):
            try:
                result = prepare_session_data(entry, tokenizer)
            except Exception as exc:  # 单条坏数据不阻塞整体
                print(f"  [skip] {entry.get('id', idx)}: {type(exc).__name__}: {exc}")
                continue
            f_out.write(json.dumps({result["id"]: result}, ensure_ascii=False) + "\n")
            if (idx + 1) % 500 == 0:
                print(f"  ... {idx + 1}/{len(data)}")
    print(f"[prepare_session_data] ✅ 完成 → {out}")


if __name__ == "__main__":
    main()
