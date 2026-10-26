#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentBank 原始 parquet → 各 task 对话 JSON（服务器版，直接读本地文件）。

【与原版差异】
  原版用 `datasets.load_dataset("Solaris99/AgentBank", task)`（每次重新走 HF/缓存）；
  服务器已把整个 repo 用 huggingface-cli 下载到本地（见 download_datasets.sh），
  这里直接遍历本地 parquet（AgentBank/data/<task>/*.parquet）读取，
  网络无关、更快，输出格式与原版一致（AgentBank_<task>.json）。

【任务列表】
  alfworld/apps/gsm8k/hotpotqa/humaneval/intercode_bash/intercode_sql/iqa/math/
  mathqa/mbpp/mbpp_before/mind2web/rearrange/strategyqa/triviaqa/webarena/webshop/alfred

【分类标签（重要）】
  session_id 前缀决定 SimpleHybrid 的 task_type（见 uni_utils.classify_simple_hybrid_task_type）：
    - "alfred_"  → MULTI_TURN_AGENTIC
    - "apps_"/"mind2web_" → SINGLE_TURN_RANDOM
  因此 data.sh 只挑选 alfred（agentic 多轮）+ apps/mind2web（单轮 API 调用）两类构建混合负载。

【用法】
  /root/sl_old/bin/python prepare_agentbank_data.py \
      --agentbank-dir /root/autodl-tmp/TaskData/AgentBankData/AgentBank \
      --out-dir . [--tasks alfred apps mind2web]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import numpy as np
except Exception:
    np = None


def _jsonable(obj):
    """把 numpy 类型 / 嵌套结构转成 JSON 可序列化对象"""
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj



def normalize_role(role):
    """角色名规范化 → user / assistant / system"""
    role = str(role).lower()
    if role in ["human", "user"]:
        return "user"
    elif role in ["gpt", "chatgpt", "bing", "bard", "assistant"]:
        return "assistant"
    elif role == "system":
        return "system"
    else:
        raise ValueError(f"Unknown role: {role}")


def read_task_parquet(task_dir: Path):
    """读一个 task 目录下所有 parquet → 行字典列表（与原版 subset 语义一致）"""
    parquet_files = sorted(task_dir.glob("*.parquet"))
    assert parquet_files, f"没有 parquet: {task_dir}"
    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    # 每行转成 {column: value} 字典（对齐原版 dict(zip(subset.keys(), values))）
    return [dict(zip(df.columns, row)) for row in df.itertuples(index=False, name=None)]


def main() -> None:
    p = argparse.ArgumentParser(description="AgentBank parquet → 对话 JSON")
    p.add_argument("--agentbank-dir", default="/root/autodl-tmp/TaskData/AgentBankData/AgentBank",
                   help="AgentBank repo 本地目录")
    p.add_argument("--out-dir", default=".", help="输出目录")
    p.add_argument("--tasks", nargs="*",
                   default=["alfred", "apps", "mind2web"],
                   help="要导出的 task（默认只导出 benchmark 用到的三个）")
    args = p.parse_args()

    # 兼容两种布局：AgentBank/data/<task>/ 或 AgentBank/<task>/
    cand = Path(args.agentbank_dir) / "data"
    data_root = cand if cand.is_dir() else Path(args.agentbank_dir)
    assert data_root.is_dir(), f"找不到 {data_root}，请先运行 download_datasets.sh"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        task_dir = data_root / task
        if not task_dir.is_dir():
            print(f"[skip] 无 task 目录: {task}")
            continue
        sample_list = read_task_parquet(task_dir)
        # 角色规范化（与 prepare_agentbank_data.py 一致）
        for sample in sample_list:
            if "conversations" in sample:
                for msg in sample["conversations"]:
                    if "from" in msg:
                        msg["from"] = normalize_role(msg["from"])
        out_path = out_dir / f"AgentBank_{task}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(sample_list), f, indent=2, ensure_ascii=False, default=str)
        print(f"[AgentBank] ✅ {task}: {len(sample_list)} 条 → {out_path}")


if __name__ == "__main__":
    main()
