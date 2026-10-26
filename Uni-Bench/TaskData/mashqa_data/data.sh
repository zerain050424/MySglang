#!/usr/bin/env bash
# =============================================================================
# MASHQA 数据准备（服务器版）
# 流程: {train,test,val}_webmd_squad_v2_full.json → prepare_data.py → prepare_session_data.py
# 产物: TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl
# 说明: id 前缀 "QA-"（→ SINGLE_TURN 结构化单轮队列）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/../.." && pwd)"
TOKENIZER="/root/autodl-tmp/model/Qwen2.5-7B-Instruct"
PY="/root/sl_old/bin/python"
RAW_DIR="/root/autodl-tmp/TaskData/mashqa_data"
OUT_DIR="$UB/TaskData/mashqa_data/Qwen2.5-7B-Instruct"
mkdir -p "$OUT_DIR"
cd "$UB/TaskData/mashqa_data"

[ -f "$RAW_DIR/train_webmd_squad_v2_full.json" ] || { echo "❌ 未找到 raw 数据"; exit 1; }

# 1) 原始 JSON → 单轮对话 JSON
$PY prepare_data.py --input-dir "$RAW_DIR" --output mashqa.json

# 2) 转 session_data
$PY "$UB/prepare_session_data.py" \
    --raw_data_path  "$UB/TaskData/mashqa_data/mashqa.json" \
    --save_data_path "$OUT_DIR/mashqa_session_data.jsonl" \
    --tokenizer_path "$TOKENIZER"

echo "✅ MASHQA session 数据完成:"
echo "   $OUT_DIR/mashqa_session_data.jsonl"
