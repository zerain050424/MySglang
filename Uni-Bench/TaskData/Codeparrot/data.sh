#!/usr/bin/env bash
# =============================================================================
# Codeparrot(APPS) 数据准备（服务器版）
# 流程: test.jsonl（已下载 1.29GB）→ prepare_data.py → prepare_session_data.py
# 产物: TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl
# 说明: 默认 id 前缀 "Code-"（→SINGLE_TURN，与原版 UniCache 一致）；
#       想归入 SINGLE_TURN_RANDOM 队列可把 ID_PREFIX 改成 apps_
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/../.." && pwd)"
TOKENIZER="/root/autodl-tmp/model/Qwen2.5-7B-Instruct"
PY="/root/sl_old/bin/python"
RAW="/root/autodl-tmp/TaskData/Codeparrot/test.jsonl"
ID_PREFIX="${ID_PREFIX:-Code-}"          # 可覆盖: ID_PREFIX=apps_ bash data.sh
OUT_DIR="$UB/TaskData/Codeparrot/Qwen2.5-7B-Instruct"
mkdir -p "$OUT_DIR"
cd "$UB/TaskData/Codeparrot"

[ -f "$RAW" ] || { echo "❌ 未找到 $RAW ，请先下载"; exit 1; }

# 1) 原始 jsonl → 单轮对话 JSON（只取 introductory 难度）
$PY prepare_data.py --input "$RAW" --output Code.json --id-prefix "$ID_PREFIX"

# 2) 转 session_data
$PY "$UB/prepare_session_data.py" \
    --raw_data_path  "$UB/TaskData/Codeparrot/Code.json" \
    --save_data_path "$OUT_DIR/Code_session_data.jsonl" \
    --tokenizer_path "$TOKENIZER"

echo "✅ Codeparrot session 数据完成:"
echo "   $OUT_DIR/Code_session_data.jsonl"
