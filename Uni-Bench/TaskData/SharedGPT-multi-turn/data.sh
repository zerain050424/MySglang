#!/usr/bin/env bash
# =============================================================================
# ShareGPT（多轮文本）数据准备（服务器版）
# 流程: raw json → clean_ShareGPT_V3.py → deduplicate.py → prepare_session_data.py
# 产物: TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
# 说明: tokenizer 用服务器的 Qwen2.5-7B-Instruct（与 sglang server 一致）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/../.." && pwd)"          # Uni-Bench 根目录
TOKENIZER="/root/autodl-tmp/model/Qwen2.5-7B-Instruct"  # 服务器模型（可改）
PY="/root/sl_old/bin/python"                        # UniCache 部署环境 python
RAW="/root/autodl-tmp/TaskData/SharedGPT-multi-turn/ShareGPT_V3_unfiltered_cleaned_split.json"
OUT_DIR="$UB/TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct"
mkdir -p "$OUT_DIR"
cd "$UB/TaskData/SharedGPT-multi-turn"

# 1) 原始数据检查（若未下载，提示先跑 download_datasets.sh）
[ -f "$RAW" ] || { echo "❌ 未找到 $RAW ，请先下载 ShareGPT 原始数据"; exit 1; }

# 2) 清洗：role 规范化 + 过滤非法多轮 → ShareGPT_cleaned_valid.json
$PY clean_ShareGPT_V3.py --raw_data_path "$RAW" --clean_data_path ShareGPT_cleaned_valid.json

# 3) 去重：去掉首轮重复的会话 → ShareGPT_cleaned_valid_deduplicated.json
$PY deduplicate.py   # 内部硬编码读/写本目录下文件

# 4) 转 session_data（逐 turn token 化，Qwen2.5-7B tokenizer）
$PY "$UB/prepare_session_data.py" \
    --raw_data_path  "$UB/TaskData/SharedGPT-multi-turn/ShareGPT_cleaned_valid_deduplicated.json" \
    --save_data_path "$OUT_DIR/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl" \
    --tokenizer_path "$TOKENIZER"

echo "✅ ShareGPT session 数据完成:"
echo "   $OUT_DIR/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl"
