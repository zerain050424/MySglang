#!/usr/bin/env bash
# =============================================================================
# AgentBank（agentic 多轮 + 单轮 API 调用）数据准备（服务器版）
# 流程: AgentBank parquet（已下载）→ prepare_agentbank_data.py → (可选 Alfred_scale_up)
#       → prepare_session_data.py
# 产物:
#   TaskData/AgentBankData/Qwen2.5-7B-Instruct/
#     AgentBank_alfred_scale_5000_session_data.jsonl   (agentic 多轮, id=alfred_*)
#     AgentBank_apps_session_data.jsonl                (单轮 API, id=apps_*)
#     AgentBank_single_api_session_data.jsonl          (apps+mind2web 合并, 可选)
# 说明: 分类规则见 uni_utils.classify_simple_hybrid_task_type:
#       alfred_*→MULTI_TURN_AGENTIC, apps_*/mind2web_*→SINGLE_TURN_RANDOM
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/../.." && pwd)"
TOKENIZER="/root/autodl-tmp/model/Qwen2.5-7B-Instruct"
PY="/root/sl_old/bin/python"
AB_DIR="/root/autodl-tmp/TaskData/AgentBankData/AgentBank"   # huggingface-cli 下载的 repo
OUT_DIR="$UB/TaskData/AgentBankData/Qwen2.5-7B-Instruct"
mkdir -p "$OUT_DIR"
cd "$UB/TaskData/AgentBankData"

[ -d "$AB_DIR" ] || { echo "❌ 未找到 $AB_DIR ，请先运行 download_datasets.sh"; exit 1; }

# 1) parquet → 各 task 对话 JSON（只导 benchmark 需要的 alfred/apps/mind2web）
$PY prepare_agentbank_data.py --agentbank-dir "$AB_DIR" --out-dir . \
    --tasks alfred apps mind2web

# 2) alfred 扩量到 5000（增加 agentic 负载规模；会改首轮内容避免跨会话前缀共享）
if [ -f AgentBank_alfred.json ]; then
    $PY Alfred_scale_up.py
fi

# 3) 转 session_data（alfred 扩量版 + apps）
for f in AgentBank_alfred_scale_5000.json AgentBank_apps.json; do
    [ -f "$f" ] || { echo "  [skip] $f 不存在"; continue; }
    base="$(basename "$f" .json)"
    $PY "$UB/prepare_session_data.py" \
        --raw_data_path  "$UB/TaskData/AgentBankData/$f" \
        --save_data_path "$OUT_DIR/${base}_session_data.jsonl" \
        --tokenizer_path "$TOKENIZER"
done

# 4) 可选：apps + mind2web 合并成"单轮 API 混合"负载
if [ -f "$OUT_DIR/AgentBank_apps_session_data.jsonl" ] && \
   [ -f "$OUT_DIR/AgentBank_mind2web_session_data.jsonl" ]; then
    (cd "$OUT_DIR" && $PY "$UB/TaskData/AgentBankData/merge_app_mind2web.py")
fi

echo "✅ AgentBank session 数据完成 → $OUT_DIR/"
