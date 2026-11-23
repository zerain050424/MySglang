#!/usr/bin/env bash
# =============================================================================
# Heter-Bench 一键数据准备：下载原始数据 → 生成各负载 session_data
#
# 用法（在 Heter-Bench 根目录执行）:
#   bash TaskData/prepare_all.sh
#   PY=/path/to/venv/bin/python TOKENIZER=/path/to/model bash TaskData/prepare_all.sh
#   SKIP_DOWNLOAD=1 PY=/path/to/venv/bin/python bash TaskData/prepare_all.sh
#
# 环境变量（透传给 download_datasets.sh 与各 data.sh）:
#   PY             Python 解释器（默认 python3；同时用于下载与处理，需含
#                  huggingface_hub、transformers，AgentBank 另需 pandas；
#                  若下载与处理分环境，请先单跑 download_datasets.sh，再配 SKIP_DOWNLOAD=1）
#   TOKENIZER      tokenizer/模型路径（默认取 scripts/config.sh 的 MODEL）
#   RAW_DIR        原始数据目录（默认 <Heter-Bench>/TaskData/raw）
#   HF_ENDPOINT    HF 镜像地址（默认 https://hf-mirror.com）
#   SKIP_DOWNLOAD  设为 1 时跳过下载阶段，仅运行处理脚本
#   DATASETS       空格分隔的数据集子集（可选：sharegpt apps agentbank mashqa；
#                  仅控制下载阶段，处理阶段仍会依次执行全部 4 个 data.sh）
# =============================================================================

# 原样透传给子脚本，由各子脚本决定各自默认值
export PY TOKENIZER RAW_DIR HF_ENDPOINT

set -eu
UB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$UB"

echo "=============================================================="
echo "  Heter-Bench 数据准备"
echo "  PY          = ${PY:-python3}"
echo "  TOKENIZER   = ${TOKENIZER:-<scripts/config.sh 的 MODEL>}"
echo "  RAW_DIR     = ${RAW_DIR:-TaskData/raw}"
echo "  HF_ENDPOINT = ${HF_ENDPOINT:-https://hf-mirror.com}"
echo "=============================================================="

# 1) 原始数据下载
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    bash "$UB/TaskData/download_datasets.sh" ${DATASETS:-}
else
    echo "[skip] 已设置 SKIP_DOWNLOAD=1，跳过下载"
fi

# 2) 生成 session_data（任一脚本失败即退出）
bash "$UB/TaskData/SharedGPT-multi-turn/data.sh"
bash "$UB/TaskData/Codeparrot/data.sh"
bash "$UB/TaskData/AgentBankData/data.sh"
bash "$UB/TaskData/mashqa_data/data.sh"

echo ""
echo "全部完成。hybrid 负载产物（供 configs/hybrid_tasks_config.yaml）:"
echo "  TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl"
echo "  TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl"
echo "  TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl"
echo "  TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl"
echo "  备用: TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl"
