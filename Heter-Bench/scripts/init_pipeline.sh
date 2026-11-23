#!/usr/bin/env bash
# =============================================================================
# Heter-Bench 全流程一键端到端初始化 Pipeline
#
# 作用：打通从全新机器到数据准备完毕的全流程：
#       1. 创建轻量数据处理环境（setup_data_env.sh）
#       2. 下载 Qwen2.5-7B-Instruct 模型及分词器（download_model.sh）
#       3. 下载原始数据集并转换生成四个负载 session_data（TaskData/prepare_all.sh）
#
# 用法（在 Heter-Bench 根目录执行）:
#   bash scripts/init_pipeline.sh
#
# 断点续跑与覆盖开关:
#   SKIP_VENV=1        跳过创建/更新数据处理 venv（直接复用现有 PY）
#   SKIP_MODEL=1       跳过模型下载步骤
#   SKIP_DOWNLOAD=1    跳过原始数据集下载（仅重新执行 session_data 数据处理）
#   SOURCE=huggingface 模型下载改走 HF 镜像（默认 modelscope）
#   MODEL=/path        自定义模型存放路径（默认 <Heter-Bench>/models/Qwen2.5-7B-Instruct）
#   RAW_DIR=/path      自定义原始数据集存放路径（默认 <Heter-Bench>/TaskData/raw）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$UB"

echo "##############################################################"
echo "#                                                            #"
echo "#        Heter-Bench 全流程端到端数据准备 Pipeline          #"
echo "#                                                            #"
echo "##############################################################"

# 阶段 1: 轻量数据环境准备
if [ "${SKIP_VENV:-0}" != "1" ]; then
    echo ""
    echo ">>> [Stage 1/3] 初始化轻量数据处理环境..."
    bash "$UB/scripts/setup_data_env.sh"
else
    echo ""
    echo ">>> [Stage 1/3] 已设置 SKIP_VENV=1，跳过环境创建"
fi

# 确保提取到有效的数据环境 Python
if [ -z "${PY:-}" ]; then
    if [ -x "$UB/.venv_data/bin/python" ]; then
        PY="$UB/.venv_data/bin/python"
    else
        PY="python3"
    fi
fi
export PY

# 阶段 2: 评测模型与 Tokenizer 下载
if [ "${SKIP_MODEL:-0}" != "1" ]; then
    echo ""
    echo ">>> [Stage 2/3] 下载基准模型与 Tokenizer..."
    bash "$UB/scripts/download_model.sh"
else
    echo ""
    echo ">>> [Stage 2/3] 已设置 SKIP_MODEL=1，跳过模型下载"
fi

# 阶段 3: 数据集下载与 session_data 转换
echo ""
echo ">>> [Stage 3/3] 准备基准数据集 (下载 + 生成 session_data)..."
bash "$UB/TaskData/prepare_all.sh"

echo ""
echo "##############################################################"
echo "#  🎉 全部流水线执行完毕！                                    #"
echo "#  数据处理产物已生成至 TaskData/ 各任务目录中。             #"
echo "##############################################################"
