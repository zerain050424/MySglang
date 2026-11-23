#!/usr/bin/env bash
# =============================================================================
# Heter-Bench 评测模型下载脚本（针对 Qwen2.5-7B-Instruct）
#
# 作用：下载基准模型权重与 tokenizer 到本地指定目录。
#       支持 ModelScope（国内高速免翻墙）与 Hugging Face 镜像（HF_ENDPOINT）双源。
#
# 用法（在 Heter-Bench 根目录执行）:
#   bash scripts/download_model.sh                          # 默认使用 ModelScope 下载
#   SOURCE=huggingface bash scripts/download_model.sh       # 使用 Hugging Face 镜像下载
#   MODEL=/path/to/save bash scripts/download_model.sh      # 指定下载保存目录
#
# 环境变量:
#   MODEL          模型落盘目录（默认取 scripts/config.sh 中的 MODEL，即 <Heter-Bench>/models/Qwen2.5-7B-Instruct）
#   SOURCE         下载源选择: modelscope（默认，国内首选）或 huggingface
#   HF_ENDPOINT    HF 镜像地址（SOURCE=huggingface 时生效，默认: https://hf-mirror.com）
#   PY             Python 解释器（默认取 .venv_data/bin/python，不存在则回退至 python3）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/.." && pwd)"
source "$UB/scripts/config.sh"

SOURCE="${SOURCE:-modelscope}"
TARGET_DIR="${MODEL:-$UB/models/Qwen2.5-7B-Instruct}"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

# 默认优先使用数据环境中的 python
if [ -z "${PY:-}" ]; then
    if [ -x "$UB/.venv_data/bin/python" ]; then
        PY="$UB/.venv_data/bin/python"
    else
        PY="python3"
    fi
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=============================================================="
echo "  Heter-Bench 模型下载"
echo "  模型 ID     : $MODEL_ID"
echo "  下载源      : $SOURCE"
echo "  目标目录    : $TARGET_DIR"
echo "  Python 解释器: $PY"
if [ "$SOURCE" = "huggingface" ]; then
    echo "  HF_ENDPOINT : $HF_ENDPOINT"
fi
echo "=============================================================="

# 1) 幂等性检查：若关键模型与分词器文件已存在且非空，则视为已就绪
if [ -f "$TARGET_DIR/config.json" ] && [ -f "$TARGET_DIR/tokenizer.json" ]; then
    echo "✅ 模型文件已完整存在于: $TARGET_DIR"
    echo "   跳过重复下载。"
    exit 0
fi

mkdir -p "$TARGET_DIR"

# 2) 根据下载源调用对应 API 下载
if [ "$SOURCE" = "modelscope" ]; then
    echo "🚀 开始通过 ModelScope 下载 $MODEL_ID ..."
    "$PY" -c "
import os
import sys
try:
    from modelscope import snapshot_download
except ImportError:
    print('❌ 未检测到 modelscope 库，请先运行: bash scripts/setup_data_env.sh', file=sys.stderr)
    sys.exit(1)

model_id = '$MODEL_ID'
target_dir = '$TARGET_DIR'
print(f'下载中: {model_id} -> {target_dir}')
snapshot_download(model_id=model_id, local_dir=target_dir)
"
elif [ "$SOURCE" = "huggingface" ]; then
    echo "🚀 开始通过 Hugging Face (镜像: $HF_ENDPOINT) 下载 $MODEL_ID ..."
    "$PY" -c "
import os
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print('❌ 未检测到 huggingface_hub 库，请先运行: bash scripts/setup_data_env.sh', file=sys.stderr)
    sys.exit(1)

repo_id = '$MODEL_ID'
target_dir = '$TARGET_DIR'
print(f'下载中: {repo_id} -> {target_dir}')
snapshot_download(repo_id=repo_id, local_dir=target_dir)
"
else
    echo "❌ 不支持的下载源: $SOURCE (仅支持 modelscope 或 huggingface)" >&2
    exit 1
fi

# 3) 最终文件完整性校验
if [ -f "$TARGET_DIR/config.json" ] && [ -f "$TARGET_DIR/tokenizer.json" ]; then
    echo "=============================================================="
    echo "✅ 模型下载完成并校验成功！"
    echo "   模型与 Tokenizer 路径: $TARGET_DIR"
    echo "=============================================================="
else
    echo "❌ 下载后未检测到关键文件 (config.json / tokenizer.json)，可能下载未完全完成。" >&2
    exit 1
fi
