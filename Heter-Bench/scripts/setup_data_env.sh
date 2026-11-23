#!/usr/bin/env bash
# =============================================================================
# Heter-Bench 轻量数据处理环境初始化脚本
#
# 作用：创建专用轻量 venv 并安装数据下载、模型下载与分词所需依赖：
#       huggingface_hub, modelscope, transformers, pandas
#       （不需要 torch/cuda/sglang 依赖，启动快、体积小、无版本冲突）
#
# 用法（在 Heter-Bench 根目录执行）:
#   bash scripts/setup_data_env.sh
#   VENV_DIR=/data/venv_data bash scripts/setup_data_env.sh
#
# 环境变量:
#   VENV_DIR       虚拟环境生成目录（默认: <Heter-Bench>/.venv_data）
#   PIP_INDEX_URL  PyPI 镜像源地址（默认: 清华大学开源镜像站）
#   SYS_PY         用于创建 venv 的系统 Python 解释器（默认: python3）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/.." && pwd)"

SYS_PY="${SYS_PY:-python3}"
VENV_DIR="${VENV_DIR:-$UB/.venv_data}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "=============================================================="
echo "  Heter-Bench 轻量数据处理环境初始化"
echo "  系统 Python : $SYS_PY"
echo "  目标虚拟环境: $VENV_DIR"
echo "  PyPI 镜像   : $PIP_INDEX_URL"
echo "=============================================================="

# 1) 检查系统 Python 与 venv 模块
if ! command -v "$SYS_PY" >/dev/null 2>&1; then
    echo "❌ 未找到系统 Python 命令: $SYS_PY" >&2
    exit 1
fi

if ! "$SYS_PY" -c "import venv" >/dev/null 2>&1; then
    echo "❌ 当前 Python ($SYS_PY) 缺失 venv 模块，请先安装（例如 sudo apt install python3-venv）" >&2
    exit 1
fi

# 2) 创建虚拟环境（幂等）
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if [ -x "$PY" ] && [ -x "$PIP" ]; then
    echo "ℹ️  虚拟环境目录已存在: $VENV_DIR"
else
    echo "📦 创建虚拟环境: $VENV_DIR ..."
    "$SYS_PY" -m venv "$VENV_DIR"
fi

# 3) 安装或补齐依赖
echo "📦 更新 pip 并安装数据处理依赖..."
"$PIP" install -U pip -i "$PIP_INDEX_URL"
"$PIP" install -U \
    huggingface_hub \
    modelscope \
    transformers \
    pandas \
    -i "$PIP_INDEX_URL"

# 4) 校验环境完整性
echo "🔍 校验数据处理环境依赖..."
"$PY" -c "
import huggingface_hub
import modelscope
import transformers
import pandas
print('✅ 依赖加载正常:')
print('   huggingface_hub:', huggingface_hub.__version__)
print('   modelscope     :', modelscope.__version__)
print('   transformers   :', transformers.__version__)
print('   pandas         :', pandas.__version__)
"

echo "=============================================================="
echo "✅ 轻量数据环境准备就绪！"
echo "   Python 解释器路径: $PY"
echo "   后续使用可指定: export PY=\"$PY\""
echo "=============================================================="
