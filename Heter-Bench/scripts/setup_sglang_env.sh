#!/usr/bin/env bash
# =============================================================================
# Heter-Bench 完整 SGLang 运行与评测环境初始化脚本
#
# 作用：搭建包含 PyTorch (CUDA)、FlashInfer 以及本仓库 SGLang 可编辑安装的
#       完整生产/评测 Python 虚拟环境，并输出/配置 env.sh。
#
# 用法（在 Heter-Bench 根目录执行）:
#   bash scripts/setup_sglang_env.sh
#   VENV_DIR=/data/venv_sglang bash scripts/setup_sglang_env.sh
#
# 环境变量:
#   VENV_DIR       虚拟环境生成目录（默认: <Heter-Bench>/.venv_sglang）
#   PIP_INDEX_URL  PyPI 镜像源地址（默认: 清华开源镜像站）
#   TORCH_CUDA     安装 PyTorch 时对应的 CUDA 版本（默认: cu124，可选 cu121 / cu118 等）
#   SYS_PY         用于创建 venv 的系统 Python 解释器（默认: python3）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_DIR="$(cd "$UB/.." && pwd)"

SYS_PY="${SYS_PY:-python3}"
VENV_DIR="${VENV_DIR:-$UB/.venv_sglang}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

echo "=============================================================="
echo "  Heter-Bench 完整 SGLang 运行环境初始化"
echo "  仓库根目录  : $ROOT_DIR"
echo "  目标虚拟环境: $VENV_DIR"
echo "  CUDA 规格   : $TORCH_CUDA"
echo "  PyPI 镜像   : $PIP_INDEX_URL"
echo "=============================================================="

# 1) 检查编译工具链（Ninja, GCC/G++）
echo "🔍 检查系统基础依赖..."
for tool in ninja gcc g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "⚠️  警告: 未在 PATH 中找到 $tool，SGLang 运行时编译算子可能需要，请确保系统已安装（sudo apt install -y ninja-build build-essential）"
    fi
done

# 2) 创建独立虚拟环境
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if [ -x "$PY" ] && [ -x "$PIP" ]; then
    echo "ℹ️  虚拟环境已存在: $VENV_DIR"
else
    echo "📦 创建虚拟环境: $VENV_DIR ..."
    "$SYS_PY" -m venv "$VENV_DIR"
fi

# 3) 安装 PyTorch 与核心算子依赖
echo "📦 升级 pip 与 wheel..."
"$PIP" install -U pip wheel setuptools -i "$PIP_INDEX_URL"

echo "📦 安装 PyTorch ($TORCH_CUDA)..."
"$PIP" install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

# 4) 安装 flashinfer 与推理辅助依赖
echo "📦 安装 flashinfer 与辅助依赖..."
"$PIP" install flashinfer -i "https://flashinfer.ai/whl/$TORCH_CUDA/torch2.4" || {
    echo "ℹ️  flashinfer 在线源未命中，尝试从官方 PyPI 安装..."
    "$PIP" install flashinfer -i "$PIP_INDEX_URL" || true
}

# 5) 以 editable 方式安装本仓库 SGLang
echo "📦 安装本仓库 SGLang (editable 模式)..."
"$PIP" install -e "$ROOT_DIR/python" -i "$PIP_INDEX_URL"

# 6) 安装评测辅助依赖
echo "📦 安装 Heter-Bench 评测所需依赖..."
"$PIP" install -U aiohttp requests pyyaml tqdm -i "$PIP_INDEX_URL"

# 7) 验证环境
echo "🔍 校验 SGLang 与 Torch 环境..."
"$PY" -c "
import torch
import sglang
print('✅ SGLang 运行环境就绪:')
print('   Torch 版本   :', torch.__version__, 'CUDA 可用:', torch.cuda.is_available())
print('   SGLang 路径  :', sglang.__file__)
"

echo "=============================================================="
echo "✅ 完整 SGLang 运行环境搭建完成！"
echo "   虚拟环境路径: $VENV_DIR"
echo "   Python 解释器: $PY"
echo ""
echo "💡 提示：运行评测服务前，可使环境生效："
echo "   source $VENV_DIR/bin/activate"
echo "   或在 scripts/env.sh 中指定 PY=\"$PY\""
echo "=============================================================="
