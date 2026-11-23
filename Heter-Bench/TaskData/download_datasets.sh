#!/usr/bin/env bash
# =============================================================================
# 原始数据集下载入口：补齐默认环境变量并调用 download_datasets.py
#
# 用法（在 Heter-Bench 根目录执行）:
#   bash TaskData/download_datasets.sh                  # 下载全部数据集
#   bash TaskData/download_datasets.sh sharegpt apps    # 仅下载指定数据集
#
# 可选数据集: sharegpt / apps / agentbank / mashqa（别名详见 download_datasets.py）
#
# 环境变量:
#   PY             Python 解释器（默认 python3；仅需 huggingface_hub）
#   RAW_DIR        原始数据落盘目录（默认 <Heter-Bench>/TaskData/raw）
#   HF_ENDPOINT    HF 镜像地址（默认 https://hf-mirror.com，已设置时不覆盖）
# =============================================================================
set -eu
UB="$(cd "$(dirname "$0")/.." && pwd)"

PY="${PY:-python3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "UB          = $UB"
echo "PY          = $PY"
echo "HF_ENDPOINT = $HF_ENDPOINT"
echo "RAW_DIR     = ${RAW_DIR:-TaskData/raw}"

exec "$PY" "$UB/TaskData/download_datasets.py" "$@"
