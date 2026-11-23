#!/usr/bin/env bash
# =============================================================================
# 模拟器一键实验：tail-stress 长尾 TTFT 压力测试（使用 best_configs.md 推荐配置）
#
# 用法:
#   bash Heter-Bench/mixexp/scripts/sim/run_tail_stress.sh
#   ARMS="lru,chain_lru,heter_rgate,opt" bash Heter-Bench/mixexp/scripts/sim/run_tail_stress.sh
#
# 环境变量:
#   PY      Python 解释器（默认 python3）
#   ARMS    实验臂（默认 lru,chain_lru,heter_rgate,opt）
#   CAP     容量（默认 198000）
# =============================================================================
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TOOL="$REPO_ROOT/Heter-Bench/mixexp/tools/build_tail_stress.py"
cd "$REPO_ROOT"

PY="${PY:-python3}"
ARMS="${ARMS:-lru,chain_lru,heter_rgate,opt}"
CAP="${CAP:-198000}"

echo "[sim-tail-stress] arms=$ARMS cap=$CAP"
SIM_REPO="$REPO_ROOT" PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python" \
    "$PY" "$TOOL" \
    --cap "$CAP" \
    --arms "$ARMS" \
    --floor 8192 --weight 0.02 --gate multiturn \
    --rg-k 0.6 --rg-rmin 0.3 --pf --pf-beta 1.0 \
    "$@"
