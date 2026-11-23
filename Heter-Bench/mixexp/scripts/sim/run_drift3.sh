#!/usr/bin/env bash
# =============================================================================
# 模拟器一键实验：drift3 构成漂移（含 gate 检查与完整矩阵）
#
# 用法:
#   bash Heter-Bench/mixexp/scripts/sim/run_drift3.sh           # 默认跑 bg_10 的 gate 检查
#   MODE=matrix BG=05,10,20 bash Heter-Bench/mixexp/scripts/sim/run_drift3.sh
#
# 环境变量:
#   PY      Python 解释器（默认 python3）
#   MODE    运行模式: gate（默认）或 matrix（完整矩阵 + stationary + reverse）
#   BG      背景倍率列表，逗号分隔（默认 10，matrix 模式可设为 05,10,20）
#   ARMS    策略臂（默认 lru,heter_cache,opt）
#   SEED    随机种子（默认 42）
# =============================================================================
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python3}"
MODE="${MODE:-gate}"
BG="${BG:-10}"
ARMS="${ARMS:-lru,heter_cache,opt}"
SEED="${SEED:-42}"

IFS=',' read -r -a _BGS <<< "$BG"

for b in "${_BGS[@]}"; do
    PLAN="Heter-Bench/TaskData/drift3/bg_${b}.json"
    if [ ! -f "$PLAN" ]; then
        echo "❌ 未找到 plan 文件: $PLAN（请先运行 build_drift3_data.py 生成数据）" >&2
        exit 1
    fi
    if [ "$MODE" = "gate" ]; then
        echo "[sim-drift3] gate check: plan=$PLAN"
        PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python" \
            "$PY" "$SCRIPT_DIR/drift3_run.py" --plan "$PLAN" --gate --seed "$SEED" "$@"
    else
        echo "[sim-drift3] matrix run: plan=$PLAN arms=$ARMS seed=$SEED"
        PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python" \
            "$PY" "$SCRIPT_DIR/drift3_run.py" --plan "$PLAN" --arms "$ARMS" --seed "$SEED" --stationary --reverse "$@"
    fi
done
