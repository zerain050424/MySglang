#!/usr/bin/env bash
# =============================================================================
# 模拟器一键实验：exp3 四 cell（scan / struct / chat / agentic）策略对比
#
# 用法（仓库根目录或 Heter-Bench 目录均可执行）:
#   bash Heter-Bench/mixexp/scripts/sim/run_exp3.sh
#   CELLS="chat" ARMS="lru,chain_lru,opt" bash Heter-Bench/mixexp/scripts/sim/run_exp3.sh
#
# 环境变量:
#   PY      Python 解释器（默认 python3）
#   CELLS   逗号分隔的 cell 列表（默认 scan,struct,chat,agentic）
#   ARMS    逗号分隔的策略臂（默认 lru,heter_cache,chain_lru,heter_rgate）
#   CAP     token 池容量（默认 198000）
#   PAGE    page size（默认 8）
# =============================================================================
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python3}"
CELLS="${CELLS:-scan,struct,chat,agentic}"
ARMS="${ARMS:-lru,heter_cache,chain_lru,heter_rgate}"
CAP="${CAP:-198000}"
PAGE="${PAGE:-8}"
OUT="${OUT:-Heter-Bench/mixexp/result/sim_exp3_summary.json}"

echo "[sim-exp3] cells=$CELLS arms=$ARMS cap=$CAP page=$PAGE"
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python" \
    "$PY" "$SCRIPT_DIR/sim_exp3_cells.py" \
    --cells "$CELLS" \
    --arms "$ARMS" \
    --cap "$CAP" \
    --page "$PAGE" \
    --out "$OUT" \
    "$@"
