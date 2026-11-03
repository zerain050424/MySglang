#!/usr/bin/env bash
# Phase-0 门控：不过门控不要跑正式矩阵。
#   bash scripts/run_gate.sh                 # auto（有 sglang 就跑真后端）
#   bash scripts/run_gate.sh --backend fake  # 无依赖机器：只验链路
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
if [ -f ../env.sh ]; then
  # shellcheck disable=SC1091
  source ../env.sh
fi
PY="${WTL_PY:-${PY:-python}}"
export PYTHONPATH="$HERE:$HERE/../../python:${PYTHONPATH:-}"
exec "$PY" -m wtl.gate "$@"
