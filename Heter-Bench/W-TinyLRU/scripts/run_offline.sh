#!/usr/bin/env bash
# 离线矩阵（L1）：无 GPU。所有参数透传给 `python -m wtl.run_matrix`。
#
#   bash scripts/run_offline.sh --arms all --workloads sharegpt,alfred,codeparrot \
#        --ratios 0.3,0.6,1.0 --seeds 42,43,44 --jobs 12
#
#   bash scripts/run_offline.sh --mode synth --arms A0,A2,A3 --synth-depths 512,2048,8192
#
# 环境变量：
#   WTL_PY        指定解释器（默认优先用 Heter-Bench/scripts/env.sh 的 PY，否则 python）
#   UNIBENCH_ROOT 数据根目录（默认 ../）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# 实验机上的 Heter-Bench 环境（venv / LD_LIBRARY_PATH）；不存在就跳过
if [ -f ../../scripts/env.sh ] && [ "${WTL_SKIP_UNIBENCH_ENV:-0}" != "1" ]; then
  # shellcheck disable=SC1091
  source ../../scripts/env.sh
fi
PY="${WTL_PY:-${PY:-python}}"

export PYTHONPATH="$HERE:$HERE/../../python:${PYTHONPATH:-}"
mkdir -p result/offline result

echo "[run_offline] python=$PY cwd=$HERE"
exec "$PY" -m wtl.run_matrix "$@"
