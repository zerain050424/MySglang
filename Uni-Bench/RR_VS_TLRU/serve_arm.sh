#!/usr/bin/env bash
# 启动一个实验臂（burst/pause 场景）。
#   bash serve_arm.sh <arm> <max_total_tokens> [config.json]
set -euo pipefail

cd "$(dirname "$0")"
BENCH_DIR="$(cd .. && pwd)"          # Uni-Bench/
source "$BENCH_DIR/env.sh"           # $PY + 库路径

ARM=${1:?usage: serve_arm.sh <arm> <max_total_tokens> [config.json]}
BUDGET=${2:?usage: serve_arm.sh <arm> <max_total_tokens> [config.json]}
CONFIG=${3:-configs/arm_${ARM}.json}

MODEL=${MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}
PORT=${PORT:-30000}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [ ! -f "$CONFIG" ]; then
  echo "[serve_arm] config not found: $CONFIG" >&2
  exit 2
fi

CFG=$(tr -d '\n' < "$CONFIG")
echo "[serve_arm] arm=$ARM budget=$BUDGET port=$PORT config=$CONFIG"
echo "[serve_arm] policy_config=$CFG"

cd "$BENCH_DIR/.."
exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --radix-eviction-policy simple_hybrid \
  --enable-simple-hybrid \
  --radix-eviction-policy-config "$CFG" \
  --max-total-tokens "$BUDGET" \
  --enable-metrics \
  --port "$PORT" \
  ${EXTRA_ARGS}
