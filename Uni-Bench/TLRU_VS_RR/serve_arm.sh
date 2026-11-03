#!/usr/bin/env bash
# 启动一个实验臂的 sglang server。
#
#   bash serve_arm.sh <arm> <max_total_tokens> [config.json]
#
# 三个臂共用 simple_hybrid 后端与 simple_hybrid 策略，差别只在
# --radix-eviction-policy-config 指向的 JSON。JSON 走 "$CFG" 单参数传递，
# 不做字符串拼接（避免 bash 花括号展开/词分割把 JSON 拆坏）。
set -euo pipefail

cd "$(dirname "$0")"
BENCH_DIR="$(cd .. && pwd)"          # Uni-Bench/
source "$BENCH_DIR/env.sh"           # 提供 $PY 与库路径

ARM=${1:?usage: serve_arm.sh <arm> <max_total_tokens> [config.json]}
BUDGET=${2:?usage: serve_arm.sh <arm> <max_total_tokens> [config.json]}
CONFIG=${3:-configs/${ARM}.json}

MODEL=${MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}
PORT=${PORT:-30000}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [ ! -f "$CONFIG" ]; then
  echo "[serve_arm] config not found: $CONFIG" >&2
  exit 2
fi

# 去掉换行，保证整个 JSON 是单个 argv
CFG=$(tr -d '\n' < "$CONFIG")

echo "[serve_arm] arm=$ARM budget=$BUDGET port=$PORT config=$CONFIG"
echo "[serve_arm] policy_config=$CFG"

cd "$BENCH_DIR/.."                   # sglang 仓库根
exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --radix-cache-backend simple_hybrid \
  --radix-eviction-policy simple_hybrid \
  --enable-simple-hybrid \
  --radix-eviction-policy-config "$CFG" \
  --max-total-tokens "$BUDGET" \
  --enable-metrics \
  --port "$PORT" \
  ${EXTRA_ARGS}
