#!/usr/bin/env bash
# 启动一个 sglang server。服务器环境修正见 env.sh。
#
# 用法：
#   POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 bash serve.sh
#   POLICY=lru          MAX_TOTAL_TOKENS=100000 bash serve.sh
#
# 注意：simple_hybrid 需要同时加 --enable-simple-hybrid（选择 SimpleHybridRadixCache）
#      和 --radix-eviction-policy=simple_hybrid（选择 SimpleHybridStrategy），缺一不可。
set -eu
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$BENCH_DIR/env.sh"
# shellcheck source=config.sh
source "$BENCH_DIR/config.sh"

MODEL=${MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}
PORT=${PORT:-30000}
POLICY=${POLICY:-lru}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-200000}
LOG=${LOG:-$BENCH_DIR/result/log/server_${POLICY}_${MAX_TOTAL_TOKENS}.log}

mkdir -p "$(dirname "$LOG")"

# simple_hybrid 需要额外的 --enable-simple-hybrid
EXTRA=""
if [ "$POLICY" = "simple_hybrid" ]; then
  EXTRA="--enable-simple-hybrid"
fi
# 可选附加参数（如 --radix-cache-backend=classic），用 EXTRA_ARGS 传入
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "[serve] policy=$POLICY max_total_tokens=$MAX_TOTAL_TOKENS port=$PORT"
echo "[serve] log -> $LOG"

# 需要从 sglang 仓库根目录运行（python -m sglang.launch_server）
cd "$BENCH_DIR/.."

"$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --radix-eviction-policy "$POLICY" \
  --enable-metrics \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --port "$PORT" \
  $EXTRA $EXTRA_ARGS \
  2>&1 | tee "$LOG"
