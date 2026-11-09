#!/usr/bin/env bash
# 启动一个 sglang server。服务器环境修正见 env.sh。
#
# 用法：
#   POLICY=heter_cache MAX_TOTAL_TOKENS=100000 bash scripts/serve.sh
#   POLICY=lru          MAX_TOTAL_TOKENS=100000 bash scripts/serve.sh
#
# 注意：heter_cache 需要同时加 --enable-heter-cache（选择 HeterCacheRadixCache）
#      和 --radix-eviction-policy=heter_cache（选择 HeterCacheStrategy），缺一不可。
set -eu
BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=env.sh
source "$BENCH_DIR/scripts/env.sh"
# shellcheck source=config.sh
source "$BENCH_DIR/scripts/config.sh"

MODEL=${MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}
PORT=${PORT:-30000}
POLICY=${POLICY:-lru}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-200000}
LOG=${LOG:-$BENCH_DIR/result/log/server_${POLICY}_${MAX_TOTAL_TOKENS}.log}

mkdir -p "$(dirname "$LOG")"

# heter_cache 需要额外的 --enable-heter-cache。
# POLICY_CONFIG 传策略参数 JSON（含空格，必须作为一个参数整体传给 CLI）。
# EXTRA_ARGS 是自由附加参数（如 --attention-backend triton），按空格拆分。
SERVER_ARGS=()
if [ "$POLICY" = "heter_cache" ]; then
  SERVER_ARGS+=(--enable-heter-cache)
fi
if [ -n "${POLICY_CONFIG:-}" ]; then
  SERVER_ARGS+=(--radix-eviction-policy-config "$POLICY_CONFIG")
fi
EXTRA_ARGS=${EXTRA_ARGS:-}
if [ -n "$EXTRA_ARGS" ]; then
  read -r -a _EXTRA <<< "$EXTRA_ARGS"
  SERVER_ARGS+=("${_EXTRA[@]}")
fi

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
  "${SERVER_ARGS[@]}" \
  2>&1 | tee "$LOG"
