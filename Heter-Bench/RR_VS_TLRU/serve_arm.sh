#!/usr/bin/env bash
# 启动一个实验臂（burst/pause 场景）。
#   bash serve_arm.sh <arm> <max_total_tokens> [config.json]
set -euo pipefail

cd "$(dirname "$0")"
BENCH_DIR="$(cd .. && pwd)"          # Heter-Bench/
source "$BENCH_DIR/scripts/env.sh"           # $PY + 库路径

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

# 绝对化：后面会 cd 到仓库根，相对路径会失效（曾因此把原生 TLRU 误判成 heter_cache）
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
CFG=$(tr -d '\n' < "$CONFIG")
echo "[serve_arm] arm=$ARM budget=$BUDGET port=$PORT config=$CONFIG"
echo "[serve_arm] policy_config=$CFG"

cd "$BENCH_DIR/.."

# 原生 TLRU：必须在 UnifiedRadixCache 上跑（registry 会硬校验），因此不能加
# --enable-heter-cache；config 也只接受 threshold / next_prompt_estimate。
# 判断方式：按 config 内容（threshold 是 TLRUStrategy 独有的顶层键），
# 这样新增阈值档位（a4/a5/...）不必改脚本；也可用 NATIVE_TLRU=1 强制。
NATIVE=0
if [ "$ARM" = "a3_native_tlru" ] || [ "${NATIVE_TLRU:-0}" = "1" ]; then NATIVE=1; fi
if grep -q '"threshold"' "$CONFIG" 2>/dev/null; then NATIVE=1; fi
if [ "$NATIVE" = "1" ]; then
  echo "[serve_arm] native TLRU on UnifiedRadixCache"
  exec "$PY" -m sglang.launch_server \
    --model-path "$MODEL" \
    --radix-eviction-policy tlru \
    --radix-eviction-policy-config "$CFG" \
    --max-total-tokens "$BUDGET" \
    --enable-metrics \
    --port "$PORT" \
    ${EXTRA_ARGS}
fi

exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --radix-eviction-policy heter_cache \
  --enable-heter-cache \
  --radix-eviction-policy-config "$CFG" \
  --max-total-tokens "$BUDGET" \
  --enable-metrics \
  --port "$PORT" \
  ${EXTRA_ARGS}
