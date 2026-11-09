#!/usr/bin/env bash
# Cascade 矩阵：逐格 起 server → 标定(每策略一次) → 跑 bench → 停 server
# 用法: bash tailshot/run_cascade.sh [cell ...]      默认只跑 R3
#       OUT=/path SEED=42 TAGSUF=_s42 bash tailshot/run_cascade.sh R3
#
# 说明：--attention-backend triton 是环境必需（flashinfer 预编译在 CUDA 13 头文件下报错），
#       所有格子统一使用，避免不同格子后端不一致。
set -u
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${OUT:-$BENCH/result/cascade}
SEED=${SEED:-42}
TAGSUF=${TAGSUF:-}
CELLS=${*:-R3}
PORT=${PORT:-30000}
mkdir -p "$OUT"

run_one () {
  local CFG=$1 TAG=$2
  echo "=== [$TAG] server up $(date +%T) ==="
  POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 PORT=$PORT \
    LOG=$OUT/server_${TAG}.log EXTRA_ARGS="--attention-backend triton" \
    nohup bash "$BENCH/serve.sh" > /dev/null 2>&1 &
  for _ in $(seq 1 120); do
    sleep 5
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  done
  if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "=== [$TAG] server FAILED，见 $OUT/server_${TAG}.log ==="
    return 1
  fi
  echo "=== [$TAG] server healthy $(date +%T)，池=$(grep -o 'max_total_num_tokens=[0-9]*' "$OUT/server_${TAG}.log" | tail -1) ==="

  if [ ! -f "$OUT/calib_simple_hybrid.json" ]; then
    /root/sl_old/bin/python "$BENCH/calibrate.py" \
      --base-url "http://127.0.0.1:$PORT" --output "$OUT/calib_simple_hybrid.json"
  fi

  echo "=== [$TAG] bench start $(date +%T) ==="
  /root/sl_old/bin/python "$BENCH/bench_unicache.py" \
    --tasks-config "$BENCH/tailshot/cascade_${CFG}.yaml" \
    --concurrency 32 --arrival-scale 1 --wait-scale 1 --seed "$SEED" \
    --base-url "http://127.0.0.1:$PORT" \
    --output "$OUT/${TAG}${TAGSUF}.json" 2>&1 | tee "$OUT/${TAG}${TAGSUF}.run.log"
  echo "=== [$TAG] bench done $(date +%T) ==="

  pkill -f "sglang.launch_server" 2>/dev/null
  sleep 10
}

for c in $CELLS; do
  run_one "$c" "$c"
done
echo "=== CASCADE ALL DONE $(date +%T) ==="
