#!/usr/bin/env bash
# Tailshot 矩阵：2 策略 × (对照 + 3 搅拌强度)。
# 用法: bash tailshot/run_matrix.sh <输出根目录>
set -u
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-$BENCH/result/tailshot}
mkdir -p "$OUT"
cd "$BENCH/.."

run_one () {
  local POLICY=$1 CFG=$2 TAG=$3
  local PORT=30000
  echo "=== [$TAG] server up (policy=$POLICY) $(date +%T) ==="
  POLICY=$POLICY MAX_TOTAL_TOKENS=100000 PORT=$PORT \
    LOG=$OUT/server_${TAG}.log EXTRA_ARGS="$EXTRA_ARGS" nohup bash "$BENCH/serve.sh" > /dev/null 2>&1 &
  local SPID=$!
  # 等待健康（最长 10 分钟）
  for i in $(seq 1 120); do
    sleep 5
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  done
  if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "=== [$TAG] server FAILED to become healthy ==="; kill $SPID 2>/dev/null; return 1
  fi
  echo "=== [$TAG] server healthy $(date +%T) ==="

  # 冷启动标定（每策略一次，随 server 生命周期）
  if [ ! -f "$OUT/calib_${POLICY}.json" ]; then
    /root/sl_old/bin/python "$BENCH/calibrate.py" \
      --base-url "http://127.0.0.1:$PORT" \
      --output "$OUT/calib_${POLICY}.json"
  fi

  echo "=== [$TAG] bench start $(date +%T) ==="
  /root/sl_old/bin/python "$BENCH/bench_unicache.py" \
    --tasks-config "$CFG" --concurrency 32 \
    --arrival-scale 1 --wait-scale 1 \
    --base-url "http://127.0.0.1:$PORT" \
    --output "$OUT/${TAG}.json"
  echo "=== [$TAG] bench done $(date +%T) ==="

  pkill -f "sglang.launch_server" 2>/dev/null
  sleep 15  # 等 GPU 释放
}

for POLICY in lru simple_hybrid; do
  run_one $POLICY "$BENCH/tailshot/tailshot_control.yaml"      "${POLICY}_control"
  run_one $POLICY "$BENCH/tailshot/tailshot_F150.yaml"         "${POLICY}_F150"
  run_one $POLICY "$BENCH/tailshot/tailshot_F400.yaml"         "${POLICY}_F400"
  run_one $POLICY "$BENCH/tailshot/tailshot_F1200.yaml"        "${POLICY}_F1200"
done
echo "=== MATRIX ALL DONE $(date +%T) ==="
