#!/usr/bin/env bash
set -u
BENCH=/root/sg-unicache/Uni-Bench
OUT=$BENCH/result/tailshot_bigv
cd /root/sg-unicache/Uni-Bench
while ! grep -q "MATRIX ALL DONE" result/tailshot_bigv/matrix.log 2>/dev/null; do sleep 60; done
sleep 30

run_one () {
  local POLICY=$1 CFG=$2 TAG=$3
  local PORT=30000
  echo "=== [$TAG] server up (policy=$POLICY) $(date +%T) ==="
  POLICY=$POLICY MAX_TOTAL_TOKENS=100000 PORT=$PORT \
    LOG=$OUT/server_${TAG}.log EXTRA_ARGS="--mem-fraction-static 0.79" \
    nohup bash "$BENCH/serve.sh" > /dev/null 2>&1 &
  for i in $(seq 1 120); do
    sleep 5
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  done
  if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "=== [$TAG] server FAILED ==="; return 1
  fi
  echo "=== [$TAG] bench start $(date +%T) ==="
  /root/sl_old/bin/python "$BENCH/bench_unicache.py" \
    --tasks-config "$CFG" --concurrency 32 \
    --arrival-scale 1 --wait-scale 1 \
    --base-url "http://127.0.0.1:$PORT" \
    --output "$OUT/${TAG}.json"
  echo "=== [$TAG] bench done $(date +%T) ==="
  pkill -f "sglang.launch_server" 2>/dev/null
  sleep 15
}

run_one lru "$BENCH/tailshot/tailshot_F2000.yaml" "lru_F2000"
run_one simple_hybrid "$BENCH/tailshot/tailshot_F2000.yaml" "simple_hybrid_F2000"
echo "=== F2000 APPEND DONE $(date +%T) ==="
