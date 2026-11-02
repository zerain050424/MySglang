#!/usr/bin/env bash
# LargeScale 单格驱动：起 server → 跑 bench → 停 server
#
# 用法:
#   POOL=alfred_r12_s1 POLICY=base WAIT_SCALE=0.05 SEED=42 bash tailshot/run_large.sh
#   POOL=alfred_r12_s1 POLICY=drr  WAIT_SCALE=0.02 SEED=42 bash tailshot/run_large.sh
#
# POLICY: base = 纯 UniCache（fair_drr=false）
#         drr  = 会话级公平轮询（fair_drr=true，仅 agentic 队列）
set -u
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
POOL=${POOL:?need POOL}
POLICY=${POLICY:-base}
WAIT_SCALE=${WAIT_SCALE:-0.02}
SEED=${SEED:-42}
OUT=${OUT:-$BENCH/result/large}
PORT=${PORT:-30000}
TAG=${TAG:-${POOL}_${POLICY}_s${WAIT_SCALE}_seed${SEED}}
mkdir -p "$OUT"

if [ -n "${POLICY_CFG:-}" ]; then
  EXTRA="--radix-eviction-policy-config ${POLICY_CFG}"
elif [ "$POLICY" = "drr" ]; then
  EXTRA='--radix-eviction-policy-config {"fair_drr":true}'
else
  EXTRA=''
fi

echo "=== [$TAG] server up $(date +%T) ==="
POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 PORT=$PORT \
  LOG=$OUT/server_${TAG}.log EXTRA_ARGS="--attention-backend triton ${MEM_FRAC:+--mem-fraction-static $MEM_FRAC} ${EXTRA}" \
  setsid nohup bash "$BENCH/serve.sh" > /dev/null 2>&1 &
for _ in $(seq 1 120); do
  sleep 5
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
done
if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
  echo "=== [$TAG] server FAILED，见 $OUT/server_${TAG}.log ==="
  exit 1
fi
echo "=== [$TAG] healthy $(date +%T)，池=$(grep -o 'max_total_num_tokens=[0-9]*' "$OUT/server_${TAG}.log" | tail -1) ==="

echo "=== [$TAG] bench start $(date +%T) ==="
/root/sl_old/bin/python "$BENCH/bench_unicache.py" \
  --tasks-config "$BENCH/tailshot/large/${POOL}.yaml" \
  --concurrency 32 --arrival-scale 1 --wait-scale "$WAIT_SCALE" --seed "$SEED" \
  --base-url "http://127.0.0.1:$PORT" \
  --output "$OUT/${TAG}.json" 2>&1 | tee "$OUT/${TAG}.run.log"
echo "=== [$TAG] bench done $(date +%T) ==="

pgrep -f "sglang[.]launch_server" | xargs -r kill 2>/dev/null
sleep 8
echo "=== [$TAG] ALL DONE $(date +%T) ==="
