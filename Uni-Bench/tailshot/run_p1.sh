#!/usr/bin/env bash
# P1 批次：ρ ∈ {1.2,2.0,3.0} × 池种子 {1,2} × 策略 {base,drr}
# 每格：建池（含 gap_s）→ 起 server → 跑 bench → 停 server → 汇总指标
set -u
BENCH="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BENCH"
OUT=result/large
mkdir -p "$OUT"
for RHO in ${RHOS:-1.2 2.0 3.0}; do
  for PSEED in ${PSEEDS:-1 2}; do
    POOL="alfred_r${RHO}_s${PSEED}"
    if [ ! -f "TaskData/large/${POOL}.jsonl" ]; then
      echo "=== build $POOL $(date +%T) ==="
      /root/sl_old/bin/python exp/build_large_pools.py \
        --source alfred --pool 36000 --rho "$RHO" --model claude --seed "$PSEED" \
        --max-turns 20 --max-gap 90 --name "$POOL" 2>&1 | tail -4
    fi
    for POL in base drr; do
      TAG="${POOL}_${POL}"
      POOL="$POOL" POLICY="$POL" WAIT_SCALE="${WAIT_SCALE:-0.02}" SEED=42 TAG="$TAG" \
        bash tailshot/run_large.sh > "$OUT/${TAG}.driver.log" 2>&1
      /root/sl_old/bin/python exp/summarize_large.py \
        --csv "$OUT/${TAG}.csv" --wait-scale "${WAIT_SCALE:-0.02}" \
        --json-out "$OUT/${TAG}.metrics.json" > "$OUT/${TAG}.summary.log" 2>&1
      echo "--- $TAG done $(date +%T) ---"
      grep -E "hit_agentic|max_recompute|cold_hit_min|deficit_max|deficit_gini" "$OUT/${TAG}.summary.log" | sed 's/^/    /'
    done
  done
done
echo "=== P1 ALL DONE $(date +%T) ==="
