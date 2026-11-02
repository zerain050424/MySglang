#!/usr/bin/env bash
# 全量：A 复制(3 池种子) + B κ 扫描 + C n_a 扫描 + D pool 73k + E n_a=4
cd /root/sg-unicache/Uni-Bench
OUT=result/burst; mkdir -p $OUT
run () {  # run <pool> <arm> <cfg> <memfrac>
  local POOL=$1 ARM=$2 CFG=$3 MF=${4:-}
  POOL=$POOL POLICY=$ARM WAIT_SCALE=1.0 SEED=42 TAG=${POOL}_${ARM} POLICY_CFG="$CFG" MEM_FRAC=$MF \
    bash tailshot/run_large.sh > $OUT/${POOL}_${ARM}.driver.log 2>&1
  mv result/large/${POOL}_${ARM}.json result/large/${POOL}_${ARM}.csv $OUT/ 2>/dev/null
  echo "  done ${POOL}_${ARM} $(date +%T)"
}
build () { /root/sl_old/bin/python exp/build_burst_pool.py "$@" | tail -2; }

echo "=== B: κ 扫描 (s1n3) $(date +%T) ==="
run s1n3 rr '{"fair_rr":true,"kappa":0.3}'
run s1n3 rr '{"fair_rr":true,"kappa":0.7}'
echo "=== C: n_a=8 @36k (thrash) $(date +%T) ==="
build --seed 20251028 --n-active 8 --name n8
for A in base rr; do
  [ "$A" = rr ] && C='{"fair_rr":true}' || C=''
  run burst_n8 $A "$C"
done
echo "=== D: pool 73k, n_a=8 $(date +%T) ==="
for A in base rr; do
  [ "$A" = rr ] && C='{"fair_rr":true}' || C=''
  run burst_n8 $A "$C" 0.79
done
echo "=== E: n_a=4 @36k $(date +%T) ==="
build --seed 20251027 --n-active 4 --name n4
for A in base rr; do
  [ "$A" = rr ] && C='{"fair_rr":true}' || C=''
  run burst_n4 $A "$C"
done
echo "=== A: 复制（池种子 2/3）$(date +%T) ==="
for S in 20251029 20251030; do
  build --seed $S --n-active 3 --name "s$S"
  for A in base rr; do
    [ "$A" = rr ] && C='{"fair_rr":true}' || C=''
    run burst_s$S $A "$C"
  done
done
echo "=== FULL DONE $(date +%T) ==="
