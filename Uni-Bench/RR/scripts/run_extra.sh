#!/usr/bin/env bash
cd /root/sg-unicache/Uni-Bench
OUT=result/burst
run () { local POOL=$1 ARM=$2 CFG=$3 MF=${4:-} TAG=$5
  POOL=$POOL POLICY=$ARM WAIT_SCALE=1.0 SEED=42 TAG=$TAG POLICY_CFG="$CFG" MEM_FRAC=$MF \
    bash tailshot/run_large.sh > $OUT/$TAG.driver.log 2>&1
  mv result/large/$TAG.json result/large/$TAG.csv $OUT/ 2>/dev/null
  echo "  done $TAG $(date +%T)"
}
echo "=== 补跑 C: n_a=8 @36k (thrash) ==="
for A in base rr; do
  [ "$A" = rr ] && C='{"fair_rr":true}' || C=''
  run burst_n8 $A "$C" "" thrash_n8_$A
done
echo "=== 补跑 B: κ=0.3 ==="
run s1n3 rr '{"fair_rr":true,"kappa":0.3}' "" kappa03
echo "=== 补跑完 $(date +%T) ==="
