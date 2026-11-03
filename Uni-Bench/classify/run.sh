#!/usr/bin/env bash
# 会话级分类准确率：主实验 + 阈值扫描。
# 纯离线（不启动 server、不占 GPU），依赖 /root/sl_old/bin/python。
set -eu
BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-/root/sl_old/bin/python}
OUT=${OUT:-$BENCH_DIR/result/classify}
cd "$BENCH_DIR/classify"

echo "[1/4] 真实 trace（$BENCH_DIR/result/phased_load_prediction_C.csv）"
"$PY" run_eval.py --source trace --out "$OUT"

echo "[2/4] 任务数据回放（每负载 1500 会话，按负载的轮间隔模型）"
"$PY" run_eval.py --source task --max-per-workload 1500 --gap-model per_workload \
    --tag task --out "$OUT"

echo "[3/4] 阈值扫描（真实 trace）"
"$PY" sweep.py --source trace --tag trace --out "$OUT"

echo "[4/4] 阈值扫描（任务数据 + slow-gap 场景）"
"$PY" sweep.py --source task --max-per-workload 1500 --tag task --out "$OUT"

echo "[done] 结果在 $OUT"
