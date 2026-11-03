#!/usr/bin/env bash
# TLRU vs RR 实验矩阵：臂 × KV 预算 × seed。
#
#   bash run_matrix.sh                 # 3 臂 × 3 预算 × 3 seed
#   BUDGETS=200000 SEEDS=42 ARMS=a1_rr bash run_matrix.sh
#
# 幂等：result/<arm>/<budget>/<seed>.json 已存在则跳过，可中断续跑。
set -euo pipefail

cd "$(dirname "$0")"
BENCH_DIR="$(cd .. && pwd)"
source "$BENCH_DIR/env.sh"

ARMS=${ARMS:-"a0_control a1_rr a2_tlru"}
BUDGETS=${BUDGETS:-"100000 200000 400000"}
SEEDS=${SEEDS:-"42 43 44"}

PORT=${PORT:-30000}
BASE_URL="http://127.0.0.1:${PORT}"

DATA=${DATA:-$BENCH_DIR/TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl}
MU=${MU:-1.81}
SIGMA=${SIGMA:-1.092}
NUM_SESSIONS=${NUM_SESSIONS:-500}
CONC=${CONC:-32}
ARRIVAL_SCALE=${ARRIVAL_SCALE:-1}
WAIT_SCALE=${WAIT_SCALE:-1}
TIMEOUT_S=${TIMEOUT_S:-1800}
# 允许为某个臂指定不同的 config（例如扫 TLRU 阈值）
TLRU_CONFIG=${TLRU_CONFIG:-configs/a2_tlru.json}
RR_CONFIG=${RR_CONFIG:-configs/a1_rr.json}

if [ ! -f "$DATA" ]; then
  echo "[matrix] data not found: $DATA" >&2
  exit 2
fi

config_of() {
  case "$1" in
    a1_rr)   echo "$RR_CONFIG" ;;
    a2_tlru) echo "$TLRU_CONFIG" ;;
    *)       echo "configs/$1.json" ;;
  esac
}

stop_server() {
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
  sleep 3
  # 兜底：等端口真正释放
  for _ in $(seq 1 20); do
    curl -sf "$BASE_URL/health" >/dev/null 2>&1 || return 0
    sleep 2
  done
  echo "[matrix] WARN: port $PORT still alive after stop" >&2
}

wait_ready() {
  for _ in $(seq 1 120); do
    curl -sf "$BASE_URL/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

cleanup() { stop_server; }
trap cleanup EXIT

for budget in $BUDGETS; do
  for arm in $ARMS; do
    cfg=$(config_of "$arm")
    for seed in $SEEDS; do
      out="result/${arm}/${budget}"
      mkdir -p "$out"
      if [ -f "$out/${seed}.json" ]; then
        echo "[matrix] skip ${arm}/${budget}/seed=${seed} (done)"
        continue
      fi

      stop_server
      echo "[matrix] start ${arm} budget=${budget} seed=${seed} cfg=${cfg}"
      PORT=$PORT bash serve_arm.sh "$arm" "$budget" "$cfg" \
        > "$out/server_${seed}.log" 2>&1 &
      SERVER_PID=$!

      if ! wait_ready; then
        echo "[matrix] ERROR: server not ready; tail follows" >&2
        tail -40 "$out/server_${seed}.log" >&2 || true
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
      fi

      "$PY" "$BENCH_DIR/bench_unicache.py" \
        --data-path "$DATA" \
        --mu "$MU" --sigma "$SIGMA" \
        --num-sessions "$NUM_SESSIONS" \
        --concurrency "$CONC" \
        --arrival-scale "$ARRIVAL_SCALE" \
        --wait-scale "$WAIT_SCALE" \
        --seed "$seed" \
        --timeout-s "$TIMEOUT_S" \
        --base-url "$BASE_URL" \
        --output "$out/${seed}.json" \
        2>&1 | tee "$out/bench_${seed}.log"

      # 抽 profile 关键行（server 端每 update_interval 打一次）
      grep -E "victim_src:|tlru:|one_shot_evicted=|pool=" "$out/server_${seed}.log" \
        > "$out/profile_${seed}.txt" || true

      # 复现信息
      "$PY" - "$out/${seed}.json" "$arm" "$budget" "$seed" "$cfg" <<'PY'
import json, sys, pathlib
out, arm, budget, seed, cfg = sys.argv[1:6]
meta = {
    "arm": arm, "max_total_tokens": int(budget), "seed": int(seed),
    "policy_config": json.loads(pathlib.Path(cfg).read_text()),
    "policy_config_path": cfg,
}
pathlib.Path(out).with_name(f"meta_{seed}.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2))
PY

      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      stop_server
      echo "[matrix] done ${arm}/${budget}/seed=${seed}"
    done
  done
done

echo "[matrix] all done -> result/"
