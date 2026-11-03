#!/usr/bin/env bash
# RR vs T-LRU burst/pause 矩阵：臂 × 回归方式 × 预算 × seed。
#
#   bash run_matrix.sh                      # 全矩阵
#   ARMS="a1_rr a2_tlru" RESUME_MODES="simultaneous batched" \
#   BUDGETS=60000 SEEDS="42 43 44" bash run_matrix.sh
#
# 幂等：result/<arm>/<mode>/<budget>/<seed>.json 已存在则跳过。
set -euo pipefail

cd "$(dirname "$0")"
BENCH_DIR="$(cd .. && pwd)"
source "$BENCH_DIR/env.sh"

ARMS=${ARMS:-"a0_control a1_rr a2_tlru"}
RESUME_MODES=${RESUME_MODES:-"simultaneous batched"}
BUDGETS=${BUDGETS:-"60000 100000"}
SEEDS=${SEEDS:-"42 43 44"}

# burst/pause 形状（三个臂必须完全一致）
TASKS_CONFIG=${TASKS_CONFIG:-configs/burst_pause.yaml}
BURST_DURATION_S=${BURST_DURATION_S:-60}
WAVE_SIZE=${WAVE_SIZE:-4}
WAVE_INTERVAL_S=${WAVE_INTERVAL_S:-10}
CONC=${CONC:-32}
WAIT_SCALE=${WAIT_SCALE:-1}
TIMEOUT_S=${TIMEOUT_S:-1800}

PORT=${PORT:-30000}
BASE_URL="http://127.0.0.1:${PORT}"

stop_server() {
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
  sleep 3
  for _ in $(seq 1 20); do
    curl -sf "$BASE_URL/health" >/dev/null 2>&1 || return 0
    sleep 2
  done
  echo "[matrix] WARN: port $PORT still alive" >&2
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

for mode in $RESUME_MODES; do
  for budget in $BUDGETS; do
    for arm in $ARMS; do
      for seed in $SEEDS; do
        out="result/${arm}/${mode}/${budget}"
        mkdir -p "$out"
        if [ -f "$out/${seed}.json" ]; then
          echo "[matrix] skip ${arm}/${mode}/${budget}/seed=${seed}"
          continue
        fi

        stop_server
        echo "[matrix] start arm=${arm} mode=${mode} budget=${budget} seed=${seed}"
        PORT=$PORT bash serve_arm.sh "$arm" "$budget" "configs/arm_${arm}.json" \
          > "$out/server_${seed}.log" 2>&1 &
        SERVER_PID=$!

        if ! wait_ready; then
          echo "[matrix] ERROR: server not ready" >&2
          tail -40 "$out/server_${seed}.log" >&2 || true
          kill "$SERVER_PID" 2>/dev/null || true
          exit 1
        fi

        "$PY" "$BENCH_DIR/bench_unicache.py" \
          --tasks-config "$TASKS_CONFIG" \
          --burst-duration-s "$BURST_DURATION_S" \
          --resume-mode "$mode" \
          --resume-wave-size "$WAVE_SIZE" \
          --resume-wave-interval-s "$WAVE_INTERVAL_S" \
          --concurrency "$CONC" \
          --wait-scale "$WAIT_SCALE" \
          --seed "$seed" \
          --timeout-s "$TIMEOUT_S" \
          --base-url "$BASE_URL" \
          --output "$out/${seed}.json" \
          2>&1 | tee "$out/bench_${seed}.log"

        grep -E "victim_src:|tlru:|one_shot_evicted=|pool=" "$out/server_${seed}.log" \
          > "$out/profile_${seed}.txt" || true

        "$PY" - "$out" "$arm" "$mode" "$budget" "$seed" <<'PY'
import json, pathlib, sys
out, arm, mode, budget, seed = sys.argv[1:6]
meta = {"arm": arm, "resume_mode": mode,
        "max_total_tokens": int(budget), "seed": int(seed)}
pathlib.Path(out, f"meta_{seed}.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2))
PY

        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        stop_server
        echo "[matrix] done ${arm}/${mode}/${budget}/seed=${seed}"
      done
    done
  done
done

echo "[matrix] all done -> result/"
