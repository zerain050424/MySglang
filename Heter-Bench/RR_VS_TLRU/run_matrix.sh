#!/usr/bin/env bash
# RR vs T-LRU burst/pause 矩阵：臂 × 回归方式 × 预算 × seed。
#
#   bash run_matrix.sh                       # 快跑（默认 2 seed，约 1 小时）
#   SEEDS="42" bash run_matrix.sh            # 更快的试点
#   SEEDS="42 43 44" bash run_matrix.sh      # 全量
#
# 预算标定（实测）：单 run 常驻峰值 35.7k tokens，去重后就是真实工作集。
#   BUDGET=22000 -> 1.62x（约 13.7k 被淘汰）
#   BUDGET=30000 -> 1.19x（约 5.7k 被淘汰，最敏感的温和压力）
#   再高（60k/100k）就没有淘汰发生，三个臂不可能有差异。
#
# 时间由三部分构成，且最大的一项是 server 启动（约 1.5 min/次）：
#   启动   ~90 s   ← 只依赖 (arm, budget)，默认按该组合复用，seed/mode 之间用
#                    /flush_cache 隔开（36 次启动 -> 6 次）
#   bench  ~110 s
#   flush  ~2 s
#
# 幂等：result/<arm>/<mode>/<budget>/<seed>.json 已存在则跳过，可中断续跑。
set -euo pipefail

cd "$(dirname "$0")"
BENCH_DIR="$(cd .. && pwd)"
source "$BENCH_DIR/scripts/env.sh"

ARMS=${ARMS:-"a0_control a1_rr"}
RESUME_MODES=${RESUME_MODES:-"simultaneous batched"}
BUDGETS=${BUDGETS:-"22000 30000"}
SEEDS=${SEEDS:-"42 43"}

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

# server 只依赖 (arm, budget)，与 resume_mode / seed 无关：
# 默认按 (arm, budget) 复用同一个 server，seed 之间用 /flush_cache 隔开。
GROUP_BY_SERVER=${GROUP_BY_SERVER:-1}
FLUSH_TIMEOUT_S=${FLUSH_TIMEOUT_S:-120}

flush_cache() {
  # GET / POST 都支持。失败不致命，但必须显式告警——否则 seed 之间会串味。
  if curl -sf -m "$FLUSH_TIMEOUT_S" -X POST \
      "$BASE_URL/flush_cache?timeout=$FLUSH_TIMEOUT_S" >/dev/null 2>&1; then
    return 0
  fi
  if curl -sf -m "$FLUSH_TIMEOUT_S" \
      "$BASE_URL/flush_cache?timeout=$FLUSH_TIMEOUT_S" >/dev/null 2>&1; then
    return 0
  fi
  echo "[matrix] WARN: flush_cache 失败，seed 之间的隔离可能被破坏" >&2
  return 1
}

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

start_server() {  # $1=arm $2=budget
  # POLICY_CONFIG 可覆盖该臂的配置（扫 TLRU 阈值时用）
  PORT=$PORT bash serve_arm.sh "$1" "$2" "${POLICY_CONFIG:-configs/arm_$1.json}"
}

run_one() {  # $1=arm $2=mode $3=budget $4=seed $5=当前 server 的日志路径
  local arm=$1 mode=$2 budget=$3 seed=$4 log=${5:-}
  local out="result/${arm}/${mode}/${budget}"
  mkdir -p "$out"
  if [ -f "$out/${seed}.json" ]; then
    echo "[matrix] skip ${arm}/${mode}/${budget}/seed=${seed}"
    return 0
  fi

  flush_cache || true
  # 复用 server 时日志是合一的：记下当前行号，bench 结束后只取新增部分，
  # 这样每个 seed 的 profile 与 server 日志切片都是「本轮自己的」。
  local mark=0
  if [ -n "$log" ] && [ -f "$log" ]; then
    mark=$(wc -l < "$log")
  fi
  echo "[matrix] bench arm=${arm} mode=${mode} budget=${budget} seed=${seed}"
  "$PY" "$BENCH_DIR/scripts/bench_hetercache.py" \
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

  if [ -n "$log" ] && [ -f "$log" ]; then
    tail -n "+$((mark + 1))" "$log" > "$out/server_${seed}.log"
    grep -E "victim_src:|tlru:|one_shot_evicted=|pool=" "$out/server_${seed}.log" \
      > "$out/profile_${seed}.txt" || true
  fi

  "$PY" - "$out" "$arm" "$mode" "$budget" "$seed" <<'META'
import json, pathlib, sys
out, arm, mode, budget, seed = sys.argv[1:6]
meta = {"arm": arm, "resume_mode": mode,
        "max_total_tokens": int(budget), "seed": int(seed)}
pathlib.Path(out, f"meta_{seed}.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2))
META
}

cleanup() { stop_server; }
trap cleanup EXIT

for budget in $BUDGETS; do
  for arm in $ARMS; do
    pending=0
    for mode in $RESUME_MODES; do
      for seed in $SEEDS; do
        [ -f "result/${arm}/${mode}/${budget}/${seed}.json" ] || pending=$((pending + 1))
      done
    done
    if [ "$pending" -eq 0 ]; then
      echo "[matrix] skip ${arm}/${budget}（所有 mode/seed 已完成）"
      continue
    fi

    if [ "$GROUP_BY_SERVER" = "1" ]; then
      stop_server
      shared_log="result/${arm}/__server_${budget}.log"
      mkdir -p "result/${arm}"
      echo "[matrix] start server arm=${arm} budget=${budget}（${pending} 次运行共用）"
      start_server "$arm" "$budget" > "$shared_log" 2>&1 &
      SERVER_PID=$!
      if ! wait_ready; then
        echo "[matrix] ERROR: server 未就绪" >&2
        tail -40 "$shared_log" >&2 || true
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
      fi
      for mode in $RESUME_MODES; do
        for seed in $SEEDS; do
          run_one "$arm" "$mode" "$budget" "$seed" "$shared_log"
        done
      done
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      stop_server
    else
      for mode in $RESUME_MODES; do
        for seed in $SEEDS; do
          [ -f "result/${arm}/${mode}/${budget}/${seed}.json" ] && continue
          stop_server
          out="result/${arm}/${mode}/${budget}"
          mkdir -p "$out"
          start_server "$arm" "$budget" > "$out/server_${seed}.log" 2>&1 &
          SERVER_PID=$!
          if ! wait_ready; then
            echo "[matrix] ERROR: server 未就绪" >&2
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
          fi
          run_one "$arm" "$mode" "$budget" "$seed" "$out/server_${seed}.log"
          kill "$SERVER_PID" 2>/dev/null || true
          wait "$SERVER_PID" 2>/dev/null || true
          stop_server
        done
      done
    fi
  done
done

echo "[matrix] all done -> result/"
