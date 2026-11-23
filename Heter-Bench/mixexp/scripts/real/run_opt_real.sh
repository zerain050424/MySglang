#!/usr/bin/env bash
# =============================================================================
# 真机一键实验：OPT 离线最优上界重放（Capture -> UseMap -> Replay）
#
# 流程（严格遵循 README.md 5.2 节重锚定规范）:
#   1) 若无 capture CSV（默认 full_lru.csv），先用 lru 跑一次采集到达时刻
#   2) 设定重放起点 R（当前 + REPLAY_MARGIN_S），生成对齐时间轴的 use_map
#   3) 启动 POLICY=opt 服务并等至 R 时刻发起重放
#
# 用法:
#   CELL=chat bash Heter-Bench/mixexp/scripts/real/run_opt_real.sh
#
# 环境变量:
#   CELL               目标 cell（默认 chat）
#   PAGE_SIZE          页大小，必须与 server --page-size 一致（默认 1）
#   MAX_TOTAL_TOKENS   显存池大小（默认 198000）
#   REPLAY_MARGIN_S    预留给模型加载的缓冲秒数（默认 180）
#   SKIP_CAPTURE=1     若已有 full_lru.csv，跳过第一步采集
# =============================================================================
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BENCH_DIR"

source "$BENCH_DIR/scripts/env.sh"
source "$BENCH_DIR/scripts/config.sh"

CELL="${CELL:-chat}"
PAGE_SIZE="${PAGE_SIZE:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-198000}"
PORT="${PORT:-30000}"
WARMUP_S="${WARMUP_S:-1000}"
REPLAY_MARGIN_S="${REPLAY_MARGIN_S:-180}"
EXTRA_ARGS="${EXTRA_ARGS:---kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --page-size ${PAGE_SIZE}}"
BASE_URL="http://127.0.0.1:${PORT}"

OUT_DIR="mixexp/result/exp3/${CELL}"
TASKS_YAML="TaskData/exp3/${CELL}/tasks.yaml"
CAPTURE_CSV="$OUT_DIR/full_lru.csv"
USE_MAP="TaskData/exp3/${CELL}/use_map_p${PAGE_SIZE}.json"

mkdir -p "$OUT_DIR"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        pkill -P "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local max_wait=300
    local i=0
    while [ $i -lt $max_wait ]; do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "❌ server 异常退出" >&2; return 1; }
        sleep 2
        i=$((i + 2))
    done
    echo "❌ server 启动超时" >&2
    return 1
}

# 1) Capture run（如果已有 CSV 且设了 SKIP_CAPTURE=1 则跳过）
if [ "${SKIP_CAPTURE:-0}" != "1" ] || [ ! -f "$CAPTURE_CSV" ]; then
    echo ">>> [OPT 1/3] 采集到达时刻 (lru capture run)..."
    POLICY=lru MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" PORT="$PORT" \
        EXTRA_ARGS="$EXTRA_ARGS" LOG="$OUT_DIR/server_full_lru.log" \
        bash "$BENCH_DIR/scripts/serve.sh" >/dev/null 2>&1 &
    SERVER_PID=$!
    wait_for_server

    "$PY" "$BENCH_DIR/scripts/bench_heap.py" \
        --tasks-config "$TASKS_YAML" \
        --base-url "$BASE_URL" \
        --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s "$WARMUP_S" \
        --output "$OUT_DIR/full_lru.json"

    cleanup
    sleep 3
else
    echo ">>> [OPT 1/3] 复用已有 capture CSV: $CAPTURE_CSV"
fi

# 2) 构建重锚定 use_map
R=$(( $(date +%s) + REPLAY_MARGIN_S ))
echo ">>> [OPT 2/3] 生成 use_map (重放锚点 R=$R, page_size=$PAGE_SIZE)..."
"$PY" "$SCRIPT_DIR/build_use_map_from_bench.py" \
    --cell "$CELL" \
    --csv "$CAPTURE_CSV" \
    --tasks "$TASKS_YAML" \
    --page-size "$PAGE_SIZE" \
    --out "$USE_MAP" \
    --replay-start "$R"

# 3) 起 OPT 服务并等待 R 时刻重放
echo ">>> [OPT 3/3] 启动 OPT server 并等待至重放起点..."
POLICY_CFG="{\"use_map_path\":\"${USE_MAP}\",\"block_size\":${PAGE_SIZE}}"
POLICY=opt POLICY_CONFIG="$POLICY_CFG" MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" PORT="$PORT" \
    EXTRA_ARGS="$EXTRA_ARGS" LOG="$OUT_DIR/server_full_opt.log" \
    bash "$BENCH_DIR/scripts/serve.sh" >/dev/null 2>&1 &
SERVER_PID=$!
wait_for_server

NOW=$(date +%s)
if [ "$R" -gt "$NOW" ]; then
    WAIT_S=$(( R - NOW ))
    echo "[OPT] server 已就绪，等待 ${WAIT_S}s 至锚定重放起点 R=${R}..."
    sleep "$WAIT_S"
else
    echo "⚠️  [OPT] 当前已超过锚定时刻 R ($NOW > $R)，建议调大 REPLAY_MARGIN_S！"
fi

"$PY" "$BENCH_DIR/scripts/bench_heap.py" \
    --tasks-config "$TASKS_YAML" \
    --base-url "$BASE_URL" \
    --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s "$WARMUP_S" \
    --output "$OUT_DIR/full_opt.json"

cleanup
echo "✅ OPT 真机重放完成 -> $OUT_DIR/full_opt.json"
