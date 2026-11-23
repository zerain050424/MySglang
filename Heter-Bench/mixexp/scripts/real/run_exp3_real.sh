#!/usr/bin/env bash
# =============================================================================
# 真机一键实验：exp3（scan / struct / chat / agentic）策略对比与结果汇总
#
# 流程（对每个 cell × arm）：
#   1) 后台拉起 serve.sh 并轮询 /health 等待就绪
#   2) 调用 scripts/bench_heap.py 发射全量负载
#   3) 安全停服并释放显存
#   4) 全部臂完成后自动调用 mixexp/scripts/real/analyze.py 输出汇总表
#
# 用法（在仓库根目录或 Heter-Bench 目录执行）:
#   bash Heter-Bench/mixexp/scripts/real/run_exp3_real.sh
#   CELL=chat ARMS="lru hc_full chain_lru" bash Heter-Bench/mixexp/scripts/real/run_exp3_real.sh
#
# 环境变量:
#   CELL               目标 cell（默认 chat；可用空格传多个，如 "scan struct chat agentic"）
#   ARMS               要跑的实验臂（默认 "lru hc_full chain_lru"；可选 hc_norr / heter_rgate）
#   MAX_TOTAL_TOKENS   显存池大小（默认 198000）
#   PORT               服务端口（默认 30000）
#   WARMUP_S           预热统计过滤窗口秒数（默认 1000）
#   EXTRA_ARGS         透传给 serve.sh 的附加参数（默认 "--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85"）
# =============================================================================
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BENCH_DIR"

source "$BENCH_DIR/scripts/env.sh"
source "$BENCH_DIR/scripts/config.sh"

CELLS="${CELL:-chat}"
ARMS="${ARMS:-lru hc_full chain_lru}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-198000}"
PORT="${PORT:-30000}"
WARMUP_S="${WARMUP_S:-1000}"
EXTRA_ARGS="${EXTRA_ARGS:---kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85}"
BASE_URL="http://127.0.0.1:${PORT}"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[real] 清理后台 server (pid=$SERVER_PID)..."
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
    echo "[real] 等待服务端口 ${BASE_URL}/health 就绪..."
    while [ $i -lt $max_wait ]; do
        if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
            echo "[real] server 已就绪 (${i}s)"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "❌ server 进程提前退出，请检查日志。" >&2
            return 1
        fi
        sleep 2
        i=$((i + 2))
    done
    echo "❌ 等待 server 就绪超时 (${max_wait}s)" >&2
    return 1
}

for cell in $CELLS; do
    TASKS_YAML="TaskData/exp3/${cell}/tasks.yaml"
    [ -f "$TASKS_YAML" ] || { echo "❌ 缺少 $TASKS_YAML" >&2; exit 1; }
    OUT_DIR="mixexp/result/exp3/${cell}"
    mkdir -p "$OUT_DIR"

    for arm in $ARMS; do
        POLICY_CFG=""
        case "$arm" in
            lru)
                POL="lru"
                ;;
            hc_full)
                POL="heter_cache"
                POLICY_CFG='{"reuse":{"enabled":true}}'
                ;;
            hc_norr)
                POL="heter_cache"
                POLICY_CFG='{"reuse":{"enabled":false}}'
                ;;
            chain_lru)
                POL="chain_lru"
                POLICY_CFG='{"floor_tokens":8192,"weight":0.02,"gate":"multiturn","tail_first":true}'
                ;;
            heter_rgate)
                POL="heter_cache"
                POLICY_CFG='{"regret_gate":{"enabled":true,"k":0.6,"r_min":0.3,"window_s":120,"priced_fallback":true,"floor_tokens":8192,"weight":0.02,"beta":1.0}}'
                ;;
            *)
                echo "❌ 不支持的 arm: $arm" >&2
                exit 1
                ;;
        esac

        echo ""
        echo "=============================================================="
        echo "  [real-exp3] cell=$cell  arm=$arm  policy=$POL"
        echo "=============================================================="

        LOG="$OUT_DIR/server_full_${arm}.log"
        if [ -n "$POLICY_CFG" ]; then
            POLICY="$POL" POLICY_CONFIG="$POLICY_CFG" \
                MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" PORT="$PORT" \
                EXTRA_ARGS="$EXTRA_ARGS" LOG="$LOG" \
                bash "$BENCH_DIR/scripts/serve.sh" >/dev/null 2>&1 &
        else
            POLICY="$POL" \
                MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" PORT="$PORT" \
                EXTRA_ARGS="$EXTRA_ARGS" LOG="$LOG" \
                bash "$BENCH_DIR/scripts/serve.sh" >/dev/null 2>&1 &
        fi
        SERVER_PID=$!

        wait_for_server

        "$PY" "$BENCH_DIR/scripts/bench_heap.py" \
            --tasks-config "$TASKS_YAML" \
            --base-url "$BASE_URL" \
            --arrival-scale 1.0 \
            --wait-scale 1.0 \
            --max-inflight 0 \
            --warmup-s "$WARMUP_S" \
            --output "$OUT_DIR/full_${arm}.json"

        cleanup
        sleep 3
    done

    echo ""
    echo ">>> [real-exp3] 汇总 cell=$cell 结果:"
    "$PY" "$SCRIPT_DIR/analyze.py" --dir "$OUT_DIR" --scale full --tokens "$MAX_TOTAL_TOKENS"
done
