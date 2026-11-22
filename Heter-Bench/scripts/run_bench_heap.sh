#!/usr/bin/env bash
# =============================================================================
# run_bench_heap.sh: Heter-Bench 堆发射器端到端自动化运行脚本
# 
# 流程：
# 1. 检查/加载环境配置 (config.sh / env.sh)
# 2. 检查或启动 SGLang 推理服务器 (serve.sh)，支持 heter_cache / lru 等
# 3. 轮询等待服务器就绪 (/health)
# 4. 执行 bench_heap.py 堆发射器进行真实墙钟压测
# 5. 退出时自动清理本次启动的后台服务进程并展示结果
#
# 用法示例：
#   # 1) 默认跑混合负载 (heter_cache):
#   bash Heter-Bench/scripts/run_bench_heap.sh
#
#   # 2) 跑 LRU 策略对照，自定义显存容量与任务配置:
#   POLICY=lru MAX_TOTAL_TOKENS=198000 \
#     TASKS_CONFIG=Heter-Bench/mixexp/TaskData/exp3/chat/tasks.yaml \
#     OUTPUT=Heter-Bench/result/exp3_chat_lru.json \
#     bash Heter-Bench/scripts/run_bench_heap.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HETER_BENCH_DIR="${ROOT_DIR}/Heter-Bench"

# -----------------------------------------------------------------------------
# 1. 环境与参数默认值
# -----------------------------------------------------------------------------
if [ -f "${SCRIPT_DIR}/config.sh" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/config.sh"
fi
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/env.sh"
fi

PY="${PY:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="http://${HOST}:${PORT}"

POLICY="${POLICY:-heter_cache}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-100000}"
POLICY_CONFIG="${POLICY_CONFIG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASKS_CONFIG="${TASKS_CONFIG:-${HETER_BENCH_DIR}/configs/hybrid_tasks_config.yaml}"
OUTPUT="${OUTPUT:-${HETER_BENCH_DIR}/result/bench_heap_${POLICY}_$(date +%Y%m%d_%H%M%S).json}"

MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
WAIT_SCALE="${WAIT_SCALE:-1.0}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
WARMUP_S="${WARMUP_S:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
TIMEOUT_S="${TIMEOUT_S:-1800}"

# 确保输出目录存在
mkdir -p "$(dirname "${OUTPUT}")"
LOG_FILE="$(dirname "${OUTPUT}")/server_${POLICY}.log"

# -----------------------------------------------------------------------------
# 2. 检查与启动 SGLang 服务器
# -----------------------------------------------------------------------------
SERVER_PID=""

cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        echo -e "\n[run_bench] 正在停止后台 Server (PID: ${SERVER_PID})..."
        kill -15 "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        echo "[run_bench] Server 已终止。"
    fi
}
trap cleanup EXIT INT TERM

is_server_ready() {
    curl -s --connect-timeout 2 "${BASE_URL}/health" >/dev/null 2>&1
}

if is_server_ready; then
    echo "[run_bench] 检测到已存在运行中的 Server (${BASE_URL})，直接复用。"
else
    echo "[run_bench] 启动后台 Server: POLICY=${POLICY}, MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}, PORT=${PORT}"
    echo "[run_bench] 日志输出重定向至: ${LOG_FILE}"

    export POLICY MAX_TOTAL_TOKENS PORT LOG="${LOG_FILE}"
    if [ -n "${POLICY_CONFIG}" ]; then
        export POLICY_CONFIG
    fi
    if [ -n "${EXTRA_ARGS}" ]; then
        export EXTRA_ARGS
    fi

    # 启动 serve.sh
    bash "${SCRIPT_DIR}/serve.sh" > "${LOG_FILE}" 2>&1 &
    SERVER_PID=$!

    echo "[run_bench] 等待 Server 初始化并就绪..."
    READY=0
    for ((i=1; i<=300; i++)); do
        if is_server_ready; then
            READY=1
            break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[run_bench] 错误: Server 进程已异常退出，请检查日志: ${LOG_FILE}"
            exit 1
        fi
        sleep 2
    done

    if [ "${READY}" -ne 1 ]; then
        echo "[run_bench] 错误: 等待 Server 启动超时 (600s)，日志: ${LOG_FILE}"
        exit 1
    fi
    echo "[run_bench] Server 已成功就绪 (${BASE_URL})！"
fi

# -----------------------------------------------------------------------------
# 3. 运行堆压测发射器
# -----------------------------------------------------------------------------
echo "========================================================================"
echo "[run_bench] 开始执行 bench_heap 压测"
echo "  Tasks 配置    : ${TASKS_CONFIG}"
echo "  Base URL      : ${BASE_URL}"
echo "  Max Inflight  : ${MAX_INFLIGHT}"
echo "  Wait Scale    : ${WAIT_SCALE}"
echo "  Arrival Scale : ${ARRIVAL_SCALE}"
echo "  Warmup (s)    : ${WARMUP_S}"
echo "  结果输出      : ${OUTPUT}"
echo "========================================================================"

"${PY}" "${SCRIPT_DIR}/bench_heap.py" \
    --tasks-config "${TASKS_CONFIG}" \
    --base-url "${BASE_URL}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --wait-scale "${WAIT_SCALE}" \
    --arrival-scale "${ARRIVAL_SCALE}" \
    --warmup-s "${WARMUP_S}" \
    --progress-every "${PROGRESS_EVERY}" \
    --timeout-s "${TIMEOUT_S}" \
    --output "${OUTPUT}"

echo "========================================================================"
echo "[run_bench] 压测执行完成！"
echo "  JSON 摘要文件 : ${OUTPUT}"
CSV_OUTPUT="${OUTPUT%.json}.csv"
if [ -f "${CSV_OUTPUT}" ]; then
    echo "  CSV 逐行文件  : ${CSV_OUTPUT}"
fi
echo "========================================================================"
