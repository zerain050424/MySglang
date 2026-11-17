#!/usr/bin/env bash
# =============================================================================
# 用 SGLang Simulator 跑 HeterCache 的离线仿真（CPU-only）。
#
# 干什么：
#   1. 起一个 sglang simulator server（真 Scheduler + 真 RadixCache 装配链，
#      只把 model forward 换成延迟预测器），带 --enable-heter-cache；
#   2. 用 benchmark/simulator/bench_serving.py 回放一条 Autobench trace；
#   3. 把服务端日志和 benchmark 产物收到 OUTPUT_DIR，供核对 [HeterCache profile]。
#
# 前置：~/venvs/sim 环境（见 Heter-Bench/docs/sglang-simulator-hetercache.md
#       的「环境」一节）。这里不替你装依赖。
#
# 用法：
#   bash Heter-Bench/scripts/run_sglang_simulator_heter.sh \
#       --trace /tmp/simgpt/trace.jsonl \
#       --out   /tmp/simgpt/run-sid \
#       [--port 30082] [--max-total-tokens 8192]
#
#   # 想加/减 sglang 参数，用 -- 之后透传：
#   bash ... -- --chunked-prefill-size -1
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_PY="${SIM_PY:-$HOME/venvs/sim/bin/python}"
MODEL="${MODEL:-$REPO/tools/sglang-simulator/test/assets/qwen3-8b}"
# tokenizer 必须与 trace 的 token 化口径一致：HeterCache 靠 token 相等判前缀命中。
TOKENIZER="${TOKENIZER:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ 2>/dev/null | head -1)}"
SIM_CFG="${SIM_CFG:-$REPO/tools/sglang-simulator/examples/sim_configs/ml.json}"

TRACE=""
OUT=""
PORT="${PORT:-30082}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace) TRACE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$TRACE" ]] || { echo "--trace 不存在: $TRACE" >&2; exit 2; }
[[ -x "$SIM_PY" ]] || { echo "找不到 venv python: $SIM_PY（改 SIM_PY=...）" >&2; exit 2; }
[[ -n "$TOKENIZER" ]] || { echo "找不到 tokenizer（改 TOKENIZER=...）" >&2; exit 2; }
OUT="${OUT:-/tmp/sglang-simulator-heter-$(date +%s)}"
mkdir -p "$OUT"

export PYTHONPATH="$REPO/tools/sglang-simulator/cpu_shim:$REPO/python"
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR="$OUT"

SERVER_LOG="$OUT/server.log"
: > "$SERVER_LOG"

"$SIM_PY" -m sglang_simulator.simulation.sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$TOKENIZER" \
  --sim-config-path "$SIM_CFG" \
  --port "$PORT" \
  --max-total-tokens "$MAX_TOKENS" \
  --max-running-requests 8 \
  --enable-heter-cache \
  --radix-eviction-policy heter_cache \
  --disable-overlap-schedule \
  "${EXTRA_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  sleep 1
  pkill -f "[s]glang_simulator.simulation.sglang.launch_server" 2>/dev/null || true
}
trap cleanup EXIT

echo "[run] 起服务中（日志 $SERVER_LOG）..."
READY=0
for _ in $(seq 1 240); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/model_info" 2>/dev/null)" == "200" ]]; then
    READY=1
    break
  fi
  sleep 1
done
if [[ "$READY" != "1" ]]; then
  echo "[run] 服务 240s 未就绪，看 $SERVER_LOG 尾部：" >&2
  tail -30 "$SERVER_LOG" >&2
  exit 1
fi

# 确认缓存实现是 HeterCacheRadixCache。漏了 --enable-heter-cache 会静默退到
# UnifiedRadixCache，而 C_RadixCacheClockHook 不覆盖它 —— 秒级判据全失效。
IMPL_LINE="$(grep -m1 'Tree cache initialized' "$SERVER_LOG" || true)"
echo "[run] $IMPL_LINE"
if [[ "$IMPL_LINE" != *"impl=HeterCacheRadixCache"* ]]; then
  echo "[run] 警告：缓存实现不是 HeterCacheRadixCache，本次结果不能当 HeterCache 用" >&2
fi

# 注意：不要传 --max-concurrency。OFFLINE 模式要把整条 trace 一次性交给
# scheduler，由模拟器按逻辑时钟排程；压并发会让它等不齐请求而卡死。
"$SIM_PY" "$REPO/benchmark/simulator/bench_serving.py" \
  --simulator-mode offline \
  --backend sglang \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --dataset-name autobench \
  --dataset-path "$TRACE" \
  --use-trace-timestamps \
  --num-prompts "${NUM_PROMPTS:-1000000}" \
  --ready-check-timeout-sec 120 \
  --output-file "$OUT/benchmark.json" 2>&1 | tee "$OUT/bench.log" | tail -25

echo
echo "[run] ===== [HeterCache profile]（$OUT/server.log）====="
grep -A8 'HeterCache profile' "$OUT/server.log" | tail -30 || echo "（没有打印：模拟时间跨度未超过 PROFILE_INTERVAL=30s）"
echo
echo "[run] 产物目录: $OUT"
