#!/usr/bin/env bash
# 真机（L2）：逐臂起服务 → 跑 Uni-Bench 客户端 → 收 TTFT / 命中率 / struct 埋点。
#
#   ARMS="A0 A2 A3" MODES="hybrid sharegpt" TOKENS_LIST="100000" SEEDS="42" \
#     bash scripts/run_online.sh
#
# 关键点
#   * --enable-simple-hybrid 选 backend，--radix-eviction-policy 选策略，两者缺一不可；
#   * 每臂独立的 SGLANG_STRUCT_PROFILE，便于 analyze 分组；
#   * 结果命名 bench_<arm>_<mode>_<tokens>k_s<seed>.json，analyze 从文件名解析元数据。
#
# 环境变量：MODEL / DATA / PORT / CONC / WAIT_SCALE（来自 ../config.sh，可覆盖）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # W-TinyLRU
cd "$HERE"

if [ -f ../env.sh ]; then
  # shellcheck disable=SC1091
  source ../env.sh
fi
if [ -f ../config.sh ]; then
  # shellcheck disable=SC1091
  set +u; source ../config.sh; set -u
fi
PY="${WTL_PY:-${PY:-python}}"
PORT="${PORT:-30000}"
MODEL="${MODEL:?需要 MODEL（或在 ../config.sh 里配置）}"
CONC="${CONC:-32}"
WAIT_SCALE="${WAIT_SCALE:-1.0}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
SEEDS="${SEEDS:-42}"
TOKENS_LIST="${TOKENS_LIST:-100000}"
ARMS="${ARMS:-A0 A2 A3}"
MODES="${MODES:-hybrid sharegpt}"
CLIENT="${CLIENT:-$HERE/../bench_unicache.py}"
OUTDIR="${OUTDIR:-$HERE/result/online}"
mkdir -p "$OUTDIR"

export PYTHONPATH="$HERE:${PYTHONPATH:-}"

wait_health() {
  local url="http://127.0.0.1:${PORT}/health"
  for _ in $(seq 1 300); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# 实验臂 -> 策略与配置（与 wtl/arms.py 保持一致，改参数时两边同时改）
arm_policy() {
  case "$1" in
    A0) echo "lru|" ;;
    A1) echo 'slru|{"protected_threshold":2}' ;;
    A2) echo 'simple_hybrid|{"slru":{"share_min":1000000}}' ;;
    A3) echo 'simple_hybrid|{"slru":{"share_min":2}}' ;;
    A4) echo 'simple_hybrid|{"slru":{"share_min":3}}' ;;
    *) echo "??|" ;;
  esac
}

bench_mode_args() {
  case "$1" in
    hybrid)   echo "--tasks-config $HERE/../hybrid_tasks_config.yaml" ;;
    sharegpt) echo "--data-path $HERE/../TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl" ;;
    alfred)   echo "--data-path $HERE/../TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl" ;;
    codeparrot) echo "--data-path $HERE/../TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl" ;;
    *) echo "" ;;
  esac
}

run_one() {
  local arm="$1" mode="$2" tokens="$3" seed="$4"
  local spec policy cfg
  spec="$(arm_policy "$arm")"
  policy="${spec%%|*}"
  cfg="${spec#*|}"
  [ "$policy" = "??" ] && { echo "!! 未知实验臂 $arm"; return 1; }

  local tag="${arm}_${mode}_${tokens}k_s${seed}"
  local slog="$OUTDIR/server_${tag}.log"
  local prof="$OUTDIR/struct_${tag}.jsonl"
  local out="$OUTDIR/bench_${tag}.json"
  [ -f "$out" ] && { echo "== skip $tag（已有结果）"; return 0; }

  local extra=()   # bash 4.3 兼容：展开时用 ${extra[@]+...}
  if [ "$policy" = "simple_hybrid" ]; then
    extra+=(--enable-simple-hybrid)
    extra+=(--radix-eviction-policy-config "$cfg")
  fi

  echo "=================================================="
  echo ">>> $tag  policy=$policy  pool=${tokens}k"
  echo "=================================================="
  SGLANG_STRUCT_PROFILE="$prof" \
  SGLANG_STRUCT_PROFILE_TAG="$tag" \
  SGLANG_STRUCT_PROFILE_EVERY="${PROFILE_EVERY:-30}" \
  "$PY" -m wtl.serve \
    --model-path "$MODEL" \
    --radix-eviction-policy "$policy" \
    --max-total-tokens "$tokens" \
    --enable-metrics \
    --port "$PORT" ${extra[@]+"${extra[@]}"} > "$slog" 2>&1 &
  local pid=$!

  if ! wait_health; then
    echo "!! 启动失败/超时，见 $slog"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return 1
  fi

  # shellcheck disable=SC2046
  "$PY" "$CLIENT" $(bench_mode_args "$mode") \
    --base-url "http://127.0.0.1:${PORT}" \
    --concurrency "$CONC" \
    --wait-scale "$WAIT_SCALE" \
    --seed "$seed" \
    --output "$out" || echo "!! bench 失败：$tag"

  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep "${COOLDOWN_S:-5}"
  echo "{\"tag\":\"$tag\",\"arm\":\"$arm\",\"policy\":\"$policy\",\"config\":$([ -n "$cfg" ] && echo "$cfg" || echo null),\"tokens\":$tokens,\"seed\":$seed,\"mode\":\"$mode\"}" >> "$OUTDIR/manifest.jsonl"
}

rc=0
for tokens in $TOKENS_LIST; do
  for mode in $MODES; do
    for seed in $SEEDS; do
      for arm in $ARMS; do
        run_one "$arm" "$mode" "$tokens" "$seed" || rc=1
      done
    done
  done
done

echo "[run_online] 结果目录：$OUTDIR"
echo "[run_online] 下一步：python -m wtl.analyze --offline result/offline --online $OUTDIR"
exit $rc
