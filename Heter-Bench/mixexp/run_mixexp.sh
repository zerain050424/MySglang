#!/usr/bin/env bash
# 混合负载实验：规模阶梯 × 实验臂。
#
# 用法：
#   bash mixexp/run_mixexp.sh --scale smoke                 # 冒烟：全臂跑通
#   bash mixexp/run_mixexp.sh --scale quick --arms lru,hc_full
#   bash mixexp/run_mixexp.sh --scale full                  # 全量（耗时长）
#   bash mixexp/run_mixexp.sh --scale mid --mode drift      # 组成漂移
#   bash mixexp/run_mixexp.sh --scale smoke --dry-run       # 只打印命令
#
# 每个臂独占一次 server：起 server → 等 /health → 跑 bench → 停 server。
# 输出：result/mixexp/<scale>_<arm>.json + server_<scale>_<arm>.log
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$HERE/.." && pwd)"
# shellcheck source=../scripts/env.sh
source "$BENCH_DIR/scripts/env.sh"
# shellcheck source=../scripts/config.sh
source "$BENCH_DIR/scripts/config.sh"

SCALE=smoke
MODE=hybrid            # hybrid | drift
ARMS=all
REPS=1
DRY_RUN=0
TOKENS_OVERRIDE=""
CONC_OVERRIDE=""
WAIT_OVERRIDE=""
OUTDIR="${OUTDIR:-$BENCH_DIR/result/mixexp}"
PORT="${PORT:-30000}"

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --scale)  SCALE="$2"; shift 2 ;;
    --mode)   MODE="$2"; shift 2 ;;
    --arms)   ARMS="$2"; shift 2 ;;
    --reps)   REPS="$2"; shift 2 ;;
    --tokens) TOKENS_OVERRIDE="$2"; shift 2 ;;
    --conc)   CONC_OVERRIDE="$2"; shift 2 ;;
    --wait-scale) WAIT_OVERRIDE="$2"; shift 2 ;;
    --out)    OUTDIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

# 规模阶梯：数据量由 configs/mix_<scale>.yaml 的 max_record 决定，
# 这里只给配套的池容量/并发/轮间压缩。三者必须同向放大，否则
# 小规模时工作集压不满池子，各臂会打平（实验无效）。
case "$SCALE" in
  smoke) DEF_TOKENS=30000;  DEF_CONC=8;  DEF_WAIT=0.1 ;;
  quick) DEF_TOKENS=60000;  DEF_CONC=16; DEF_WAIT=0.3 ;;
  mid)   DEF_TOKENS=120000; DEF_CONC=24; DEF_WAIT=1.0 ;;
  full)  DEF_TOKENS=200000; DEF_CONC=32; DEF_WAIT=1.0 ;;
  *) echo "未知规模：$SCALE（可选 smoke|quick|mid|full）" >&2; exit 2 ;;
esac

TOKENS=${TOKENS_OVERRIDE:-$DEF_TOKENS}
CONC=${CONC_OVERRIDE:-$DEF_CONC}
WAIT_SCALE=${WAIT_OVERRIDE:-$DEF_WAIT}

if [ "$MODE" = "drift" ]; then
  CFG="$HERE/configs/drift_phases.yaml"
  CFG_FLAG="--phases-config"
else
  CFG="$HERE/configs/mix_${SCALE}.yaml"
  CFG_FLAG="--tasks-config"
fi
[ -f "$CFG" ] || { echo "缺少配置：$CFG" >&2; exit 1; }
[ -f "$HERE/arms.tsv" ] || { echo "缺少 $HERE/arms.tsv" >&2; exit 1; }

if [ "$ARMS" = "all" ]; then
  SELECTED=$(awk -F'\t' '!/^#/ && NF>=2 {print $1}' "$HERE/arms.tsv")
else
  SELECTED=$(printf '%s' "$ARMS" | tr ',' '\n')
fi

mkdir -p "$OUTDIR"
cd "$BENCH_DIR/.."

wait_health() {
  local url="http://127.0.0.1:${PORT}/health"
  for _ in $(seq 1 300); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

echo "=================================================="
echo " mixexp  scale=$SCALE  mode=$MODE  tokens=$TOKENS"
echo "         conc=$CONC  wait_scale=$WAIT_SCALE  reps=$REPS"
echo "         config=$CFG"
echo "         arms=$(echo $SELECTED | tr '\n' ' ')"
echo "=================================================="

FAILED=""
for arm in $SELECTED; do
  LINE=$(awk -F'\t' -v a="$arm" '!/^#/ && $1==a {print; exit}' "$HERE/arms.tsv")
  [ -n "$LINE" ] || { echo "!! 未定义的臂：$arm" >&2; FAILED="$FAILED $arm"; continue; }

  POLICY=$(printf '%s' "$LINE" | cut -f2)
  PCFG=$(printf '%s' "$LINE" | cut -f3)
  DESC=$(printf '%s' "$LINE" | cut -f4)
  [ "$PCFG" = "-" ] && PCFG=""

  SERVER_ARGS=(--model-path "$MODEL" --radix-eviction-policy "$POLICY"
               --enable-metrics --max-total-tokens "$TOKENS" --port "$PORT")
  [ "$POLICY" = "heter_cache" ] && SERVER_ARGS+=(--enable-heter-cache)
  [ -n "$PCFG" ] && SERVER_ARGS+=(--radix-eviction-policy-config "$PCFG")

  for rep in $(seq 1 "$REPS"); do
    SUFFIX=""
    [ "$REPS" -gt 1 ] && SUFFIX="_r${rep}"
    OUT="$OUTDIR/${SCALE}_${arm}${SUFFIX}.json"
    SLOG="$OUTDIR/server_${SCALE}_${arm}${SUFFIX}.log"

    echo
    echo "--------------------------------------------------"
    echo ">>> $SCALE / $arm  ($POLICY)  rep=$rep"
    echo "    $DESC"
    echo "--------------------------------------------------"

    if [ "$DRY_RUN" = "1" ]; then
      printf '  "$PY" -m sglang.launch_server'
      printf ' %q' "${SERVER_ARGS[@]}"
      printf '\n'
      echo "  \"\$PY\" \"$BENCH_DIR/scripts/bench_hetercache.py\" \\"
      echo "        $CFG_FLAG \"$CFG\" --base-url http://127.0.0.1:${PORT} \\"
      echo "        --concurrency $CONC --arrival-scale 1 --wait-scale $WAIT_SCALE \\"
      echo "        --output \"$OUT\""
      continue
    fi

    "$PY" -m sglang.launch_server "${SERVER_ARGS[@]}" > "$SLOG" 2>&1 &
    SRV_PID=$!

    if ! wait_health; then
      echo "!! server 起不来（或超时），见 $SLOG" >&2
      kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null
      FAILED="$FAILED ${arm}${SUFFIX}"
      continue
    fi

    "$PY" "$BENCH_DIR/scripts/bench_hetercache.py" \
      "$CFG_FLAG" "$CFG" \
      --base-url "http://127.0.0.1:${PORT}" \
      --concurrency "$CONC" \
      --arrival-scale 1 \
      --wait-scale "$WAIT_SCALE" \
      --output "$OUT" \
      || { echo "!! bench 失败：$arm" >&2; FAILED="$FAILED ${arm}${SUFFIX}"; }

    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    sleep 5
  done
done

[ -n "$FAILED" ] && echo "失败的臂：$FAILED" >&2

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "（dry-run：未执行。去掉 --dry-run 开始跑。）"
  exit 0
fi

echo
"$PY" "$HERE/analyze.py" --dir "$OUTDIR" --scale "$SCALE" --tokens "$TOKENS" || true
