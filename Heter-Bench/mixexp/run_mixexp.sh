#!/usr/bin/env bash
# 混合负载实验：规模阶梯 × 实验臂。
#
# 前置：bash mixexp/prepare_data.sh 生成 *.nosys.jsonl（去掉 Qwen 公共 system 段）
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
ARMS_FILE="${ARMS_FILE:-$HERE/arms.tsv}"          # 方案清单，可换成本实验专用的一份
CONFIGS_DIR="${CONFIGS_DIR:-$HERE/configs}"       # 配置目录，可换成本实验专用的一组
MEM_FRACTION="${MEM_FRACTION:-}"                  # sglang --mem-fraction-static，空=不给
ACTUAL_TOKENS=""                                  # server 实际分配到的池容量（可能被显存钳制）

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
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
    --arms-file)    ARMS_FILE="$2"; shift 2 ;;
    --configs-dir)  CONFIGS_DIR="$2"; shift 2 ;;
    --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
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
  CFG="$CONFIGS_DIR/drift_phases.yaml"
  CFG_FLAG="--phases-config"
else
  CFG="$CONFIGS_DIR/mix_${SCALE}.yaml"
  CFG_FLAG="--tasks-config"
fi
[ -f "$CFG" ] || { echo "缺少配置：$CFG" >&2; exit 1; }
[ -f "$ARMS_FILE" ] || { echo "缺少方案清单：$ARMS_FILE" >&2; exit 1; }

if [ "$ARMS" = "all" ]; then
  SELECTED=$(awk -F'\t' '!/^#/ && NF>=2 {print $1}' "$ARMS_FILE")
else
  SELECTED=$(printf '%s' "$ARMS" | tr ',' '\n')
fi

mkdir -p "$OUTDIR"
cd "$BENCH_DIR/.."

# 数据预检：配置里的 data_path 必须都在。缺了立刻停——
# 否则要等 server 起完、bench 跑到第一个任务才报错。
check_data_paths() {
  # 只读 yaml，不需要 sglang 依赖；$PY 不在时退回本机 python3，
  # 免得把「解释器缺失」误报成「数据缺失」。
  local check_py="$PY"
  command -v "$check_py" >/dev/null 2>&1 || check_py=python3
  "$check_py" - "$CFG" "$BENCH_DIR" <<'PYCHECK'
import sys
from pathlib import Path
import yaml

cfg, root = Path(sys.argv[1]), Path(sys.argv[2])
data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
missing = [
    task["data_path"]
    for task in data.get("tasks", [])
    if not (root / task["data_path"]).exists()
]
if missing:
    print("缺少数据文件：")
    for path in missing:
        print("  " + path)
    sys.exit(1)
PYCHECK
}

if [ "$DRY_RUN" != "1" ] && ! check_data_paths; then
  echo
  echo "先准备数据：bash $HERE/prepare_data.sh" >&2
  exit 2
fi

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
  LINE=$(awk -F'\t' -v a="$arm" '!/^#/ && $1==a {print; exit}' "$ARMS_FILE")
  [ -n "$LINE" ] || { echo "!! 未定义的臂：$arm" >&2; FAILED="$FAILED $arm"; continue; }

  POLICY=$(printf '%s' "$LINE" | cut -f2)
  PCFG=$(printf '%s' "$LINE" | cut -f3)
  DESC=$(printf '%s' "$LINE" | cut -f4)
  [ "$PCFG" = "-" ] && PCFG=""

  SERVER_ARGS=(--model-path "$MODEL" --radix-eviction-policy "$POLICY"
               --enable-metrics --max-total-tokens "$TOKENS" --port "$PORT")
  [ -n "$MEM_FRACTION" ] && SERVER_ARGS+=(--mem-fraction-static "$MEM_FRACTION")
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

    # 实际池容量可能被显存钳制，与 --max-total-tokens 不一致，必须记下来
    ACTUAL_TOKENS=$(grep -oE 'max_total_num_tokens=[0-9]+' "$SLOG" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "$ACTUAL_TOKENS" ] && [ "$ACTUAL_TOKENS" != "$TOKENS" ]; then
      echo "!! 池容量被钳制：请求 $TOKENS token，实际 $ACTUAL_TOKENS token（显存限制）" >&2
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
"$PY" "$HERE/analyze.py" --dir "$OUTDIR" --scale "$SCALE" --tokens "${ACTUAL_TOKENS:-$TOKENS}" || true
