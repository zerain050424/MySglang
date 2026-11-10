#!/usr/bin/env bash
# exp3：四类混合负载（含公共前缀、长尾会话、周期性突发）
#
# 默认参数：
#   池容量   --mem-fraction-static 0.85 + --kv-cache-dtype fp8_e4m3（约 198k token）
#   并发     32          轮间不压缩（--wait-scale 1.0）
#   方案     lru,hc_full（hc_norr 备用）
#   输出     mixexp/result/exp3/
#
# 用法：
#   bash mixexp/run_exp3.sh --build-data            # 先构造四类数据，再跑
#   bash mixexp/run_exp3.sh --dry-run               # 只打印命令
#   bash mixexp/run_exp3.sh --arms lru,hc_full      # 显式指定方案
#   bash mixexp/run_exp3.sh --mode drift            # 相位模式（含 20 秒突发窗口）
#   ARMS=lru,hc_full,hc_norr bash mixexp/run_exp3.sh
#
# 覆盖项：CONC / MEM_FRACTION / KV_DTYPE / OUTDIR / ARMS 环境变量，或直接透传
# run_mixexp.sh 的参数（例如 --conc 48）。
#
# 注意：显存比例 0.9 在本机（RTX 4090 24 GB）会在 CUDA graph 捕获阶段 OOM，
#       0.85 是可用上限；fp8 KV 让同一显存装下约两倍 token。
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$HERE/.." && pwd)"

ARMS="${ARMS:-lru,hc_full}"
CONC="${CONC:-32}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
OUTDIR="${OUTDIR:-$HERE/result/exp3}"
DATA_DIR="$BENCH_DIR/TaskData/exp3"

BUILD=0
DRY=0
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    --build-data) BUILD=1 ;;
    --dry-run) DRY=1; PASSTHRU+=("$arg") ;;
    *) PASSTHRU+=("$arg") ;;
  esac
done

if [ "$BUILD" = "1" ]; then
  PY_BUILD="${PY:-python3}"
  echo ">>> 构造 exp3 数据（TaskData/exp3）"
  if [ "$DRY" = "1" ]; then
    "$PY_BUILD" "$HERE/tools/build_exp3_data.py" --dry-run || exit 1
  else
    "$PY_BUILD" "$HERE/tools/build_exp3_data.py" || exit 1
  fi
elif [ ! -d "$DATA_DIR" ]; then
  echo "!! 缺少 $DATA_DIR，先执行：bash mixexp/run_exp3.sh --build-data" >&2
  exit 2
fi

exec bash "$HERE/run_mixexp.sh" \
  --scale full \
  --arms "$ARMS" \
  --arms-file "$HERE/arms_exp3.tsv" \
  --configs-dir "$HERE/configs/exp3" \
  --mem-fraction "$MEM_FRACTION" \
  --conc "$CONC" \
  --server-arg "--kv-cache-dtype" \
  --server-arg "$KV_DTYPE" \
  --out "$OUTDIR" \
  --skip-existing \
  "${PASSTHRU[@]+"${PASSTHRU[@]}"}"
