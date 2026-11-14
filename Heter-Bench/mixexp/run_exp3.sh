#!/usr/bin/env bash
# exp3：四类「单类 token 主导」的 cell（scan / struct / chat / agentic）
#
# 每个 cell 让一类负载占 ≥60% 的 prompt token，其余三类均分剩下的部分。
# 主导按 token 计，与会话数无关（401 条单轮冲刷会话的 token 可能比 96 条 agentic
# 会话还少）。四套数据与四份 tasks.yaml 都由 build_exp3_data.py 反解生成。
#
# 默认参数：
#   池容量   --mem-fraction-static 0.85 + --kv-cache-dtype fp8_e4m3（约 198k token）
#   并发     32          轮间不压缩（--wait-scale 1.0）
#   方案     lru,hc_full（hc_norr 备用）
#   输出     mixexp/result/exp3/<cell>/
#
# 用法：
#   bash mixexp/run_exp3.sh --build-data            # 准备数据 + 构造四个 cell，再跑
#   bash mixexp/run_exp3.sh --dry-run               # 只打印命令
#   bash mixexp/run_exp3.sh --mix agentic           # 只跑 agentic 主导 cell
#   bash mixexp/run_exp3.sh --mix all --arms lru,hc_full
#   bash mixexp/run_exp3.sh --mix legacy            # 旧的单组成（22/35/22/22）
#   bash mixexp/run_exp3.sh --mode drift            # 相位模式（仅 legacy 组成）
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
MIX="all"
DOMINANT_SHARE="${DOMINANT_SHARE:-0.60}"
TOTAL_TOKENS="${TOTAL_TOKENS:-25e6}"

BUILD=0
DRY=0
PASSTHRU=()
MODE="hybrid"
# --mix / --mode 需要吃掉一个值；--build-data / --dry-run 由本脚本处理，其余透传
while [ $# -gt 0 ]; do
  case "$1" in
    --build-data) BUILD=1; shift ;;
    --dry-run)    DRY=1; PASSTHRU+=("$1"); shift ;;
    --mix)        MIX="$2"; shift 2 ;;
    --mode)       MODE="$2"; shift 2 ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

# 展开 cell 列表：all → 四个；legacy → 旧的单组成一份
if [ "$MIX" = "all" ]; then
  CELLS="scan struct chat agentic"
elif [ "$MIX" = "legacy" ]; then
  CELLS="legacy"
else
  CELLS="$(printf '%s' "$MIX" | tr ',' ' ')"
fi
for cell in $CELLS; do
  case "$cell" in
    scan|struct|chat|agentic|legacy) ;;
    *) echo "未知 cell：$cell（可选 scan|struct|chat|agentic|all|legacy）" >&2; exit 2 ;;
  esac
done

if [ "$MODE" = "drift" ] && [ "$CELLS" != "legacy" ]; then
  echo "--mode drift 只支持 --mix legacy：相位文件是按旧单组成写的" >&2
  exit 2
fi

PY_BUILD="${PY:-python3}"
BUILDER="$HERE/tools/build_exp3_data.py"

if [ "$BUILD" = "1" ]; then
  # 先走 prepare_data.sh：剥 Qwen system 段，并剔掉多轮源里轮数 < MULTI_TURN_MIN
  # 的会话（否则这些会话终身判 TRANS，会以单轮流量身份挤占 one-shot 带）。幂等。
  echo ">>> 准备数据（剥 Qwen system + 剔除多轮源的短会话）"
  if [ "$DRY" = "1" ]; then
    echo "    bash $HERE/prepare_data.sh（幂等，已存在则跳过）"
  else
    bash "$HERE/prepare_data.sh" || exit 1
  fi

  echo ">>> 构造 exp3 数据（主导份额 ≥${DOMINANT_SHARE}，预算 ${TOTAL_TOKENS} token）"
  for cell in $CELLS; do
    if [ "$cell" = "legacy" ]; then
      cmd=("$PY_BUILD" "$BUILDER")
    else
      cmd=("$PY_BUILD" "$BUILDER" --mix "$cell"
           --dominant-share "$DOMINANT_SHARE" --total-tokens "$TOTAL_TOKENS")
    fi
    if [ "$DRY" = "1" ]; then
      printf '    %q' "${cmd[@]}"; echo
    else
      "${cmd[@]}" || exit 1
    fi
  done
fi

fail=0
for cell in $CELLS; do
  if [ "$cell" = "legacy" ]; then
    CFG_ARGS=(--configs-dir "$HERE/configs/exp3" --scale full)
    CELL_OUT="$OUTDIR"
  else
    TASKS="$DATA_DIR/$cell/tasks.yaml"
    if [ ! -f "$TASKS" ]; then
      if [ "$DRY" = "1" ]; then
        echo "（dry-run）$TASKS 尚未生成；真实运行时将执行："
        echo "    bash mixexp/run_exp3.sh --build-data --mix $cell"
        echo "    bash mixexp/run_mixexp.sh --config TaskData/exp3/$cell/tasks.yaml \\"
        echo "        --scale full --arms $ARMS --arms-file mixexp/arms_exp3.tsv --out $OUTDIR/$cell"
        continue
      fi
      echo "!! 缺少 $TASKS，先执行：bash mixexp/run_exp3.sh --build-data --mix $cell" >&2
      fail=1
      continue
    fi
    # 生成的 tasks.yaml 里已含 data_path / max_record / lambda_rate
    CFG_ARGS=(--config "$TASKS" --scale full)
    CELL_OUT="$OUTDIR/$cell"
  fi

  echo "=================================================="
  echo ">>> exp3 cell=$cell  out=$CELL_OUT"
  echo "=================================================="
  bash "$HERE/run_mixexp.sh" \
    "${CFG_ARGS[@]}" \
    --mode "$MODE" \
    --arms "$ARMS" \
    --arms-file "$HERE/arms_exp3.tsv" \
    --mem-fraction "$MEM_FRACTION" \
    --conc "$CONC" \
    --server-arg "--kv-cache-dtype" \
    --server-arg "$KV_DTYPE" \
    --out "$CELL_OUT" \
    --skip-existing \
    "${PASSTHRU[@]+"${PASSTHRU[@]}"}" || fail=1
done

exit "$fail"
