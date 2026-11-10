#!/usr/bin/env bash
# exp2：池容量 mem-fraction 0.85 + ShareGPT 全部 >5 轮 + caps.agentic 0.6
#
# 用法：
#   bash mixexp/run_exp2.sh --dry-run                 # 先看命令
#   bash mixexp/run_exp2.sh --arms lru,hc_full_v2     # 建议先跑两个方案（约 6 h）
#   bash mixexp/run_exp2.sh                           # 三个方案全跑（约 9 h）
#
# 输出：mixexp/result/exp2/  （与 exp1 归档互不覆盖）
#
# 显存比例说明（本机 RTX 4090 24 GB 实测）：
#   0.704 → 池 35 975 token（exp1 实际值）
#   0.85  → 池 99 036 token（可用，本脚本默认）
#   0.90  → 池 120 633 token，但 CUDA graph 捕获阶段 OOM，服务起不来
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

exec bash "$HERE/run_mixexp.sh" \
  --scale full \
  --arms "${ARMS:-lru,hc_full,hc_full_v2}" \
  --arms-file "$HERE/arms_exp2.tsv" \
  --configs-dir "$HERE/configs/exp2" \
  --mem-fraction "${MEM_FRACTION:-0.85}" \
  --out "${OUTDIR:-$HERE/result/exp2}" \
  "$@"
