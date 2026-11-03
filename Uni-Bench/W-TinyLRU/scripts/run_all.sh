#!/usr/bin/env bash
# 一键全流程：Phase-0 门控 → 离线矩阵 → 分析（在线部分需自己跑 run_online.sh）
#
#   bash scripts/run_all.sh              # 全量（耗时较长）
#   bash scripts/run_all.sh --quick      # 小规模冒烟：20 session × 2 臂 × 1 压力
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="${WTL_PY:-${PY:-python}}"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

QUICK=0
for arg in "$@"; do [ "$arg" = "--quick" ] && QUICK=1; done

if [ "$QUICK" = "1" ]; then
  echo "== [1/3] gate（quick）"
  "$PY" -m wtl.gate --backend "${WTL_BACKEND:-auto}" || exit 1
  echo "== [2/3] offline（quick）"
  "$PY" -m wtl.run_matrix --mode synth --arms A0,A3 --synth-depths 512 \
      --synth-breadths 4 --ratios 0.3 --seeds 42 --jobs 2 \
      --backend "${WTL_BACKEND:-auto}" --datadir result/data || exit 1
  echo "== [3/3] analyze"
  "$PY" -m wtl.analyze --offline result/offline --online result/online || exit 1
  echo "== done: result/analysis/summary.md"
  exit 0
fi

echo "== [0/4] prefix_stats（真实负载的分层前缀长度）"
"$PY" -m wtl.prefix_stats \
  --data-path ../TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl --task AgentBank_alfred \
  --data-path ../TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl --task Codeparrot \
  --data-path ../TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl --task ShareGPT \
  --num-sessions 500 --out result/prefix_stats.json || exit 1

echo "== [1/4] gate"
"$PY" -m wtl.gate --backend "${WTL_BACKEND:-auto}" || exit 1

echo "== [2/4] offline matrix（A0-A11 × 3 压力 × 3 负载 × 3 seed）"
"$PY" -m wtl.run_matrix \
  --arms all --workloads sharegpt,alfred,codeparrot \
  --ratios 0.3,0.6,1.0 --seeds 42,43,44 \
  --prefix-stats result/prefix_stats.json \
  --jobs "${JOBS:-12}" || exit 1

echo "== [2b/4] 合成前缀阶梯（场景 A）"
"$PY" -m wtl.run_matrix --mode synth --arms A0,A1,A2,A3 \
  --synth-depths 512,2048,8192 --synth-breadths 3,10 \
  --ratios 0.3,0.6 --backend "${WTL_BACKEND:-auto}" \
  --datadir result/data --jobs "${JOBS:-12}" || exit 1

echo "== [3/4] analyze（在线结果缺失时自动跳过对应判据）"
"$PY" -m wtl.analyze --offline result/offline --online result/online \
  --struct 'result/online/struct_*.jsonl' || exit 1

echo "== [4/4] 完成：result/analysis/summary.md"
echo "   在线 TTFT 请另跑：ARMS='A0 A2 A3' bash scripts/run_online.sh"
