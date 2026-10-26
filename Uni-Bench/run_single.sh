#!/usr/bin/env bash
# =============================================================================
# 单一负载 benchmark 集成脚本
# 把 5 个单负载数据集的压测集成到一个脚本，按「策略」输出到独立目录。
#
# 用法（server 需已用同一 policy 启动，见 serve.sh）：
#   bash run_single.sh lru             # → result/LRU/
#   bash run_single.sh simple_hybrid   # → result/UniCache/
#   POLICY=lfu bash run_single.sh      # → result/LFU/
#
# 可调环境变量：
#   NUM_SESSIONS=2000   每个负载的 session 数
#   CONC=32             并发会话数（mashqa 单独用 MASHQA_CONC=8，减少兄弟问题 racing）
#   WAIT_SCALE=1        多轮负载轮间等待缩放（1=真实 lognormal；单轮负载此参数无效）
#   PORT=30000          server 端口
# =============================================================================
set -eu

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=env.sh
source "$BENCH_DIR/env.sh"
# shellcheck source=config.sh
source "$BENCH_DIR/config.sh"

cd "$BENCH_DIR"

POLICY="${1:-${POLICY:-lru}}"
PORT=${PORT:-30000}
BASE_URL="http://127.0.0.1:${PORT}"
NUM_SESSIONS=${NUM_SESSIONS:-2000}
CONC=${CONC:-32}
MASHQA_CONC=${MASHQA_CONC:-8}
WAIT_SCALE=${WAIT_SCALE:-1}

# policy → 输出目录名（LRU / UniCache / ...）
case "$POLICY" in
  lru)           DIR="LRU" ;;
  simple_hybrid) DIR="UniCache" ;;
  lfu)           DIR="LFU" ;;
  slru)          DIR="SLRU" ;;
  *)             DIR="$(echo "$POLICY" | tr 'a-z' 'A-Z')" ;;
esac
OUTDIR="$BENCH_DIR/result/$DIR"
mkdir -p "$OUTDIR"

# 健康检查：server 未就绪则退出（提示先起 server）
if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
  echo "!! server 未就绪（$BASE_URL/health），请先起 server："
  echo "   POLICY=$POLICY MAX_TOTAL_TOKENS=100000 bash $BENCH_DIR/serve.sh"
  exit 1
fi

echo "=================================================="
echo ">>> 单一负载 benchmark | policy=$POLICY → $OUTDIR"
echo ">>> num_sessions=$NUM_SESSIONS concurrency=$CONC wait_scale=$WAIT_SCALE"
echo "=================================================="

# run <name> <data_path> [bench_unicache.py 额外参数...]
run() {
  local name="$1" data="$2"; shift 2
  echo "--------------------------------------------------"
  echo ">>> [$DIR] $name"
  "$PY" "$BENCH_DIR/bench_unicache.py" \
    --data-path "$data" \
    --base-url "$BASE_URL" \
    --output "$OUTDIR/bench_${name}.json" \
    "$@"
}

# ---- 多轮负载（--wait-scale 生效）----
run sharegpt "TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl" \
    --concurrency "$CONC" --num-sessions "$NUM_SESSIONS" --wait-scale "$WAIT_SCALE"

run alfred "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl" \
    --concurrency "$CONC" --num-sessions "$NUM_SESSIONS" --wait-scale "$WAIT_SCALE"

# ---- 单轮负载（--wait-scale 无效，省略）----
run codeparrot "TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl" \
    --concurrency "$CONC" --num-sessions "$NUM_SESSIONS"

run apps "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl" \
    --concurrency "$CONC" --num-sessions "$NUM_SESSIONS"

# mashqa：必须 --no-shuffle（同文档问题相邻），并降低并发减少兄弟问题并发竞争
run mashqa "TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl" \
    --concurrency "$MASHQA_CONC" --num-sessions "$NUM_SESSIONS" --no-shuffle

# ---- 汇总 ----
echo ""
"$PY" - "$OUTDIR" "$DIR" << 'PYEOF'
import glob, json, os, sys
outdir, dname = sys.argv[1], sys.argv[2]
print(f"==================== {dname} 汇总 ====================")
print(f"{'file':26s} {'hit%':>8s} {'reqs':>8s} {'err':>7s} {'wall_s':>8s}")
for p in sorted(glob.glob(os.path.join(outdir, "bench_*.json"))):
    d = json.load(open(p, encoding="utf-8"))
    print(f"{os.path.basename(p):26s} {str(d.get('cache_hit_rate_pct')):>8s} "
          f"{str(d.get('num_requests')):>8s} {str(d.get('num_errors')):>7s} "
          f"{str(d.get('wall_time_s')):>8s}")
print("====================================================")
PYEOF
