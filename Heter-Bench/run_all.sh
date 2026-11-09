#!/usr/bin/env bash
# Uni-Bench 实验矩阵：policy × max_total_tokens，自动启停 server 并汇总命中率。
#
# 用法：
#   bash run_all.sh
#   POLICIES="lru simple_hybrid" TOKENS_LIST="50000 200000" bash run_all.sh
#
# 输出：Uni-Bench/result/bench_<policy>_<tokens>.json + 终端汇总表
set -u

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=config.sh
# shellcheck source=env.sh
source "$BENCH_DIR/env.sh"
# shellcheck source=config.sh
source "$BENCH_DIR/config.sh"

OUTDIR=${OUTDIR:-$BENCH_DIR/result}
mkdir -p "$OUTDIR"

# python -m sglang.launch_server 需要从仓库根运行
cd "$BENCH_DIR/.."
REPO_ROOT="$(pwd)"

wait_health() {
  local url="http://127.0.0.1:${PORT}/health"
  for _ in $(seq 1 300); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_one() {
  local policy="$1" tokens="$2"
  local extra=""
  [ "$policy" = "simple_hybrid" ] && extra="--enable-simple-hybrid"

  echo "=================================================="
  echo ">>> policy=$policy  max_total_tokens=$tokens"
  echo "=================================================="

  local server_log="$OUTDIR/server_${policy}_${tokens}.log"
  "$PY" -m sglang.launch_server \
    --model-path "$MODEL" \
    --radix-eviction-policy "$policy" \
    --enable-metrics \
    --max-total-tokens "$tokens" \
    --port "$PORT" $extra > "$server_log" 2>&1 &
  local pid=$!

  if ! wait_health; then
    echo "!! server 启动失败（或超时），见 $server_log"
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 1
  fi

  "$PY" "$BENCH_DIR/bench_unicache.py" \
    --data-path "$DATA" \
    --base-url "http://127.0.0.1:${PORT}" \
    --concurrency "$CONC" \
    --wait-scale "$WAIT_SCALE" \
    --output "$OUTDIR/bench_${policy}_${tokens}.json"

  # 停 server
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 5
}

for tokens in $TOKENS_LIST; do
  for policy in $POLICIES; do
    run_one "$policy" "$tokens" || echo "!! 跳过 policy=$policy tokens=$tokens"
  done
done

# ---- 汇总 ----
"$PY" - "$OUTDIR" << 'PYEOF'
import glob, json, os, sys

outdir = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(outdir, "bench_*.json"))):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    rows.append((
        os.path.basename(p),
        d.get("cache_hit_rate_pct"),
        d.get("ttft_p50_ms"),
        d.get("ttft_p99_ms"),
        d.get("num_requests"),
        d.get("num_errors"),
    ))

print("\n=================== Uni-Bench 汇总 ===================")
print(f"{'file':38s} {'hit%':>7s} {'ttft_p50':>9s} {'ttft_p99':>9s} {'reqs':>7s} {'err':>5s}")
for r in rows:
    print(f"{r[0]:38s} {str(r[1]):>7s} {str(r[2]):>9s} {str(r[3]):>9s} {str(r[4]):>7s} {str(r[5]):>5s}")
print("=====================================================")
print("提示：对比 simple_hybrid 与基线的 hit% 与 ttft；cache 越小差异应越明显。")
PYEOF
