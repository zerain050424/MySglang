#!/usr/bin/env bash
set -euo pipefail

# 远端路径：实验机上的 checkout 仍叫 /root/sg-unicache/Uni-Bench，
# 本地目录已改名 Heter-Bench。迁移时连同这里、config.sh、tailshot/ 下的同名路径一起改。
HETER_BENCH_DIR="/root/sg-unicache/Uni-Bench"
SRV_SESSION="hetercache_srv"
BENCH_SESSION="hetercache_bench"
SERVER_URL="http://127.0.0.1:30000"

cd "${HETER_BENCH_DIR}"

# ── 0) 清场 ────────────────────────────────────────────────
rm -f result/node_labels_phased.jsonl result/phased_load_prediction_C.*

# 清理可能残留的同名 tmux 会话
tmux has-session -t "${SRV_SESSION}" 2>/dev/null && tmux kill-session -t "${SRV_SESSION}" || true
tmux has-session -t "${BENCH_SESSION}" 2>/dev/null && tmux kill-session -t "${BENCH_SESSION}" || true

# ── 1) 起 server（tmux；约 40s 就绪）─────────────────────────
tmux new -d -s "${SRV_SESSION}" "cd ${HETER_BENCH_DIR} \
 && SGLANG_WORKLOAD_OCC_LOG=1 SGLANG_WORKLOAD_OCC_INTERVAL=30 \
    SGLANG_NODE_LABEL_DUMP=${HETER_BENCH_DIR}/result/node_labels_phased.jsonl \
    SGLANG_NODE_LABEL_INTERVAL=30 \
    POLICY=heter_cache MAX_TOTAL_TOKENS=100000 \
    bash serve.sh"

echo "Waiting for server at ${SERVER_URL} ..."
TIMEOUT=600   # 最多等 10 分钟
ELAPSED=0
until curl -sf "${SERVER_URL}/get_model_info" >/dev/null; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
    echo "ERROR: server did not become ready within ${TIMEOUT}s" >&2
    exit 1
  fi
done
echo SERVER_READY

# ── 2) 起 bench（另一 tmux；~15–18 min）─────────────────────
tmux new -d -s "${BENCH_SESSION}" "cd ${HETER_BENCH_DIR} \
 && source env.sh \
 && \"\$PY\" bench_hetercache.py --phases-config phased_load_prediction.yaml \
      --base-url ${SERVER_URL} --concurrency 32 \
      --arrival-scale 1 --wait-scale 1 \
      --output result/phased_load_prediction_C.json \
      > result/phased_run_C.log 2>&1"

echo "Bench started in tmux session '${BENCH_SESSION}'."
echo "Monitor with: tmux attach -t ${BENCH_SESSION}"
echo "Server session: ${SRV_SESSION}"