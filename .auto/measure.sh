#!/bin/bash
# D-SLRU 对照测量：A0(LRU) / A1(上游 SLRU) / A3(两带)，两个绝对池容量。
# 需要实验机（离线回放依赖真实 sglang + torch）。
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 语法快检（<1s，出错立即失败）
python -m py_compile \
  python/sglang/srt/mem_cache/victim_override.py \
  python/sglang/srt/mem_cache/evict_policy.py \
  python/sglang/srt/mem_cache/uni_utils.py

# 2) 远端对照
# shellcheck disable=SC1091
set -a; source /home/rain/.slru_remote_env; set +a
exec python .auto/remote_eval.py
