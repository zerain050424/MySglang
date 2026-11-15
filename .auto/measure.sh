#!/bin/bash
# HeterCache vs LRU 配对测量：本地编排，远端（双 4090）执行。
# 用法（cell 参数原样透传给 remote_eval.py）：
#   ./.auto/measure.sh --mode smoke                       # 链路冒烟（无指标结论）
#   ./.auto/measure.sh --cells fast_mid                   # 见 remote_eval.py CELLS 注册表
# 凭据：source ~/.heter_remote_env（仓库外，不提交）。
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f "$HOME/.heter_remote_env" ] || { echo "缺少 ~/.heter_remote_env（HET_HOST/HET_PORT/HET_USER/HET_PW）" >&2; exit 2; }

# 1) 语法快检（<1s，出错立即失败）
python3 -m py_compile \
  python/sglang/srt/mem_cache/evict_policy.py \
  python/sglang/srt/mem_cache/heter_cache/override.py \
  python/sglang/srt/mem_cache/heter_cache/utils.py \
  python/sglang/srt/mem_cache/heter_cache/selector.py \
  python/sglang/srt/mem_cache/heter_cache/radix_cache.py \
  .auto/remote_eval.py

# 2) 远端配对测量
set -a; source "$HOME/.heter_remote_env"; set +a
exec python3 .auto/remote_eval.py "$@"
