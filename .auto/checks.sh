#!/bin/bash
# 语义与淘汰循环 characterization 单测（本地可跑的 torch-free 子集）。
#
# 注意：test_session_type_classifier.py / test_heter_cache_abs_offset.py /
# test_cache_reset_isolation.py 需要完整 sglang 导入链，本机 conda 环境因
# huggingface-hub==1.33 冲突跑不了（AGENTS.md §5.1）；它们在远端 checks 阶段执行
#（remote_eval.py 同步后运行 sl_old 环境 pytest）。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=$PWD/python python3 -m pytest -q \
  test/manual/test_queue_selector.py \
  test/manual/test_share_one_shot.py \
  Heter-Bench/mixexp/tools/tests/test_build_exp3_data.py \
  2>&1 | tail -3
PYTHONPATH=$PWD/python python3 test/manual/test_evict_loop.py 2>&1 | tail -1
