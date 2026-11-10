#!/bin/bash
# 语义与淘汰循环 characterization 单测（本地，torch-free）。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=$PWD/python python3 -m pytest -q \
  test/manual/test_quota_selector.py \
  test/manual/test_slru_one_shot.py \
  test/manual/test_session_type_classifier.py \
  Heter-Bench/mixexp/tools/tests/test_build_exp3_data.py \
  2>&1 | tail -4
PYTHONPATH=$PWD/python python3 test/manual/test_evict_loop.py 2>&1 | tail -2
