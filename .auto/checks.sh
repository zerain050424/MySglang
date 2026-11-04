#!/bin/bash
# 语义与淘汰循环的 characterization 单测（本地，无需 GPU）。
set -euo pipefail
cd "$(dirname "$0")/.."
python .auto/run_unit.py test/manual/test_slru_one_shot.py 2>&1 | tail -3
python test/manual/test_evict_loop.py 2>&1 | tail -3
python Uni-Bench/W-TinyLRU/tests/test_smoke.py 2>&1 | tail -2
