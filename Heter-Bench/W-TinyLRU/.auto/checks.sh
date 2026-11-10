#!/bin/bash
# 回放器不变量：语法、树/分配器账目一致。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="/root/sg-unicache/python:/root/sl_old/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-/opt/conda/bin/python}
"$PY" -m py_compile wtl/trace.py wtl/replay.py wtl/backends.py
"$PY" .auto/invariants.py 2>&1 | tail -2
