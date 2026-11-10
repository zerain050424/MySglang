#!/bin/bash
# 回放器保真度：CPU 回放 vs 真机，逐 task 命中率偏差。
set -euo pipefail
cd "$(dirname "$0")/.."
/opt/conda/bin/python .auto/measure.py
