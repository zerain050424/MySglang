#!/usr/bin/env bash
# =============================================================================
# Uni-Bench 服务器环境（被 serve.sh / run_all.sh source）
# 1. 定义变量 PY，指向 UniCache 部署的 venv 里的 python：/root/sl_old/bin/python。
# 2. 优先从虚拟环境里加载ninja等
# 3. 修改默认库的查找路径
# =============================================================================
export PY="/root/sl_old/bin/python"                        # UniCache 部署 python
export PATH="/root/sl_old/bin:$PATH"                       # ninja / gcc 等
SP="/root/sl_old/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$SP/torch/lib:$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cusparselt/lib:$SP/nvidia/nccl/lib:$SP/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
