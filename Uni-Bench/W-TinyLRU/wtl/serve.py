"""``python -m wtl.serve`` —— 带 struct 埋点的 sglang 服务启动器。

等价于 ``python -m sglang.launch_server``，但先安装 ``wtl.struct_profile``。
**不修改框架代码**：埋点只是把 ``SimpleHybridRadixCache.evict`` 包了一层。

用法（在 Uni-Bench/W-TinyLRU 下）：

    PY=/root/sl_old/bin/python
    SGLANG_STRUCT_PROFILE=result/struct_A3.jsonl \\
    SGLANG_STRUCT_PROFILE_TAG=A3_hybrid_100k \\
    $PY -m wtl.serve --model-path $MODEL \\
        --enable-simple-hybrid --radix-eviction-policy simple_hybrid \\
        --radix-eviction-policy-config '{"tiny_lru":{"policy":"wtinylru","k":3}}' \\
        --max-total-tokens 100000 --enable-metrics --port 30000

注意：``PYTHONPATH`` 需要包含仓库的 ``python/`` 目录（与直接 ``python -m
sglang.launch_server`` 的要求一致）。
"""

from __future__ import annotations

import os
import sys

from .compat import apply_all as apply_compat
from .struct_profile import install_from_env


def main() -> None:
    # 框架缺陷补丁（WTL_COMPAT=0 可关闭）：不装的话，
    # 命中「window 晋升」路径时 struct_queue_policy 会无限递归把服务打挂。
    apply_compat()
    install_from_env()

    # 与 sglang/launch_server.py 的 __main__ 块保持同一路径
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    from sglang.launch_server import run_server

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
