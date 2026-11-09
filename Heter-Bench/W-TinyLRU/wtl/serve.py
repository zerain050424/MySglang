"""``python -m wtl.serve`` —— 带 struct 埋点的 sglang 服务启动器。

等价于 ``python -m sglang.launch_server``，但先安装 ``wtl.struct_profile``。
**不修改框架代码**：埋点只是把 ``HeterCacheRadixCache.evict`` 包了一层。

用法（在 Heter-Bench/W-TinyLRU 下）：

    PY=/root/sl_old/bin/python
    SGLANG_STRUCT_PROFILE=result/struct_A3.jsonl \\
    SGLANG_STRUCT_PROFILE_TAG=A3_hybrid_100k \\
    $PY -m wtl.serve --model-path $MODEL \\
        --enable-heter-cache --radix-eviction-policy heter_cache \\
        --radix-eviction-policy-config '{"slru":{"share_min":2}}' \\
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
    # 框架缺陷补丁（WTL_COMPAT=0 可关闭）：保留给旧实现/其他模块的兼容修复，
    # 标签版 TinyLRU 自身不需要它。
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
