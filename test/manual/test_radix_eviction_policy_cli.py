"""``--radix-eviction-policy`` 的 CLI 白名单回归（纯 argparse，不需要 torch）。

单独成文件：``test_sim_opt.py`` 依赖 torch，而这条是纯参数解析问题——真机 OPT
流程（``Heter-Bench/mixexp/README`` §2）第一道门就是它。
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))


def test_opt_is_accepted_by_the_server_cli():
    """``--radix-eviction-policy opt`` 必须在 argparse 阶段被接受。

    回归点：``RADIX_EVICTION_POLICY_CHOICES`` 原先没有 ``"opt"``，策略工厂明明
    注册了 ``OPTStrategy``，服务却起不来（``invalid choice: 'opt'``）——
    ``mixexp/README`` §2 的「采时刻 → 造 use map → 带 map 重放」整条流程就断在
    参数解析这一步。
    """

    import argparse

    from sglang.srt.arg_groups.choices import RADIX_EVICTION_POLICY_CHOICES

    assert "opt" in RADIX_EVICTION_POLICY_CHOICES

    server_args = pytest.importorskip("sglang.srt.server_args")
    parser = argparse.ArgumentParser()
    server_args.ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model-path", "/tmp/_sim_opt_cli_probe", "--radix-eviction-policy", "opt"]
    )
    assert args.radix_eviction_policy == "opt"
