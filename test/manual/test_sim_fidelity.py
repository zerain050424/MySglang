"""把 sim 的整套自检接进 pytest（纯 CPU，不需要 GPU / 模型 / 数据集）。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_fidelity.py

自检逻辑全部在 ``sim/__main__.py``（M0 装载 / M1 时钟 / M2 分配器 / M3 驱动 /
M4 保真），这里只负责跑它并把输出原样贴进失败信息，避免两套断言各写一份。
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run_sim() -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("STRAT_LOG_LEVEL", "quiet")
    env["PYTHONPATH"] = os.path.join(ROOT, "python") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "sim"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_sim_selfcheck_passes():
    proc = _run_sim()
    assert proc.returncode == 0, f"python -m sim 失败：\n{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout.splitlines()[-1]


def test_fidelity_stress_report():
    """在 CPU 下直接校验树拓扑不变量、reconcile 后账本一致性与复现指纹。"""
    from sim.analysis.fidelity import run_fidelity
    from sim.core.config import SimConfig

    report = run_fidelity(SimConfig(capacity=2048, seed=11))
    assert not report.violations, report.violations
    assert report.ledger["exact_after_reconcile"], report.ledger
    assert report.matched, "两次保真压测指纹不一致"


def test_sim_selfcheck_is_reproducible():
    """两次独立进程的完整输出必须逐字相同（含全部指纹）。"""

    first, second = _run_sim(), _run_sim()
    strip = lambda text: "\n".join(  # noqa: E731 - 环境相关行不参与比较
        line for line in text.splitlines()
        if not line.strip().startswith(("repo", "python  ", "torch", "stub", "^"))
    )
    assert strip(first.stdout) == strip(second.stdout), "两次 run 的输出不一致"
