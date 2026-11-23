#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heter-Bench 原始数据集下载（huggingface_hub 实现，断点续传 + 镜像友好）。

【四个数据集的来源与落位】（默认 RAW_DIR = <Heter-Bench>/TaskData/raw，可 env 覆盖）

  ShareGPT    HF datasets anon8231489123/ShareGPT_Vicuna_unfiltered
              单文件 ShareGPT_V3_unfiltered_cleaned_split.json (~1.4GB)
              → <RAW_DIR>/SharedGPT-multi-turn/

  Codeparrot  HF datasets codeparrot/apps
              单文件 test.jsonl (~1.29GB，test split，introductory 过滤在处理侧做)
              → <RAW_DIR>/Codeparrot/

  AgentBank   HF datasets Solaris99/AgentBank
              整仓 snapshot（parquet，prepare_agentbank_data.py 遍历
              AgentBank/data/<task>/*.parquet）
              → <RAW_DIR>/AgentBank/

  MASHQA      HF datasets VivekNaga/Medicaldata
              train/test/val 三个 webmd_squad_v2_full.json（LFS，共约 200MB）
              → <RAW_DIR>/mashqa_data/

【幂等】目标文件已存在且大小合理（> MIN_SIZE）时跳过，重复跑不重复下载；
        huggingface_hub 自带断点续传（未完成的临时文件会接着下）。

【镜像】遵循 HF_ENDPOINT 环境变量（国内建议 export HF_ENDPOINT=https://hf-mirror.com）。
        注意：hf_hub_download 会先把文件落到 hub 缓存 (~/.cache/huggingface)
        再拷到 local_dir，两个位置都要够放。

【用法】
    python3 download_datasets.py                    # 全部下载
    python3 download_datasets.py sharegpt apps      # 只下指定数据集
    RAW_DIR=/data/raw python3 download_datasets.py  # 自定义落盘目录

可选数据集别名：sharegpt / apps(codeparrot) / agentbank / mashqa
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MASHQA_FILES = [
    "train_webmd_squad_v2_full.json",
    # test_webmd_squad_v2_full.json 上游仓库不存在（hf 与镜像均 404），
    # prepare_data.py 对缺失文件自动跳过，无需下载。
    "val_webmd_squad_v2_full.json",
]

# 单文件下载：alias -> (repo_id, filename, 子目录, 最小合理字节数)
# 大小阈值只为挡「0 字节 / 损坏文件」，不是精确值。
FILE_SPECS = {
    "sharegpt": (
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        "SharedGPT-multi-turn",
        500 * 1024 * 1024,  # 实际 ~1.4GB
    ),
    "apps": (
        "codeparrot/apps",
        "test.jsonl",
        "Codeparrot",
        500 * 1024 * 1024,  # 实际 ~1.29GB
    ),
}
# MASHQA 的三个文件各自较小（10~150MB），单独给阈值
MASHQA_MIN_SIZE = 5 * 1024 * 1024

ALIASES = ["sharegpt", "apps", "agentbank", "mashqa"]


def _raw_dir() -> Path:
    raw = os.environ.get("RAW_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    # 默认：<本脚本所在目录>/raw（即 TaskData/raw）
    return Path(__file__).resolve().parent / "raw"


def _dest_ok(path: Path, min_size: int) -> bool:
    """已存在且大小合理 → 视为下载完成。"""
    return path.is_file() and path.stat().st_size >= min_size


def _download_file(repo_id: str, filename: str, dest_dir: Path, min_size: int) -> None:
    from huggingface_hub import hf_hub_download

    dest = dest_dir / filename
    if _dest_ok(dest, min_size):
        print(f"  [skip] {dest.name} 已存在（{dest.stat().st_size / 1e9:.2f} GB）")
        return
    print(f"  下载 {repo_id}/{filename} → {dest_dir}")
    got = hf_hub_download(repo_id=repo_id, filename=filename,
                          repo_type="dataset", local_dir=dest_dir)
    # hf_hub_download 在 local_dir 模式下直接返回落盘路径（新版）；保险起见校验目标
    got_path = Path(got)
    if got_path != dest and got_path.is_file():
        got_path.replace(dest)
    if not _dest_ok(dest, min_size):
        raise RuntimeError(f"{dest} 下载后大小异常（{_size_str(dest)}，阈值 {min_size}）")


def _download_agentbank(dest_dir: Path) -> None:
    """AgentBank：整仓 snapshot。prepare_agentbank_data.py 期望
    <agentbank-dir>/data/<task>/*.parquet，故下到 raw/AgentBank/。"""
    from huggingface_hub import snapshot_download

    # 已有 parquet 就视为完成（整仓重下代价太大）
    existing = list(dest_dir.glob("data/**/*.parquet")) or list(dest_dir.rglob("*.parquet"))
    if existing:
        print(f"  [skip] {dest_dir} 已有 {len(existing)} 个 parquet")
        return
    print(f"  下载 Solaris99/AgentBank 整仓 → {dest_dir}")
    snapshot_download(repo_id="Solaris99/AgentBank",
                      repo_type="dataset", local_dir=dest_dir)
    if not existing and not list(dest_dir.rglob("*.parquet")):
        raise RuntimeError(f"{dest_dir} 下载后未发现 parquet 文件")


def _size_str(path: Path) -> str:
    if not path.exists():
        return "missing"
    n = path.stat().st_size
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="*", choices=ALIASES,
                    help=f"要下载的数据集（缺省=全部）：{ALIASES}")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("❌ 缺少 huggingface_hub，请先安装：pip install -U huggingface_hub",
              file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"HF_ENDPOINT = {endpoint}")

    raw_dir = _raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"RAW_DIR = {raw_dir}")

    wanted = args.datasets or ALIASES
    failed: list[str] = []

    for name in wanted:
        print(f"\n=== {name} ===")
        try:
            if name == "agentbank":
                _download_agentbank(raw_dir / "AgentBank")
            elif name == "mashqa":
                for fn in MASHQA_FILES:
                    _download_file("VivekNaga/Medicaldata", fn,
                                   raw_dir / "mashqa_data", MASHQA_MIN_SIZE)
            else:
                repo_id, filename, sub, min_size = FILE_SPECS[name]
                _download_file(repo_id, filename, raw_dir / sub, min_size)
        except Exception as e:  # 单个数据集失败不影响其余，但汇总退出码
            failed.append(name)
            print(f"  ❌ {name} 下载失败：{e}", file=sys.stderr)

    print("\n=== 汇总 ===")
    for name, (repo_id, filename, sub, _) in FILE_SPECS.items():
        if name in wanted:
            print(f"  {name:10s} {_size_str(raw_dir / sub / filename)}")
    if "agentbank" in wanted:
        n = len(list((raw_dir / 'AgentBank').rglob("*.parquet")))
        print(f"  {'agentbank':10s} {n} parquet")
    if "mashqa" in wanted:
        for fn in MASHQA_FILES:
            print(f"  {'mashqa':10s} {fn}: {_size_str(raw_dir / 'mashqa_data' / fn)}")

    if failed:
        print(f"\n❌ 失败：{failed}（重跑本脚本可断点续传）", file=sys.stderr)
        return 1
    print("\n✅ 原始数据就绪，接着跑 bash TaskData/prepare_all.sh 处理")
    return 0


if __name__ == "__main__":
    sys.exit(main())
