#!/usr/bin/env python3
"""远端数据准备（在实验机上运行）：从 /root/TaskData.zip 流式抽取 → 剥 Qwen system
段 → 写入 /root/heter-autoresearch/Heter-Bench/TaskData/，再构造 exp3 四类负载，
最后输出 sha256 清单（冻结用）。

设计：
- 全程流式（zipfile 逐行），不落盘原始 jsonl，节省一半磁盘。
- 剥离逻辑复用仓库内 strip_qwen_system.strip_session（mode=qwen，只剥首段）。
- 与 strip_qwen_system 的差异：流式版本无论是否有改动都写 .nosys（内容等价：
  无 system 段时 nosys == 原文）。
- gt5 = num_turns > 5（chat 桶 6–8 / 10–15 的并集），exp3 chat 源。
- 汇总 head 去重文件只写前 NOSYS_HEAD_LINES 行，仅供 smoke 链路验证（mix_smoke
  的 max_record ≤ 10），不用于任何结论。

用法（实验机）：
  /root/sl_old/bin/python /root/heter-autoresearch/.auto/remote_data_prep.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path("/root/heter-autoresearch")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Heter-Bench" / "tools"))

from strip_qwen_system import strip_session  # noqa: E402

ZIP_PATH = Path("/root/TaskData.zip")
TASKDATA = REPO / "Heter-Bench" / "TaskData"
NOSYS_HEAD_LINES = 200

# zip 内成员 → (输出相对 TaskData 的路径, 行过滤器)
MEMBERS = [
    ("root/sg-unicache/Uni-Bench/TaskData/AgentBankData/Qwen2.5-7B-Instruct/"
     "AgentBank_apps_session_data.jsonl",
     "AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.nosys.jsonl", None),
    ("root/sg-unicache/Uni-Bench/TaskData/AgentBankData/Qwen2.5-7B-Instruct/"
     "AgentBank_alfred_scale_5000_session_data.jsonl",
     "AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.nosys.jsonl",
     None),
    ("root/sg-unicache/Uni-Bench/TaskData/Codeparrot/Qwen2.5-7B-Instruct/"
     "Code_session_data.jsonl",
     "Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.nosys.jsonl", None),
    # ShareGPT：完整剥离 → gt5 过滤版本（exp3 chat 源）
    ("root/sg-unicache/Uni-Bench/TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
     "ShareGPT_cleaned_valid_deduplicated_session_data.jsonl",
     "SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
     "ShareGPT_cleaned_valid_deduplicated_gt5_session_data.nosys.jsonl", "gt5"),
]

MANIFEST = REPO / ".auto" / "data_manifest.sha256"


def out_path(rel: str) -> Path:
    p = TASKDATA / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def stream_one(zf: zipfile.ZipFile, member: str, rel: str, flt) -> dict:
    src = zf.open(member)
    dst = out_path(rel)
    stats = {"sessions": 0, "written": 0, "turns": 0, "changed": 0,
             "tokens_removed": 0, "emptied": 0}
    h = hashlib.sha256()
    with dst.open("w", encoding="utf-8") as out:
        with io.TextIOWrapper(src, encoding="utf-8") as text:
            for line in text:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 每行单键 {sid: session}；strip/过滤作用于内层 session
                sid, record = next(iter(obj.items()))
                stats["sessions"] += 1
                changed, removed, emptied = strip_session(record, "qwen")
                stats["turns"] += int(record.get("num_turns") or 0)
                stats["changed"] += changed
                stats["tokens_removed"] += removed
                stats["emptied"] += int(emptied)
                if flt == "gt5" and int(record.get("num_turns") or 0) <= 5:
                    continue
                blob = json.dumps({sid: record}, ensure_ascii=False)
                out.write(blob + "\n")
                h.update((blob + "\n").encode("utf-8"))
                stats["written"] += 1
    stats["sha256"] = h.hexdigest()
    stats["out"] = rel
    return stats


def main() -> int:
    if not ZIP_PATH.is_file():
        print(f"缺少 {ZIP_PATH}", file=sys.stderr)
        return 2
    results = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member, rel, flt in MEMBERS:
            print(f">>> {member} -> {rel} (filter={flt})", flush=True)
            r = stream_one(zf, member, rel, flt)
            print("   ", {k: v for k, v in r.items() if k != "out"}, flush=True)
            results.append(r)

    # smoke 用 head 汇总文件（仅链路验证；max_record<=10，200 行足够）
    gt5_rel = MEMBERS[3][1]
    src = out_path(gt5_rel)
    head_rel = ("SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
                "ShareGPT_cleaned_valid_deduplicated_session_data.nosys.jsonl")
    dst = out_path(head_rel)
    h = hashlib.sha256()
    n = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if n >= NOSYS_HEAD_LINES:
                break
            fout.write(line)
            h.update(line.encode("utf-8"))
            n += 1
    print(f">>> head {n} -> {head_rel}", flush=True)
    results.append({"out": head_rel, "sha256": h.hexdigest(), "written": n})

    # exp3 构造（tokenizer 在 sl_old 环境）
    env = dict(os.environ)
    print(">>> build_exp3_data.py", flush=True)
    cmd = ["/root/sl_old/bin/python",
           str(REPO / "Heter-Bench/mixexp/tools/build_exp3_data.py"),
           "--tokenizer", "/root/autodl-tmp/model/Qwen2.5-7B-Instruct"]
    proc = subprocess.run(cmd, cwd=str(REPO / "Heter-Bench"), env=env)
    if proc.returncode != 0:
        print("build_exp3_data 失败", file=sys.stderr)
        return 1

    # exp3 产物也进清单
    exp3dir = TASKDATA / "exp3"
    for p in sorted(exp3dir.glob("*.jsonl")):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        results.append({"out": str(p.relative_to(TASKDATA)), "sha256": h.hexdigest()})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['sha256']}  {r['out']}\n")
    print(f"清单已写入 {MANIFEST}")
    for r in results:
        print(f"{r['sha256'][:12]}  {r['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
