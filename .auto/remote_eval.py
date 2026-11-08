#!/usr/bin/env python3
"""把本地改动同步到实验机，跑 D-SLRU 对照，输出 METRIC 行。

背景：离线回放需要真实 `sglang` 与 torch，本机环境坏（torchvision 版本冲突），
所以测量在实验机上做，本脚本负责上传 → 跑 → 取指标。

只上传**策略相关**的三个文件，不动实验机上的 git 状态。
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import paramiko

HOST = os.environ["SLRU_HOST"]
PORT = int(os.environ.get("SLRU_PORT", "42369"))
USER = os.environ.get("SLRU_USER", "root")
PW = os.environ["SLRU_PW"]
REPO = "/root/sg-unicache"
EXP = REPO + "/Uni-Bench/W-TinyLRU"
POOLS = [150000, 250000]
ARMS = ["A0", "A1", "A3"]
TPL = 2048 + 256  # prefix_depth + variant_len

UPLOAD = [
    "python/sglang/srt/mem_cache/victim_override.py",
    "python/sglang/srt/mem_cache/evict_policy.py",
    "python/sglang/srt/mem_cache/uni_utils.py",
]

REMOTE_METRICS = r'''
import json, sys
TPL = 2048 + 256
out = {}
for name in sys.argv[1:]:
    d = json.load(open("result/ar/%s.json" % name))
    s = d["summary"]
    reqs = [x for x in (d.get("rows") or []) if "noise" not in (x.get("sid") or "")]
    n = len(reqs) or 1
    intact = sum(1 for x in reqs if x["cached_tokens"] >= TPL) / n * 100
    tpl_tok = sum(min(x["cached_tokens"], TPL) for x in reqs) / (n * TPL) * 100
    rest_cap = sum(x["prompt_tokens"] - TPL for x in reqs) or 1
    rest = sum(max(0, x["cached_tokens"] - TPL) for x in reqs) / rest_cap * 100
    out[name] = dict(intact=round(intact, 2), tpl_tok=round(tpl_tok, 2),
                     priv_tok=round(rest, 2), hit=round(s["hit_pct"], 2),
                     evicts=s["evictions"], picks=(s.get("slru") or {}).get("slru_picks"))
print(json.dumps(out))
'''


def main() -> int:
    t0 = time.time()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
    sftp = cli.open_sftp()

    for rel in UPLOAD:
        sftp.put(rel, f"{REPO}/{rel}")
    with sftp.open(f"{EXP}/ar_metrics.py", "w") as f:
        f.write(REMOTE_METRICS)

    cmds = [
        f"cd {EXP}",
        "source ../env.sh",
        'export PYTHONPATH="$PWD:$PWD/../../python:${PYTHONPATH:-}"',
        "mkdir -p result/ar",
    ]
    for pool in POOLS:
        for arm in (ARMS if pool == POOLS[-1] else ["A1", "A3"]):
            cmds.append(
                f'"$PY" -m wtl.replay --arm {arm} '
                "--data-path result/data/dslru3/session_data.jsonl "
                "--synth-meta result/data/dslru3/meta.json "
                f"--num-sessions -1 --seed 42 --pool-tokens {pool} "
                f"--backend real --no-shuffle --quiet --out result/ar/{arm}_{pool}.json"
            )
    cmds.append(
        '"$PY" ar_metrics.py '
        + " ".join(f"{a}_{p}" for p in POOLS for a in ARMS)
    )

    _, so, se = cli.exec_command(" && ".join(cmds), timeout=900)
    out = so.read().decode()
    err = se.read().decode()
    rc = so.channel.recv_exit_status()
    sftp.close()
    cli.close()
    if rc != 0:
        print("REMOTE FAILED rc=%d\n%s\n%s" % (rc, out[-2000:], err[-3000:]))
        return rc

    data = json.loads(out.strip().splitlines()[-1])

    def g(arm, pool, key):
        return data[f"{arm}_{pool}"][key]

    a3_min = min(g("A3", p, "intact") for p in POOLS)
    a1_min = min(g("A1", p, "intact") for p in POOLS)
    gap_min = min(g("A3", p, "intact") - g("A1", p, "intact") for p in POOLS)

    print(f"METRIC template_intact_min={a3_min:.2f}")
    print(f"METRIC gap_min_vs_upstream={gap_min:.2f}")
    print(f"METRIC intact_min_A1={a1_min:.2f}")
    print(f"METRIC intact_min_A0={min(g('A0', p, 'intact') for p in POOLS):.2f}")
    print(f"METRIC tpl_tok_min={min(g('A3', p, 'tpl_tok') for p in POOLS):.2f}")
    print(f"METRIC priv_tok_250k={g('A3', POOLS[-1], 'priv_tok'):.2f}")
    print(f"METRIC priv_tok_A1_250k={g('A1', POOLS[-1], 'priv_tok'):.2f}")
    print(f"METRIC hit_250k={g('A3', POOLS[-1], 'hit'):.2f}")
    print(f"METRIC hit_A0_250k={g('A0', POOLS[-1], 'hit'):.2f}")
    print(f"METRIC picks_total={sum(int(g(a, p, 'picks') or 0) for p in POOLS for a in ('A3',)):d}")
    print(f"METRIC eval_seconds={time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
