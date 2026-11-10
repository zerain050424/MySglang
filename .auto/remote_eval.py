#!/usr/bin/env python3
"""远端配对测量驱动：同步代码 → 远端跑 run_mixexp（LRU vs HeterCache 臂）→ 拉
结果 JSON → 本地算 METRIC 行。

设计要点：
- 上传是**全量覆盖**预登记文件清单（无状态）：autoresearch discard 回滚后远端不会残留旧实现。
- 臂的启停、健康等待、server 日志全部复用仓库内 `run_mixexp.sh`（预登记驱动）。
- 指标计算只在本文件（本地、受版本控制），远端只产出原始结果 JSON。
- 凭据全部来自环境变量（~/.heter_remote_env source），不写入任何文件。

用法：
  python3 .auto/remote_eval.py --cells fast_mid            # 见 CELLS
  python3 .auto/remote_eval.py --mode smoke                # 链路冒烟，不判读
  python3 .auto/remote_eval.py --list-cells
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import time
from typing import Any, Dict, List, Optional

import paramiko

# ---------------------------------------------------------------- 目标机配置
REPO = "/root/heter-autoresearch"          # 远端独立 clone（不碰 /root/sg-unicache）
EXP = REPO + "/Heter-Bench"
PORT = 30001                               # 远端 server 端口（独立）
GPU = "0"                                  # 配对比较在同一张卡上
PY = "/root/sl_old/bin/python"
MODEL = "/root/autodl-tmp/model/Qwen2.5-7B-Instruct"

# 远端语义单测（完整 sglang 导入链，本机环境跑不了、远端 sl_old 可以）。
REMOTE_TESTS = [
    "test/manual/test_session_type_classifier.py",
    "test/manual/test_heter_cache_abs_offset.py",
    "test/manual/test_cache_reset_isolation.py",
]

# 预登记上传清单：策略/测量链路的全部文件，每次全量覆盖。
UPLOAD = [
    "python/sglang/srt/mem_cache/evict_policy.py",
    "python/sglang/srt/mem_cache/victim_override.py",
    "python/sglang/srt/mem_cache/heter_utils.py",
    "python/sglang/srt/mem_cache/quota_selector.py",
    "python/sglang/srt/mem_cache/queue_occupancy.py",
    "python/sglang/srt/mem_cache/heter_cache_radix_cache.py",
    "python/sglang/srt/mem_cache/heter_bench_bridge.py",
    "Heter-Bench/scripts/bench_hetercache.py",
    "Heter-Bench/scripts/env.sh",
    "Heter-Bench/scripts/config.sh",
    "Heter-Bench/mixexp/run_mixexp.sh",
    "Heter-Bench/mixexp/run_exp3.sh",
    "Heter-Bench/mixexp/analyze.py",
    "Heter-Bench/mixexp/arms_exp3.tsv",
    "Heter-Bench/mixexp/configs/exp3/mix_full.yaml",
    "Heter-Bench/mixexp/configs/exp3/drift_phases.yaml",
    "Heter-Bench/mixexp/tools/build_exp3_data.py",
]

# 主指标口径：可复用 task（预登记，禁止按 session_id 前缀临时猜）。
REUSABLE_TASKS = ["ShareGPT", "AgentBank_alfred", "Struct"]
SCAN_TASKS = ["AgentBank_apps", "Codeparrot"]

# ---------------------------------------------------------------- cell 注册表
# 冻结原则：cell 定义写入本文件并随实验提交；改 cell = 新的测量协议，必须重跑基线。
# 压力轴 = 池容量 tokens（fp8 KV，mem-fraction 0.85，实际池以 server 日志为准）；
# 时间轴 = 到达/轮间等待等比例压缩（SCALE 同时作用于两臂，保持相对结构）。
CELLS: Dict[str, Dict[str, Any]] = {
    # 冒烟：验证链路，只看不判
    "smoke": dict(scale="smoke", mode="hybrid", tokens=30000, conc=8, wait=0.1, rep=None),
    # 快速迭代档（mid 规模、压缩轮间等待）
    "fast_mid": dict(scale="mid", mode="hybrid", tokens=120000, conc=24, wait=0.3, rep=None),
    # exp3 全规模稳态，压缩 0.1（arrivals 与 waits 同比压缩）
    "exp3_low": dict(scale="full", mode="hybrid", tokens=180000, conc=32, wait=0.1, rep=None,
                     cfg="exp3/mix_full.yaml", arrival=0.1),
    "exp3_mid": dict(scale="full", mode="hybrid", tokens=120000, conc=32, wait=0.1, rep=None,
                     cfg="exp3/mix_full.yaml", arrival=0.1),
    "exp3_high": dict(scale="full", mode="hybrid", tokens=70000, conc=32, wait=0.1, rep=None,
                      cfg="exp3/mix_full.yaml", arrival=0.1),
    # exp3 相位漂移（burst 窗口），压缩 0.1 —— 预登记确认档
    "exp3_drift": dict(scale="full", mode="drift", tokens=120000, conc=32, wait=0.1, rep=None,
                       arrival=0.1),
}

ARMS_DEFAULT = ["lru", "hc_full"]


# ---------------------------------------------------------------- 远端执行
def connect() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        os.environ["HET_HOST"],
        port=int(os.environ["HET_PORT"]),
        username=os.environ["HET_USER"],
        password=os.environ["HET_PW"],
        timeout=30,
    )
    return cli


def run_remote(cli: paramiko.SSHClient, cmd: str, timeout: int = 7200,
               quiet: bool = False) -> int:
    """流式执行远端命令，stdout/stderr 直通本地。返回 exit code。"""
    transport = cli.get_transport()
    chan = transport.open_session()
    chan.settimeout(timeout)
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    buf = []
    while True:
        data = chan.recv(65536)
        if not data:
            break
        text = data.decode("utf-8", "replace")
        buf.append(text)
        if not quiet:
            sys.stdout.write(text)
            sys.stdout.flush()
    rc = chan.recv_exit_status()
    chan.close()
    if rc != 0:
        tail = "".join(buf)[-4000:]
        print(f"[remote rc={rc}] ...{tail}", file=sys.stderr)
    return rc


def sftp_mkdirs(sftp, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def sync_files(cli: paramiko.SSHClient) -> None:
    sftp = cli.open_sftp()
    sftp_mkdirs(sftp, EXP)
    for rel in UPLOAD:
        local = os.path.join(os.getcwd(), rel)
        remote = posixpath.join(REPO, rel)
        d = posixpath.dirname(remote)
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp_mkdirs(sftp, d)
        sftp.put(local, remote)
    sftp.close()


# ---------------------------------------------------------------- 指标
def arm_metrics(path: str) -> Optional[Dict[str, float]]:
    """从结果 JSON 计算：可复用宏平均、各 task 命中率、TTFT 分位、吞吐。"""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("num_errors", 0):
        print(f"[WARN] {path}: {d['num_errors']} 个错误请求", file=sys.stderr)
    per = d.get("per_task") or {}
    if not per:
        return None
    reusable = [per[t]["cache_hit_rate_pct"] for t in REUSABLE_TASKS if t in per]
    real = [per[t]["cache_hit_rate_pct"] for t in REUSABLE_TASKS[:2] if t in per]
    if not reusable:
        return None
    m: Dict[str, float] = {
        "reusable_token_hit_pct": sum(reusable) / len(reusable),
        "real_reusable_hit_pct": sum(real) / len(real) if real else 0.0,
        "overall_token_hit_pct": float(d.get("cache_hit_rate_pct") or 0.0),
        "ttft_p50_ms": float(d.get("ttft_p50_ms") or 0.0),
        "ttft_p99_ms": float(d.get("ttft_p99_ms") or 0.0),
        "requests": float(d.get("num_requests") or 0),
        "errors": float(d.get("num_errors") or 0),
        "prompt_tokens": float(d.get("prompt_tokens") or 0),
    }
    for t, stats in per.items():
        key = "hit_" + t.replace("AgentBank_", "").lower()
        m[key] = float(stats["cache_hit_rate_pct"])
    return m


def report(arms_res: Dict[str, Optional[Dict[str, float]]], cell: str,
           t0: float) -> int:
    base = arms_res.get("lru")
    cand = next((v for k, v in arms_res.items() if k != "lru" and v), None)
    if base is None or cand is None:
        print("METRIC status=incomplete")
        return 1
    d = cand["reusable_token_hit_pct"] - base["reusable_token_hit_pct"]
    print(f"METRIC reusable_token_hit_pct={cand['reusable_token_hit_pct']:.3f}")
    print(f"METRIC delta_vs_lru={d:.3f}")
    for k in ("real_reusable_hit_pct", "overall_token_hit_pct",
              "ttft_p50_ms", "ttft_p99_ms", "requests", "errors"):
        print(f"METRIC cand_{k}={cand[k]:.3f}")
        print(f"METRIC lru_{k}={base[k]:.3f}")
    for t in REUSABLE_TASKS + SCAN_TASKS:
        key = "hit_" + t.replace("AgentBank_", "").lower()
        if key in cand and key in base:
            print(f"METRIC cand_{key}={cand[key]:.2f}")
            print(f"METRIC lru_{key}={base[key]:.2f}")
    print(f"METRIC eval_seconds={time.time() - t0:.1f}")
    return 0


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="逗号分隔的 cell 名（见 CELLS）")
    ap.add_argument("--mode", choices=["smoke", "hybrid", "drift"])
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--conc", type=int)
    ap.add_argument("--wait", type=float)
    ap.add_argument("--arms", default=",".join(ARMS_DEFAULT))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--list-cells", action="store_true")
    args = ap.parse_args()

    if args.list_cells:
        for name, c in CELLS.items():
            print(f"{name}: {c}")
        return 0

    # 解析 cells → 实验列表
    runs: List[Dict[str, Any]] = []
    if args.cells:
        for name in args.cells.split(","):
            c = CELLS.get(name)
            if c is None:
                print(f"未知 cell：{name}", file=sys.stderr)
                return 2
            runs.append(dict(c, name=name))
    else:
        runs.append(dict(
            scale="mid", mode=args.mode or "hybrid",
            tokens=args.tokens or 120000, conc=args.conc or 24,
            wait=args.wait if args.wait is not None else 0.3,
            name="adhoc", rep=None))

    t0 = time.time()
    cli = connect()
    try:
        sync_files(cli)

        # 远端语义单测（sl_old 环境；失败立即停，不进入基准）。
        test_cmd = (
            f"cd {REPO} && source Heter-Bench/scripts/env.sh && "
            f"PYTHONPATH={REPO}/python {PY} -m pytest -q "
            + " ".join(REMOTE_TESTS)
            + " 2>&1 | tail -5"
        )
        if run_remote(cli, test_cmd, timeout=900, quiet=True) != 0:
            print("METRIC status=remote_tests_failed")
            return 1
        print("[remote tests] PASSED", flush=True)

        for r in runs:
            scale, mode = r["scale"], r["mode"]
            tokens, conc, wait = r["tokens"], r["conc"], r["wait"]
            cfg = r.get("cfg") or "exp3/mix_full.yaml" if mode == "drift" or scale == "full" \
                else None
            outdir = f"{EXP}/mixexp/result/auto/{r['name']}"
            tag = f"auto_{r['name']}"
            arms = args.arms.replace(",", " ")
            env = (
                f"export PY={PY} MODEL={MODEL} PORT={PORT} "
                f"CUDA_VISIBLE_DEVICES={GPU} "
                f"ARMS='{arms}' ARMS_FILE={EXP}/mixexp/arms_exp3.tsv "
                f"CONFIGS_DIR={EXP}/mixexp/configs OUTDIR={outdir} "
                f"SKIP_EXISTING=0 "
            )
            cmd = [
                f"cd {REPO}/Heter-Bench",
                "source scripts/env.sh",
                "mkdir -p " + outdir,
                env + f"bash mixexp/run_mixexp.sh --scale {scale} --mode {mode} "
                      f"--tokens {tokens} --conc {conc} --wait-scale {wait} "
                      f"--reps {args.reps} --out {outdir} "
                      f"--mem-fraction 0.85 "
                      f"--server-arg --kv-cache-dtype --server-arg fp8_e4m3 "
                      f"--arms-file {EXP}/mixexp/arms_exp3.tsv",
            ]
            rc = run_remote(cli, " && ".join(cmd), timeout=7200)
            if rc != 0:
                print(f"METRIC status=remote_fail cell={r['name']}")
                return rc

            # 拉结果
            sftp = cli.open_sftp()
            arms_res: Dict[str, Optional[Dict[str, float]]] = {}
            for arm in args.arms.split(","):
                for rep in range(1, args.reps + 1):
                    suffix = f"_r{rep}" if args.reps > 1 else ""
                    remote = f"{outdir}/{tag}_{scale}_{arm}{suffix}.json"
                    local = f"/tmp/het_{r['name']}_{arm}{suffix}.json"
                    try:
                        sftp.get(remote, local)
                    except FileNotFoundError:
                        print(f"[missing] {remote}", file=sys.stderr)
                        arms_res[arm + suffix] = None
                        continue
                    m = arm_metrics(local)
                    arms_res[arm + suffix] = m
            sftp.close()
            rc = report(arms_res, r["name"], t0)
            if rc != 0:
                return rc
    finally:
        cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
