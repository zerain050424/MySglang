from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # 让 import bench_unicache 可用

import aiohttp
import bench_unicache as bu            # 复用 Stats / run_session / one_request

DATA = (
    HERE / "TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct"
    / "ShareGPT_cleaned_valid_deduplicated_session_data.jsonl"
)


def load_first_n(path: Path, n: int):
    """读文件前 n 行"""
    sessions = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                sessions.append(list(json.loads(line).values())[0])
    return sessions


async def main(args) -> None:
    args.arrival_config = bu.load_arrival_config(args.arrival_config)
    sessions = load_first_n(DATA, args.n)
    print(f"[first{args.n}] loaded {len(sessions)} sessions from {DATA}")

    stats = bu.Stats()
    sem = asyncio.Semaphore(args.concurrency)
    url = args.base_url.rstrip("/") + "/generate"

    async with aiohttp.ClientSession() as http:
        tasks = [
            asyncio.create_task(bu.run_session(http, sem, s, args, stats, url))
            for s in sessions
        ]
        await asyncio.gather(*tasks)

    summary = stats.summary()
    print("\n=============== ShareGPT 前 %d 条结果 ===============" % len(sessions))
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print("=====================================================")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="只跑 ShareGPT 前 N 条 session")
    p.add_argument("--n", type=int, default=10, help="session 条数")
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout-s", type=float, default=1800.0)
    p.add_argument("--wait-scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--arrival-config",
        default=str(HERE / "arrival_config.yaml"),
        help="到达间隔 YAML 路径（模仿 UniCache tasks_config.yaml）",
    )
    p.add_argument("--mu", type=float, default=None,
                   help="轮间等待 lognormal mean 全局覆盖；默认 None=按负载类型取")
    p.add_argument("--sigma", type=float, default=None,
                   help="轮间等待 lognormal sigma 全局覆盖；默认 None=按负载类型取")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))