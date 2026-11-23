#!/usr/bin/env python3
"""把一次 capture run（建议 lru 臂）的逐请求 CSV 固化成 OPT 的 use map。

OPT 上界的唯一输入是 ``{block_hash: [使用时刻]}``（``sim/opt_map.py``）。本脚本
只做「时刻回填」这一层：数据读取与 hash 规则全部复用现成模块——

    CSV 行 ``(task, session_id, turn, ts)``  →  ``sim.session_data.load_tasks``
    读出的 token 序列按轮回填时刻            →  ``sim.opt_map.build_use_map``

两个时刻域的换算写在这里：CSV 的 ``ts`` 是 ``time.time()``（epoch 秒），server 的
``_now()`` 是 ``time.monotonic``（``radix_cache.py``，CLOCK_MONOTONIC，开机起算）。
Linux 上两者系统级同源、只差一个常数偏移；``--offset auto``（缺省）在构建时刻
自测该偏移——**要求采集与构建在同一次内核启动周期内**，中间休眠/重启会让偏移
漂移，换算出的时刻会越界，脚本按不可信数据直接报错。

用法::

    python mixexp/tools/build_use_map_from_bench.py --cell chat

    # 显式写全（默认值即由此展开；--csv/--tasks/--out 都相对 Heter-Bench 根解析）
    python mixexp/tools/build_use_map_from_bench.py \\
        --cell chat \\
        --csv mixexp/result/exp3/chat/full_lru.csv \\
        --tasks TaskData/exp3/chat/tasks.yaml \\
        --page-size 1 \\
        --out TaskData/exp3/chat/use_map_p1.json

产出交给 server：``POLICY=opt`` +
``POLICY_CONFIG='{"use_map_path":"...","block_size":<--page-size>}'``。
``block_size`` 必须等于起 server 时的 ``--page-size``（缓存构造时显式校验）。

**时刻必须重锚定到重放时间轴（``--replay-start``）。** capture 与重放发生在不同
墙钟时刻，而 server 的 ``_now()`` 是绝对 monotonic：若 map 里的时刻留在 capture
时间轴上，重放时所有块的下一次使用都在过去 → 全部查作 ``+∞`` → OPT 静默退化成
任意序（``OPTStrategy`` 注释里警告的那种失败）。所以 map 时刻 = 相对延迟
``d_i`` + 重放起点；做法是先选一个未来的重放起点 ``R``（epoch 秒），本脚本把
``d_i = ts_i - C``（C 为 capture 起点）重锚到 ``R``，然后 server 起好、压测端
``sleep`` 到 ``R`` 再发射。常数级偏差（shell 延迟、首轮完成时刻近似起点）只移动
``now`` 边界，不改变块间次序，无害。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent.parent   # mixexp/tools → Heter-Bench
REPO_ROOT = BENCH_DIR.parent                                # sim/ 与 python/ 在仓库根

DEFAULT_CELL = "chat"
DEFAULT_PAGE_SIZE = 1
DEFAULT_MIN_COVERAGE = 0.9


def _add_paths() -> None:
    for path in (str(REPO_ROOT), str(REPO_ROOT / "python")):
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把 capture run 的逐请求 CSV 固化成 OPT 的 use map"
    )
    p.add_argument("--cell", default=DEFAULT_CELL,
                   help="cell 名；决定 --csv/--tasks/--out 的默认路径（默认 chat）")
    p.add_argument("--csv", default=None,
                   help="capture run 的逐请求 CSV（缺省 mixexp/result/exp3/<cell>/full_lru.csv）")
    p.add_argument("--tasks", default=None,
                   help="tasks.yaml（缺省 TaskData/exp3/<cell>/tasks.yaml）")
    p.add_argument("--out", default=None,
                   help="use map 输出路径（缺省 TaskData/exp3/<cell>/use_map_p<pagesize>.json）")
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE,
                   help="切块粒度（token），必须等于起 server 时的 --page-size（默认 1）")
    p.add_argument("--offset", default="auto",
                   help="wall→monotonic 偏移：auto=现在实测（默认），或显式秒数")
    p.add_argument("--replay-start", default="auto", metavar="EPOCH|auto",
                   help="重放 run 的起点（epoch 秒）：把 capture 时间轴重锚到它；"
                        "auto=现在。建议先定一个未来时刻 R，起完 server 后 sleep 到 R 再发射")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="采集率下限（被回填的轮 / 数据总轮数），低于它报错退出（默认 0.9）")
    return p.parse_args()


def _resolve(path: str) -> Path:
    """相对路径一律相对 Heter-Bench 根解析（与 tasks.yaml 的 data_path 同口径）。"""
    path = Path(path)
    return path if path.is_absolute() else BENCH_DIR / path


def measure_offset(explicit: str) -> float:
    """wall → monotonic 的常数偏移。

    auto：现在实测。前提是 CSV 的采集发生在同一次内核启动周期内——偏移只在
    机器休眠/重启后漂移，漂了会在下面的越界检查里显式炸出来，不会静默。
    """
    if explicit != "auto":
        return float(explicit)
    offset = time.time() - time.monotonic()
    print(f"[offset] wall - monotonic = {offset:.3f}s（auto 实测）")
    return offset


def load_capture(csv_path: Path) -> tuple[dict, dict]:
    """读逐请求明细 → {(task, sid, turn): ts}。

    同键重复行（理论上不该有：一个 turn 一行）取首行，重复数单独报。
    """
    rows: dict[tuple[str, str, int], float] = {}
    dup = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["task"], row["session_id"], int(row["turn"]))
            if key in rows:
                dup += 1
                continue
            rows[key] = float(row["ts"])
    if not rows:
        raise SystemExit(f"{csv_path} 里没有可用的明细行（缺 ts 列？）")
    per_task: dict[str, int] = {}
    for task, _sid, _turn in rows:
        per_task[task] = per_task.get(task, 0) + 1
    return rows, per_task


def main() -> int:
    args = parse_args()
    _add_paths()

    from sim.opt_map import build_use_map, write_use_map
    from sim.workload.session_data import load_tasks

    csv_path = _resolve(args.csv or f"mixexp/result/exp3/{args.cell}/full_lru.csv")
    tasks_path = _resolve(args.tasks or f"TaskData/exp3/{args.cell}/tasks.yaml")
    out_path = _resolve(
        args.out or f"TaskData/exp3/{args.cell}/use_map_p{args.page_size}.json"
    )
    for path in (csv_path, tasks_path):
        if not path.exists():
            raise SystemExit(f"缺少输入：{path}")

    offset = measure_offset(args.offset)
    rows, per_task = load_capture(csv_path)
    # 重锚定：capture 起点 C（首行完成时刻，与真 t0 差首轮延迟，秒级且为常数）
    # 与重放起点 R 之间的墙钟差就是整条时间轴的平移量。
    capture_start = min(rows.values())
    replay_start = time.time() if args.replay_start == "auto" else float(args.replay_start)
    shift = replay_start - capture_start
    if shift < 0:
        raise SystemExit(
            f"--replay-start ({replay_start:.0f}) 早于 capture 起点（{capture_start:.0f}）："
            "重放不可能发生在采集之前，检查传值"
        )
    print(f"[shift] 重锚定 +{shift:.0f}s（capture {capture_start:.0f} → 重放 {replay_start:.0f}）")
    with tasks_path.open(encoding="utf-8") as handle:
        task_specs = json.load(handle) if tasks_path.suffix == ".json" else _yaml(handle)

    sessions = load_tasks(task_specs, base_dir=str(BENCH_DIR))
    if not sessions:
        raise SystemExit(f"{tasks_path} 里没有会话")

    # CSV 与 tasks.yaml 的 task 名必须能对上：对不上说明采错了 cell / 配错文件。
    data_tasks = {s.task_name for s in sessions}
    unknown = set(per_task) - data_tasks
    if unknown:
        raise SystemExit(
            f"CSV 里的 task {sorted(unknown)} 不在 {tasks_path} 中："
            "采的 cell 与给的 tasks.yaml 不是同一份负载"
        )

    # 时刻回填。Session/Turn 是 frozen dataclass：用 replace，不能直接赋值。
    # 重锚定后 map 时刻本来就落在未来（重放还没发生），所以只查负值：
    # offset 漂移（重启/休眠）会把 ts - offset 拖成大负数，在这里显式炸出来。
    fixed, captured, total = [], 0, 0
    dropped_turn = 0
    for session in sessions:
        turns = []
        for idx, turn in enumerate(session.turns, 1):     # CSV 的 turn 列 1 起
            total += 1
            ts = rows.get((session.task_name, session.session_id, idx))
            if ts is None:
                dropped_turn += 1
                continue
            t = ts + shift - offset
            if t < 0.0:
                raise SystemExit(
                    f"换算出负时刻 t={t:.1f}s（{session.session_id} 第 {idx} 轮）："
                    "--offset 已过期（采集后机器重启/休眠过），请重测或显式传值"
                )
            turns.append(replace(turn, t=t))
            captured += 1
        if turns:
            fixed.append(replace(session, turns=tuple(turns)))

    coverage = captured / total if total else 0.0
    if coverage < args.min_coverage:
        raise SystemExit(
            f"采集率 {coverage:.1%} < --min-coverage {args.min_coverage:.0%}"
            f"（captured={captured} / total={total}，丢弃 {dropped_turn} 轮）："
            "capture run 与 tasks.yaml 大概率不是同一条到达流，map 不可信"
        )

    write_use_map(str(out_path), build_use_map(fixed, page_size=args.page_size))
    with out_path.open(encoding="utf-8") as handle:
        n_blocks = len(json.load(handle))

    times = [t.t for s in fixed for t in s.turns]
    print(
        f"[use map] 会话 {len(fixed)}/{len(sessions)}  轮 {captured}/{total}"
        f"（coverage {coverage:.1%}）  块 {n_blocks}  页 {args.page_size}"
        f"  时刻范围 [{min(times):.0f}, {max(times):.0f}]s（monotonic 域，已锚定重放起点）"
    )
    print(f"[use map] → {out_path}")
    print(
        "[use map] 起 server 时 --page-size 必须等于 "
        f"{args.page_size}，POLICY_CONFIG 的 block_size 同值"
    )
    return 0


def _yaml(handle) -> list:
    import yaml

    return yaml.safe_load(handle)["tasks"]


if __name__ == "__main__":
    sys.exit(main())
