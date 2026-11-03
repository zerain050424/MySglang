"""实验臂矩阵（设计文档 §6）。

单一事实来源：离线回放、真机启动、分析脚本都从这里读配置，
避免「跑了 A4 却记成 A3」这类实验事故。

约定
----
* ``policy`` 直接对应 ``--radix-eviction-policy``；
* ``config`` 直接对应 ``--radix-eviction-policy-config``（即策略构造 kwargs）；
* ``A2`` 是隔离框架增益的基线（simple_hybrid + tiny_lru=off），
  ``A0/A1`` 是上游能力边界（lru / slru），``A3`` 是主对象。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Optional

# 默认参数（与 STRUCT_QUEUE_WTINYLRU.md §7.1 一致）
DEFAULT_K = 3
DEFAULT_WINDOW_ENTRIES = 64
DEFAULT_EPOCH_EVENTS = 8192

# 注意：tiny_lru 缺省 policy="off"。这里显式写全，便于逐臂对照。
_WTL = lambda **kw: {
    "tiny_lru": {
        "policy": "wtinylru",
        "k": kw.pop("k", DEFAULT_K),
        "window_entries": kw.pop("window_entries", DEFAULT_WINDOW_ENTRIES),
        "epoch_events": kw.pop("epoch_events", DEFAULT_EPOCH_EVENTS),
        "admission_tie": kw.pop("admission_tie", "reject"),
        "admission_metric": kw.pop("admission_metric", "share"),
        "inherit_share": kw.pop("inherit_share", True),
        **kw,
    }
}


@dataclasses.dataclass(frozen=True)
class Arm:
    id: str
    label: str
    policy: str
    config: Optional[dict]
    note: str = ""
    # 判定「策略未被行使」时用来跳过结论（见设计文档 §8.5）
    expect_struct: bool = False

    @property
    def tiny_lru(self) -> dict:
        return (self.config or {}).get("tiny_lru", {})

    @property
    def is_baseline(self) -> bool:
        return self.id in ("A0", "A2")

    def config_json(self) -> Optional[str]:
        return None if self.config is None else json.dumps(self.config, sort_keys=True)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "policy": self.policy,
            "config": self.config,
            "note": self.note,
            "expect_struct": self.expect_struct,
        }


ARMS: Dict[str, Arm] = {
    "A0": Arm("A0", "lru", "lru", None, "SGLang 默认基线", False),
    "A1": Arm(
        "A1", "slru", "slru", {"protected_threshold": 2},
        "上游分段 LRU：能力上限参照（无准入/无 aging）", False,
    ),
    "A2": Arm(
        "A2", "hybrid_off", "simple_hybrid", {"tiny_lru": {"policy": "off"}},
        "隔离框架增益：多队列 + alpha，struct 关闭", False,
    ),
    "A3": Arm(
        "A3", "wtl_default", "simple_hybrid", _WTL(),
        "主对象：k=3, window=64, epoch=8192", True,
    ),
    "A4": Arm("A4", "wtl_k2", "simple_hybrid", _WTL(k=2),
              "保护门槛更低（更早进 protected）", True),
    "A5": Arm("A5", "wtl_k5", "simple_hybrid", _WTL(k=5),
              "保护门槛更高（只有强共享才留）", True),
    "A6": Arm("A6", "wtl_win16", "simple_hybrid", _WTL(window_entries=16),
              "观察窗过小：准入几乎不发生", True),
    "A7": Arm("A7", "wtl_win256", "simple_hybrid", _WTL(window_entries=256),
              "观察窗过大：策略迟钝，逼近 A2", True),
    "A8": Arm("A8", "wtl_admit", "simple_hybrid", _WTL(admission_tie="admit"),
              "平票放宽：看后悔率是否上升", True),
    "A9": Arm("A9", "wtl_age512", "simple_hybrid", _WTL(epoch_events=512),
              "激进 aging：旧模板快速失权", True),
    "A10": Arm("A10", "wtl_noage", "simple_hybrid", _WTL(epoch_events=10**9),
               "关闭 aging：暴露频数永不衰减的代价", True),
    "A11": Arm("A11", "wtl_share_per_token", "simple_hybrid",
               _WTL(admission_metric="share_per_token"),
               "长度归一假设检验（设计文档 §5.4）", True),
}

DEFAULT_ARMS = ["A0", "A2", "A3"]
ALL_ARMS = list(ARMS)


def get(arm_id: str) -> Arm:
    try:
        return ARMS[arm_id.strip().upper()]
    except KeyError:
        raise KeyError(
            f"未知实验臂 {arm_id!r}；可用：{', '.join(ARMS)}"
        ) from None


def select(expr: Optional[str]) -> List[Arm]:
    """解析 ``A0,A3-A5`` / ``all`` / ``default`` 表达式，保持输入顺序。"""
    if not expr or expr in ("default", "DEFAULT"):
        return [ARMS[a] for a in DEFAULT_ARMS]
    if expr in ("all", "ALL", "*"):
        return [ARMS[a] for a in ALL_ARMS]
    out: List[Arm] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("A-"):
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo.strip().lstrip("Aa")), int(hi.strip().lstrip("Aa"))
            for i in range(lo_i, hi_i + 1):
                out.append(get(f"A{i}"))
        else:
            out.append(get(part))
    # 去重保序
    seen, uniq = set(), []
    for a in out:
        if a.id not in seen:
            seen.add(a.id)
            uniq.append(a)
    return uniq


# 压力点：pool_ratio = pool_tokens / 校准工作集（见 replay 的 --pool-ratio）
DEFAULT_POOL_RATIOS = [0.3, 0.6, 1.0]
SANITY_POOL_RATIO = 2.0  # §10 无压力对照点：各臂必须趋同


if __name__ == "__main__":  # 便于人工核对矩阵
    for a in ARMS.values():
        print(f"{a.id:4s} {a.label:22s} {a.policy:14s} {a.config_json()}")
