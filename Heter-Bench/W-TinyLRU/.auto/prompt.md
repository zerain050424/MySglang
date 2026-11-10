# Autoresearch: CPU 回放器完全还原真机行为

## Objective

`Heter-Bench/W-TinyLRU/wtl/replay.py --backend real` 在 CPU 上跑的是与真机**同一份**
`HeterCacheRadixCache` 与同一套策略代码（不加载模型、不起 server、不占 GPU，full 档
24,716 请求 / 5,110 万 prompt token 约 55 秒，真机 11,330 秒）。

目标：让它在**不针对指标调参**的前提下，逐 task 复现真机的命中率。当前最大的缺口在
`heter_cache` 系列（agentic 队列差 16pp），`lru` 已经到 0.1pp。

## Metrics

- **Primary**: `max_dev_pp`（pp，lower better）
  = 所有 (scale, arm, task) 组合里 |模拟命中率 − 真机命中率| 的最大值。
  包含 full/lru、full/hc_full、smoke/lru 三个 case 的全部 task，不排除任何点。
- **Secondary**:
  - `dev_full_hc_full` / `dev_full_lru` / `dev_smoke_lru`（按 case 的最大偏差）
  - `sim_wall_s`（回放耗时，超过 ~600s 就失去意义）
  - `worst_case`

## How to Run

`./.auto/measure.sh` — 依次回放三个 case，输出 `METRIC name=value`。

参数一律从真机产物里读，不许手填：
- allocator 容量 ← `server_<scale>_<arm>.log` 的 `max_total_num_tokens`
- 并发数、wait-scale、arrival-scale、e2e ← `result/mixexp/<scale>_<arm>.json`

## Files in Scope

- `Heter-Bench/W-TinyLRU/wtl/backends.py` — 回放后端（`RealBackend.acquire`：匹配 /
  分配 / 淘汰 / 插入 / 锁）。
- `Heter-Bench/W-TinyLRU/wtl/trace.py` — 到达调度与并发会话模型。
- `Heter-Bench/W-TinyLRU/wtl/replay.py` — 回放主循环与 CLI。
- `Heter-Bench/mixexp/calibrate_sim.py` — 标定报告（可选同步更新）。
- `.auto/invariants.py` — 不变量检查（可扩充）。

## Off Limits

- **不许改产品代码**：`python/sglang/srt/mem_cache/**`、`Heter-Bench/scripts/bench_hetercache.py`
  是基准，不是被优化对象。回放器必须适配它们，不能反过来。
- **不许按 trace 特判**：不得根据 session id / task 名 / 固定长度分支。
- **不许为了贴数字放松口径**：不得剔除 task、不得改分母、不得挑选 task 子集。
- 不得修改 `result/mixexp/*.json`（真机结果）。

## Constraints

- `./.auto/checks.sh` 必须通过：树内 token == 分配器 used（池索引不得在多个节点重复），
  以及语法检查。
- 每轮改动必须能写出一句**机制解释**：真机的哪条代码路径没有被模拟到。
  没有机制解释的参数调整（例如直接扫 `--inflight-ms` 找最佳值）不算进展。
- 偏差只允许由"补上真机行为"产生，不允许由"拟合真机数字"产生。

## What's Been Tried

| # | 改动 | max_dev_pp | 判定 |
|---|---|---|---|
| 0 | 基线（无限并发 + 手工 match/insert + 全量插 response） | 33.4（smoke/ShareGPT） | — |
| 1 | `--concurrency`：信号量会话调度（对齐 bench 的 `asyncio.Semaphore`） | lru full 7.49 → 63.5 | keep |
| 2 | `--inflight-reserve`：扣掉在途请求占用的 allocator 容量 | lru full +0.07 | keep |
| 3 | 命中前缀 `inc_lock_ref`：修复「匹配后被淘汰 → 用已释放索引重建」 | 修复死缺陷（树内 token 曾涨到池容量 4×） | keep |
| 4 | 插入前 `session_registry.touch`：对齐 `_adopt` 的 L/d 重置 | hc_full fallback 68% → 0.5% | keep |
| 5 | **观测时钟纳入仿真**（`queue_occupancy` / `cache_telemetry`） | agentic q_occ 15.7k → 26.6k（真机 32.1k） | keep |
| 6 | **response 段不驻留，只做临时占用** | **smoke 33.44 → 0.25** | keep |
| 7 | `--service-ms`（每请求 e2e） | 变差（52.6 vs 59.2） | discard：mu/sigma 已含服务时长 |
| 8 | `split_prompt` 二次插入 + 叶子时间戳偏移 | 无变化（同一仿真时刻创建，时间戳相同） | discard |

### 当前状态

| case | 模拟 | 真机 | Δ |
|---|---|---|---|
| smoke / lru | 94.20 | 94.44 | −0.25（逐 task ≤0.25） |
| full / lru | 61.7? | 59.17 | 见 measure 输出（Codeparrot +2.5 是主要来源） |
| full / hc_full | ~28 | 48.29 | **−20.5（靶心）** |

### 已知缺口（hc_full / agentic）

逐轮对比（`full_hc_full.csv` vs 回放 rows）：**第 1–6 轮几乎完全一致**（Δ ≤ 2pp），
之后随轮次放大：t20 −13pp、t30 −17pp、t40 −25pp。短会话已复现，长会话链保留率偏低。

按会话聚合（turn≥10 的平均命中率）：真机中位 0.51 / p10 0.25；回放 0.36 / p10 0.12。
是**整体水位下移**，不是少数会话归零。0.51 恰好等于 RR 的 `loss_cap = 0.5L`，
说明真机基本每轮被削到上限为止；回放削得更频繁。

淘汰归因探针（整轮 38.66M token）：

| 占比 | 节点标签 / one-shot / 会话类型 | 会话 |
|---|---|---|
| 35.05M (90.6%) | QAGENT / 0 / AGENTIC | alfred |
| 2.40M | QSTRUCT / 1 / STRUCT | 非 alfred |
| 0.69M | QSTRUCT / 1 / STRUCT | alfred |
| 0.40M | QCHAT / 0 / CHAT | alfred |

即：**RR 轮询就是削 alfred 的主刀**（17,222 次，平均 2,035 token/次），
one-shot 只占 8%。所以问题不是「别的路径偷删」，而是**回放需要的淘汰次数比真机多**。

回放 vs 真机终局账本：

| 指标 | 回放 | 真机 |
|---|---|---|
| agentic occ 中位 | 26.6k | 32.4k |
| struct 累计命中 | 0.73M | **8.49M** |
| struct 终局 occ（其中 one-shot） | 1.1k | 23.4k（23.0k） |
| chat 终局 occ | 5.4k | 17 |
| agentic 累计淘汰 | 32.4M | 22.5M |
| 账本存活会话 | 37 | **586** |

下一轮的三个候选（按可疑度，都还没试）：

1. **struct 标签归属**：真机 struct 命中 8.49M 远超 apps+code 理论上限（约 1.2M），
   说明 `fork → QSTRUCT 黏性` 把大量命中记到 struct；回放只有 0.73M，说明回放的
   分叉拓扑与真机不同。分叉点决定 `struct_tier` 的保护与 one-shot 判定，可能是
   水位差的一半来源。
2. **账本存活 37 vs 586**：`charge` 在会话状态被 `sweep` 后会**重建**一个 TRANS
   状态（`SessionState()` 默认值），此后该会话的节点全部变成 one-shot。真机账本
   存活 586，回放 37，说明回放的 sweep 触发频率远高于真机。
3. **淘汰频次**：回放每个请求平均触发 0.86 次 RR 淘汰（17,222 / 20k），真机按
   token 量推算约 0.55 次。多出来的部分来自 miss → 重插 → 再淘汰的正反馈。
