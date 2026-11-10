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

逐轮对比（`full_hc_full.csv` vs 回放 rows）显示：**第 1–6 轮几乎完全一致**
（Δ ≤ 2pp），之后随轮次线性扩大：t20 −13pp、t30 −17pp、t40 −25pp。
即短会话已经复现，长会话的**链保留率**偏低。

回放 vs 真机的终局账本：

| 指标 | 回放 | 真机 |
|---|---|---|
| agentic occ 中位 | 26.6k | 32.4k |
| struct 累计命中 | 0.73M | **8.49M** |
| struct 终局 occ（其中 one-shot） | 1.1k | 23.4k（23.0k） |
| chat 终局 occ | 5.4k | 17 |
| agentic 累计淘汰 | 32.4M | 22.5M |
| sessions（账本存活） | 37 | **586** |

三个方向（按可疑度）：

1. **struct 的标签归属**：真机 struct 命中 8.49M 远超 apps+code 的理论上限
   （约 1.2M），说明大量命中被 `fork → QSTRUCT 黏性` 规则吸到 struct 名下；
   回放只有 0.73M，说明回放的**分叉拓扑**和真机不同（同一批 token 在回放里
   没有形成多分支）。
2. **会话账本存活数 37 vs 586**：`sweep` 在回放里把会话删得太干净
   （`empty_grace=60` + 节点被清空）。账本消失会让 `is_one_shot` 保守返回
   False，同时 RR 的候选集变化。
3. **长会话的保留率**：RR 的 `loss_cap = 0.5L` 上限解释不了 25% 的保留率，
   说明还有别的路径在削 agentic（需要按 session 统计淘汰量归因）。
