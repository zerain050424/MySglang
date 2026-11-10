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
| 0 | 基线（无限并发 + 手工 match/insert） | 无明显意义（lru 51.7pp） | — |
| 1 | `--concurrency`：信号量会话调度（对齐 bench 的 `asyncio.Semaphore`） | lru 4.3 → 0.07 | keep |
| 2 | `--inflight-reserve`：扣掉在途请求占用的 allocator 容量 | lru +0.07 | keep |
| 3 | 命中前缀 `inc_lock_ref`：修复"匹配后被淘汰→用已释放索引重建"的重复占位 | 修复死缺陷（树内 token 涨到池容量 4×） | keep |
| 4 | 插入前 `session_registry.touch`：对齐 `_adopt` 的 L/d 重置 | hc_full 23.9%（fallback 从 68% 降回千分之三） | keep |
| 5 | `--inflight-ms`：在途请求保持 lock_ref | hc_full 32.7%（真机 48.3） | 方向对，量不够 |

### 已知缺口（下一轮的靶心）

真机 hc_full 的 agentic 队列 `q_occ` 长期停在 32k（cap 18k，1.8×），回放里会被削到
18.9k。淘汰总量：真机 15.6M，回放 29.8M —— 回放**淘汰得太狠**，命中率被正反馈放大。
可疑的未建模路径（按优先级）：

1. `cache_finished_req` / `_adopt` 的真实生命周期（分配 → 插入 → `free_kv_row` → `unpin`），
   回放目前是手工 `match + alloc + insert`，`touch` 还是我补的。
2. 请求自身的 KV 行在 decode 期间不可淘汰（属于 `req.kv`，不只是命中前缀）。
3. `Req` 元数据（`tool_call` / `origin_input_ids`）没传，分类器只能靠间隔判定，
   `tool_call_count` 恒 0。
4. 在途请求的持续时间：真机 e2e 均值 1.41s，但 `unpin` 发生在完成时刻，
   回放用的是固定窗口。
