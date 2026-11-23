# exp3 四 cell 复跑：chat 老化 TTL 180 → 300

> 【消融实验专题 · 参数微调】chat 老化 TTL 180→300 倒 U 曲线平台顶实测验证

## 实验环境

| 项 | 值 |
|---|---|
| 代码 | `W1` @ 本次提交（`DEFAULT_AGEING_TTL_S["chat"] = 300`，其余与 `engine_full` 基线一致） |
| 仿真器 | `Simulator.run_engine`，`EngineConfig(schedule_policy="fcfs")`，`max_prefill_tokens=16384`、chunked prefill 关 |
| page / 容量 | page=8、cap=198,000 token（exp3 的压力点，与基线同值） |
| 负载 | exp3 四个 cell：`scan`（均衡）/ `struct` 主导 60% / `chat` 主导 60% / `agentic` 主导 60% |
| 服务模型 | `tailshot/calib_lru.json`，decode 40 tok/s |
| OPT | 时间轴固定点迭代至 `dispatched` 零残差（scan 3 轮、struct 2 轮、chat/agentic 3 轮），`map_miss = 0` |
| 随机种子 | 0（与基线一致） |

驱动：`/root/sim_runs/sim_engine_cell.py`（`CELL=<cell> PAGE=8 CAP=198000`），产物
`/root/sim_runs/engine_full/<cell>_p8_cap198000/compare.json`；基线存放在
`/root/sim_runs/engine_full_ttl180/` 以便逐项对照。

## 参数设置

本次唯一改动：`python/sglang/srt/mem_cache/heter_cache/utils.py` 的
`DEFAULT_AGEING_TTL_S["chat"]` 由 180s 调至 300s（依据 `drift3_report.md` §4.10 的 TTL 扫描
倒 U 曲线的平台顶）。LRU 与 OPT 不读该参数，因此它们的数字应与基线**逐位相同**——本次实测
确实逐位相同，等于顺手做了一次确定性与改动隔离的检查。

## 结果

| cell | LRU | **heter(300)** | heter(180) | Δ | OPT | heter−LRU | OPT−heter |
|---|---|---|---|---|---|---|---|
| scan（均衡） | 29.65% | **33.59%** | 32.69% | **+0.90pp** | 38.42% | +3.94pp（+3.03） | +4.84pp（+5.74） |
| struct 主导 | 81.64% | **82.41%** | 81.56% | **+0.85pp** | 83.49% | +0.77pp（**−0.08**） | +1.08pp（+1.93） |
| chat 主导 | 57.28% | **58.07%** | 58.90% | **−0.83pp** | 79.45% | +0.79pp（+1.62） | +21.38pp（+20.55） |
| agentic 主导 | 81.07% | **82.36%** | 81.70% | **+0.66pp** | 84.43% | +1.29pp（+0.62） | +2.07pp（+2.73） |

按 prompt token 加权的四 cell 合计：

| 臂 | chat TTL=300 | chat TTL=180 | Δ |
|---|---|---|---|
| LRU | 62.55% | 62.55% | 0.00pp（未读该参数，预期如此） |
| **heter_cache** | **64.24%** | 63.84% | **+0.40pp** |
| OPT | 71.54% | 71.54% | 0.00pp（同上） |
| heter−LRU | **+1.69pp** | +1.29pp | +0.40pp |
| OPT−heter | +7.31pp | +7.70pp | −0.39pp |

## 对比分析

1. **三个 cell 明确改善，只有 chat 主导 cell 变差。** scan +0.90pp、struct +0.85pp、
   agentic +0.66pp；chat 主导 −0.83pp。和 drift3（P2 chat 自身 −1.2~−1.7pp、struct/agentic
   由负转正）是同一条机制的两面：老化 TTL 抬高 = 陈旧 chat 前缀多留一会儿，代价付在 chat
   池内，收益落在别的队列不被打扰。
2. **struct cell 由「唯一输给 LRU 的 cell」翻正**：−0.08pp → +0.77pp。这是本次最有价值的一项，
   因为 P1/struct 一直是 heter 相对 LRU 的短板（drift3 里也是 −0.5~−0.9pp）。
3. **chat 主导 cell 是唯一回退**，但它对 LRU 仍是正收益（+0.79pp），且该 cell 的 OPT−heter
   高达 **21.38pp**——说明 chat 主导场景的差距主体不在这 1pp 的 TTL 上，而在「整条前缀被
   清运」的结构性问题（`drift3_report.md` §3.5 的 regret 归因同样指向这一点）。
4. **加权合计 +0.40pp**，代价是 OPT−heter 收窄 0.39pp（离上界略远一点），heter−LRU 拉开
   0.40pp。方向上属于「更早地偏向背景队列、少保护 chat」，是政策取舍而非纯增益。

## 复核

- LRU / OPT 四 cell 数字与基线逐位相同 ⇒ 改动只经过 heter 的策略路径，无副作用。
- OPT 三条 run 的时间轴残差均收敛到 0，`map_miss=0`，上界可信。
- 单测：`test_queue_selector.py` 25 passed、`test_evict_loop` ALL PASSED；
  `test_sim_clock_injection.py` 因缺 `sglang_simulator` 包在 import 阶段失败（基线如此）。
