# 代价加权上界诊断：TTFT 目标值不值得投

## 一、实验环境

| 项 | 值 |
|---|---|
| 代码 | `W1`（新增 `OptParams.cost_map_path/cost_mode`、`build_use_map(cost_model=, cost_of=)`、`write_block_costs`） |
| 仿真器 | `run_engine` + `EngineConfig(fcfs)`、page=8、cap=198k |
| 代价模型 | `qwen3coder30b_blksz_16` = `linear:865,2`（逐块 prefill 代价，论文共用同一组系数） |
| 指标 | `compute_saving_ratio` = Σ(命中块代价) / Σ(prompt 块代价) |
| 负载 | exp3 `chat` / `agentic` / `scan`（seed=0）+ drift3 `bg=1.0×`（seed=42） |

## 二、方法

1. 用一次 LRU run 冻结时间轴，`build_use_map(..., cost_model=CM, cost_of=costs)` 同时产出
   `{hash: [t...]}` 与 `{hash: cost}`；
2. **hit-optimal OPT**（原口径）迭代到 `dispatched` 零残差；
3. **代价加权 OPT**：同一张 use map，只改排序键：
   - `protect_deep`：`key = -next_use / cost`（贵的块更晚被淘汰）
   - `paper`：`key = -next_use * cost`（论文 BeladyCompute 字面式）
4. 三条臂都在 `cost_model=qwen3` 下跑，读 `compute_saving_ratio` 打分（策略与指标解耦）。

## 三、结果

| 负载 | LRU hit / compute | **OPT hit-optimal** | OPT protect_deep | OPT paper | **gap（deep − hit）** |
|---|---|---|---|---|---|
| chat | 57.28 / 60.31 | **79.45 / 81.17** | 72.82 / 75.05 | 70.92 / 71.45 | **−6.12pp** |
| agentic | 81.07 / 85.16 | **84.43 / 87.93** | 84.43 / 87.93 | 83.74 / 86.85 | **0.00pp** |
| scan | 29.65 / 35.53 | **38.42 / 44.28** | 38.37 / 44.22 | 37.96 / 43.68 | **−0.06pp** |
| drift3 bg=1.0× | 65.20 / 67.56 | **71.65 / 73.81** | 71.65 / 73.81 | 71.65 / 73.81 | **0.00pp** |

（heter_cache 参考：chat 58.12 / 61.29、agentic 82.36 / 86.30、scan 33.59 / 39.47、drift3 66.84 / 69.12）

## 四、结论

1. **代价加权的 gap 在本项目负载上 ≤ 0**（−6.12 / 0.00 / −0.06 / 0.00），
   即"为深块的更高重算代价单独加权"没有任何可回收空间。这与论文在其余 5 条 trace 上
   gap ≤ 1.1 点、只在其长上下文单轮 trace 上有 +12.6 点，是同一结论的另一份实测。
2. **所有臂 `compute > hit`**（如 scan 44.28 vs 38.42、chat 81.17 vs 79.45）：
   命中**天然集中在深块**（多轮会话是追加式，越深越容易命中；miss 多在浅层新前缀）。
   所以"深度贵"这件事已经被命中结构吸收了，再按深度加权只是在同一排序上加扰动。
3. **chat 上 protect_deep 反而亏 6.12pp**，机制清楚：保护深块＝放弃浅块，而 radix 是
   前缀连续的——浅块一没，整条链作废（本项目实测：`depth≤1 页` 的淘汰贡献 71% 的 regret）。
   hit-optimal Belady 的 `min(各块 next_use)` 已经隐式编码了"下游体量"（浅块被更多未来请求
   共享 → min 更小 → 更该保护），再乘一个随深度增长的成本是**重复计价**。
4. 因此：**把 TTFT p90 当首要目标这件事，不能通过"深度加权淘汰"实现**。
   论文的正例来自 block-addressable 缓存（可留深块、重算浅块），radix 的叶结构不同，
   这条经验不能直接搬。

## 五、对 p90 目标的含义

仍可动的只剩两处：

1. **容量/超订比**：已经测过（chat cell 198k → 396k 让 OPT−heter 从 21.35pp 缩到 3.51pp）；
2. **miss 的形状**（不是深度）：把"整条前缀丢失"换成"部分丢失"——即按"下游依赖体量"
   选受害者（浅块优先保护、砍链尾），而不是按"单块代价"。这是 tail-stress 负载给出的
   920ms → 184ms 空间所在，与深度加权是两件事。
3. 排队那 48% 的 p90 只能靠调度（LPM / 小批优先），不在淘汰策略范围内。
