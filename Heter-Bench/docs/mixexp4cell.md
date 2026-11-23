# exp3 四 Cell 混合实验分析报告（`mixexp4cell`）

> **数据来源**：
> - 汇总指标：[`Heter-Bench/mixexp/result/sim_exp3_summary.json`](../mixexp/result/sim_exp3_summary.json)
> - 逐请求明细：`Heter-Bench/mixexp/result/sim_exp3_records/{scan,struct,chat,agentic}/*.records.jsonl`（共 28 份，由 `queue_hit_report.py` 与跨 cell 聚合脚本按 `8319.6 tok/s` prefill 速率统计）
> - 实验配置：4 个主导 cell（`scan` / `struct` / `chat` / `agentic`）× 7 个策略臂（`opt`、`chain_lru`、`heter_rgate`、`heter_rgate_chain`、`fifo`、`slru`、`lfu`），总计 116,303 次请求/cell 组合，每 cell 约 25M prompt tokens，仿真池容量 `cap = 198,000 tokens`、`page = 8`。

---

## 一、 4-Cell 总体汇总

| Cell（请求数） | `opt` (hit / P90) | `chain_lru` | `heter_rgate` | `heter_rgate_chain` | `fifo` | `slru` | `lfu` |
|---|---|---|---|---|---|---|---|
| **scan** (46,712) | **38.53%** / 100.3ms | **37.03%** / 102.6ms | 35.67% / 104.8ms | 35.67% / 104.8ms | 32.58% / 110.3ms | 28.15% / 117.6ms | 27.73% / 117.1ms |
| **struct** (21,141) | **83.66%** / 74.0ms | **83.07%** / 76.0ms | 82.86% / 76.3ms | 82.86% / 76.3ms | 82.45% / 78.1ms | 74.96% / 106.5ms | 74.53% / 106.7ms |
| **chat** (29,622) | **81.54%** / 67.4ms | **68.07%** / 108.7ms | 64.15% / 125.8ms | 63.94% / 126.6ms | 62.46% / 140.1ms | 32.21% / 195.9ms | 31.43% / 194.4ms |
| **agentic** (18,828) | **84.54%** / 77.1ms | **83.82%** / 79.2ms | 83.28% / 81.6ms | 83.28% / 81.6ms | 82.84% / 84.0ms | 73.21% / 122.7ms | 66.52% / 152.7ms |

**核心结论**：
- **`chain_lru`** 在除 `opt` 外的所有策略中全场领先，尤其在 `struct`（83.07%）、`agentic`（83.82%）和 `scan`（37.03%）上均逼近 `opt`。
- **`heter_rgate` / `heter_rgate_chain`** 整体显著优于 `fifo`、`slru`、`lfu`，但在 4 个 cell 上均略逊于 `chain_lru`（`scan` 低 1.36pp，`chat` 低 3.92pp / P90 慢 17.1ms）。
- **`fifo`** 表现稳健且远超 `slru`/`lfu`；**`slru` / `lfu`** 在 `chat`（32% / 31%）和 `scan`（28%）下严重退化。

---

## 二、 按会话类型聚合表现（跨 4 Cell 统计）

将 4 个 cell 的逐请求记录按会话类型（`struct`、`agentic`、`chat`、`trans`）合并统计，`whole_loss` 表示多轮会话 `turn > 0` 时 `hit == 0` 的续轮整链丢失比例：

### 2.1 `struct`（单轮跨会话共享前缀，12,058 请求）

| 策略 | 命中率 | TTFT P50 (ms) | TTFT P90 (ms) | TTFT P99 (ms) |
|---|---|---|---|---|
| `opt` | 96.59% | 20.6 | 30.9 | **89.0** |
| `lfu` / `slru` | **96.64%** | 20.8~20.9 | 31.7 | 148.9~153.7 |
| `chain_lru` | 93.07% | 21.0 | 32.0 | 269.2 |
| `heter_rgate_chain` | 92.01% | 21.2 | 32.2 | 270.9 |
| `fifo` | 91.99% | 21.3 | 32.3 | 270.6 |
| `heter_rgate` | 91.98% | 21.2 | 32.3 | 271.0 |

* 高频共享模板是 `lfu` / `slru` 的唯一舒适区（靠频次把模板死锁在池内，命中率甚至略高于 `opt`），代价是彻底饿死 `chat`。
* `heter_rgate*` 在 `struct` 上与 `fifo` 持平（~92%），比 `chain_lru` 低约 1pp，主要损失发生在 `scan` cell（`heter_rgate` 的 `struct` 桶掉至 **85.49%** / P90 **251.9ms**，而 `chain_lru` 为 **94.86%** / P90 **39.1ms**）。

### 2.2 `agentic`（短间隔多轮长链，6,761 请求）

| 策略 | 命中率 | TTFT P50 (ms) | TTFT P90 (ms) | TTFT P99 (ms) | 续轮整链丢失率 |
|---|---|---|---|---|---|
| `opt` | **98.13%** | 20.3 | 32.6 | **56.8** | **0.00%** |
| `heter_rgate` | **98.10%** | 20.8 | 32.8 | 78.1 | 0.12% |
| `heter_rgate_chain` | 98.08% | 20.6 | 32.9 | 78.1 | 0.15% |
| `chain_lru` | 98.06% | 20.5 | 33.0 | **72.3** | **0.04%** |
| `fifo` | 98.05% | 20.8 | 33.3 | 96.1 | 0.16% |
| `slru` | 95.63% | 26.6 | 67.3 | 165.1 | 1.36% |
| `lfu` | 89.13% | 28.9 | 135.4 | **736.0** | **4.55%** |

* `agentic` 轮间极短（T50 ≈ 6.1s），所有非频次策略的命中率均在 **98.05%–98.13%**、P90 稳定在 **33ms** 左右；**`heter_rgate` 仅差 `opt` 0.03pp**。
* `lfu` 因新轮次 `hit_count` 低被频繁提前驱逐，P99 恶化至 **736.0ms**。

### 2.3 `chat`（长间隔多轮对话，30,051 请求 —— 真正的胜负手）

| 策略 | 命中率 | TTFT P50 (ms) | TTFT P90 (ms) | TTFT P99 (ms) | 续轮整链丢失率 |
|---|---|---|---|---|---|
| `opt` | **93.33%** | 17.7 | **30.3** | **98.4** | **0.79%** |
| `chain_lru` | **80.82%** | 20.5 | **78.8** | 227.9 | **18.01%** |
| `heter_rgate` | 75.83% | 21.4 | 108.6 | 235.6 | 24.26% |
| `heter_rgate_chain` | 75.63% | 21.4 | 109.4 | 235.2 | 24.15% |
| `fifo` | 70.26% | 21.7 | 142.7 | 245.4 | 25.00% |
| `slru` | 16.13% | 89.7 | 209.2 | 287.6 | 77.86% |
| `lfu` | 14.26% | 90.7 | 209.5 | 289.5 | 75.15% |

* **全实验差距几乎全由 `chat` 决定**：`slru`/`lfu` 整链丢失高达 **75%–78%**；`heter_rgate` 较 `fifo` 提升 **+5.57pp**（P90 快 34.1ms），但较 `chain_lru` 仍有 **4.99pp** 差距（P90 慢 29.8ms，整链丢失多 6.25pp）。

### 2.4 `trans`（单轮无复用背景流量，67,433 请求）

| 策略 | 命中率 | TTFT P50 (ms) | TTFT P90 (ms) | TTFT P99 (ms) |
|---|---|---|---|---|
| `opt` | 0.19% | 50.5 | 101.7 | 189.5 |
| `chain_lru` | 0.03% | 50.7 | 102.9 | 198.8 |
| `heter_rgate_chain` | 0.04% | 50.9 | 103.8 | 204.4 |
| `heter_rgate` | 0.04% | 50.9 | 103.9 | 203.7 |
| `fifo` | 0.03% | 51.0 | 104.8 | 210.7 |
| `slru` | 0.04% | 51.5 | 107.4 | 216.5 |
| `lfu` | 0.04% | 51.6 | 107.9 | 225.1 |

---

## 三、 关键机理诊断

### 3.1 为什么 `heter_rgate` 与 `heter_rgate_chain` 几乎重合，且在 `chat` / `scan` 落后 `chain_lru`？

1. **Stage 0 / 跨队列仲裁屏蔽了队内策略**：
   - 在 `scan` 与 `struct` cell 中，`victim_overrides` 的 `reuse:*` / `chain:*` `picks` 全部为 **0**；在 `agentic` cell 中也仅有 **10–12 次**。绝大多数淘汰在到达 `chat`/`agentic` 队内选点之前就已被 Stage 0（one-shot）或 `struct` 队列消化，导致 `heter_rgate` 与 `heter_rgate_chain` 指标完全一致。
   - 在 `scan` cell 下，`struct` 桶命中率跌到 **85.49%**（P90 251.9ms），说明部分可跨会话复用的 `struct` 模板在 Stage 0 或 VD 仲裁下被过早清理。
2. **`chat` cell 中 `RegretGate` 对 `struct` 大量弃权，压力集中转嫁给 `chat`**：
   - `chat/heter_rgate` 的 `regret_gate_stats` 显示：`abstain_struct = 8,544`、`abstain_chat = 1,092`、`abstain_agentic = 146`、`cheapest_pick = 0`（从未触发全弃权 `PricedFallback`）；与此同时 `reuse:chat` 被选中 **24,299 次**（`chain:chat` 22,539 次）。
   - 后悔门频繁跳过 `struct`，把大量淘汰配额硬压给 `chat` 队列，最终 `chat` cell 内的 `ShareGPT` 桶命中率降至 **68.61%**（`heter_rgate_chain` 为 **68.27%**，对比 `chain_lru` 为 **75.91%**），`chat` 队列容量被抽干后，队内 `ChainEvict` 无法挽回总量缺口。

### 3.2 为什么 `FIFO`（`node.creation_time`）意外强健？

1. **新写入轮次的天然保护窗（碾压 `LFU` / `SLRU`）**：`LFU`/`SLRU` 会将刚插入、仅 `hit_count=1` 的多轮新节点立即当作低频垃圾淘汰；而 `FIFO` 下新写入节点的 `creation_time` 最新，天然获得一整个「池周转周期」的保护窗，足以覆盖 `agentic` 与部分 `chat` 的轮间空窗。
2. **Radix 树「仅淘汰叶子」拓扑 + `creation_time` 的弱链形保护**：浅层祖先只要仍有活跃子节点就不会成为叶子，不会提前断裂；而单轮 `trans` 冲刷与已结束的老会话叶子按创建时间自然出池，不会像 `LFU` 那样被历史高频僵尸节点永久占坑。
3. **`FIFO` 的天花板**：因完全不刷新访问时间，在 `scan` 下会把刚被重读过的 `struct` 老节点按 `creation_time` 踢出（`struct` 桶仅 **80.81%** / P90 **259.4ms**）；在 `chat` 下无法保护轮间略长于池周转时间的活跃会话（`chat` 桶 **70.26%** / P90 **142.7ms**）。

---

## 四、 后续改进方向（面向命中率与 TTFT P90）

1. **收紧 `chat` cell 下的 `RegretGate` 弃权偏置**：排查 `abstain_struct` 高达 8,544 次的原因，避免在 `cheapest_pick=0` 时把淘汰压力单边倾倒进 `chat` 队列（或允许更早进入全局 `PricedFallback` 跨队比价）。
2. **保护 `scan` 下的 `struct` 模板**：降低 `struct` 低共享阈值节点的误判清理，防止 `struct` 桶 P90 从 39ms 退化至 252ms。
3. **利用已有但未接入决策的会话/位置信号**：将 `SessionState.gaps`（个体轮间节奏）、`SessionState.d / L`（受损比例）、`TreeNode.reuse_self / reuse_cross` 与 `recompute / abs_start`（断链重算代价）接入 `RegretGate` 与队内排序，缩小与 `chain_lru` / `opt` 的差距。
