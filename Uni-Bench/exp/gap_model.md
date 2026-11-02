# Agentic 会话间隔的分解模型（工具感知）

> 数据来源：`cachewise-coding-traces/parsed_traces`（Claude Code 真实用户，94 会话，
> 20,634 llm_call / 11,617 tool_call，带真实时间戳与 `execution_time_ms`）
> 分析可复现：见本文 §5 的两个统计口径。

## 1. 实测：间隔不是单一分布，按「有没有工具」分成两桶

| 桶 | n | p50 | p90 | lognormal (mu, sigma) |
|---|---|---|---|---|
| **无工具**（模型答完，等人/思考） | 661（6.9%） | **49.7 s** | 248.5 s | **4.09 / 1.58** |
| **有工具**（总间隔） | 8,938（93.1%） | **7.85 s** | 42.8 s | 2.34 / 1.11 |
| 有工具（扣掉工具耗时后的残余） | 8,938 | 6.22 s | 20.5 s | **2.02 / 0.87** |

两桶中位数差 **6.3 倍**。把两类混在一起拟合（我们之前对 cachewise 全局拟合得到 2.47/1.21），
得到的是一个"两头都不对"的折中值。

每轮工具调用数分布：`0 → 6.9%`、`1 → 87.0%`、`2 → 4.8%`、`≥3 → 1.9%`。

## 2. 实测：逐工具耗时差异巨大，必须分开

| 工具 | n | p50 | p90 | mu | sigma |
|---|---|---|---|---|---|
| Bash | 4,048 | 1.08 s | **31.0 s** | 0.65 | **2.41** |
| Edit | 2,002 | 0.08 s | 9.93 s | −1.19 | 2.32 |
| Read | 1,912 | 0.06 s | 0.41 s | −2.37 | 1.42 |
| TodoWrite | 1,649 | 0.05 s | 0.12 s | −2.58 | 1.39 |
| Grep | 643 | 0.08 s | 0.22 s | −2.30 | 0.99 |
| Write | 395 | 0.08 s | **26.9 s** | −0.89 | 2.68 |
| ExitPlanMode | 226 | **24.8 s** | 95.2 s | 3.37 | 1.56 |
| WebFetch | 93 | 11.6 s | 43.0 s | 2.73 | 1.02 |
| WebSearch | 63 | **36.2 s** | 57.1 s | 3.61 | 0.43 |

三类明显分层：

```
快工具  Read / Grep / Glob / TodoWrite / BashOutput     p50 0.05~0.08 s   → 对间隔几乎无影响
中工具  Bash / Edit / Write                             p50 0.08~1.1 s，但 p90 27~31 s（σ 2.3~2.7）
慢工具  ExitPlanMode / WebFetch / WebSearch             p50 11~36 s
```

## 3. 对已知常数的解释

三个"看起来打架"的常数，其实是同一模型在三类场景下的取值：

| 场景 | 桶构成 | 等效 (mu, sigma) | 对应常数 |
|---|---|---|---|
| ALFWorld（alfred）：每轮都有环境动作，耗时 ~0.1 s，无人工等待 | 几乎全在"有工具"，残余主导 | ~2.0 / 0.9 | **UniCache agentic 1.81 / 1.092** |
| Claude Code（人在环）：93% 有工具 + 6.9% 等人 | 混合 | 2.34 / 1.11（综合）→ 全局拟合 2.47 / 1.21 | 实测 |
| 纯人工等待（无工具轮） | 100% 无工具 | 4.09 / 1.58 | 百炼 text 4.78 / 1.28（更偏人类节奏） |

⇒ **不是"谁错了"，而是缺一个状态维度**。用户的拆分正是补上这一维。

## 4. 模型定义

```
gap(t → t+1) = tool_phase(t) + think_phase

tool_phase(t) = Σ_{k ∈ tools(t)} dur(tool_k)
                若 tools(t) 为空 → 0
dur(tool_k)   ~ 逐工具经验分布（§2 表；按名称查，未知工具回落到"API 工具" ≈ 1 s）
think_phase   ~ lognormal(2.02, 0.87)          （来自"扣掉工具耗时后的残余"）
                若 tools(t) 为空 → think_phase ~ lognormal(4.09, 1.58)（人工等待桶）
```

三个实现要点：

1. **工具耗时用经验分布采样，不要用 lognormal 拟合**。Bash/Edit/Write 的 σ 是 2.3~2.7，
   lognormal 会显著低估 `p90`（如 Bash p90 实测 31 s，lognormal(0.65, 2.41) 的 p90 ≈ 30 s，
   但单点采样时经验分位数更稳、且天然有上界）。
2. **并行工具求和**：一轮里可能有 2~5 个工具（4.8% + 1.9%），`tool_phase` 取和；
   实测 `corr(log gap, log tool_sum) = 0.644`，说明加性假设成立但不是全部。
3. **工具占比要诚实**：`tool_sum / gap` 的中位数只有 **0.02** —— 大多数轮次里工具是"快工具"，
   加性项几乎不改变间隔。分解的收益集中在两处：
   - **长尾**：慢工具把 p90/p99 拉长（单 lognormal 拟合不出来）；
   - **状态判断**：知道"此刻有 Bash 在飞行"，就能判断"未来 10 s 内不可能回来"。

## 5. 与 P 的关系：条件化的 P

现在的 `P(idle) = 1 − CDF(idle)` 用的是"下一次间隔"的无条件分布。有了工具状态后应改成：

```
P(idle | state) =
    S_tool(idle)                      若上一轮有工具在飞行（state = tool:k）
    S_nothink(idle)                   若上一轮无工具（在等人）
其中 S_tool 用 tool_phase 的剩余时间分布 + think_phase 卷积
```

效果（用实测参数算的无条件混合 vs 单 l 模型）：

| idle | 混合（条件化） | 单一 lognormal(2.47,1.21) | UniCache(1.81,1.092) |
|---|---|---|---|
| 10 s | 0.538 | 0.555 | 0.326 |
| 30 s | 0.204 | 0.221 | 0.073 |
| 90 s | 0.051 | 0.047 | 0.007 |
| 300 s | **0.012** | 0.004 | 0.0002 |

无条件混合与单一拟合差别不大（因为"无工具"只占 6.9%）；**真正的差别在长尾和状态化**：
`P(300 s)` 差 3 倍，而"正卡在 Slow 工具里"的那一段，单模型会低估 P 一个数量级。

## 6. 在 Uni-Bench 里怎么落地

| 步骤 | 内容 |
|---|---|
| **D1 数据侧** | 给每条 session 的每轮标 `tools: ["Bash", ...]`。APIGen-MT 从 `function_call` 消息提取工具名（它带 `tools` 定义）；alfred 没有 API 工具，标成对环境动作 `["env_action"]`（≈0.1 s） |
| **D2 落表** | `exp/tool_durations.json`：来自 cachewise §2 的经验分位数（每工具 p10/p50/p90/p99 + 采样用原始样本） |
| **D3 driver** | `gap_mode: tool_aware`：`gap = Σ dur(tool) + lognorm(2.02, 0.87)`；无工具轮用 `lognorm(4.09, 1.58)`；未知工具回落 1 s |
| **D4 验证** | 生成的间隔分布要与 cachewise 同桶的 p50/p90/p99 对齐（容差 15%），且"每轮工具数分布"一致 |

## 7. 风险

1. **工具名不匹配**：APIGen-MT 的工具是业务 API（`search_flights` 之类），cachewise 是编码工具。
   需要一张映射表或统一回落到 `api_call ≈ 1 s`；否则会把航班查询当成 Bash 跑 30 s。
2. **工具内部长尾**：Bash p90 31 s 里有编译/测试，属于真实长尾，不能截断 —— 截断会系统性低估
   "长 idle"，而长 idle 正是我们关心的淘汰场景。
3. **error/retry**：cachewise 里 `is_error` 存在但没有重试链，重试会额外增加间隔，当前模型不含。
4. **人工等待占比**：cachewise 只有 6.9% 无工具轮，但真实 to-C 场景（百炼 text）是 100% 人工节奏。
   我们的 agentic 负载（alfred/APIGenMT）更接近"自驱"，所以 6.9% 这个比例**不能直接搬到 bench**，
   要按目标场景单独设 `π_nothink`。

## 8. 结论

用户提出的拆法**成立且有数据支撑**：

- 两桶中位差 6.3 倍（49.7 s vs 7.85 s），混拟合必然失真；
- 逐工具 p50 从 0.05 s 到 36 s，跨 3 个数量级，必须按工具名查表；
- 但这套分解的收益**不在中位数**（多数轮工具只占 2%），而在**长尾**与**"工具在飞行中"的状态判断**。

⇒ 建议实现顺序：先 D1+D3（能生成工具感知的间隔），再 D4 对齐验证，最后才把 `P` 改成条件化版本。

---

# S1 落地结果（2026-10-27）

## 产物

```
exp/build_tool_durations.py     # 可复现：tracelab + cachewise 合并，逆 CDF 分位表
exp/tool_durations.json         # 127 KB，552,899 次工具调用 / 97 个工具
```

数据量从 cachewise 的 11.6k 提升到 **552,899**（tracelab 541k + cachewise 11.6k），
每个工具输出 `{n, class, min, p10/p50/p90/p99, max, q[0..100], sources}`，
`q` 供下游按逆 CDF 线性插值采样（文件小、且不丢分布形状）。

## 规模最大的工具

| 工具 | class | n | p50 | p90 | p99 |
|---|---|---|---|---|---|
| Bash | medium | 162,256 | 0.33 s | 20.38 | 289.5 |
| exec_command | medium | 157,318 | 0.24 | 1.20 | 30.2 |
| Read | fast | 65,818 | 0.03 | 0.25 | 3.1 |
| write_stdin | interactive | 54,766 | 5.00 | 42.58 | 300.0 |
| Edit | fast | 38,661 | 0.07 | 2.18 | 64.5 |
| WebFetch | medium | 3,645 | 6.26 | 19.78 | 150.5 |
| **Agent** | subagent | 2,536 | **30.25** | 177.4 | 869.6 |
| WebSearch | slow | 1,686 | 17.14 | 76.1 | 232.4 |
| **AskUserQuestion** | human | 1,400 | **74.85** | 549.6 | 3019.8 |
| ExitPlanMode | human | 863 | 69.22 | 474.4 | 2062.4 |
| TaskOutput | subagent | 636 | **117.08** | 420.1 | 600.3 |

## 分类汇总

| class | 工具数 | 调用数 | 含义 |
|---|---|---|---|
| medium | 31 | 323,384 | 真机器耗时（Bash / exec_command / WebFetch …） |
| fast | 53 | 165,912 | < 0.2 s，对间隔几乎无影响 |
| interactive | 2 | 55,878 | `write_stdin`（p50 5 s）/ Monitor，等终端交互 |
| **human** | 3 | 2,522 | `AskUserQuestion` / `ExitPlanMode` / `request_user_input` —— **是等用户** |
| slow | 5 | 1,772 | WebSearch / Task 等 |

## 对模型定义的一处修正

1. **`human` 类不能当工具耗时**。`AskUserQuestion` p50 74.9 s、`ExitPlanMode` p50 69.2 s
   本质是"等用户回答/批准"，应并入**无工具桶**（人工等待 mu=4.09/1.58），
   否则会把人工等待当机器耗时、把 P 算错。此前基于 cachewise 的结论里
   `ExitPlanMode` 被当成"慢工具"，现予修正。
2. `Agent`（p50 30 s）/ `TaskOutput`（117 s）/ `wait_agent`（96.6 s）按 p50 自动归入 `slow`，
   与其它慢工具一样计入 `Σ dur`，不做特殊建模。

## driver 侧的用法（S2 待实现）

```
gap(t → t+1) = Σ_{k ∈ tools(t), class(k) ∈ {fast, medium, slow, interactive}} dur(k) + ε
              若 tools(t) 为空        → 走无工具桶（人工等待）
              human 类（AskUserQuestion / ExitPlanMode）照常计入 Σ dur —— 它本来就是真实等待

dur(k) ~ 逆 CDF 采样 exp/tool_durations.json:tools[k].q
ε      ~ lognorm(2.02, 0.87)（来自"有工具轮扣掉工具耗时后的残余"）
```

---

# S2 落地结果（2026-10-27）：gap 模型接进 driver

## 产物

| 文件 | 内容 |
|---|---|
| `bench_unicache.py` | 新增 `ToolGapSampler`（约 50 行）；`gap_mode: tool_aware`；`--tool-durations` |
| `exp/build_tool_durations.py` | 除逐工具表外，另输出两个**轮级桶** |
| `exp/tool_durations.json` | 552,899 次工具 / 97 工具 + `buckets.{nothink,eps}` |
| `exp/make_tool_labels.py` | 给 session_data 每轮打 `tools` 标签（apigen / alfred / auto） |

## 两个轮级桶（cachewise 语义，与 driver 一致）

| 桶 | n | p50 | p90 | 含义 |
|---|---|---|---|---|
| `eps` | 8,917 | 6.21 s | 20.43 | 有工具轮：间隔扣掉工具耗时后的残余 |
| `nothink` | 647 | 49.11 s | 215.71 | 无工具轮：模型答完、等用户 |

采样器优先用桶的**经验分位表**（`buckets[k].q`），没有时才退回 lognormal
—— 因为 lognormal 拟合会高估尾部（4.09/1.58 的 p90 算出 452 s，实测只有 248 s）。

## 采样验证（生成 vs 实测）

用 cachewise 的工具名占比构造混合（6.9% 无工具），各采样 6 万次：

| 桶 | 生成 p50 / p90 | cachewise 实测 p50 / p90 | 偏差 |
|---|---|---|---|
| 有工具 | 7.40 / 37.8 | 7.85 / 42.8 | −6% / −12% |
| 无工具 | 47.65 / 217.8 | 49.7 / 248.5 | −4% / −12% |

`p90` 偏低 12% 来自「每轮只采一个工具」的简化（真实有 4.8%+1.9% 的多工具轮）。

**alfred 场景**（每轮一个 `env_action` ≈ 0.1 s）：生成 **p50 = 6.33 s**，UniCache 假设 6.1 s
⇒ 独立复现了那组常数，也说明它就是"自驱 agent + 0.1 s 环境动作"的等效值。

## 标注验证

| 数据 | 抽样 | 有工具的轮 | 提取结果 |
|---|---|---|---|
| alfred（`victim_pool12_C`） | 4 会话 / 104 轮 | **100%** | `env_action` × 104 |
| APIGen-MT（gz 直读） | 100 会话 / 728 轮 | **48.8%** | `get_reservation_details` 121、`get_user_details` 60、`update_reservation_baggages` 35 … |

APIGen-MT 的业务 API 不在工具表里，按 `tool_fallback.api_call = 1.0 s` 回落。

## 用法

```yaml
# tasks config 里开
gap_mode: tool_aware
default_tools: [env_action]        # 数据没标 tools 时的兜底
```

```bash
python bench_unicache.py --tasks-config ... --wait-scale 1 \
  --tool-durations exp/tool_durations.json
```

`wait = Σ 工具耗时(逆CDF采样) + eps(经验分位) × wait_scale`；无工具轮用 `nothink` 桶。

---

## 尾部量化修正（2026-10-27）

**问题**：`q[]` 原为 101 个等距百分位点，逆 CDF 在 `[q99, q100]` 之间线性插值，
而这段跨越 3 个数量级 —— 等于把极端值密度凭空拉高。实测（tracelab+cachewise 全量重算）：

| 对象 | KS | p50 原/采 | p99.9 原/采 |
|---|---|---|---|
| Bash | 0.006 | 0.33/0.33 | **1304 → 3227** |
| exec_command | 0.015 | 0.24/0.24 | **156 → 3020** |
| Read | 0.022 | 0.03/0.03 | **141 → 922** |
| `eps` 桶 | 0.021 | 6.21/6.25 | **566 → 3086** |

中位数完全保真，但 ≥p99 放大 4–19 倍。对本实验是致命混淆：级联的触发条件正是
"最冷会话"（尾部），会凭空制造冷离群。

**修法**：分位表改为 `(p, q)` 双表，p 非均匀：`0..99` 步长 1、`99.1..99.9` 步长 0.1、
`99.91..99.99` 步长 0.01，共 120 个等级；采样器用 `bisect` 定位区间后线性插值。

**修后（同口径线性分位点比较，40 万次采样）**

| 对象 | p99 | p99.9 | p99.95 | p99.99 |
|---|---|---|---|---|
| Bash | 289.8/286.1 | 1303.0/1303.8 | 1911.5/1935.5 | 3061.2/3168.5 |
| Read | 3.14/3.15 | 140.9/139.5 | 394.6/340.6 | 756.7/769.7 |
| exec_command | 30.2/29.6 | 155.3/171.1 | 249.9/254.0 | 1823.5/1826.1 |
| `eps` | 93.0/92.5 | 552.1/547.6 | 1284.8/1272.5 | 2363.1/2289.4 |

**p99–p99.99 最大相对误差 13.7%**（原 4–19 倍），KS 仍 ≤ 0.022。文件 127 KB → 282 KB。

---

## ε 的 harness 依赖（2026-10-27 补充）

`exp/agent_trace_report.md` §4 用 tracelab 的逐轮时间戳重新分解了 `gap = 工具耗时 + ε`：

| harness | ε p50 | ε p90 | 说明 |
|---|---|---|---|
| claude（Claude Code 类） | **0** | **0** | 工具一返回立刻再请求，离场时间 = 工具耗时 |
| codex | 0 | 7.3 s | 少量调度残余 |
| cachewise | 5.48 s | 26.8 s | 含排队/思考 |

当前 `tool_durations.json` 的 ε 桶来自 cachewise（6.21 s），对 claude 类 harness **高估约 50×**
（自驱轮 0.12 s → 6.3 s）。这不是错误而是**保守上界**：用它跑实验等价于假设"每个工具之后
还要再等 6 s"，会让会话看起来更冷。大规模实验 P3 增加第 ④ 个 idle 模型（ε=0）量化这个敏感性。
