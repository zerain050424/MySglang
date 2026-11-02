# Agent 会话 trace 特征报告：轮间间隔与工具调用

> 数据：tracelab（claude 280,636 轮 / codex 349,649 轮）+ cachewise（9,401 轮）
> 口径：一轮 = 一次模型请求；`gap` = 本轮首个事件 − 上一轮末个事件（请求到请求）
> 脚本：`exp/trace_features.py`（解析 → `exp/data/rounds.parquet`）、`exp/trace_report.py`（统计 + 绘图）

---

## 1. 数据与口径

| 源 | 会话 | 轮 | 触发类型可得 | 工具耗时配对 |
|---|---|---|---|---|
| claude（tracelab） | 4,682 | 280,636 | 100% | **100%**（会话级，result 落在下一轮） |
| codex（tracelab） | 2,638 | 349,649 | 93.8% | 52%（大量 result 无对应 call 事件） |
| cachewise | 78 | 9,401 | 100% | 100% |

两个必须说明的口径问题：

1. **claude 的 `tool_call` 与 `tool_result` 分属相邻两轮**（call 在轮尾、result 在下一轮开头），
   所以工具时长必须**会话级配对**，并把时长记在发起调用的那一轮名下。按轮内配对会把
   claude 的 `tool_sum` 中位数算成 0.04 s（真值 0.13 s，且 100% 可配）。
2. codex 有 20 万条**孤立 tool_result**（对应 call 事件不在 timing_events 里），
   因此 codex 的工具耗时是**下界**，其 `tool/gap` 比值不可直接与 claude 比较。

---

## 2. 结论摘要

| # | 结论 | 关键数字 |
|---|---|---|
| **C1** | 间隔是**双峰**的，不是单峰重尾 | claude 自驱轮 p50 **0.12 s** vs 人工轮 p50 **210 s**，差 **1700×** |
| **C2** | 自驱轮的间隔≈工具执行时间，harness 几乎不留空档 | claude `tool/gap` 中位 **0.92**；残余 `eps` p50 **0**、p90 **0** |
| **C3** | 人工轮占 9–14%，是长尾的唯一来源 | claude 13.7% / codex 9.2% / cachewise 7.3%；`gap>300s` 中 claude 82%、codex 97% 来自人工轮 |
| **C4** | 会话内自相关很弱 | lag-1 ACF：claude 0.087 / codex 0.169 / cachewise 0.158 |
| **C5** | 会话内离散度远大于会话间 | 会话内 CV 中位 **20 / 11 / 4.5**；会话中位间隔跨 p10–p90 仅 0.03→4.2 s |
| **C6** | 真实上下文为 12–15 万 token，前缀复用 ≈1 | claude input p50 **149,927**、p90 543,605；prefix/input **0.9934** |

---

## 3. 轮间间隔：双峰结构

| 源 / 触发 | n | p50 | p90 | p99 | >60 s | >300 s |
|---|---|---|---|---|---|---|
| claude / 自驱（tool_result） | 242,253 | **0.12** | 12.3 | 272 | 4.7% | 0.9% |
| claude / 人工（user_message） | 38,327 | **210.0** | 1035 | 26,985 | 69.1% | 26.6% |
| codex / 自驱 | 295,861 | 2.21 | 11.0 | 114 | 1.5% | 0.1% |
| codex / 人工 | 32,218 | **54.6** | 825 | 19,875 | 48.2% | 21.1% |
| cachewise / 自驱 | 8,716 | 6.46 | 31.8 | 210 | 4.9% | 0.6% |
| cachewise / 人工 | 685 | 9.29 | 61.1 | 579 | 10.2% | 2.3% |

![fig1](figs/fig01_gap_dist.png)
![fig2](figs/fig02_gap_cdf.png)

触发类型占比与长尾占比：

![fig3](figs/fig03_trigger_tail.png)

**读法**：claude 的自驱轮 p50 = 0.12 s —— 工具一返回就立刻再请求；
人工轮 p50 = 210 s（3.5 分钟），p99 达 7.5 小时。两桶相差 3 个数量级，
用单一 lognormal 拟合两桶会把自驱轮算成几十秒（UniCache 的 `(1.81, 1.092)` 就是这种混合的产物）。

---

## 4. 时间预算：机器循环 vs 人工循环

| 源 / 触发 | eps p50 | eps p90 | eps p99 |
|---|---|---|---|
| claude / 自驱 | **0.00** | **0.0** | 0.0 |
| codex / 自驱 | 0.00 | 7.3 | 20.1 |
| cachewise / 自驱 | 5.48 | 26.8 | 204.6 |
| claude / 人工 | 209.7 | 1034 | 26,985 |
| codex / 人工 | 54.6 | 825 | 19,875 |

![fig13](figs/fig13_eps_human.png)

`eps = gap − 上一轮发出的工具耗时`。**claude 的 eps 恒为 0** ⇒ harness 在工具返回的下一毫秒
就发出请求，会话在缓存里的"离场时间"完全等于工具执行时间。
codex 有 0–7 s 的调度残余，cachewise 有 5.5 s（含排队/思考）。
**结论：ε 是 harness 属性，不是一个可跨数据集复用的常数**（我们此前用 cachewise 的 6.21 s 作通用值，
对 claude 类 harness 高估约 50×）。

> 口径限制：claude 有 45%、codex 有 65% 的轮出现 `tool_sum > gap`，因为该轮有多个工具调用、
> 部分结果落在更后面的轮次。因此 `eps` 只能读作"下界意义上的残余"，不宜做精确分解。

---

## 5. 会话内的时间结构

### 5.1 间隔与轮位置无关

![fig8](figs/fig08_gap_by_pos.png)

claude 中位间隔在各轮位置恒为 0.1–0.3 s，codex 从 1.5 s 缓慢升到 3.0 s（后期工具更重）。
**没有"会话越到后面越慢"的现象** ⇒ 会话可用平稳的轮类型混合近似。

### 5.2 会话内自相关很弱

![fig9](figs/fig09_acf.png)

lag-1：claude 0.087、codex 0.169、cachewise 0.158；lag≥3 后基本落在 ±0.1 内。
**推论**：按轮独立采样间隔（我们大规模实验的做法）是可接受的近似，不会系统性低估"连续冷"。

### 5.3 会话内离散度 ≫ 会话间差异

![fig10](figs/fig10_session_hetero.png)

| 源 | 会话数 | 会话中位间隔 p10/p50/p90 | 会话内 CV 中位 |
|---|---|---|---|
| claude | 3,356 | 0.03 / **0.11** / 4.2 s | **19.9** |
| codex | 2,396 | 0.02 / 1.16 / 10.4 s | 11.3 |
| cachewise | 77 | 4.8 / 6.4 / 9.4 s | 4.5 |

CV 20 说明"会话内中位间隔 0.1 s、但个别间隔上千秒"是常态 —— 间隔由**每轮的类型**（自驱/人工）
决定，而不是由"会话性格"决定。

> **对 P_i 设计的直接影响**：不能用一个会话的历史间隔去估计"它下一次何时回来"的 hazard；
> 应该用「下一轮是自驱还是人工」的条件分布。人工轮的触发在会话内近似随机（13.7%），
> 这才是 P_i 真正需要建模的不确定性来源。

---

## 6. 工具调用特征

### 6.1 频次：两个生态

![fig4](figs/fig04_tool_freq.png)

| claude | n | codex | n | cachewise | n |
|---|---|---|---|---|---|
| Bash | 158,610 | exec_command | 251,630 | Bash | 4,049 |
| Read | 63,984 | write_stdin | 75,975 | Edit | 2,002 |
| Edit | 36,713 | exec | 46,754 | Read | 1,912 |
| TaskUpdate | 8,853 | apply_patch | 32,332 | TodoWrite | 1,650 |
| Write | 8,625 | wait | 11,404 | Grep | 643 |
| WebFetch | 3,636 | shell_command | 7,624 | Write | 396 |
| Agent | 2,546 | shell | 5,416 | ExitPlanMode | 226 |
| WebSearch | 1,710 | update_plan | 3,473 | MultiEdit | 223 |

claude 与 cachewise（都是 Claude Code 系列）工具集高度重叠（Bash/Read/Edit/Write/Grep + TodoWrite），
codex 是另一套（exec_command/write_stdin/apply_patch）。**跨 harness 合并工具表时，
同一个语义动作会分裂成 2–3 个名字**（Bash vs exec_command vs shell_command）。

### 6.2 时长：跨 5 个数量级

![fig5](figs/fig05_tool_survival.png)

| 工具 | n | p50 | p90 | p99 |
|---|---|---|---|---|
| Read | 65,818 | 0.03 | 0.25 | 3.1 |
| Edit | 38,661 | 0.07 | 2.2 | 64.5 |
| exec_command | 157,318 | 0.24 | 1.2 | 30.2 |
| Bash | 162,256 | 0.33 | 20.4 | 289.5 |
| write_stdin | 54,766 | **5.0** | 42.6 | 300 |
| WebFetch | 3,645 | 6.3 | 19.8 | 150 |
| WebSearch | 1,686 | 17.1 | 76.1 | 232 |
| Agent | 2,536 | **30.3** | 177 | 870 |
| AskUserQuestion | 1,400 | **74.9** | 550 | 3020 |

尾部差别极大：Read 的 p99 只有 3 s，而 `write_stdin`（等交互终端）与 `Agent`（起子会话）
的 p50 就是 5–30 s。**分布形状本身是多峰的**（快工具的 spike + 慢工具的重尾），
这也是为什么用分位表而不是 lognormal。

### 6.3 每轮工具数：以单工具为主，长尾到 10+

![fig6](figs/fig06_tools_per_round.png)

| 源 | 1 | 2 | 3 | 4 | ≥5 |
|---|---|---|---|---|---|
| claude | 89.5% | 7.1% | 2.2% | 0.8% | 0.4% |
| codex | 79.3% | 7.2% | 8.5% | 4.1% | 0.9% |
| cachewise | 93.5% | 4.7% | 1.3% | 0.4% | 0.2% |

### 6.4 间隔与工具耗时：claude/codex 强相关

![fig7](figs/fig07_gap_vs_toolsum.png)

| 源 | n | `tool/gap` 中位 | log-log 相关 |
|---|---|---|---|
| claude | 238,572 | 0.92 | 0.32 |
| codex* | 158,707 | 1.45 | — |
| cachewise | 8,702 | **0.017** | 0.09 |

claude 的间隔几乎完全由工具耗时决定（比值中位 0.92）；cachewise 相反（1.7%）——
因为 cachewise 的 `gap` 是 llm_call→llm_call，工具耗时已被剔除，真正的空档在 `eps` 里（5.5 s）。
**同一指标在两个数据集里语义不同，不能混用。**

---

## 7. 会话形态与 token 结构

| 源 | 会话 | 会话时长 p50 / p90 | 轮数 p50 / p90 | 空闲占比中位 |
|---|---|---|---|---|
| claude | 4,682 | 3.1 min / 4.1 h | 12 / 113 | 17% |
| codex | 2,638 | 10.0 min / 6.8 h | 23 / 246 | 59% |
| cachewise | 78 | 36.5 min / 20 h | 64 / 303 | 62% |

![fig11](figs/fig11_concurrency.png)

并发会话数（会话跨度 = 首个到最后一个请求）：

| 源 | 观测窗口 | peak | p99 | p50 |
|---|---|---|---|---|
| claude | 2025-10-03 ~ 2026-07-24 | **247** | 148 | 5 |
| codex | 2025-09-23 ~ 2026-07-24 | 191 | 53 | 5 |

真实部署下同时活跃的 agent 会话是 **百级**，而不是我们 bench 里的 12 条。

![fig12](figs/fig12_tokens.png)

| 源 | input_tokens p50 / p90 / p99 | prefix/input 中位 | 每轮新增 token 中位 |
|---|---|---|---|
| claude | **149,927** / 543,605 / 925,226 | **0.9934** | 993 |
| codex | 123,900 / 208,962 / 240,111 | 0.9902 | 1,086 |

---

## 8. 对 UniCache 实验的五条直接影响

1. **间隔模型必须分桶，且 ε 按 harness 参数化。**
   `gap = 工具耗时 + ε`，ε 实测：claude **0 s**、codex 0–7 s、cachewise 5.5 s。
   当前 `tool_durations.json` 用 cachewise 的 ε（6.21 s）作通用值，
   对 claude 类会话把自驱轮间隔高估约 50×（0.12 → 6.3 s）。
2. **人工轮是长尾的唯一来源，必须单独建模。**
   `gap>300 s` 中 claude 82%、codex 97% 由人工轮触发；其 p50 为 210 s / 55 s / 9 s。
   占轮数 7–14%，但决定了全部冷会话。
3. **P_i 不能用会话历史 hazard。**
   会话内 CV 11–20 vs 会话间中位间隔差异仅 1 个数量级 ⇒ 间隔由轮类型决定。
   P_i 应建模为「下一轮为人工轮的概率 × 人工等待分布」，其中人工概率 ≈ 0.09–0.14。
4. **按轮独立采样是安全的。** lag-1 ACF ≤ 0.17，我们的 i.i.d. 采样不会系统性低估冷会话聚集。
5. **bench 的规模与真实差 1–2 个数量级。**
   单轮上下文 150k（我们 2.4k）、并发会话百级（我们 12）、前缀复用 0.99（我们接近 1）。
   级联实验的结论外推到真实部署时，必须承认这一差距（详见 `exp/LargeScale.md` 的规模扫描）。

---

## 9. 局限

1. codex 有 20 万条孤立 tool_result，工具耗时为**下界**，其 `tool/gap` 比值偏高（1.45）。
2. `eps` 只在下界意义下可读（45%/65% 的轮 `tool_sum > gap`，多工具跨轮所致）。
3. cachewise 只有 78 个可用会话，其人工轮只有 685 个样本，p90 以上不稳定。
4. tracelab 无正文，无法核对工具参数与失败重试；`is_error` 字段未纳入分析。
5. claude 的 4,682 个会话中 p50 只有 12 轮，长会话（p90 113 轮）才是缓存压力的来源，
   分位数聚合会掩盖这部分。

---

## 附：产物

```
exp/data/rounds.parquet        676,880 轮（30 MB，含 gap / eps / tool_sum / token 列）
exp/agent_gap_stats.json       全部统计表
exp/figs/fig01..fig13.png      13 张图
exp/trace_features.py          解析脚本（--reparse 重建 parquet）
exp/trace_report.py            统计 + 绘图
```
