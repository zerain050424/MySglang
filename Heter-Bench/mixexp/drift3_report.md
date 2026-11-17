# drift3 结果：构成漂移下 LRU / heter_cache / OPT（engine FCFS · page=8 · seed=42）

> 时间：2026-11-17 · 分支 `W1` · 单 seed（42），三段主导漂移
> 载体：`sim/` 的 `EngineSimulator`（FCFS 成批 prefill），跑生产 `HeterCacheRadixCache`
> 数据：`Heter-Bench/TaskData/drift3/`（15,309 会话 / 125 MB，`build_drift3_data.py` 生成）
> 设计 + gate：`Heter-Bench/mixexp/drift3_design.md`（gate 6/6 PASS）
> 结论速览：**背景越强，heter 的收益单调放大**（+0.13 → +1.49 → +3.95pp），收益几乎全部来自 chat 相位（+2.1 → +5.4 → +8.1pp）；**漂移本身很贵**：chat 相位稳态比同构成不切换低 16~25pp；**路径依赖可忽略**（逆序差 ≤0.2pp）。

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| 机器 | bitahub 容器 `bitahub-a21813790891307008572215`（13 vCPU / 503 GB，纯 CPU 仿真） |
| 代码 | `/root/sg-unicache` @ `W1`（含 `arrival.mode=segments`、`evict` 平票确定化） |
| 解释器 | `/root/sl_old/bin/python`（3.12 + torch） |
| 服务时长 | `Uni-Bench/result/tailshot/calib_lru.json`：prefill 8,319.6 tok/s，decode 40 tok/s |
| 驱动 | `Heter-Bench/mixexp/tools/drift3_run.py --plan bg_{05,10,20}.json --seed 42 --stationary --reverse` |

## 2. 参数设置

时间轴：`warmup 300s → P1 1200s → P2 1200s → P3 1200s → cooldown 300s`（全程 4,200s，≈6.9M prompt token）。

| 相位 | 主导 | 构成目标（bg=1×） |
|---|---|---|
| warmup / cooldown | — | 四类各 25% |
| P1 | **struct** | struct 55% + chat 10% + agentic 10% + **trans 25%** |
| P2 | **chat**（ShareGPT） | chat 55% + struct 10% + agentic 10% + trans 25% |
| P3 | **agentic**（AgentBank_alfred） | agentic 55% + struct 10% + chat 10% + trans 25% |

- 自变量：**trans 背景注入速率** × {0.5, 1, 2}（前景 λ 恒定，只加背景 ⇒ 背景份额 14% / 25% / 40%）；
- 常量：capacity 198,000、page_size **8（三臂统一）**、FCFS（`max_prefill_tokens=16384`，chunked 关）、seed 42；
- 对照：stationary（同构成不切换，取后 300s 稳态）、逆序（agentic→chat→struct）；
- 按评审意见：VD 视野消融本轮**不做**。

## 3. 结果

### 3.1 总体命中率与 OPT 上界

| 背景 | LRU | heter_cache | OPT（上界） | heter−LRU | OPT−LRU | OPT−heter |
|---|---|---|---|---|---|---|
| 0.5× | 76.92% | 77.05% | **81.50%** | +0.13pp | +4.58pp | +4.45pp |
| 1.0× | 65.20% | 66.69% | **71.65%** | +1.49pp | +6.45pp | +4.96pp |
| 2.0× | 47.49% | 51.43% | **57.38%** | **+3.95pp** | +9.89pp | +5.94pp |

### 3.2 分相位（hit / 相位末 300s 稳态）

| 背景 | 臂 | P1 struct | P2 chat | P3 agentic |
|---|---|---|---|---|
| 0.5× | LRU | 0.832 / 0.796 | 0.667 / 0.614 | 0.784 / 0.806 |
| | heter | 0.825 / 0.787 | **0.688 / 0.658** | 0.783 / 0.801 |
| | OPT | 0.838 / 0.798 | 0.771 / 0.803 | 0.839 / 0.835 |
| 1.0× | LRU | 0.724 / 0.683 | 0.523 / 0.485 | 0.681 / 0.716 |
| | heter | 0.724 / 0.678 | **0.577 / 0.567** | 0.690 / 0.719 |
| | OPT | 0.738 / 0.691 | 0.660 / 0.718 | 0.756 / 0.759 |
| 2.0× | LRU | 0.569 / 0.523 | 0.323 / 0.305 | 0.507 / 0.546 |
| | heter | 0.583 / 0.534 | **0.404 / 0.387** | 0.548 / 0.570 |
| | OPT | 0.599 / 0.547 | 0.506 / 0.560 | 0.623 / 0.625 |

### 3.3 漂移代价（drift 稳态 − 同构成 stationary 稳态，heter_cache）

| 背景 | P1 struct | P2 chat | P3 agentic |
|---|---|---|---|
| 0.5× | −7.5pp | **−16.1pp** | −3.2pp |
| 1.0× | −7.8pp | **−18.5pp** | −4.8pp |
| 2.0× | −8.9pp | **−24.9pp** | −6.3pp |

（stationary 对照取 run 的后 300s 稳态，剔除冷启动；同 λ、同构成、不切换。）

### 3.4 对照

| 背景 | 正向 LRU / heter | 逆序 LRU / heter | 差 |
|---|---|---|---|
| 0.5× | 0.7692 / 0.7705 | 0.7705 / 0.7721 | ≤ +0.16pp |
| 1.0× | 0.6520 / 0.6669 | 0.6513 / 0.6683 | ≤ +0.14pp |
| 2.0× | 0.4749 / 0.5143 | 0.4748 / 0.5133 | ≤ +0.10pp |

## 4. 对比分析

### 4.1 背景越强，heter 的收益越大——假设被证实，且几乎全在 chat 相位

| 背景 | heter−LRU（overall） | P1 struct | P2 chat | P3 agentic |
|---|---|---|---|---|
| 0.5× | +0.13pp | −0.70pp | **+2.09pp** | −0.05pp |
| 1.0× | +1.49pp | −0.09pp | **+5.42pp** | +0.93pp |
| 2.0× | +3.95pp | +1.42pp | **+8.12pp** | +4.00pp |

单调性成立：背景越大（池周转越快），按"未来复用价值"保护越划算；而在低背景（0.5×）下，
池子宽裕到 LRU 的 recency 近似最优，heter 的保护反而是净负担（P1 甚至 −0.70pp）。
**chat 是唯一在所有背景档都稳定受益的相位**——它的共享前缀最长、被 recency 误杀的损失最大，
与 exp3 静态 cell 上的结论一致（chat cell 上 OPT 余量 22pp）。

### 4.2 漂移代价远大于构成效应，且集中的 chat 相位

P2(chat) 的稳态比"同构成不切换"低 **16~25pp**，是全场最大损失；P1(struct) −7.5~−8.9pp、
P3(agentic) −3.2~−6.3pp。机制解释：前一个相位（struct / trans 主导）把 chat 赖以命中的
**跨会话长前缀**挤掉了，chat 相位开始时这些块需要重新积累——而 chat 的单会话跨度 ≈500s，
1200s 的相位里只有约 2 代会话，恢复窗口不够。这也是 pos 4.3 里 T90 指标不敏感的原因
（主导类在自己的相位里始终处于"重新积累"状态，没有明显的"恢复到稳态"平台）。

推论（对策略设计的含义）：只按"下一步大盘命中率"调策略不够，**相位切换的瞬态**才是
损失主项；能跨相位保住长前缀的策略（正是 heter 的共享保护段）才有价值——这解释了
为什么 heter 的收益集中在 chat、且随背景增强。

### 4.3 OPT 上界随背景扩大：+4.45 → +5.94pp（相对 heter）、+4.58 → +9.89pp（相对 LRU）

高背景下选择余量被放大：池子越紧，"该淘汰谁"的信息价值越高。同时 OPT 在 P2 的稳态
（0.718 @1×、0.560 @2×）明显高于 heter（0.567 / 0.387）——chat 相位是策略差距最大的地方。

### 4.4 路径依赖可忽略

逆序（agentic→chat→struct）与正向的 overall 差 ≤0.2pp，与 heter−LRU 的差（0.13~3.95pp）
相比是噪声级。**结论不依赖相位顺序**（至少在这套构成与容量下）。

### 4.5 T90 指标本轮不好用（诚实记录）

用"主导类 60s 滚动命中率首次进入稳态−2pp 并连续 3 窗"定义 T90，结果多数相位是 0 或
360s、chat 在 OPT 下是 660s。原因是相位内构成同质、真正的构成变化只发生在边界一个瞬间，
而"稳态"取相位末 300s——切分点太粗。下一轮改成**逐相位用 60s 分箱画曲线**并直接报
"边界后第 1/2/3 个 100s 的命中率"，比单一 T90 更有信息量。

## 5. 局限

1. **单 seed**（42）：所有差异的置信度未做配对检验；tips：`drift3_run.py --seed 43` 即可复现第二轮。
2. agentic 每相位只有 ~28 个会话（51k token/会话），构成份额 σ≈10pp（gate 已按 2σ 容差）。
3. page_size=8 与生产默认 1 不同（OPT 需要）；四臂统一保证可比，但 page 粒度会翻转 heter 的符号。
4. stationary 对照是**独立 run**（从 0 冷启动），本文只取其末 300s 稳态；严格口径应在同一 run 内做 A/B。
5. 不含 TTFT / 重算税，只有命中率；regret 账本本轮未开。

## 6. 产物

| 路径 | 内容 |
|---|---|
| `/root/sim_runs/drift3/all_s42.json` | 三个背景档 × 三臂 × 分相位（hit/稳态/淘汰/分类型） |
| `/root/sim_runs/drift3/bg_{0.5,1,2}x_s42.json` | 单背景档明细（含 stationary / reverse） |
| `/root/sim_runs/drift3/stationary_steady.json` | stationary 对照（all + 后 300s 稳态） |
| `/root/sim_runs/drift3/use_map_*_it*.json` | OPT 各轮 use map |
| `/root/sim_runs/drift3_matrix.log` | 运行日志 |
