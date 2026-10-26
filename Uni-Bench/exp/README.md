# Uni-Bench 实验设计总纲

> 位置：`Uni-Bench/exp/README.md`
> 范围：本目录收录 UniCache(SimpleHybrid) 在 sglang 上的两个受控实验设计。
> 状态：Tailshot 已实现并产数（`result/tailshot*`）；Cascade 已设计并完成首次实测（`result/cascade/R3.*`），完整矩阵待跑。

---

## 0. 资产地图

| 类别 | 路径 | 说明 |
|---|---|---|
| 压测驱动 | `bench_unicache.py` | asyncio + aiohttp，混合负载，逐请求 CSV |
| 起 server | `serve.sh` / `env.sh` / `config.sh` | `POLICY` × `MAX_TOTAL_TOKENS` |
| 数据 | `TaskData/` | alfred(agentic) / apps(random) / Codeparrot(single) / ShareGPT(text) |
| 标定 | `calibrate.py` | 冷启动 prefill 速率 → `ttft = a + b·L` |
| 实验 1 | `tailshot/` + `result/tailshot*` | 已完成的 Tailshot 矩阵 |
| 实验 2 | `exp/`（本文件） | Cascade 设计 + 首次实测 |

硬件基线：RTX 4090 24G / Qwen2.5-7B-Instruct(bf16) / 池实测 35,975 或 73,120 tok（`mem-fraction` 默认 vs 0.79）。`calibrate`: **8319 tok/s ⇒ 0.1202 ms/token**，截距 −82 ms。

---

# 实验 1 · Tailshot（已完成）

## 1.1 目标

定量暴露 **node-level recency 的尾部驱逐缺陷**：多轮长对话在轮间空窗被外来流挤出缓存、尾部（最新 turn）叶子先被驱逐，下一轮到达时整段历史重算 KV，TTFT 恶化。对照组：`simple_hybrid`（UniCache 迁移版：lognormal 复用概率 + alpha 在线平衡 + 两带驱逐）。

## 1.2 设计要点

拆成两个可控变量（而不是直接喂一条混合负载）：

1. **重访间隔**：受害者轮间 `lognormal(mu=3.4012, sigma=0.01)` ⇒ **30.0s**（抖动 ±0.3s）
2. **池周转速率**：`F` = 搅拌流 offered tok/s，判据为 `池容量 / F` vs `30s`

| 档 | F | 周转 `36K/F` | 与 30s 的关系 |
|---|---|---|---|
| control | 0 | ∞ | 无搅拌 |
| F150 | 150 | 240s | 8× 余量 → 无感 |
| F400 | 400 | 90s | 3× 余量 → 局部失效 |
| F1200 | 1200 | 30s | 正好等于重访间隔 → 每轮洗池 |

三状态模型：① `F<F*` 阈下免疫 ② 局部失效 ③ 阈上崩溃（整链重算）。

## 1.3 负载构造

| 流 | 数据 | λ | mu/sigma | max_record |
|---|---|---|---|---|
| 受害者 | `victim_pool.jsonl`（4 条 34 轮 8.3k） | 0.5 | 3.4012 / 0.01 | 4 |
| 搅拌 1 | apps（单轮 random，均值 398 tok） | `F/(2·570)` | 0 / 0 | 88→641 |
| 搅拌 2 | Codeparrot（单轮，均值 748 tok） | 同上 | 0 / 0 | 88→641 |

会话内串行、会话间并发；受害者长占 4 槽，搅拌用余下 28 槽。

## 1.4 指标

CSV 逐请求：
`task / session_id / turn / prompt_tokens / cached_tokens / recompute_tokens / ttft_ms / e2e_ms / gap_since_prev_end_s`

聚合 JSON：
`cache_hit_rate_pct`、`ttft_p50/p90/p99_ms`、`e2e_mean/p99_ms`、`per_task[].cache_hit_rate_pct`、`wall_time_s`、`request_throughput`

## 1.5 实测结果（摘）

小受害者（池 36k）：

| F | LRU 重算率 med | LRU TTFT p50/max | hybrid 重算率 med | hybrid TTFT p50/max |
|---|---|---|---|---|
| 0 | 8.6% | 42 / 73 ms | 8.6% | 39 / 63 ms |
| 400 | 8.5% | 53 / 149 ms | 8.6% | 54 / 86 ms |
| **1200** | **98.1%** | **166 / 307 ms** | **8.8%** | **54 / 119 ms** |

大受害者（池 73k，F2000 为全程搅拌）：

| F | LRU 重算率 | LRU TTFT p50/p99/max | hybrid 重算率 | hybrid TTFT p50/p99/max |
|---|---|---|---|---|
| 1200 | 6.3% | 64 / 99 / 125 ms | 6.3% | 68 / 140 / 156 ms |
| **2000** | **99.4%** | **495 / 2043 / 2340 ms** | 6.1% | 78 / 164 / 339 ms |

结论：LRU 在 `池周转 ≈ 重访间隔` 处阈值型崩溃；hybrid 在同位置基本不动。

## 1.6 已知陷阱（引用结果前必读）

1. `MAX_TOTAL_TOKENS=100000` 实际未生效（实际 35,975）⇒ 阈值与卡绑定
2. 全局 `cache_hit_rate_pct` / 全局 TTFT 分位被搅拌流污染，只有 `per_task` + 受害者子集能下结论
3. 只有 4 条受害者 / 52 个有效轮 ⇒ `p99` ≈ `max`，尾部显著性有限
4. `calibrate` 负截距使小长度段低估 ~30%，仅适合量级判断
5. `gap_since_prev_end_s` = 本轮完成 − 上轮完成，含本轮耗时，只能做有效性自检
6. `F` 是 offered 而非实际写入（实测 LRU 65.4 / 213.6 / 792.0 at F=150/400/1200）
7. `simple_hybrid` 必须同时给 `--radix-eviction-policy simple_hybrid` 与 `--enable-simple-hybrid`
8. `max_record` 取文件前 N 行而非随机抽样（Codeparrot 前 88 行均值高 26%）

---

# 实验 2 · Cascade（设计 + 首次实测）

## 2.1 动机：Tailshot 结构上看不到级联

用现有 CSV 复核（victim=4 条 alfred，34 轮，`cached_tokens≤40 且 turn>1` 记为 wipe）：

| run | wipe 数 | 按会话分布 | HHI |
|---|---|---|---|
| `simple_hybrid_F1200` | 0 | — | — |
| `simple_hybrid_F2000` | 0 | — | — |
| `lru_F2000` | 73 | 20 / 18 / 18 / 17 | **0.251** |

`HHI = Σ(share_i²)`，`1/N = 0.25` ⇒ 完全均摊。原因：Tailshot 给所有受害者同一个 30s，`last_access_time` 几乎相同 ⇒ 分数并列 ⇒ `min()` 事实轮转。

> **结论：对称 idle 把这个缺陷结构性地掩盖了。打破 idle 对称性才是本实验的核心。**

## 2.2 假设与预注册判据

| 编号 | 假设 | 成立应观察到 | 被推翻 |
|---|---|---|---|
| **H1** | 异构 idle 下冷会话整链被抽干，损失集中 | `HHI≥0.5`、`WipeRate≥0.3`、`LossQuantum≥0.8`、`SimultaneityIndex≥0.85` | `HHI≈1/N` 且 wipe 均匀 |
| **H2** | 与 idle 水平无关，只与异质度有关 | `HHI(R2,全90s)≤0.4` 且 `HHI(R3,同μ/σ)≥0.5`；`HHI(R4)≥HHI(R3)` | 两组 HHI 相同 ⇒ 归因错误 |
| **H3'** | 尾部由"最大单会话重算"驱动，而非总重算量 | `Spearman(p99档, max_session_recompute)>0.9`；run 内 TTFT max 落在 wipe 后 ≤1 turn | 两者无相关 |

预注册接受条件（跑前写死）：

```
R1: HHI ≤ 0.40 且 WipeRate ≤ 0.05        ← 仪表校验，不过则实验作废
R3: HHI ≥ 0.50 且 WipeRate ≥ 0.30 且 LossQuantum ≥ 0.8 且 SimultaneityIndex ≥ 0.85
```

## 2.3 会话时间轴：burst → pause → resume

```
t=0.00   12 victim 到达（组内 0.05s 错开）
t=0~24   turn1..20，gap=1.0s（fixed）        → 链长 4,533 tok/条
t=24.0   屏障：最后一条越过 turn20 → 释放 stirrer
+10s     H 组（P=10）resume turn21..26
+30s     W 组（P=30）resume
+90s     C 组（P=90）resume   ← 主观测点
```

- 只有 resume 段（turn21..26）进主分析；burst 段是全热内部对照
- 每格墙钟 ≈ `20 + P_max + 6 ≈ 2 min`，与"长链"解耦（链长由轮数决定，不由墙钟决定）

### 三条流的到达律

| 流 | 到达律 | 基准 | 理由 |
|---|---|---|---|
| victim(12) | 确定性 `0.05·i`，`gap_mode=fixed` `gap=1s` | 事件屏障 | 三组必须同相位；Poisson 会打散 pause |
| stirrer | Poisson（沿用 λ） | 屏障释放后 | 压力只作用在 pause 内，保住 burst 对照 |
| resume | 由 `pause_s` 决定 | 屏障 + P | 顺序由 P 严格保证 |

屏障用 `asyncio` 事件（计数版），不用墙钟：每轮时延随 prompt 增长（0.1→0.6s），20 轮漂移 5~8s。

并发：`concurrency=32`，victim 长占 12 槽，stirrer 峰值并发 ≈0.5，不排队。不可把 concurrency 降到 12，否则 stirrer 排队、`U` 不可控。

## 2.4 压力预算（决定 F）

```
burst 结束驻留 = 12 × 4,533 = 54,396 tok = 池 73,120 的 74%   ⇒ 无淘汰 ✓
```

有效写入比 ≈0.65（现有实测），90s 窗口：

| F | U@90s | 摊薄每会话损失 | 损失率 | 链等价 |
|---|---|---|---|---|
| 150 | 8.8k | 0.73k | 16% | 1.9 |
| **300** | **17.5k** | **1.46k** | **32%** | **3.9** ← 主格 |
| 450 | 26.3k | 2.19k | 48% | 5.8 |
| 1200 | 70k | 5.8k | 129% | 15.5 ❌ 全员打穿 |
| 2000 | 117k | 9.7k | 215% | 26 ❌ 无判别力 |

判别区要求 `2·chain ≤ U ≤ 6·chain` ⇒ **F∈[150,450]**。stirrer `λ = F/(2×570)`。

> 不能沿用 Tailshot 的 F 档：那是 4×8.3k / 36k 池的口径。

**实际取值（2026-10-26 实施）**：victim 从 `AgentBank_alfred_scale_5000` 切 26 轮、
链长 ~2,470（比上表假设的 4.5k 短），故改用默认池 35,975：

```
12 × 2,470 = 29,672 tok = 池的 82%   ⇒ burst 段不淘汰（实测 burst wipe = 0）
搅拌 F_eff ≈ 126 tok/s × 90s ≈ 11.4k ⇒ 约 4.6 条链的淘汰量
```

上表 73k / F=300 的组合对应"4.5k 链 + 12 条"的假设，实施时按实际链长重算。

## 2.5 矩阵：5 格 × 3 repeats

| # | `P`(H/W/C) | F | 策略 | 用途 |
|---|---|---|---|---|
| **R1** | {30,30,30} | 300 | hybrid | 度量校验格（须 HHI≈1/3、Wipe≈0） |
| **R2** | {90,90,90} | 300 | hybrid | 水平对照（分离"水平"vs"异质"） |
| **R3** | {10,30,90} | 300 | hybrid | 主格 |
| **R4** | {10,30,90} | 450 | hybrid | 剂量-响应 |
| **R5** | {10,30,90} | 0（H 组延到 60 轮自压） | hybrid | 排除搅拌流混杂 |

主比较是配对的：**R3 的 C 组 vs R2 的 C 组**（同 P=90、同 U，唯一差别是其他会话热不热）。

## 2.6 仪表（最小改动，未实现）

| # | 内容 | 必要性 |
|---|---|---|
| **M1** | CSV 加绝对 `ts` | 论证"同一时刻别会话仍满命中" |
| **M2** | CSV 加 `queue_time_ms` / `prefill_ms`（`meta_info` 已有） | 排除"尾部只是排队" |
| **M3** | task 配置 `gap_mode/gap_s/pause_after_turn/pause_s/arrival.mode/offset_s/gate_after` | 时间轴 + 门控 |
| M4(可选) | server 侧 `sessions_touched per evict()` 直方图 | 直接观测原子性 |

## 2.7 指标定义

| 指标 | 定义 | 期望 |
|---|---|---|
| **Wipe** | `turn>1 且 cached_tokens ≤ 40`（40≈仅 31-token system prompt） | 级联指纹 |
| **WipeRate** | `#wipe / #valid_turns` | 对称≈0；冷组≈1 |
| **HHI** | `Σ_g(wipe_g/Σwipe)²` | 对称 0.33 → 级联 1.0 |
| **CliffRate** | `P(cached_t < 0.5·cached_{t-1})` | 断崖而非渐进 |
| **LossQuantum** | resume 首轮 `recompute` 中位数 ÷ 链长 | ≈1.0 = 整链 |
| **SimultaneityIndex** | 每个 wipe 时刻其他存活会话的 `cached/prompt` 中位数 | 级联≈0.9+ |
| **p99_resume_ttft** | 仅 resume 段 | 主终点 |

## 2.8 归因三步法

对每个 wipe 事件（会话 s，turn t）：

1. **连续性**：`cached_{t-1} − cached_t ≈ chain_len`（一轮内掉整条，而非逐轮缓降）
2. **同时性**：`SimultaneityIndex ≈ 0.9+`（用 M1 的 `ts`）⇒ 池里有位置，是别人占着
3. **可替代性**：wipe 后 ≤1 turn 出现该 run 的 TTFT max

三条同时满足 = 级联；任一不满足则降级为"容量不足"或"整体不均衡"，需重新归因。

## 2.9 统计与判废

- repeats R=3（独立重启 server；受害者切片 seed 42/43/44）
- `p99_resume_ttft` bootstrap 1000 次给 95% CI
- 判废：`gap` 中位数偏离设定 P > ±15%；`num_errors>0`；全局命中率 <50%（thrash 区）
- 全局 `cache_hit_rate_pct` 不作任何结论依据

## 2.10 自检表（每次 run）

| 检查 | 合格线 |
|---|---|
| burst 段 wipe（`cached≤40, turn≤20`） | **0** |
| burst 段 `gap` p50 | `1.0+平均时延`，σ<0.2s |
| pause 实际时长 p50 | `P ± 15%` |
| stirrer `U_real` vs 预算 | ±25% |
| 组间 burst 结束时刻差 | <1.5s |
| `num_errors` | 0 |

## 2.11 时间预算

| 阶段 | 内容 | 预估 |
|---|---|---|
| smoke | 4 条/组 × 9 轮，P={5,15,45}，R1/R3，1 repeat | ~10 min |
| 主矩阵 | 5 格 × 3 repeats（每格 2 min 跑 + 1.5 min 起停） | ~55 min |
| 离线分析 | `analyze_cascade.py` | ~1h 一次性 |
| 合计机时 | | **≈65 min** |

## 2.12 证伪预案（必须提前写死）

| 观察 | 结论修正 |
|---|---|
| R3 的 `HHI ≤ 0.4` | 级联在真实负载下不发生，撤回"原子化"结论；给逐事件 argmin 序列定位原因 |
| wipe 有但 `SimultaneityIndex < 0.5` | 是容量不足，不是集中淘汰；结论改为"容量利用率一般" |
| `p99` 与 `max_session_recompute` 无相关 | 尾部与集中度无关，只与总重算量有关 ⇒ 优化优先级改为"提高总命中率" |

## 2.13 稳健性臂（可选）

staggered 变体：victim i 在 t=6·i 秒到达（i=0..11），其余不变。此时 pause 错开 6s 网格，检验结论是否依赖相位对齐；约 +3 min/次，3 格 × 1 repeat。

## 2.14 改动清单（未实现，待批准）

1. `bench_unicache.py`：CSV 加 `ts/queue_time_ms/prefill_ms`；支持 `gap_mode/gap_s/pause_after_turn/pause_s`；`arrival.mode=fixed` + `gate_after` 屏障
2. `TaskData/victim_pool12_{H,W,C}.jsonl`：从 `AgentBank_alfred_scale_5000` 切 12 条 ≥26 轮（不截断 prompt）
3. `tailshot/cascade_R1..R5.yaml`
4. `tailshot/run_cascade.sh`（复用 `run_matrix.sh` 骨架）
5. `analyze_cascade.py`（Wipe/HHI/Cliff/LossQuantum/Simultaneity/Spearman + 出图）

## 2.15 首次实测（2026-10-26，R3）

### A. 压力带校准：v1 无效 → v2 有效

v1 用 apps + Codeparrot（属 **one-shot 带**）作搅拌，F=200。跑完 profile：

```
workload      hit     evicted      occ        alpha
single       4984      33798      1183(-381)   1.742
random        480       8295      1116(-1116)     -
agentic    310885       2407     33581         1.201
pool=35880  one_shot_evicted=10078
```

one-shot 带吃掉了 42k 淘汰量，agentic 只掉 2,407（≈ 每轮增量），victim wipe = 0。这不是策略出错，而是两带设计的正常行为：只要 one-shot 供给不断，压力就到不了可调度带。

v2 把搅拌换成 `ShareGPT-multi-turn-*` 单轮（`MULTI_TURN_TEXT`，**可调度带**），20 条 × 601 tok，`F_eff ≈ 126 tok/s`（λ=0.20，mean prompt 601）。

### B. 结果

| 组 | pause | wipes | resume hits（中位） | LossQuantum | 备注 |
|---|---|---|---|---|---|
| hot | 10s | 0 | **0.944** | 0.046 | 完全无损 |
| warm | 30s | 0 | **0.935** | 0.046 | 完全无损 |
| cold | 90s | 4/4 | **0.012** | **0.805** | 整链被抽干 |

- `HHI(组) = 1.00`（淘汰全部集中在 cold 组）；`HHI(会话) = 0.25`（组内 4 条各掉一次）
- `SimultaneityIndex = 0.95`：cold 被抽干的同一时刻，其他会话仍有 95% 命中
- `resume TTFT p99 = 265.5 ms`（全局 p50 = 54.1 ms）
- **TTFT 分解**：`victim prefill p99 = 173.0 ms`，`queue p99 = 1.17 ms` ⇒ 尾部是重算，不是排队
- `burst 段 wipe = 0`（内部对照成立）
- `Spearman(loss_quantum, max_resume_ttft) = 0.77`

冷组逐会话逐轮 `cached_tokens`（四条同型，以一条为例）：

```
turn19 = 1607   turn20 = 1665   ← burst 末
turn21 =   25                   ← 90s pause 后首个 resume：整链没了
turn22 = 1855   turn26 = 2349   ← 重新累积
```

`cached_20 − cached_21 = 1640 ≈ 链长`，一轮内掉整条 —— 级联指纹成立。

### C. 指标口径修正（预注册阈值重订）

`WipeRate` 在"每会话只暂停一次"的设计下上限只有 `1/26 ≈ 0.04`，不能作主指标。预注册条件 `WipeRate ≥ 0.3` 口径错误，改为：

```
H1（修订）：cold resume_hit ≤ 0.10 且 hot resume_hit ≥ 0.80
            且 HHI(组) ≥ 0.80 且 SimultaneityIndex ≥ 0.85
```

对应产物：`result/cascade/R3.{json,csv,run.log,analysis.json}`，无效首跑留档 `R3_v1_oneshot.*`。

---

## 2.16 R1 / R2 / R3 三格对照（2026-10-26）

同一搅拌（ShareGPT 单轮，`F_eff≈126 tok/s`）、同一池（35,975）、同一数据，只改 pause：

| 格 | pause (H/W/C) | wipe 数 | HHI(组) | resume 命中 H/W/C | 重算总量 | 重算占比 H/W/C |
|---|---|---|---|---|---|---|
| **R1** | 30 / 30 / 30 | 0 | — | 0.943 / 0.935 / 0.946 | 7,225 | 34% / 36% / 30% |
| **R2** | 90 / 90 / 90 | 9 | 0.556 | 0.552 / 0.743 / 0.633 | 38,926 | 32% / 25% / 43% |
| **R3** | 10 / 30 / 90 | 4 | **1.000** | 0.944 / 0.935 / **0.012** | 14,632 | 17% / 18% / **66%** |

同一会话逐轮 `cached/prompt`：

```
R1 cold  ... 19:1607/1660  20:1665/1733  21:1753/1834  22:1855/1915   ← 全程只差增量
R2 cold  ... 19:1600/1660  20:1665/1733  21:1467/1834  24:1917/2192   ← 局部掉块
R2 hot   ... 19:1649/1711  20:1728/1763  21:1092/1826  26:  24/2401   ← 满链被清（落在哪条随机）
R3 cold  ... 19:1607/1660  20:1665/1733  21:  25/1834  22:1855/1915   ← 首个 resume 满链清零
R3 hot   ... 19:1649/1711  20:1721/1763  21:1768/1826  26:2262/2401   ← 全程无损
```

### 结论与诚实修正

- **H1（原子化/级联）成立**：R3 的损失 100% 落在 cold 组（`HHI(组)=1.0`，cold resume 命中
  1.2%、hot 94.4%），`SimultaneityIndex=0.95`，`prefill p99=173 ms / queue p99=1.17 ms`。
- **H2（"异质比同质更糟"）不成立**：R2（全 90s）总重算 38.9k、TTFT p99 574 ms，
  都**高于** R3（14.6k、266 ms）。异质 idle 的作用是**重新分配**损失（集中到最冷会话），
  不是放大总损失。原假设里"尾 TTFT 因集中而放大"这一条被数据推翻。
- R1 的 profile 中 `agentic evicted=15,099` 与 victim 零损失不矛盾：那是 victim 跑完后
  搅拌继续注入、把遗留链清掉的**收尾期淘汰**，不产生任何 victim 请求。

修正后的对外表述：

> 在混合负载下，node-level recency 淘汰会让**最冷的会话整条链一次清零**，
> 同批其他会话完全不受影响（实测 1.2% vs 94.4%）。这是**公平性 / 可预测性**问题
> （暂停久的用户承担 100% 的损失），而**不是**总吞吐或绝对尾延迟问题。

产物：`result/cascade/{R1,R2,R3}.{json,csv,run.log}`、`all.analysis.json`、`server_R{1,2,3}.log`。

---

## 3. 两套实验的关系

| | Tailshot | Cascade |
|---|---|---|
| 自变量 | 池周转 `F` vs 重访间隔（**对称 30s**） | idle **异质度** `R` |
| 受害者 | 4 条 × 8.3k / 34 轮 | 12 条 × 2.5k / 20+6 轮，3 组 |
| 主指标 | 重算率、TTFT p99 | **HHI / resume 命中 / LossQuantum / Simultaneity** |
| 已证 | LRU 在 `周转≈间隔` 处阈值崩溃 | 级联在异构 idle 下发生（R3），但总损失低于同质长 idle（R2） |
| 互补性 | 证明"存在尾部"，但**结构上无法**显示集中度（实测 HHI=0.251=1/N） | 直接度量集中度 |

**Tailshot 的结论 = "缓存会失效"；Cascade 的结论 = "失效是集中摊在某一会话身上的"**。
两者组合才能支撑"会话级摊薄预算"这一修复方向。

---

## 4. 附录：参数速查

```yaml
硬件:   RTX4090 24G, Qwen2.5-7B-Instruct bf16 (28 层)
池:     35,975 (默认 mem-fraction) | 73,120 (0.79)
标定:   8319 tok/s = 0.1202 ms/token, 截距 -82 ms
数据:   alfred +248.8 tok/轮; turn12≈2,266, turn20≈4,533, turn24≈5,665, turn34≈8,325
server: --radix-eviction-policy {lru|simple_hybrid} [--enable-simple-hybrid]
        --max-total-tokens N --enable-metrics --port 30000
驱动:   concurrency=32, arrival_scale=1, wait_scale=1
```