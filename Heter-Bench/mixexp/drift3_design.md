# drift3 · 构成漂移实验（设计 + seed=42 gate 结果）

> 【漂移测试专题 · 负载设计】struct / 多轮 chat / agentic 轮换主导的构成漂移负载构造与门禁预检
> 时间：2026-11-17 · 分支 `W1`
> 载体：`sim/` 的 `EngineSimulator`（FCFS 成批 prefill），跑生产 `HeterCacheRadixCache`
> 数据：`Heter-Bench/TaskData/drift3/`（由 `mixexp/tools/build_drift3_data.py` 生成）
> 状态：**gate 6/6 PASS**，等待批准后跑正式矩阵

## 1. 目标

让 **struct / 多轮对话（ShareGPT）/ Agentic** 依次在各自相位做主导，`trans`
（单轮低复用）作为**背景流量**、其注入速率是唯一可控变量。回答：

1. 切换主导负载后，各策略的命中率多久回到新稳态（**恢复时间 T90**）？
2. 与 OPT 上界的差距在切换窗口里是否被放大？
3. 背景注入速率变化时，heter_cache 相对 LRU 的收益是单调放大还是有最优区间？

## 2. 预检：四类数据落哪条队列（实测 `session_type` 分布）

| 数据 | turns/会话 | prompt token/会话 | 实测队列 |
|---|---|---|---|
| `apps_exp3` | 1 | 391 | **trans 99%** |
| `struct_exp3` | 1 | 2,064 | **struct 94%** / trans 6% |
| `chat_exp3`（ShareGPT，`mu=4.15`） | 7.8 | 6,118 | **chat 86%** / trans 14%（首轮） |
| `AgentBank_alfred`（20–40 轮切片） | 20–40 | 51,372 | **agentic 93%** / chat 6% |

推论：三类主导 + 一类背景各自压在一条队列上，机制解释干净。两个坑：**struct 是单轮**，
它进 STRUCT 靠跨会话共享；**分类只在 heter/opt 臂记账**（lru 臂 `session_type` 恒 `none`），
且 chat 的 CHAT 判定依赖 `mu/sigma`（少写 mu 会全被判成 AGENTIC）。

## 3. 时间轴与构成

```
0        300s        1500s        2700s        3900s   4200s
| warmup  |  P1 struct |  P2 chat  | P3 agentic | cooldown |
```

| 相位 | 长度 | 目标构成（prompt token 份额，bg=1×） |
|---|---|---|
| warmup | 300s | 四类各 25% |
| P1 | 1200s | **struct 55%** + chat 10% + agentic 10% + trans 25% |
| P2 | 1200s | **chat 55%** + struct 10% + agentic 10% + trans 25% |
| P3 | 1200s | **agentic 55%** + struct 10% + chat 10% + trans 25% |
| cooldown | 300s | 四类各 25%（只看回落，不参与构成判据） |

- 1200s = 10× VD 视野（120s）≈ 10× chat T50（118s），够到稳态；
- 吞吐按 exp3 标定 **≈2100 tok/s** ⇒ 每相位 2.52M token、全程 ≈6.9M token；
- **只加背景不动前景**：λ_trans × {0.5, 1, 2}，前景 λ 恒定 ⇒ 总负载随背景升高（14% → 25% → 40% 的 trans 份额）。

## 4. 到达模型

`arrival.mode="segments"`（`sim/schedule.py`，已落地并单测）：分段常数强度的非齐次
Poisson；`λ=0` 是静默段；**末段静默 ⇒ 窗口外的会话不发射**（切片截尾语义）。
每个 `(类, 相位)` 是一个独立 task（`name = "struct@p1"` 这类），`data_path` 指向该相位的
互斥切片文件——同一 jsonl 不能复用（`session_id` 重复会让 `Emitter` 直接报错）。

## 5. 变量与对照

| 类别 | 变量 | 取值 |
|---|---|---|
| 自变量 1 | 背景速率 λ_trans | 0.5× / 1× / 2× |
| 自变量 2 | 主导类（相位） | struct / chat / agentic |
| 自变量 3 | 策略 | `lru` / `heter_cache`（VD 默认）/ `opt`（上界） |
| 常量 | capacity / page_size / 调度 | 198,000 / 8（三臂统一）/ FCFS，`max_prefill_tokens=16384`，chunked 关 |
| 常量 | warmup / cooldown / wait_scale / seed | 300s / 300s / 1.0 / 42（+43 复现） |
| 对照 | stationary（同 token 预算、同构成、不切换） | 把"漂移代价"从"构成效应"里剥出来 |
| 对照 | 逆序（agentic→chat→struct） | 路径依赖检验 |

按评审意见：**VD 视野消融（`horizon_s` 30/300）本轮不做**。

## 6. 指标

| 指标 | 定义 |
|---|---|
| 分相位分类型命中率 | 每相位 × 每 `session_type` 的 Σhit/Σprompt |
| 稳态命中率 | 相位末 300s |
| 切换代价 | 相位首 300s − 相位末 300s |
| **恢复时间 T90** | 切换后 60s 滚动命中率首次达到「稳态−2pp」的时刻（未达到报 >1200s） |
| 上界差 | 同相位 `opt − arm` |
| 机制 | `victim_sources` / 三队占用 / VD densities / `share_counted` / 淘汰量 / `send_lag` |

## 7. gate 判据与 seed=42 实测结果（bg=1×，cap=198k，page=8）

| 编号 | 判据 | 结果 | 实测 |
|---|---|---|---|
| G1 | 构成控制（cohort 口径，容差 = max(3pp, 2σ_poisson)） | **PASS** | p1 struct 57% vs 55%、p2 chat 58% vs 55%、p3 agentic 59% vs 55%、trans 23–25%；各项均在容差内 |
| G2 | 注入速率（3σ_poisson） | **PASS** | 20 个 (类,相位) 组合全部在 3σ 内 |
| G3 | 池饱和 | **PASS** | 各相位均有淘汰且峰值到 198,000 |
| G4 | 确定性（同 seed 两次 `fingerprint`） | **PASS** | `2ce5d2b20686be9e` == 自身 |
| G5 | OPT 时间轴固定点 + 表覆盖 | **PASS** | iter2 残差 **0/11212**，`_misses/_lookups = 0/65,100,643` |
| G6 | 分类有效性（多轮类剔除首轮） | **PASS** | trans 99% / struct 94% / chat 86% / agentic 93% |

gate run（heter_cache）：hit **66.69%**，11,212 requests，188,371 steps，10,789 prefill steps，15s。

三处判据口径的说明（都是从数据里学到必须放宽的）：

1. **G1 用 cohort（会话首轮时刻）而不是请求时刻**：多轮会话必然跨相位（chat 单会话 ≈500s），
   到达侧构成会被 carry-over 拉平——那是要测量的现象，不是控制失败；
2. **G1/G2 的容差按泊松噪声给**（`2σ` / `3σ`），固定 ±3pp / ±5% 在小样本上必然误报
   （agentic 每相位只有 4~32 个会话）；
3. **cooldown 不参与 G1**：300s 窗口里 agentic 期望仅 4 个会话，构成份额天然抖动 ±8pp，
   只作诊断打印。

## 8. 需要修掉的实现前提（已落地）

**仿真的确定性**：`RadixCache.evict` 原先把 `evictable_leaves`（**set**）直接 `list()`，
迭代序取决于对象地址；仿真里同一请求命中的节点时间戳完全相同（平票极多），平票时
heapq 比较 nodes，于是同 seed 两次 run 的淘汰序列都会漂移（实测 drift3 上 4 次 lru run
得到 4 个不同 fingerprint）。改为按单调的 `TreeNode.id` 排序后，4 次 run 完全一致，且
exp3 scan 的 LRU 数字**逐位不变**（0.2965 / 7,268,856）。回归见
`test/manual/test_sim_determinism.py`（含一条"淘汰序与 set 迭代序无关"的精确断言）。

## 9. 产物与复现

```bash
cd <repo>
python Heter-Bench/mixexp/tools/build_drift3_data.py --force      # 数据 + yaml + plan.json
python Heter-Bench/mixexp/scripts/sim/drift3_run.py \
  --plan Heter-Bench/TaskData/drift3/bg_10.json --gate            # G1~G6
```

| 路径 | 内容 |
|---|---|
| `Heter-Bench/TaskData/drift3/{warmup,p1,p2,p3,cooldown}/{trans,struct,chat,agentic}.jsonl` | 15,309 条会话（125 MB，未提交） |
| `Heter-Bench/TaskData/drift3/bg_{05,10,20}.{yaml,json}` | 人读 yaml + 驱动读 plan（含每 (类,相位) 的 λ / 会话数 / 目标份额） |
| `Heter-Bench/mixexp/tools/build_drift3_data.py` | 切片 + λ 反算（λ = 目标份额 × 预算 / (秒数 × 实测 token/会话)） |
| `Heter-Bench/mixexp/scripts/sim/drift3_run.py` | 驱动 + gate |

## 10. 成本（正式矩阵）

| 项 | 单次 | 数量 | 合计 |
|---|---|---|---|
| lru（6.9M token） | ~6s | 3 λ_bg × 2 seed | <1 min |
| heter_cache | ~15s | 同上 | 1.5 min |
| opt（含 ≤3 轮时间轴迭代） | 60~150s | 同上 | 10~15 min |
| stationary 对照（3 相位 × lru/heter） | ~20s | 3×2 | 2 min |
| 逆序对照 | ~25s | 3 臂 | 1.5 min |
| **合计** | | | **≈20 min** |

## 11. 已知局限

1. **agentic 的构成份额天然粗糙**：51k token/会话 ⇒ 55% 预算只有 ~28 个会话，份额 σ≈10pp；
2. 多轮会话跨相位（carry-over）是真实负载的特征，报告会同时给出 cohort 与 arrival 两种构成；
3. page_size 取 8（OPT 需要），与生产默认 1 不同；四臂一致保证可比，但 page 粒度会翻转
   heter 的符号（见 `sim_engine_fcfs_full.md` §4.2）；
4. 不建模 chunked prefill 的 mixed batch / preemption。
