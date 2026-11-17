# 方案 A/B：one-shot 判据改造的实测（含 regret 记账修复）

## 一、实验环境

| 项 | 值 |
|---|---|
| 代码 | `W1` @ 本次提交（新增：淘汰侧 regret stamp、`reuse_self`/`reuse_cross`、`share.ageing_mode`） |
| 仿真器 | `Simulator.run_engine`、`EngineConfig(schedule_policy="fcfs")`、page=8、cap=198,000 |
| 服务模型 | `tailshot/calib_lru.json`（prefill 8319.6 tok/s、decode 40 tok/s） |
| 负载 | ① exp3 四个 cell（scan/struct/chat/agentic，seed=0）② drift3 三相漂移（bg 0.5/1/2×，seed=42，1200s 相位）|
| regret 口径 | 仓库自带 `RegretLedger`（`HETER_REGRET=1`），H=各队列 T50、复核时把 `retain_s` 放宽到 1e9 以免坟场过期 |
| 复现 | `/root/sim_runs/{verify_stamp,run_ab,run_k,calib_b}.py`；产物 `/root/sim_runs/schemeA_ab.json` |

## 二、参数设置

1. **接回淘汰侧 stamp**：`HeterCacheRadixCache.evict()` 在 `_evict_one` 之前调用
   `_stamp_eviction(victim, victim, len(victim.value), 1)`（单节点淘汰无级联，chain=1）。
   此前该方法**没有任何调用点**，账本只记请求侧（`stamps=0`），淘汰精确率无法测量。
2. **`reuse_self` / `reuse_cross`**：在 `_record_node_hit` 里按「命中请求的 session_id
   是否等于节点的 session_id」分流，`_split_node` 时随 `reuse_count` 一起继承。
3. **`share.ageing_mode = "session_life"`（方案 A）**：老化不再用「按队列 TTL」，
   改成「该会话是否终结」——`life = clamp(k × gap_p90, min_s, max_s)`，`now - last_active > life`
   且该节点跨会话共享数 < `share_min` 才算老化。证据不足（gaps < `min_gaps`）时保守不老化。
   struct 仍走 TTL（模板族、单轮、无会话证据）。默认 `ageing_mode="queue_ttl"`，行为不变。

## 三、结果

### 3.1 regret 记账修复后（步骤 1 验收）

| 负载 | stamps | 淘汰量 | regret | **误判率** | join_rate |
|---|---|---|---|---|---|
| exp3 chat cell | 45,111 | 11.17M tok | 34,376 | **0.31%** | 27.3% |
| drift3 bg=1.0× | 17,183 | 3.78M tok | 24,928 | **0.66%** | 6.8% |

按裁决来源（chat cell / drift3）：

| source | chat cell 误判率 | drift3 误判率 |
|---|---|---|
| `one_shot` 带 | **0.27%**（6.80M tok） | **0.48%**（3.46M tok） |
| `value:chat`（队内排序） | 0.14%（4.34M） | 0.00%（0.32M） |
| `value:struct` / `value:agentic` | 0% / 0.92% | — |

按深度：`depth≤1p` 0.47%（chat cell）/ 0.75%（drift3），`2-8p` 与 `9-64p` ≤0.01%，
`>64p` 恒 0.00%。**浅页淘汰贡献了几乎全部 regret，但绝对水平仍低于 1%。**

### 3.2 重读构成（步骤 2）

| 负载 | `reuse_self` | `reuse_cross` |
|---|---|---|
| drift3 bg=1.0× | 47,938（**97.8%**） | 1,057（2.2%） |
| exp3 chat cell | 96,684（**98.5%**） | 1,454（1.5%） |

会话轮间隔实测（drift3）：chat `p50 79s / p90 225s / max 600s`；trans `p50 19s`（单轮，
无可用 gap）；struct 无 gap。⇒ 隐含寿命 `k=3` 时 chat `p50 494s / p90 1094s`。

### 3.3 方案 A 的命中率（vs 现有 TTL）

四个 cell（cap=198k，seed=0）：

| cell | LRU | heter(queue_ttl) | heter(session_life, k=3) | Δ |
|---|---|---|---|---|
| scan | 29.65% | 33.59% | 33.25% | **−0.34pp** |
| struct | 81.64% | 82.41% | 82.30% | **−0.11pp** |
| chat | 57.28% | 58.10% | 57.97% | **−0.13pp** |
| agentic | 81.07% | 82.36% | 81.59% | **−0.77pp** |

drift3（seed=42）：

| 背景 | LRU | heter(ttl) | heter(life) | Δ | P2 chat（life vs ttl） |
|---|---|---|---|---|---|
| 0.5× | 76.92% | 77.23% | 77.00% | −0.23pp | 0.6858 vs 0.6861 |
| 1.0× | 65.20% | 66.84% | 66.56% | −0.28pp | 0.5632 vs 0.5723 |
| 2.0× | 47.49% | 51.74% | 51.59% | −0.15pp | 0.4038 vs 0.4057 |

### 3.4 控制「平均保留时长」后的对照（关键）

`k` 扫描（drift3 bg=1.0× / chat cell），隐含寿命取会话级 `p50/p90`：

| 配置 | 隐含寿命 p50 / p90 | drift3 hit | chat cell hit |
|---|---|---|---|
| queue_ttl（chat 300s） | 300 / 300 | **0.6684** | **0.5808** |
| session_life k=1.0 | 178 / 437 | 0.6355 | 0.5521 |
| session_life k=1.5 | 259 / 655 | 0.6534 | 0.5708 |
| session_life k=1.8 | 309 / 672 | 0.6563 | 0.5732 |
| session_life k=2.0 | 343 / 636 | 0.6567 | — |
| session_life k=3.0 | 494 / 1037 | 0.6656 | — |

## 四、对比分析

1. **判据的「局部精确率」已经到顶，这条路没有空间。** 带内淘汰在 120s 视窗内只有
   0.27%（chat cell）/ 0.48%（drift3）会被回访，即 **99.5%+ 判对了**。任何"让它更精准"
   的改造，可达上限就在这 0.3~0.5pp 的淘汰量里。这与"相对 OPT 的后悔里 97.7% 归因给带"
   并不矛盾：后者比较的是"策略之间"，前者衡量的是"这次淘汰事后有没有被立刻打脸"。
   带是**局部精准、全局次优**——它清掉的内容确实短期内没人要，但 OPT 会留得更久。
2. **跨会话共享在这两类负载里几乎不存在**：`reuse_cross` 只占 1.5~2.2%。
   ⇒ `share_min >= 2` 的保护几乎从不触发，`reuse_count > 0` 事实上等价于
   "这个会话回来过了"。这解释了为什么按 `reuse_count` 判 one-shot 会误伤活跃会话：
   它测的是"会话是否已经回来过"，而不是"一次性"。
3. **方案 A 不成立（同保留时长下更差）**：k=1.8 时隐含寿命 p50 309s ≈ 现有 TTL 300s，
   但命中率更低（drift3 0.6563 vs 0.6684、chat cell 0.5732 vs 0.5808）。
   只有当它**多留 60% 的时间**（k=3，p50 494s）才追平。原因：chat 的轮间隔 p90 只有
   225s，固定的 300s TTL 已经覆盖了绝大多数轮间隔——用会话级统计去逼近一个已经接近
   最优的常数，只能带来方差，带不来收益。
4. **真正有信号的是深度，不是时间**：`depth≤1p` 的淘汰占 regret 的 71%（chat cell）/
   67%（drift3），而它在淘汰量里只占 47%/59%——每 token 的风险是深页的 1.5~∞ 倍
   （9-64p 与 >64p 的误判率 ≤0.01%）。但注意绝对水平仍然很低，所以"按深度加权排序"
   的收益上限同样受限于这个 0.3~0.8% 的总量。
5. **结论**：在这两条负载上，继续打磨"何时算永不复用"的**判据**收益极小；
   可回收的空间在 §3.4 之外——即 **tail-stress 那类"私有长链 + 长轮间隔"** 的负载上
   （那里 chat 队内排序的误判率实测 20.1%），以及**全局排序**（OPT−heter 4.8~7.3pp）。
   所以后续若继续，建议按负载条件化：只有在"会话私有长链占比高"时才启用更贵的判据。
