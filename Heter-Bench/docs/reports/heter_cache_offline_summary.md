# HeterCache 离线仿真：四类负载 + 相位漂移的完整实验汇总

> 分支 `W1`（远程 `/root/sg-unicache`，最新提交 `c147aa6`）· 载体 `sim/` 的 `EngineSimulator`
> （跑的是 `python/sglang/srt/mem_cache/` 的生产代码，不是替身）· 数据 `Heter-Bench/TaskData/{exp3,drift3}`
> 本文是**最新汇总**：`mixexp/sim_engine_fcfs_full.md` 里的 heter 列对应 chat TTL=180 的旧配置，
> 本文以 TTL=300 为当前配置，并在 §2 给出两者的逐格对照。

---

## 1. 实验环境与口径

| 项 | 值 |
|---|---|
| 机器 | bitahub 容器（13 vCPU / 503 GB，纯 CPU 仿真，不用 GPU） |
| 解释器 | `/root/sl_old/bin/python` 3.12 |
| 仿真器 | `Simulator.run_engine` → `sim.engine.EngineSimulator`（`PrefillAdder` 预算成批 + FCFS） |
| 调度 / 预算 | `schedule_policy="fcfs"`、`max_prefill_tokens=16384`、chunked prefill 关 |
| page / 容量 | `page_size=8`、`capacity=198,000` token（exp3 的压力点） |
| 服务模型 | `Uni-Bench/result/tailshot/calib_lru.json`：prefill **8319.6 tok/s**、`ttft_intercept_ms=−82.8`（取 max(0,·)=0）、decode **40 tok/s** |
| 命中率口径 | token 加权 `Σhit / Σprompt_len` |
| OPT | 用 `use_map_from_run` 冻结时间轴并**迭代到 `dispatched` 零残差**（engine 的准入按 extend token 计预算，命中不同会改批组成） |
| 复现 | `Heter-Bench/mixexp/tools/{drift3_run.py,build_tail_stress.py}`；远程产物 `/root/sim_runs/` |

```bash
cd /root/sg-unicache
export PYTHONPATH=/root/sg-unicache:/root/sg-unicache/python PATH=/root/sl_old/bin:$PATH STRAT_LOG_LEVEL=quiet
python Heter-Bench/mixexp/scripts/sim/drift3_run.py --plan Heter-Bench/TaskData/drift3/bg_05.json \
       Heter-Bench/TaskData/drift3/bg_10.json Heter-Bench/TaskData/drift3/bg_20.json \
       --out /root/sim_runs/ttl300 --seed 42 --arms lru,heter_cache,opt
```

---

## 2. 四类负载（exp3 四个 cell，seed=0）

每格是"该 cell 最优配置"的成绩（scan/struct/agentic 用 chat TTL=300；chat cell 用 TTL=180，
因为该格 300 反而更低）。

| cell | LRU | **heter（最优）** | 配置 | OPT（上界） | heter−LRU | OPT−heter | 填充率¹ |
|---|---|---|---|---|---|---|---|
| scan（均衡） | 29.65% | **33.59%** | TTL 300 | 38.42% | **+3.94pp** | 4.83pp | **44.9%** |
| struct 主导 | 81.64% | **82.41%** | TTL 300 | 83.49% | **+0.77pp** | 1.08pp | **41.6%** |
| chat 主导 | 57.28% | **58.90%** | TTL 180 | 79.45% | **+1.62pp** | 20.55pp | **7.3%** |
| agentic 主导 | 81.08% | **82.36%** | TTL 300 | 84.43% | **+1.28pp** | 2.07pp | **38.2%** |
| **加权（prompt token）** | 62.55% | **64.24%** | TTL 300 | 71.54% | **+1.69pp** | 7.30pp | **18.8%** |

¹ 填充率 = (heter−LRU)/(OPT−LRU)：LRU 到上界之间，heter 拿走了多少。

chat TTL 180 → 300 的逐格影响：

| cell | heter(TTL180) | heter(TTL300) | Δ |
|---|---|---|---|
| scan | 32.69% | 33.59% | **+0.90pp** |
| struct | 81.56% | 82.41% | **+0.85pp** |
| chat | 58.90% | 58.07% | **−0.83pp** |
| agentic | 81.70% | 82.36% | **+0.66pp** |
| 加权 | 63.84% | 64.24% | **+0.40pp** |

LRU / OPT 的数字在两档下**逐位相同**（它们不读该参数），即这次改动只走 heter 的策略路径。

---

## 3. 相位漂移（drift3：struct → chat → agentic 轮流主导）

结构：`warmup 300s → p1 1200s(struct 主导) → p2 1200s(chat) → p3 1200s(agentic) → cooldown 300s`，
trans 作为可控背景（λ×0.5/1/2），15,309 会话 / 19,518 请求，seed=42。

### 3.1 总体命中率（chat TTL=300）

| 背景 | LRU | **heter** | OPT | heter−LRU | OPT−heter | 填充率 |
|---|---|---|---|---|---|---|
| 0.5× | 76.92% | **77.23%** | 81.50% | **+0.31pp** | 4.27pp | 6.7% |
| 1.0× | 65.20% | **66.84%** | 71.65% | **+1.64pp** | 4.81pp | 25.5% |
| 2.0× | 47.49% | **51.74%** | 57.38% | **+4.25pp** | 5.64pp | 43.0% |
| **加权（p1–p3 全 token）** | 61.65% | **64.12%** | 69.51% | **+2.47pp** | 5.39pp | **31.4%** |

### 3.2 分相位（hit，格式 LRU / heter / OPT）

| 背景 | P1 struct | **P2 chat** | P3 agentic |
|---|---|---|---|
| 0.5× | 83.21 / 83.20 / 83.78 | 66.72 / **68.61** / 77.09（+1.89） | 78.38 / 78.07 / 83.93 |
| 1.0× | 72.44 / 73.03 / 73.76 | 52.27 / **57.23** / 66.04（**+4.96**） | 68.08 / 68.84 / 75.64 |
| 2.0× | 56.91 / 58.74 / 59.93 | 32.29 / **40.57** / 50.63（**+8.28**） | 50.75 / 54.94 / 62.32 |

**收益几乎全部来自 chat 相位，并随背景压力放大**（+1.89 → +4.96 → +8.28pp）。

### 3.3 相位长度 1200s vs 2400s：漂移代价是瞬态

"漂移代价" = 漂移 run 的相位稳态 − 同构成 stationary 稳态（窗口严格限制在相位末尾 300s 内）。

| 相位长度 | P1 struct | **P2 chat**（bg 0.5/1/2×） | P3 agentic |
|---|---|---|---|
| 1200s | −3.6pp | **−9.1 / −7.1 / −8.5pp** | +5.2pp |
| 2400s | −4.6pp | **+3.4 / −1.8 / +3.3pp（≈0）** | −11.6pp |

相位翻倍后 chat 的代价降到噪声级，且 2400s 相位内 chat 的 300s 分箱是平的（末 3 箱 − 前 3 箱 = −0.019）
⇒ **不是"还没爬到"，是 1200s 截断了重建期**。

> ⚠️ 早期一版注意坑：stationary 的稳态窗口若取 `t ≥ 相位末−300`，会把到达窗口结束后
> 仍在跑的长会话尾巴（全部命中）算进来，漂移代价被高估 2~3 倍。必须严格落在相位内。

2400s 矩阵（另一次 run，OPT 固定点未完全收敛，引用前需重跑）：
LRU 73.96/60.92/43.66%、heter 72.81/61.23/44.74%、OPT 81.00/70.58/56.10%。

---

## 4. 优化尝试清单（全部有实测）

| 尝试 | 结果 | 说明 |
|---|---|---|
| **chat 老化 TTL 180 → 300** | **+0.40pp**（四 cell 加权）；drift3 +0.15~0.31pp | 唯一净正的调参；TTL 扫描是倒 U |
| TTL 扫描（drift3 bg=1×） | 0:66.06 / 60s:61.14 / 120s:64.87 / **180s:66.69** / **300s:66.84** / 600s:66.76 / 1200s:66.06 / 3600s:66.06 | >1200s 等于关闭老化，与 TTL=0 逐位相同 |
| VD 视野 `horizon_s` 120→300→600 | **逐位相同** | 三队列共享同一尺度，跨队列排序时是公因子；相位内 λ 恒定 ⇒ 速率估计也不变 |
| Stage 0 边界咨询 VD（概率门限 0.3~0.9） | **±0.0pp**（bg=1× 66.69→66.44/66.41） | chat 的 lognorm 在 20~180s 区间 p 恒为 0.4~0.7，阈值无分辨力 |
| 会话寿命判据替代 TTL（`session_life`） | **−0.11~−0.77pp**（四 cell）；drift3 −0.15~−0.28pp | 同保留时长下更差：k=1.8（p50 309s≈TTL300）时 0.6563 vs 0.6684 |
| 关掉 chat 老化（TTL=0） | **−0.6~−1.0pp** | 陈旧前缀滞留，最后由队列级 VD 整队裁决，砍掉的是活内容 |
| one-shot 豁免窗 20s → 200s | **−18.4pp / −29.7pp**（bg 1× / 2×） | 带供不上货，`_pick_victim` 掉进 VD 仲裁 |
| 关掉 one-shot 带（只走 VD） | **−34.7pp**（chat cell） | 带是本负载唯一的有效清运通道 |
| 保护"热会话里已被重读的节点" | 58.1% → **54.1%** | 保护面一宽就是给别的会话少留空间 |
| 钉住浅根（depth<256 / <512） | 58.1% → **39.6% / 45.3%** | radix 树里浅节点本就是最该走的，钉住等于强迫淘汰更新鲜的内容 |

**规律**：凡是"改阈值"的都不动（±0.0~1pp），凡是"多保护一点"的都显著变差。
可回收的只有容量与全局排序，不是判据。

---

## 5. 其它口径的测量（解释"为什么边际收益消失"）

### 5.1 相对 OPT 的后悔（drift3 bg=1.0×，逐请求 join）

| 项 | 值 |
|---|---|
| regret+ | **445,576 tok** = heter miss 的 14.9% = 全部 prompt 的 **5.0%**（= OPT−heter 的 4.96pp） |
| 赢/输请求 | 735 输 / **0 赢**（OPT 逐请求支配） |
| 凶手归类 | `one_shot:aged` **59.5%** / `one_shot:live` **38.2%** / `value:chat` 2.3% |
| 误杀窗口（t_req − t_evict） | aged p50 **85.5s**、p90 304.7s；live p50 67.0s、p90 251.0s |
| miss 起点深度 | **99.7% 落在 hit<512**（整条前缀丢失），≥2048 恒 0 |

⇒ 失败模式是"整条前缀消失"，不是"深尾巴被削"；这也解释了为什么 VD 排序/视野不起作用。

### 5.2 自后悔（仓库 `RegretLedger`，H=120s，淘汰后是否被立刻回访）

| 负载 | stamps | 淘汰量 | 误判率 | by source |
|---|---|---|---|---|
| exp3 chat cell | 45,111 | 11.17M tok | **0.31%** | 带 0.27% / value:chat 0.14% |
| drift3 bg=1.0× | 17,183 | 3.78M tok | **0.66%** | 带 0.48% / value:chat 0.00% |

按深度：`≤1 页` 0.47%/0.75%，`2-8` 与 `9-64 页` ≤0.01%，`>64 页` 0.00%。
⇒ **判据本身 99.5%+ 是对的**；"97.7% 的损失归因给带"说的是**相对 OPT 的比较劣势**，不是误判。

> 该账本此前 `stamps=0`（`_stamp_eviction` 无调用点），本轮已接回；复现时注意
> `retain_s` 默认 900s，长 run 直接 `report()` 只能看到最后 900s。

### 5.3 重读构成：`reuse_self` vs `reuse_cross`

| 负载 | self | cross |
|---|---|---|
| drift3 bg=1.0× | 47,938（**97.8%**） | 1,057（2.2%） |
| exp3 chat cell | 96,684（**98.5%**） | 1,454（1.5%） |

⇒ 这两类负载里跨会话共享几乎不存在，`share_min≥2` 的保护几乎不触发；
`reuse_count>0` 事实上等价于"这个会话回来过了"。

### 5.4 容量曲线（chat cell）：21pp 差距里约 18pp 是池子不够

| 容量 | LRU | heter | OPT | heter−LRU | OPT−heter |
|---|---|---|---|---|---|
| 198k（超订约 3×） | 57.28% | 58.07% | 79.45% | +0.79 | **+21.35pp** |
| 396k（2×） | 78.83% | 78.84% | 82.35% | +0.01 | **+3.51pp** |
| 792k（4×） | 82.05% | 79.90% | 82.35% | **−2.15** | +2.45pp |

OPT 在 396k 就饱和（82.35%）⇒ chat 工作集约 35~40 万 token，是 198k 池子的两倍。
**容量宽松后 heter 反而输 LRU 2.15pp**：带与队列裁决在富余容量下是纯损失。

### 5.5 尾部 TTFT（chat cell，̀`TTFT = 排队 + miss/8319.6`）

| 负载 | LRU | heter | OPT |
|---|---|---|---|
| 基线 p90 | 296ms | 288ms | — |
| 请求等权命中率 | **36.89%** | 34.90% | — |
| tail-stress P90（专门构造的整链负载） | **920ms** | 1620ms | **184ms** |

- 基线里排队只占 TTFT 21%，p99 请求 98.3% 是 `hit=0`（整条丢失）；
- **请求等权口径下 heter 输 LRU 2pp**：heter 多命中的是中长 prompt，少命中的是大量短请求；
- tail-stress（私有长链 + 长轮间隔）上 heter 比 LRU 更差（warm 整链丢失 74.7% vs 52.2%），
  因为 one-shot 判据几乎正好命中"私有、轮间 45s、首次命中间隔"的画像。

---

## 6. 结论

1. **heter 在 8 个格子里全部 ≥ LRU**（最优配置下；最薄是 drift3 0.5× 的 +0.31pp），
   但**填充率差别巨大**：scan 44.9% / struct 41.6% / agentic 38.2%，
   而 chat 主导只有 **7.3%**、drift3 0.5× 只有 **6.7%**。
2. **收益集中在 chat 相位**（+1.89/+4.96/+8.28pp），且随背景压力放大；struct/agentic 相位只有 0~1.8pp。
3. **漂移代价是瞬态**：1200s 相位下 chat −7~−9pp，2400s 降到 ±3pp（噪声级）。
4. **调参已到顶**：TTL 180→300 是唯一净正项（+0.40pp）；horizon / Stage-0 咨询 / 会话寿命判据全部 0 或负。
5. **剩余空间只有两处**：
   - **容量/超订比**（21.35pp → 3.51pp，容量翻倍）；
   - **全局排序 / 未来信息**（tail-stress 上 P90 920ms → 184ms 的空间；drift3 的 4.8~5.6pp）。
   在"会话私有长链占主导"的负载上，规则型策略拿不到 LRU→OPT 空间的大部分。

---

## 7. 产物与文件

| 内容 | 位置（远程） |
|---|---|
| drift3 数据与驱动 | `Heter-Bench/TaskData/drift3/`、`Heter-Bench/mixexp/scripts/sim/drift3_run.py` |
| tail-stress 负载 | `Heter-Bench/mixexp/tools/build_tail_stress.py`、`mixexp/tail_stress_design.md` |
| 四 cell 驱动 | `/root/sim_runs/sim_engine_cell.py` |
| 本轮矩阵 | `/root/sim_runs/ttl300/all_s42.json`、`/root/sim_runs/engine_full{,_ttl180}/` |
| 后悔/归因 | `/root/sim_runs/{regret_vs_opt,evict_attrib,verify_stamp}.py` |
| 报告 | `mixexp/{drift3_report,exp3_cells_ttl300,scheme_ab_report,tail_stress_design}.md` |
