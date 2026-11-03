# W-TinyLRU 效果实验方案（共享前缀保留 / 命中率 / TTFT）

对应实现：`struct_queue_policy.py` + `victim_override.py` + `evict_policy.py`（`SimpleHybridStrategy`）
设计文档：`STRUCT_QUEUE_WTINYLRU.md`
实验载体：`Uni-Bench/`（已有 hybrid / phased / 单负载三种回放）+ CPU 离线回放（本方案新增）

---

## 0. 一句话

**先在没有 GPU 的离线回放里，把「共享前缀保留度 → 重算 token → 命中率」这条因果链测准；
再在真机上用同一份 trace 只测 TTFT，并做逐请求配对归因。**
没有机制生效证据（`victim_src` 里 qstruct 占比为 0）的实验结果一律作废，不当成「没效果」。

---

## 1. 待验证假设（预注册，先写死再跑）

| # | 假设 | 可反驳的观测 |
|---|---|---|
| **H1** | 在 KV 压力下（pool / 工作集 < 1），W-TinyLRU 让**深共享前缀**（≥2k token，被 ≥3 个 rid 共享）的驻留概率显著高于 LRU / slru / `tiny_lru=off` | `R_prefix(depth≥2k)` 提升 < 3pp，或 CI 含 0 |
| **H2** | 该保留度提升**转化为** token 级命中率提升（不是靠牺牲别处换来的） | 总 hit% 提升但 `per_task` 里 ShareGPT/单轮出现 >1pp 回退（保护错配） |
| **H3** | 命中率提升按「重算 token 数 → prefill 时间」线性传导到 TTFT，p50 至少降 3%，p99 不升 | ΔTTFT_p50 CI 含 0，或 p99 升 >5% |
| **H4** | 增益随压力单调：`pool/工作集` 从 1.0 → 0.3 时增益放大；`pool ≫ 工作集` 时四臂无差异（对照必须是平的，否则测的是噪声） | 无压力场景出现「显著差异」→ 实验设计有 bug |
| **H5**（护栏/负向） | 单次文档询问被拒降级（§5.2）**不**误伤会话内多轮复用 | ShareGPT 的 turn≥2 命中率或 TTFT 回退 |

预注册判据（Primary，全部满足才判定「有效」）：

```
M1  R_prefix(depth≥2k)  ≥ +10pp          # 绝对百分点
M2  token_hit_rate      ≥ +5pp           # cached/prompt 汇总
M3  TTFT_p50            ≤ -3%            # 相对基线 lru
M5  ShareGPT per_task hit 回退 ≤ 1pp     # 护栏
M4-gate victim_src.qstruct ≥ 5% of evicted tokens 且 admit+reject > 0
```

---

## 2. 指标定义（口径先对齐，否则跑完没法比）

记号：请求 r 的 prompt token 序列 `T_r`，其**共享前缀** `P_k`（第 k 层，长度 `L_k`），
服务端返回的 `cached_tokens_r`、`prompt_tokens_r`。

### 2.1 M1 共享前缀保留度（本方案的核心指标，现有 benchmark 没测）

三层口径，从粗到细：

1. **全保留率**（最直观）
   ```
   I_k(r) = 1[ cached_tokens_r ≥ L_k ]                # 请求 r 到达时 P_k 是否完整在树上
   R_k    = Σ_{r∈S_k} I_k(r) / |S_k|                  # S_k = 以 P_k 开头的请求集合
   ```
2. **覆盖度**（区分「整段在」与「只剩前半段」，对长前缀更敏感）
   ```
   C_k(r) = min(cached_tokens_r, L_k) / L_k
   ```
   按 512-token 分桶给出 `C_bucket`，得到**保留度随深度衰减曲线**（x=前缀深度，y=保留度）。
   这张图是「高频共享前缀可以驻留」这一设计目标的直接证据。
3. **生存曲线**（时间维）
   ```
   以 30s 为桶：R_k(bucket) = 该桶内 I_k(r) 的均值
   ```
   用于看 W-TinyLRU 是否把「命中-淘汰-再命中」的锯齿（thrash）压平。

> **P_k 怎么来（不用改服务端）**：数据集里 `input_ids` 已在 jsonl 中。
> 预处理脚本对每个 task 统计**同一 task 内 turn_1 的成对 LCP**，取中位数与 5% 分位作为
> 该 task 的 `L_1`，再按 `{512, 1024, 2048, 4096}` 生成多层 `L_k`。
> 这样 `cached_tokens ≥ L_k` 就是纯客户端可算的判据。
> ShareGPT 的「共享前缀」是**会话内**的（turn 间历史），`P_k` 取该 session 第 t 轮前 t−1 轮的 token 数。

### 2.2 M2 命中率

* 客户端口径（主）：`token_hit_rate = Σcached_tokens / Σprompt_tokens`（`bench_unicache.py` 已输出，含 `per_task`）
* 服务端口径（交叉校验）：Prometheus `sglang:cache_hit_rate`（`metrics_collector.py:305`）
* **分队列口径**（机制诊断）：`SimpleHybridStrategy.profile_summary()` 的
  `workload hit / evicted / occ` 与 `victim_src` 行（`evict_policy.py:320`）

三者必须同时抓：客户端口径好但服务端口径不动 → 说明收益来自请求形态而非缓存；两者背离 → 埋点有 bug。

### 2.3 M3 TTFT

* `ttft_p50/p90/p99`（`bench_unicache.py` 已有）
* **分解**：`queue_time_ms`（排队，受吞吐影响）+ `prefill_ms`（纯重算，受命中影响）——CSV 里已有这两列
* **逐请求配对回归**（把因果坐实的唯一办法）：
  ```
  ttft_ms ≈ α + β·recompute_tokens/1024 + γ·queue_time_ms + δ·(prefix retained?)
  ```
  关注三件事：β（ms/1k 重算 token，应各臂一致，偏离说明测的不是同一种东西）、
  δ（保留住的请求相对没保留的请求多快）、以及两臂在同一 `recompute_tokens` 分桶上的残差差。

### 2.4 M4 机制诊断（决定「实验是否有效」，不是「效果好不好」）

| 指标 | 来源 | 用途 |
|---|---|---|
| `victim_src` 分布（one_shot / global / qstruct） | `profile_summary()` | **gate**：qstruct 占比 0 → 策略从未行使淘汰权，结果作废 |
| `admit / reject / forced / ages` | `StructQueueState.snapshot()["stats"]` | 证明准入、aging 真的触发过 |
| `window / probation / protected / parked` 的**条目与 token 占用** | 同上 | 证明三区结构符合设计（protected 应吃到大部分 token） |
| **淘汰后悔率**（离线精确） | 见 §4.2 事件日志 | `被淘汰 token 中在 W 个请求内又被请求的比例` —— 策略优劣的**直接**度量 |
| **保护浪费率** | 同上 | `protected 区被淘汰前从未再被复用的 token 占比` —— 保护错配的代价 |

后悔率/浪费率是一对孪生指标：W-TinyLRU 该做的是「压低后悔率、不抬高浪费率」。只报 hit% 无法区分「策略好」与「负载变了」。

### 2.5 护栏

`request_throughput`、`e2e_p99`、`num_errors`、`sglang:token_usage`、
服务端 CPU 时间（`evict()` 里策略开销，用 py-spy/cProfile 粗测）、
`StructQueueState.snapshot()` 的单次耗时（<10µs）。

---

## 3. 三层实验（成本递增，先便宜的把贵的筛掉）

| 层 | 载体 | 能测什么 | 不能测什么 | 单次成本 |
|---|---|---|---|---|
| **L0** 逻辑层 | `pytest test/manual/test_struct_queue_wtinylru.py` + 新合成 trace（几十~几百节点） | 成员/三区/准入/aging/退队**语义正确性**；INV-1..6；退化路径 | 任何性能 | 秒级 |
| **L1** 离线回放（**本方案主力**） | CPU 上直接构造 `SimpleHybridRadixCache`（无需模型/GPU，参照 `test/registered/unit/mem_cache/test_radix_cache_slru_accuracy.py` 的 `MHATokenToKVPool(size=N)` 手法） | **精确**的命中率、前缀保留度、后悔率、容量-性能曲线；全部参数消融 | TTFT、吞吐、并发调度效应 | 分钟级/组 |
| **L2** 真机 | `Uni-Bench/bench_unicache.py` + `launch_server` | TTFT 分位、排队效应、端到端吞吐 | 大量参数（太贵），只跑 L1 筛出的 4~5 臂 | 小时级/组 |
| **L3**（可选） | `tools/sglang-simulator` | 复用真 cache 实现 + 延迟预测器的离线 TTFT 预演 | 真实性弱于 L2 | 小时级 |

**L1 是新增资产的核心**：它先于 GPU 花掉大部分实验预算，让 L2 只做确认。

---

## 4. 需要新增的 4 件东西（都很小）

### 4.1 `Uni-Bench/prefix_stats.py`（数据预处理，30 行）
输入 session jsonl → 输出 `prefix_stats.json`：每 task 的 `lcp_median / lcp_p05 / 深度层 L_k`、
每 session 的 turn 前缀长度。L1/L2 的保留度分析都以它为基准。

### 4.2 `test/manual/replay_wtinylru.py`（离线回放驱动，本方案最重要的新文件，~200 行）
复用**生产代码路径**，逐请求回放 jsonl 的 `input_ids`：

```python
# 伪代码
cache = SimpleHybridRadixCache(CacheInitParams(
    disable=False, req_to_token_pool=..., token_to_kv_pool_allocator=...,
    page_size=1, eviction_policy="simple_hybrid",
    eviction_policy_config={"tiny_lru": {"policy": "wtinylru", ...}}))

for ev in trace_events:                      # 按到达时间排序（可取自 bench 的调度）
    res = cache.match_prefix(MatchPrefixParams(key=RadixKey(ev.input_ids)))
    need = len(ev.input_ids) - res.last_node_depth
    if cache.available_tokens() < need:
        cache.evict(EvictParams(num_tokens=need - cache.available_tokens()))
    cache.insert(InsertParams(key=RadixKey(ev.input_ids), value=...))
    # 记录：cached_tokens、P_k 分层命中、可用 token、时间
```

同时在 `_evict_one()` 上挂**只读**钩子，把 `(sim_time, node_key_hash, num_tokens, share_count,
struct_region, victim_src)` 写 `evictions.jsonl`。由此得到 §2.4 的后悔率/浪费率，
并可在回放结束后**精确重算**：若那些 token 没被淘汰，后续命中有多少。

> 硬性要求：**不改生产代码语义**。钩子用 subclass/monkeypatch，只在离线驱动里激活。
> 复跑一致性：同一 trace + 同一参数必须逐字节可复现（无 wall-clock 依赖；aging 已是事件驱动，正好）。

### 4.3 `SGLANG_STRUCT_PROFILE`（真机观测，env 开关，~20 行）
```
SGLANG_STRUCT_PROFILE=/tmp/struct.jsonl   # 每 SGLANG_STRUCT_PROFILE_EVERY 秒（默认 30）追加一行
SGLANG_STRUCT_PROFILE_SOURCE=max_total_tokens=100000,policy=wtinylru,k=3,...
```
在 `SimpleHybridStrategy.maybe_update_alpha()`（已有时间闸）里顺带 dump
`structState.snapshot()` + `profile_summary()`。关掉时零开销。

### 4.4 `Uni-Bench/analyze.py`（分析，~150 行）
读 `bench_*.json` + 同名 `.csv` + `prefix_stats.json` + `struct.jsonl`，输出 §7 的表格与三张图。

---

## 5. 负载与场景矩阵

### 5.1 场景 A：合成「共享前缀强度阶梯」（把机制和噪声分开）
一条合成 trace，三个杠杆，各自单调：

| 杠杆 | 取值 | 期望（可反驳） |
|---|---|---|
| 共享前缀深度 | 512 / 2k / 8k | 深度越大，LRU 越早丢，W-TinyLRU 优势单调放大（H1） |
| 共享广度（复用该前缀的 rid 数） | 3 / 10 / 50 | `share_count ≥ K` 的成员越多，protected 覆盖越好 |
| 复用间隔（两次使用同一前缀的间隔） | 与 pool 同量级 / 10× | 间隔 ≫ 淘汰周期时，任何策略都该失败（**负对照**：四臂应趋同） |

正文/body 长度按 lognormal 采样，模拟长文档干扰项（这是设计文档里「单次文档询问」的靶子）。

### 5.2 场景 B：真实 trace（`Uni-Bench/TaskData`）
* 混合：`hybrid_tasks_config.yaml`（AgentBank_apps + Codeparrot + ShareGPT + AgentBank_alfred，1:1:1:1）
* 单负载：ShareGPT（护栏 H5）、Codeparrot（单轮压测）、AgentBank_alfred（agentic 共享模板）
* 到达：`WAIT_SCALE=0`（快跑，纯容量竞争）与 `=1.0`（真实 Poisson，含时间局部性）各一遍

### 5.3 场景 C：压力扫描（H4）
`--max-total-tokens ∈ {50k, 100k, 200k}`（对齐现有 `config.sh` 的 `TOKENS_LIST`）。
先用 L1 把「工作集大小」量出来，报出 `pool/工作集 ≈ {0.3, 0.6, 1.0, 2.0}` 四个真正的压力点，
不要用 token 绝对值当横轴（换模型/换数据集就不可比）。

---

## 6. 消融矩阵（Arm）

| Arm | `radix-eviction-policy` | `tiny_lru` 配置 | 回答什么问题 |
|---|---|---|---|
| A0 | `lru` | — | 基线（SGLang 默认） |
| A1 | `slru` | — | 公开的「分段 LRU」参照，判断是否值得自研 |
| A2 | `simple_hybrid` | `{"policy":"off"}` | **隔离框架增益**：多队列 + alpha 本身的贡献 |
| A3 | `simple_hybrid` | `wtinylru, k=3, w=64, E=8192` | **默认配置**（H1-H5 主对象） |
| A4 | 同上 | `k=2` | 保护门槛更低（更早进 protected） |
| A5 | 同上 | `k=5` | 保护门槛更高（只有强共享才留） |
| A6 | 同上 | `w=16` | 观察窗过小（准入几乎不发生 → 逼近纯频数驻留） |
| A7 | 同上 | `w=256` | 观察窗过大（策略迟钝，退化为 A2 的近似） |
| A8 | 同上 | `admission_tie="admit"` | 平票时放宽，看是否抬高后悔率 |
| A9 | 同上 | `epoch_events=512` | 激进 aging（旧模板快速失权） |
| A10 | 同上 | `epoch_events=10**9` | 关闭 aging（只有清除才退队），暴露「频数永不衰减」的代价 |
| A11 | 同上 | `admission_metric="share_per_token"` | 设计文档 §5.4 的假设检验（长度归一到底需不需要） |

**L1**：A0-A11 × 压力点 {0.3,0.6,1.0} × 场景 B 三负载 × 3 seed → 约 300 组，CPU 可承受。
**L2**：L1 留下的 {A0, A2, A3, best-k, best-w} × 3 压力 × 3 seed × {hybrid, ShareGPT} ≈ 45 次真机跑。

---

## 7. 运行手册

### L1（离线，无 GPU）
```bash
# 1) 预处理共享前缀统计
python Uni-Bench/prefix_stats.py --data-path Uni-Bench/TaskData/.../*.jsonl \
    --out Uni-Bench/prefix_stats.json

# 2) 回放一组
python test/manual/replay_wtinylru.py \
    --data-path Uni-Bench/TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl \
    --pool-ratio 0.3 --num-sessions 500 --seed 42 \
    --policy simple_hybrid \
    --tiny-lru '{"policy":"wtinylru","k":3,"window_entries":64,"epoch_events":8192}' \
    --out Unibench/result/offline_A3_alfred_p03_s42.json \
    --evictions Unibench/result/evict_A3_alfred_p03_s42.jsonl
```

### L2（真机）
```bash
# 服务端（关键两 flag 缺一不可：--enable-simple-hybrid 选 backend，policy 选策略）
python -m sglang.launch_server --model-path $MODEL \
  --enable-simple-hybrid \
  --radix-eviction-policy simple_hybrid \
  --radix-eviction-policy-config '{"tiny_lru":{"policy":"wtinylru","k":3,"window_entries":64,"epoch_events":8192}}' \
  --max-total-tokens 100000 --enable-metrics --port 30000

# 客户端（已有；per-request CSV 里已含 ttft/prefill/queue/cached_tokens）
python Uni-Bench/bench_unicache.py --tasks-config Uni-Bench/hybrid_tasks_config.yaml \
  --base-url http://127.0.0.1:30000 --concurrency 32 --wait-scale 1.0 \
  --output Uni-Bench/result/bench_A3_hybrid_100k_s42.json
```
`run_all.sh` 现在只扫 `policy × tokens`，需要加一维 `ARM_JSON`（每臂一份 config），
`Uni-Bench/analyze.py` 直接吃 `result/` 目录。

---

## 8. 统计与判定

1. **配对**：同一 trace/seed/到达调度下逐请求配对（`session_id + turn` 是天然主键），
   报 `Δcached_tokens` 与 `Δttft_ms` 的均值 + bootstrap 95% CI（10k 重采样）。
2. **重复**：每配置 3 seed × 2 次（server 重启、warmup 丢弃前 10% 请求或前 60s）。
3. **多重比较**：11 臂不必互相比较；只与预注册基线 {A0, A2} 做 Holm 校正（α=0.05）。
4. **效应量优先**：TTFT 本身抖动大，先看 `recompute_tokens` 与 `R_prefix` 的效应（噪声小），
   TTFT 只作「是否传导」的确认；**禁止**用「TTFT 降了 2%」单独宣称收益。
5. **无效实验的判据**（提前写死）：`victim_src.qstruct < 5%` 或 `reject==0` → 记「策略未被行使」，
   不计入结论，只改参数重跑。

---

## 9. 交付物（表格 schema + 三张图）

`results.csv`（每行 = 一臂×压力×负载×seed）：

```
arm, policy, tiny_lru_json, pool_tokens, pool_ratio, workload, seed,
hit_pct, hit_pct_per_task[4], R_prefix@512/2k/8k, C_prefix@512/2k/8k,
regret_rate, protected_waste_rate, victim_src_qstruct_pct,
admit, reject, forced, ages,
ttft_p50/p90/p99_ms, queue_p50_ms, prefill_p50_ms, throughput, errors,
struct_snapshot_us
```

图 1：**x=池压力, y=hit%, 线=Arm**（H1/H4）
图 2：**x=前缀深度桶, y=保留度 C_k, 线=Arm**（本方案招牌图）
图 3：**TTFT CDF（A0 vs A3）+ 同一 recompute 桶的配对残差**（H3）
附表：「机制 gate 表」（§2.4）与「后悔率/浪费率」双指标表。

---

## 10. 前置检查清单（Phase 0，必须先过）

- [ ] `pytest test/manual/test_struct_queue_wtinylru.py` 全绿；开 INV 断言的调试构建再跑一遍
- [ ] `tiny_lru=off` 与 `simple_hybrid` 的逐请求轨迹**完全一致**（零开销路径没改变行为）
- [ ] 小合成 trace 上 `victim_src.qstruct > 0`、`admit/reject/ages > 0`（策略真被行使）
- [ ] L1 回放与 L2 真机在同一小 trace 上的 `cached_tokens` 逐请求一致（回放器忠实度）
- [ ] 无压力场景（pool ≫ 工作集）四臂 hit% 差异 < 0.5pp（对照是平的）
- [ ] `_evict_one` 上溯闸在 L1 里被触发过（§7.4 的正确性风险确实存在且被挡住）

## 11. 已知威胁与对策

| 威胁 | 对策 |
|---|---|
| TTFT 受排队/批处理支配，掩盖缓存效应 | 用 `queue_time_ms` / `prefill_ms` 分解 + 同桶残差；L2 用固定 concurrency & 固定输出长度 |
| 客户端 `cached_tokens` 与服务端口径不一致 | 三口径同抓（§2.2），任一背离即判数据可疑 |
| 前缀 LCP 判定把「部分命中」算成全命中 | 用覆盖度 `C_k` 而非只报二值 `I_k` |
| 参数扫描过拟合某个 seed | 3 seed + 预注册默认配置 A3 为主结论，其余只作敏感性 |
| aging 的 wall-clock 版本导致不可复现 | 现状已是事件驱动（`epoch_events`），回放器严禁引入时间依赖 |
| 保护过度 → 长尾短请求饿死 | 专门盯 `e2e_p99` 与 Codeparrot（单轮）hit%

---

## 12. 里程碑（按「先便宜后昂贵」排）

| 阶段 | 内容 | 产出 | 估时 |
|---|---|---|---|
| P0 | §10 前置检查 + 埋点 4.1/4.3 | gate 报告 | 1 天 |
| P1 | 4.2 回放器 + 场景 A 合成阶梯 | 保留度/后悔率曲线（机制是否如设计） | 2-3 天 |
| P2 | L1 全消融（A0-A11 × 3 压力 × 3 负载 × 3 seed） | 图 1/2 + 筛出 4-5 臂 | 2 天机时 |
| P3 | L2 真机 45 次跑 | 图 3 + TTFT 结论表 | 1-2 天机时 |
| P4 | 4.4 分析脚本 + 报告 | results.csv + 报告 | 1 天 |


---

## 附录 A：首次实跑的实测校准（2026-11，`Uni-Bench/W-TinyLRU`）

这一节是**跑过之后**对上面设计的修正，实施时以这里为准。

### A.1 框架缺陷（会打挂服务，建议修）

`StructQueueState._promote()` 在候选仍属 window 区时递归调用 `_drain_window()`；
probation/protected 都为空时无界递归 → `RecursionError`（实测 `struct_queue_policy.py:255`）。
设计文档 §5.2 的语义是「promote(cand) → probation 尾」，即晋升必须离开 window。
最小修复：`band = max(band, REGION_PROBATION)`，并删除 `_promote` 里的递归 `_drain_window()`。
实验套件用 `wtl/compat.py` 在进程内打同一补丁（结果 JSON 记 `compat` 字段）。

### A.2 门控判据需要改

- 原 §8.5 的「`reject==0` → 策略未被行使」**不成立**：window 条目成为叶子后往往立刻被淘汰，
  很难同时堆积到 `window_entries` 以上；A3 与 A6(w=16) 实测逐请求结果相同。
- 改用：状态机在管节点（`parked+regions>0`）且 `victim_src.qstruct>0` 或 `forced>0`。
- G4「后端忠实度」必须在**无限池**下比（有淘汰时真/假 LRU 的平票打破顺序不同，属实现细节）。

### A.3 负载侧两个必须知道的偏差

1. **一次性内容（noise / 单轮文档）会被 one-shot 硬优先带全部吃掉**：实测 `one_shot`
   占淘汰 token 的 90%+，struct 队列永远等不到「成员变叶子」，策略空转。
   要测 struct 必须用**多轮、无一次性文档**的负载。
2. **真实 TaskData 的跨会话 LCP 只有 24~317 token**（Codeparrot 317，其余 24）。
   跨会话「深共享前缀」只存在于合成梯子；真实数据上 M1 应退化为
   **会话内历史保留度**（本轮是否命中上一轮完整 prompt，已实现为 `history_retention`）。

### A.4 首批结果（离线，40+8 组）

- **A1（SLRU）明显有害**：ShareGPT 在 pool/工作集=0.3 时命中率 46.1% vs LRU 66.9%（−20.8pp），
  会话内历史保留 0.38 vs 0.74 —— 单调「只升不降」的保护在多轮会话上会挤掉别人的链。
- **A3 与 A2（`tiny_lru=off`）几乎逐请求等价**：命中率差 0~0.7pp，而 `qstruct` 已占淘汰 token
  的 5%~15.8%，说明当前实现接管了淘汰但没有改变选择结果。
- **A3 在 ShareGPT/pool=0.3 的 protected 保护浪费率 = 1.00**：protected 区被淘汰的 token
  事后一次都没有再被命中 —— 保护错配，是下一步最值得改的点。
