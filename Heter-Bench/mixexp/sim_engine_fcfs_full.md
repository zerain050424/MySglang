# sim 全量复跑：LRU / heter_cache / OPT（engine 仿真器 · FCFS · page_size=8）

> 时间：2026-11-17 · 分支 `W1`（`13a9878`，含 `70431e8` 真实 response 回放 + `6779957` eol 归一化）
> 载体：`sim/` 的 `EngineSimulator`（`sim/engine.py`，里程碑 E1），跑的是 `python/sglang/srt/mem_cache/` 的生产代码，不是替身
> 数据：`Heter-Bench/TaskData/exp3/{scan,struct,chat,agentic}` 全量（不采样），四 cell 各 ≈24.5M prompt token
> 结论速览：OPT 上界比 LRU 高 **+9.00pp**（token 加权），比 heter_cache 高 **+7.70pp**；heter_cache 在这套口径下 **3 胜 1 平**（+0.62 ~ +3.03pp）；FCFS 成批在本负载上**几乎没生效**（1.02~1.07 条/step）。

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| 机器 | bitahub 容器 `bitahub-a21813790891307008572215` |
| CPU / 内存 | 13 vCPU / 503 GB（仿真峰值 RSS 2.3 GB） |
| GPU | 1× RTX 4090 24 GB，**未使用**（纯 CPU 仿真，`sim` 不需要权重） |
| 代码 | `/root/sg-unicache` @ `W1` = `13a9878`（`70431e8` 真实 response 回放 + `6779957` eol 归一化） |
| 解释器 | `/root/sl_old/bin/python`（3.12 + torch；`ninja` 与 `libssl-dev` 已装，OPT 会用到 `hicache_hash_cpp` 扩展） |
| 仿真器 | `Simulator.run_engine` → `sim.engine.EngineSimulator`（按 SGLang `PrefillAdder` 预算成批准入 + chunked prefill + decode 步） |
| 服务时长模型 | `Uni-Bench/result/tailshot/calib_lru.json`：prefill **8319.6 tok/s**，`ttft_intercept_ms = −82.8`（取 max(0,·) → 0），decode **40 tok/s** |
| 复现脚本 | `/root/sim_runs/sim_engine_cell.py`（本文全部数字由它产出） |

```bash
cd /root/sg-unicache
export PYTHONPATH=/root/sg-unicache:/root/sg-unicache/python PATH=/root/sl_old/bin:$PATH STRAT_LOG_LEVEL=quiet
for c in scan struct chat agentic; do
  CELL=$c PAGE=8 CAP=198000 SCHED=fcfs OPT_ITERS=3 \
    /root/sl_old/bin/python -u /root/sim_runs/sim_engine_cell.py
done
# 产物：/root/sim_runs/engine_full/<cell>_p8_cap198000/{compare.json,use_map_iter{1,2,3}.json}
```

---

## 2. 参数设置

### 2.1 负载（全量）

| cell | 主导类 | requests | prompt tokens |
|---|---|---|---|
| scan | AgentBank_apps（60.0% token） | 46,119 | 24,514,344 |
| struct | Struct（≈60%） | 20,969 | 24,519,565 |
| chat | ShareGPT（60.0%） | 29,316 | 24,535,970 |
| agentic | AgentBank_apps/alfred（60.4%） | 18,757 | 25,221,485 |

每 cell 的 `tasks.yaml`（四类负载混比、`lambda_rate`、`max_record`）原样使用，`arrival_scale=1.0`、`wait_scale=1.0`（轮间空窗走每 task 的 `lognormvariate(mu, sigma)`）。

### 2.2 缓存与策略

| 参数 | 值 | 说明 |
|---|---|---|
| `capacity` | **198,000** token | 沿用 exp3 的压力点（≈ 四 cell 工作集的 0.3~0.6） |
| `page_size` | **8**，三臂统一 | OPT 的 `block_size` 必须等于缓存 page_size；只改一臂会让对比失真 |
| `oversize` | `skip` | 单请求 > 池才触发（本批 `max_turn` 13,592 < 198,000，实际未触发） |
| LRU | `eviction_policy="lru"` | 上游基线 |
| heter_cache | `eviction_policy="heter_cache"` | VD 默认参数：`horizon_s=120`、`smoothing_k=4`、`floor_tokens=0`、`seed=0` |
| OPT | `eviction_policy="opt"` + use map | `OPTStrategy`，块粒度 = page_size |

### 2.3 调度（FCFS）

`EngineConfig`：

| 字段 | 值 |
|---|---|
| `schedule_policy` | **`fcfs`** |
| `max_prefill_tokens` | 16384（EngineConfig 默认） |
| `chunked_prefill_size` | −1（关闭 chunked prefill） |
| `max_running_requests` / `max_prefill_bs` | `None`（不限制） |

一步的语义：把已到轮次压入 waiting → 按 FCFS 扫描、逐条命中并计预算（按 **extend token = need**）→ 整批匹配/挂锁完再统一 evict/alloc → 跑一次 prefill forward → 下一步优先推进 running batch 的 decode（每请求 1 token/step，TPOT 恒定）→ 请求完成时才 `_adopt(prompt+output)` 落树。

### 2.4 OPT 的时间轴（必须迭代）

engine 的准入按 `need = total − hit` 计预算，**命中不同的臂批组成不同 → 派发时刻不同**，所以 use map 的时间轴不能一次性取。做法：以一次实跑的 `dispatched` 冻结（`sim.opt_map.use_map_from_run`），跑 OPT 后比对两条 run 的 `dispatched`，不一致就用 OPT 自己的时刻重建表再跑，最多 3 轮。

---

## 3. 结果

### 3.1 命中率（客户端口径：Σcached_tokens / Σprompt_tokens）

| cell | LRU | heter_cache | **OPT（上界）** | OPT−LRU | OPT−heter | heter−LRU |
|---|---|---|---|---|---|---|
| scan | **29.65%** | **32.69%** | **38.42%** | **+8.77pp** | +5.74pp | +3.03pp |
| struct | 81.64% | 81.56% | **83.49%** | +1.85pp | +1.93pp | −0.08pp |
| chat | 57.28% | 58.90% | **79.45%** | **+22.16pp** | +20.55pp | +1.62pp |
| agentic | 81.08% | 81.70% | **84.43%** | +3.35pp | +2.73pp | +0.62pp |

**token 加权（分母 98,791,364）**

| 臂 | 命中 token | 命中率 |
|---|---|---|
| LRU | 61,791,392 | **62.55%** |
| heter_cache | 63,068,808 | **63.84%**（+1.29pp） |
| OPT | 70,679,544 | **71.54%**（**+9.00pp vs LRU**，+7.70pp vs heter） |

### 3.2 淘汰量与调度统计

| cell | 臂 | 淘汰 token | steps | prefill steps | 请求/prefill-step | 批内最大 token |
|---|---|---|---|---|---|---|
| scan | LRU / heter / OPT | 19.63M / 18.89M / **17.49M** | 448,775 / 452,501 / 459,270 | 43,161 / 43,222 / 43,329 | 1.07 | 13,968 / 14,104 |
| struct | 同上 | 6.92M / 6.94M / **6.46M** | 487,699 / 487,714 / 489,834 | 20,465 / 20,499 / 20,475 | 1.02 | 13,592 |
| chat | 同上 | 15.66M / 15.27M / **10.23M** | 500,200 / 502,171 / 526,595 | 28,129 / 28,201 / 28,364 | 1.04 | 13,592 |
| agentic | 同上 | 7.24M / 7.08M / **6.40M** | 487,201 / 487,981 / 491,255 | 18,347 / 18,367 / 18,369 | 1.02 | 13,592 |

OPT 的淘汰量在每个 cell 都最低（比 LRU 少 6~35%），与它命中率最高是同一条因果：少削了还会被用的内容。

### 3.3 OPT 的时间轴收敛与表覆盖

| cell | 迭代记录（与时间源的 `dispatched` 差异） | 最终 | `_misses / _lookups` |
|---|---|---|---|
| scan | iter1 44,937/46,119（max 7.0 s）→ iter2 8,123（max 0.41 s）→ **iter3 0** | 38.42% | **0 / 316,199,178** |
| struct | iter1 19,560/20,969（max 1.36 s）→ **iter2 0** | 83.49% | **0 / 109,877,753** |
| chat | iter1 28,676（max 9.28 s）→ iter2 15,473（max 1.02 s）→ iter3 18,437（max 0.87 s，**未完全收敛**） | 79.45%（三轮 79.47 / 79.34 / 79.45） | **0 / 118,670,902** |
| agentic | iter2 2,104（max 0.67 s）→ **iter3 0** | 84.43% | **0 / 89,417,034** |

四 cell 的 `_misses` 全为 0：use map 覆盖了树里真正的 token（prompt + 真实 response），「粒度错配」告警从此有判别力。

---

## 4. 对比分析

### 4.1 OPT 的余量按 cell 差一个数量级，原因在复用形态

- **chat（+22.16pp）**：跨会话共享的长历史被 LRU 的 recency 当垃圾挤掉——LRU 淘汰 15.66M、OPT 只 10.23M，差 5.4M token；这块是全场最大的可回收空间。
- **scan（+8.77pp）**：噪声主导（AgentBank_apps 占 60% token），池周转最快，但仍有 2.1M token 的淘汰差。
- **struct / agentic（+1.85 / +3.35pp）**：命中率本来就 81%，可回收的主要是一次性内容的尾巴，余量小。

结论：**上界不是均匀分布的**。按 cell 报「离上界还差多少」比报加权总数更有决策价值。

### 4.2 heter_cache 在这套口径下反超 LRU（3 胜 1 平），但符号对口径敏感

| 口径 | scan | struct | chat | agentic |
|---|---|---|---|---|
| page=1 · emitter 闭环（exp3 v3） | +2.07pp | −2.0pp | −4.0pp | −2.8pp |
| page=8 · engine FCFS（本文） | **+3.03pp** | −0.08pp | **+1.62pp** | **+0.62pp** |

同一份数据、同一个策略，只换 page 粒度与调度模型，chat 从 −4.0pp 翻到 +1.62pp。**报 heter 的数字必须同时给出 (page_size, 调度模型)**，否则不可比。可能机制：page=8 让节点更粗，共享保护段（`share_sessions ≥ share_min`）更容易在池内积累，VD 的视野内命中证据也更稳定——但这需要单独实验证实，本文不下结论。

### 4.3 换到 engine 后 LRU 掉了 0.7pp，代价集中在 152 条请求

同为 scan、cap=198000、page=8，只换调度器：

| 指标 | emitter (`run_scheduled`) | engine FCFS | 差 |
|---|---|---|---|
| 命中率 | 30.35% | 29.66% | **−0.69pp** |
| 命中 token | 7,439,224 | 7,266,856 | −172,368 |
| 淘汰 token | 19,472,856 | 19,636,952 | **+164,096（+0.84%）** |
| `send_lag` | — | p50 **15 ms** / p90 91 ms / **p99 6.9 s** | — |

按 `(session_id, turn)` 对齐 46,119 条：**变差 152 条、变好 11 条、相同 45,956 条**；变差的 152 条平均少命中 **1,202 token**（≈ 一整段 prompt 前缀），其中 **150 条派发更晚**。

机制：请求到点先进 waiting，**match 被推迟到下一个 prefill step 才执行**，所以「上一轮写树 → 下一轮命中」的有效间隔变成 `wait + send_lag`（**变长**，不是变短）；长出来的这段时间里别的请求写入/淘汰，长前缀被削掉 → 这条请求少命中 → 又要多插入 → 池子更紧 → 再多淘汰一层。这也解释了 +164k 淘汰与 −172k 命中的量级吻合。

注意符号并不稳定：cap=60,000 的用例里 emitter 0.9015 vs engine 0.9023（engine **反而** +0.08pp）。p50 延迟只有 15 ms，只有尾部（p99 ≈ 6.9 s）会跨过复用间隔。

### 4.4 FCFS 成批在本负载上几乎没生效

| cell | 请求/prefill-step | 批内最大 token | 单条 prompt 量级 |
|---|---|---|---|
| scan | 1.07 | 14,104 | 数百 ~ 1.3 万 |
| struct / chat / agentic | 1.02 ~ 1.04 | 13,592 | 数百 ~ 1.3 万 |

`max_prefill_tokens=16384` 与单条 prompt 同量级，所以每个 prefill step 基本只装 1 条。**这次 engine 与 emitter 的差别来自 decode/prefill 分步与准入延迟，不是批内竞态**（同批两条请求互相看不到对方新写前缀的那条语义没被行使）。要让 FCFS 的批有意义，得把 `max_prefill_tokens` 压到 2k~4k 或换短 prompt 负载。

### 4.5 OPT 对调度器不敏感，LRU 敏感 → 上界余量被放大

| 口径 | LRU | OPT |
|---|---|---|
| page=8 · emitter 开环重放（scan） | 32.86% | 38.53% |
| page=8 · engine FCFS（scan） | **29.66%** | **38.42%** |

OPT 保的是「未来还会用」的块，不受 match 时刻影响；LRU 直接吃准入延迟与淘汰放大。所以换更贴生产的调度器后，**上界余量从 +5.7pp 变成 +8.8pp**——这不是 LRU 变笨了，是它的失效机制被更真实的调度放大。

### 4.6 局限（读这些数时必须一起读）

1. **OPT 是近似上界**：淘汰单位是整个叶子节点（可跨多 page），不是逐块 Belady，真最优只会更高；
2. **chat 未完全收敛**：三轮固定点迭代残留 18,437/29,316 条派发时刻不同（max 0.87 s），命中率抖动 79.34~79.47%（±0.07pp）；
3. **page_size=8 与生产默认 1 不同**：为让 OPT 可跑（page=1 时 use map 是 2,400 万条、每次淘汰逐 token 二分，跑不动）而统一取 8，四臂一致保证可比；但 4.2 已证明 page 粒度会翻转 heter 的符号；
4. **engine 的已知差异**：不建模 chunked prefill 的 mixed batch、不建模 preemption / retraction、准入一次性预留 prompt+output（压力略高于生产）；
5. **指标只有命中率**：不含 TTFT / 重算税；这台机器没有 GPU，未跑在线端；regret 账本这轮未开（`HETER_REGRET` 仅对 `heter_cache` 有意义）。

---

## 附：与实体键相关的产物

| 路径 | 内容 |
|---|---|
| `/root/sim_runs/engine_full/<cell>_p8_cap198000/compare.json` | 各臂命中率 / 淘汰量 / steps / 收敛信息 |
| `/root/sim_runs/engine_full/<cell>_p8_cap198000/use_map_iter{1,2,3}.json` | 每轮 use map（110~220 MB/份，共 1.7 GB） |
| `/root/sim_runs/engine_<cell>.log` | 逐轮 stdout（含 `opt iterN` 的 `dispatched` 对比） |
