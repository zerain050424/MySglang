# LRU vs SimpleHybrid：搅拌压力下的前缀缓存存活能力

> 位置：`/root/sg-unicache/Uni-Bench/tailshot/`
> 结论速览：把「重访间隔」固定为 30 s，只扫「池子周转速率」——LRU 在
> `池容量 / F ≈ 30 s` 处发生**阈值型崩溃**（受害者重算率 8.5% → 98.1%，
> TTFT p99 62 → 297 ms），而 `simple_hybrid` 在同一位置**保持 8.6% 不动**
> （TTFT p99 116 ms）。

---

## 1. 实验定位
- 关注 **长间隔、多轮、上下文持续增长**的Agentic会话；
- 用一条与它**毫无语义关系**的**单轮高噪声流**持续「搅拌」KV 池；
- 只观测受害者**每轮重访时还剩多少 KV**，以及由此产生的 **TTFT 尾部**延迟。

被比较的两种淘汰策略：

| 策略 | 启动参数 | 机制 |
|---|---|---|
| `lru` | `--radix-eviction-policy lru` | 经典 LRU（RadixCache 默认） |
| `simple_hybrid` | `--radix-eviction-policy simple_hybrid` **+** `--enable-simple-hybrid`（缺一不可） | RadixCache + 多级队列淘汰：按 workload 分类（single/random/agentic/text…），带 one-shot 识别与在线 `alpha` 权重再平衡 |

### 1.1 文件地图

```
/root/sg-unicache/Uni-Bench/
├── bench_unicache.py                # 压测驱动（混合负载到达流 + 逐请求 CSV）
├── calibrate.py                     # 冷启动 prefill 速率标定（token → ms）
├── serve.sh                         # 起单个 server（POLICY / MAX_TOTAL_TOKENS）
├── config.sh / env.sh               # 默认配置 / venv 环境
├── TaskData/victim_pool.jsonl       # 受害者会话池（incremental：4 条 15–18 轮）
├── TaskData/victim_pool_full.jsonl  # 受害者会话池（full replay：3 条 34 轮 / 130K）
└── tailshot/                        # ← 实验本体
    ├── run_matrix.sh                # 8 格矩阵：2 policy × (control + F150/F400/F1200)
    ├── run_full_after.sh            # incremental 跑完后串行跑 full 矩阵
    ├── snapshot.py                  # 中途快照：派生指标全部定义在这里
    ├── tailshot_control.yaml        # 对照：只有受害者流
    ├── tailshot_F150.yaml           # F=150  token/s
    ├── tailshot_F400.yaml           # F=400  token/s
    ├── tailshot_F1200.yaml          # F=1200 token/s
    ├── tailshot_smoke.yaml          # 冒烟（40 apps，跑通链路用）
    └── tailshot_full_*.yaml         # 对应 full replay 版
```

产物（incremental 矩阵，跑完约 107 min）：

```
result/tailshot/
  <policy>_<cell>.json      # 聚合指标
  <policy>_<cell>.csv       # 逐请求明细 ← 真正的分析入口
  server_<policy>_<cell>.log# server 日志（含 [SimpleHybrid profile]）
  calib_<policy>.json       # 每策略一次的 prefill 标定
  matrix.log                # 8 格的启停时间线
```

---

## 2. 要回答的问题与判据

LRU 的结构性弱点不在**总容量**，而在**重访间隔**：只要两次访问之间的插入量
超过池容量，node 必然被挤掉。所以不能只喂一条「混合负载」，必须把问题拆成两个
可控变量：

1. **重访间隔**：受害者流的轮间等待 `lognormal(mu=3.4012, sigma=0.01)` → **30.0 s**（准确定性，抖动 ±0.3 s）；
2. **池子周转速率**：Token池容量 / 搅拌流的注入速率 **F**（token/s）,也就是从空池开始，按当前速率把一整个池子全部换一遍需要多长时间。

于是判据是：

```
池容量 / F     vs     受害者重访间隔 30 s
```

池容量 = **35,975 token**（实测，见 §15 陷阱 1）：

| 档位 | F (token/s) | 池周转 36K/F | 与 30 s 的关系 |
|---|---|---|---|
| control | 0 | ∞ | 无搅拌 |
| F150 | 150 | 240 s | 8 倍余量 → 无感 |
| F400 | 400 | 90 s | 3 倍余量 → 局部失效 |
| **F1200** | **1200** | **30 s** | **正好等于重访间隔 → 每轮洗一遍池子** |

---

## 3. 三条流如何拼到一根时间轴上

`bench_unicache.py --tasks-config` 的混合负载模式，逐函数对齐 UniCache
`vllm/simulator/simulator.py`：

1. 每个 task 读自己的 `session_data.jsonl`（`max_record` 截断），用
   `Random(seed + task_id)` **独立**打乱；
2. 用 `Poisson(lambda_rate)` 泊松过程生成该 task 的 session 首轮到达时刻
   （`expovariate(λ)` 累加，期望间隔 `1/λ`）；
3. 所有 task 的 `(arrival_time, counter, session)` 合并，按 (counter表示这个会话在全局第几个被创建)
   `(arrival_time, counter)` 升序排序 → 异构 session 在时间轴上交错；
4. 每个 session 一个协程：先 `await asyncio.sleep(arrival_offset * arrival_scale)`
   再抢 `Semaphore(concurrency=32)`；
5. **会话内逐轮串行**（下一轮依赖上一轮 KV，并发会人为压低命中率）；
   轮间等待 `rng.lognormvariate(mu, sigma) * wait_scale`；
6. `arrival_scale=1`、`wait_scale=1` 表示严格按 UniCache 真实节奏，不做加速。

`tailshot_*.yaml` 就是三条流：

| 流 | data_path | λ (session/s) | mu / sigma | max_record | 作用 |
|---|---|---|---|---|---|
| **受害者** `AgentBank_alfred` | `victim_pool.jsonl` | **0.5** | **3.4012 / 0.01** | 4 | 被测对象：4 条多轮 agentic 会话 |
| **搅拌 1** `AgentBank_apps` | `AgentBank_apps_session_data.jsonl` | 0.1316 → 0.3509 → 1.0526(增大 λ 增大注入速率) | 0 / 0 | 88 → 220 → 641 | 单轮 API 请求，中位 327.5 token |
| **搅拌 2** `Codeparrot` | `Code_session_data.jsonl` | 同上 | 0 / 0 | 88 → 220 → 641 | 单轮代码请求，中位 697.0 token |

- **会话内串行、会话间并发**；`async with sem:` 包住的是**整个 session 循环**，
  所以 4 条长Agent会话会**全程占住 4 个并发槽**（500 s 量级），搅拌流实际可用 28 槽；
- **Agentic会话 `shuffle: false`**（固定 4 条、固定顺序，可复现），
  **搅拌数据打乱 `shuffle: true`**；


## 5. 长Agentic池：为什么是这几个数

```
victim_pool.jsonl
  alfred_215-…  16 轮   首轮输入=123   末轮 prompt=1921  cached=24  重算=1897
  alfred_277-…  15 轮   首轮输入=232
  alfred_109-…  17 轮   首轮输入=113
  alfred_264-…  18 轮   首轮输入=206   末轮 prompt=2139  cached=24  重算=2115
```

| 参数 | 取值 | 理由 |
|---|---|---|
| λ = 0.5 session/s | 平均 2 s 到达一条 | 4 条在开头 ~8 s 内全部就位，之后全程被测 |
| mu = 3.4012 | e^3.4012 ≈ **30.0 s** | 重访间隔为 30 s|
| agentic（alfred） | 多轮工具调用 | 每轮 prompt 稳定增长（123→2139，+60~70/轮）|

`snapshot.py` 会把实测轮间隔中位数打出来做**有效性自检**：若显著偏离 30 s，
说明并发槽排队或调度抖动已经污染自变量，这一格应判废（实测 30.5–30.8 s，合格）。

---

## 6. 指标定义（一）：逐请求产出CSV`<tag>.csv`

`Stats.rows` 每行 = 一次 `/generate`：

| 字段 | 定义 |
|---|---|
| `task` | `classify_session()` 按 `session_id` 前缀归档 |
| `session_id` | 数据集原 id，仅作标签|
| `turn` | 会话内第几轮 |
| `prompt_tokens` | `meta_info.prompt_tokens`：本轮真正喂进 prefill 的整条序列长度|
| `cached_tokens` | `meta_info.cached_tokens`：命中的缓存|
| `recompute_tokens` | `prompt − cached`：本轮**必须重算**的 token |
| `ttft_ms` | 从 `POST` 到**第一个SSEchunk** 的墙钟时间 |
| `e2e_ms` | 请求全程（含 decode 全部 `max_new_tokens`=16~20） |
| `gap_since_prev_end_s` | 本轮**完成**时刻 − 上一轮**完成**时刻 |

---

## 9. 指标定义（二）：聚合结果JSON —— `<tag>.json`

`Stats.summary()` 字段与解读：

| 字段 | 定义 |
|---|---|
| `cache_hit_rate_pct` | `Σcached / Σprompt`总共的命中率|
| `per_task[name]` | `{num_sessions, num_requests, prompt_tokens, cached_tokens, cache_hit_rate_pct}` 单个负载的命中率|
| `ttft_p50/p90/p99_ms` | 全体请求TTFT的分位数|
| `e2e_mean_ms` / `e2e_p99_ms` | 端到端时延 |
| `num_sessions` / `num_requests` / `num_errors` | 规模与错误数 |
| `wall_time_s` / `request_throughput` | 墙钟 / req/s |
| `replay_mode` / `concurrency` / `arrival_scale` / `wait_scale` / `tasks_config` | 运行参数回显 |

## 10. 指标定义（四）：prefill 标定 —— `calibrate.py`

每次换 server 做一次（`calib_<POLICY>.json`），步骤：

1. `POST /flush_cache` 清空；
2. **丢弃一次预热长请求**（首次大 prefill 触发 kernel 编译/autotune，实测偏高 ~10 s）；
3. 对 `L ∈ {1000, 2000, 4000, 8000, 16000, 24000}`、每档 `reps=2`，
   **每次先 flush** 再发冷启动请求，测 TTFT；
4. 最小二乘拟合 `ttft_ms = a + L / rate`。
| 参数 | LRU | simple_hybrid |
|---|---|---|---|
| `prefill_rate_tok_per_s` | **8319.6** | 8331.2 |
| `ttft_ms_per_token` | **0.1202** | 0.1200 |
| `ttft_intercept_ms` | **−82.8** | −82.4 |
**用途：把重算 token折算成TTFT预算，判断TTFT延迟来自重算还是排队**

- 受害者末轮 prompt ≈ 2,139 → `2139 × 0.12 ≈ 257 ms`；
- F1200/LRU 实测受害者 `p99 = 297 ms`、`max = 307 ms` → 与「整条上下文重算」的
  物理下限吻合 ⇒ **尾部就是被打穿，不是调度排队**；
- F400/LRU 实测 `p99 = 112 ms` → 折算 ~1.6K token 重算，即只有**部分**前缀失效。

## 13. 实测结果

受害者子集（`prompt_tokens>500`，n=52，轮间隔 med ≈ 30.5 s）：

| 格 | 重算率 med | 重算率 max | TTFT p50 | TTFT p99 | 判读 |
|---|---|---|---|---|---|
| `lru_control` | 8.6% | 42.8% | 42 ms | 66 ms | 基线（8.6% 是数据自身增量，**不是 miss**） |
| `lru_F150` | 8.8% | 43.2% | 49 ms | 72 ms | 池周转 240 s ≫ 30 s，**无感** |
| `lru_F400` | 8.5% | 43.0% | 53 ms | 112 ms | 周转 90 s，p99 开始爬，**仍未破防** |
| `lru_F1200` | **98.1%** | 98.9% | **166 ms** | **297 ms** | 周转 30 s = 重访间隔 → **每轮重算整条上下文** |
| `simple_hybrid_control` | 8.6% | 43.2% | 39 ms | 62 ms | 与 LRU 同基线 |
| `simple_hybrid_F150` | 8.6% | 42.4% | 48 ms | 68 ms | 无退化 |
| `simple_hybrid_F400` | 8.6% | 42.6% | 54 ms | 83 ms | 无退化（p99 比 LRU 低 29 ms） |
| `simple_hybrid_F1200` | **8.8%** | 42.4% | 54 ms | **116 ms** | 重算率**基本不动**（8.8%），p99 为 LRU 的 39% |

聚合侧的对照（全局口径，仅作规模校验）：

| 格 | requests | 全局命中率 | wall_time_s | req/s | `per_task.AgentBank_alfred` |
|---|---|---|---|---|---|
| `lru_control` | 66 | 89.37% | 524.45 | 0.13 | 89.37% |
| `lru_F150` | 242 | 71.13% | 723.97 | 0.33 | 89.32% |
| `lru_F400` | 506 | 58.42% | 653.38 | 0.77 | 89.34% |
| `lru_F1200` | 1348 | 34.87% | 620.43 | 2.17 | **10.09%** |
| `simple_hybrid_F1200` | 1348 | 35.27% | 620.42 | 2.17 | **89.39%** |

---

## 14. 复现

```bash
cd /root/sg-unicache/Uni-Bench

# 全矩阵（incremental；默认 OUT=result/tailshot，约 107 min）
bash tailshot/run_matrix.sh 2>&1 | tee result/tailshot/matrix.log

# 中途看进度 + 受害者画像
/root/sl_old/bin/python tailshot/snapshot.py

# full replay 版（需先跑完上一步；脚本靠 "MATRIX ALL DONE" 同步）
bash tailshot/run_full_after.sh

# 单格手动跑（改旋钮用）
POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 bash serve.sh
/root/sl_old/bin/python bench_unicache.py \
  --tasks-config tailshot/tailshot_F1200.yaml \
  --concurrency 32 --replay-mode incremental \
  --arrival-scale 1 --wait-scale 1 --output result/x.json

# 最省事的冒烟（40 条 apps，验证链路）
/root/sl_old/bin/python bench_unicache.py \
  --tasks-config tailshot/tailshot_smoke.yaml \
  --concurrency 32 --output result/smoke.json
```

---

## 15. 已知问题（引用结果前务必确认）

1. **`MAX_TOTAL_TOKENS=100000` 实际没生效**：日志为
   `KV Cache is allocated. #tokens: 35975`、`max_total_num_tokens=35975`
   （`available_gpu_mem=4.93 GB`）。因此**所有 F 档的绝对阈值与这块卡绑定**；
   换卡要重估 `F_crit ≈ pool / 30 s`。
2. **全局 `cache_hit_rate_pct` / 全局 TTFT 分位被搅拌流污染**，
   只有 `per_task` + 受害者子集能下结论（F1200 全局 35% 完全不代表受害者 89%）。
3. **只有 4 条受害者、52 个有效轮**，`p99` 实际落在 `max` 附近，统计意义有限；
   严格尾部结论需扩 victim 池（如 16 条 × 更多轮）。
4. **`calibrate` 的负截距**使小长度段偏低（1K 预测 37 ms vs 实测 99 ms），
   用它换算 ~2K 级受害者重算会**低估约 30%**，只适合做量级判断。
5. **`simple_hybrid_F1200` 被跑了两次**（21:02–21:12 与 21:14–21:24），CSV/JSON
   被第二次覆盖，目录里只剩一份（`mtime 21:24:27`）。**本文所有数字取自第二次**；
   第一次那版的受害者 `p99 = 97 ms`、重算率 med = 8.6%，两次结论方向一致但尾部
   差 19 ms，引用时请注明取自哪一次。
6. **`gap_since_prev_end_s` 含本轮请求耗时**（量的是「本轮完成 − 上轮完成」），
   所以中位数是 30.5 s 而不是 30.0 s；它只能做有效性自检，不能当纯 `wait` 用。
7. **F 是 offered token/s，不是实际写入速率**（实测 LRU **65.4 / 213.6 / 792.0** token/s，
   因为搅拌流自身有 6–83% 的前缀命中）。两套口径混用会得出错误的临界值，见 §4.1。
8. **`serve.sh` 里 `POLICY=simple_hybrid` 必须同时有**
   `--radix-eviction-policy simple_hybrid` **和** `--enable-simple-hybrid`；
   且不要开 `--enable-session-radix-cache` 等 unified-only flag（启动报错）。
9. **`run_full_after.sh` 的自动化链是断的**：它靠 `grep "MATRIX ALL DONE"` 同步，
   而该行只在 `run_matrix.sh` 正常跑完时才写；实测 `matrix.log` 里伴随过
   `run_matrix.sh: line 53: syntax error near unexpected token 'done'`；
   `snapshot.py` 的 `BASE` 也硬编码为 `result/tailshot`——跑 full 矩阵要改。
   （另：`TaskData/victim_pool_full.jsonl` 于 10-20 21:41 被外部删除，full 矩阵当前起不来。）
10. **`max_record` 取的是文件前 N 行，不是随机抽样**：`load_sessions(shuffle=False)`
    读满 N 行即 `break`，随后的 `rng.shuffle()` 只在这 N 条内部打乱。若某天数据文件
    按长度/来源排序，前缀就会系统性偏斜（当前 Codeparrot 前 88 行均值比全库高 26%，
    直接表现为 §4.1 的 ④ 高于 ①）。
---

## 16. 附录：参数速查

```yaml
# 环境
model:      /root/autodl-tmp/model/Qwen2.5-7B-Instruct   (Qwen2.5-7B, bf16, 28 层)
python:     /root/sl_old/bin/python                       (venv，torch 2.13.0)
server:     --radix-eviction-policy {lru|simple_hybrid} [--enable-simple-hybrid]
            --max-total-tokens 100000 --enable-metrics --port 30000
有效池:      35,975 token（实测）

# 受害者
data:       TaskData/victim_pool.jsonl   (4 条，15–18 轮)
lambda:     0.5        mu: 3.4012 (30.0 s)   sigma: 0.01   max_record: 4
轮间隔实测:  30.5–30.8 s（含本轮请求耗时）

# 搅拌
data:       AgentBank_apps (均值 397.9 tok) + Codeparrot (均值 748.4 tok)
F:          150 / 400 / 1200 token/s
lambda:     0.1316 / 0.3509 / 1.0526  （= F / (2 × 570)）
max_record: 88 / 220 / 641
mu/sigma:   0 / 0（单轮）

# 驱动
concurrency: 32（受害者长占 4 槽）
arrival_scale: 1   wait_scale: 1   replay-mode: incremental | full

# 峰速标定
prefill:    8319.6 token/s = 0.1202 ms/token, intercept −82.8 ms
```
