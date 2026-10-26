# Uni-Bench

用 **ShareGPT 多轮 session 数据** 对 sglang 的 `SimpleHybridRadixCache`
（RadixCache + 多级队列淘汰）做前缀缓存 benchmark。

## 目录

```
Uni-Bench/
  bench_unicache.py    # 多轮压测 driver（asyncio + aiohttp）
  config.sh            # 配置（被 run_all.sh source）
  serve.sh             # 启动单个 server（指定 policy / cache 大小）
  run_all.sh           # 实验矩阵：policy × cache 大小，自动启停 + 汇总
  README.md
  result/              # 结果与 server 日志（运行时生成）
```

## 设计要点（为什么这样做）

1. **直发 `input_ids`**：直接复用 UniCache 已 tokenize 的 `session_data.jsonl`，
   与 UniCache 仿真逐 token 一致，避免 chat template 差异。
2. **`session_id` 打分类标签**：同一 session 的多轮用同一个 `session_id`，
   形如 `ShareGPT-multi-turn-*` → 触发 `classify_simple_hybrid_task_type`
   命中 `MULTI_TURN_TEXT`（进 text 队列）。
   **不使用 `extra_key` 做标签**——`extra_key` 会隔离 radix tree 命名空间，
   导致跨 session 的 31-token system prompt 无法共享。
3. **并发模型**：会话间并发（`Semaphore`），**会话内逐轮串行**（下一轮依赖上一轮的 KV）。
4. **命中率**：`Σcached_tokens / Σprompt_tokens`，取自每次响应的 `meta_info`。

## 前置条件

1. **sglang 已按前面 Phase 1–6 打好补丁**（`SimpleHybridStrategy` + `SimpleHybridRadixCache`
   + `--enable-simple-hybrid` + profile 日志）。
2. **数据**：UniCache 的 `session_data.jsonl`。生成方式：
   ```bash
   cd /home/rain/UniCache/TaskData/SharedGPT-multi-turn
   bash data.sh    # → models/<model>/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
   ```
   冒烟可直接用 `demo/ShareGPT_cleaned_valid_session_data_head20.jsonl`（20 session）。
   **tokenizer 必须与 server 加载的模型一致**。
3. **依赖**：`aiohttp`（sglang 环境已自带）。

## 快速开始（冒烟）

```bash
cd /home/rain/sglang

# 1) 起实验组 server（另一个终端）
POLICY=simple_hybrid MAX_TOTAL_TOKENS=200000 bash Uni-Bench/serve.sh

# 2) 跑 driver（20-session demo）
python Uni-Bench/bench_unicache.py \
  --data-path /home/rain/UniCache/demo/ShareGPT_cleaned_valid_session_data_head20.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 8 \
  --output Uni-Bench/result/smoke.json

# 3) 观察 server 日志里的 [SimpleHybrid profile]（每 30s）：text 队列应有 hit/evicted/alpha
```

冒烟预期：cache 调大（无淘汰）时命中率接近上限——跨 session 共享 31-token system prompt
+ 会话内轮间共享历史。

## 混合负载（模仿 UniCache 异构负载混合）

把多个异构负载按 UniCache 的「比例」（`lambda_rate` Poisson 会话到达率之比）混成
一条到达流交错压测，触发 SimpleHybrid 的多队列（text/single/random/agentic…）竞争。

```bash
cd /root/sg-unicache/Uni-Bench
# 起 server（simple_hybrid）
POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 bash serve.sh

# 跑混合负载（4 类负载 lambda_rate=1:1:1:1，见 hybrid_tasks_config.yaml）
source env.sh
"$PY" bench_unicache.py \
  --tasks-config hybrid_tasks_config.yaml \
  --base-url http://127.0.0.1:30000 \
  --concurrency 32 --arrival-scale 1 --wait-scale 1 \
  --output result/hybrid.json
```

关键参数：

| 参数 | 含义 |
|---|---|
| `--tasks-config` | 混合负载 YAML（含每个 task 的 data_path/lambda_rate/mu/sigma/max_record） |
| `--arrival-scale` | 会话到达节奏缩放：1=UniCache 真实 Poisson；0=全部立即到达；<1 加速 |
| 配置里 `lambda_rate` | 「混合比例」= 各 task 的 lambda_rate 之比（如 2:1） |

结果 JSON 含 `per_task` 字段：每个负载各自的命中率细分。

## 跑完整实验矩阵

```bash
cd /home/rain/sglang
# 编辑 Uni-Bench/config.sh：DATA 换成全量 session_data.jsonl
bash Uni-Bench/run_all.sh
```
默认矩阵：`POLICIES="lru simple_hybrid"` × `TOKENS_LIST="50000 100000 200000"`。

## 关键参数

| 参数 | 含义 | 调参建议 |
|---|---|---|
| `--concurrency` / `CONC` | 并发会话数 | 越大竞争越激烈，越能区分 policy |
| `--max-total-tokens` | KV pool token 上限 | **关键**：调到装不下工作集，淘汰才会发生 |
| `--wait-scale` | 轮间等待缩放 | 0=快速跑；1=按 lognormal(mu,sigma) 真实等待 |
| `--num-sessions` | 使用 session 数 | -1=全部 |

## 指标解读

| 指标 | 说明 |
|---|---|
| `cache_hit_rate_pct` | 前缀缓存命中率（核心） |
| `ttft_p50/p99_ms` | 首 token 时延分位 |
| `e2e_p99_ms` | 端到端时延分位 |
| `[SimpleHybrid profile]`（server 日志） | 各队列 hit/evicted/occ/alpha |

## 陷阱清单

| 陷阱 | 后果 | 规避 |
|---|---|---|
| 每 session 不同 `extra_key` | 跨 session 共享失效 | 用 `session_id`，`extra_key` 留空 |
| `max-total-tokens` 太大 | 无淘汰，policy 无差异 | 调小到装不下工作集 |
| `session_id` 无 `ShareGPT-multi-turn` 前缀 | 全落 single 队列 | session_id = 数据集原 id |
| tokenizer 不一致 | token ids 错位，前缀全 miss | 数据准备与 server 同模型 |
| 会话内并发发轮次 | 命中率虚低 | 确保会话内串行（driver 已保证） |
| 开了 `--enable-session-radix-cache` 等 unified-only flag | 启动报错 | 不要开 |

## 与其他基线的对比

把 `POLICIES` 改成 `"lru lfu slru simple_hybrid"` 即可多基线对比（这些策略都在
`RADIX_EVICTION_POLICY_CHOICES` 里，无需 `--enable-simple-hybrid`）。
