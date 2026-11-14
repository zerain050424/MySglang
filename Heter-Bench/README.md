# Heter-Bench

用 **ShareGPT 多轮 session 数据** 对 sglang 的 `HeterCacheRadixCache`
（RadixCache + 多级队列淘汰）做前缀缓存 benchmark。

## 目录

```
Heter-Bench/
  README.md               # 本文件：怎么跑
  docs/                   # 说明与实验报告，索引见 docs/README.md
  archive/                # 历史配置快照，仅供追溯

  scripts/                # 运行入口
    bench_heap.py         # 压测驱动：最小堆 + 真实墙钟，到点发、不等结束、空窗不占槽
                          # （对齐 UniCache simulator 的调度语义；只支持 --tasks-config）
    serve.sh              # 起单个 server（指定 policy / cache 大小）
    config.sh / env.sh    # 配置与环境变量（被 serve.sh source）

  configs/                # 全部 yaml / json 配置
    arrival_config.yaml                  # 轮间到达间隔，按 session_id 前缀区分负载
    hybrid_tasks_config.yaml             # 混合负载配比
    phased_load_prediction.yaml          # 分阶段负载
    phased_load_prediction_task_labels.json

  tools/                  # 数据准备与标定
    prepare_session_data.py              # 原始数据 → session_data.jsonl
    calibrate.py                         # prefill 速率标定（token → ms）
    strip_qwen_system.py                 # 剥掉每轮开头的 Qwen system 段（见 mixexp/README）
    drop_short_sessions.py               # 剔掉轮数 < MULTI_TURN_MIN 的会话，让 TRANS 只来自单轮负载

  mixexp/                 # 混合负载实验：四个「单类 token 主导」cell × 实验臂，见 mixexp/README.md
  eval/                   # 离线评估：节点级 / token 级 / label 口径
  profile/  log/          # 观测实现 / 运行日志
  classify/  exp/  tailshot/  RR_VS_TLRU/  W-TinyLRU/   # 实验目录
  TaskData/               # 会话数据集
```

## 设计要点（为什么这样做）

1. **直发 `input_ids`**：直接复用 HeterCache 已 tokenize 的 `session_data.jsonl`，
   与 HeterCache 仿真逐 token 一致，避免 chat template 差异。
2. **`session_id` 仅标识会话**：同一 session 的多轮用同一个 `session_id`，
   不再依赖客户端为请求类型打标签，由缓存状态机自动判断队列。
   HeterCache 根据节点是否复用、复用间隔和树分支自动判断
   `qtrans`、`qstruct`、`qagent`、`qchat` 四类队列。
   **不使用 `extra_key` 做标签**——`extra_key` 会隔离 radix tree 命名空间，
   导致跨 session 的 31-token system prompt 无法共享。
3. **并发模型**：会话间并发（`Semaphore`），**会话内逐轮串行**（下一轮依赖上一轮的 KV）。
4. **命中率**：`Σcached_tokens / Σprompt_tokens`，取自每次响应的 `meta_info`。

## 前置条件

1. **sglang 已按前面 Phase 1–6 打好补丁**（`HeterCacheStrategy` + `HeterCacheRadixCache`
   + `--enable-heter-cache` + profile 日志）。
2. **数据**：HeterCache 的 `session_data.jsonl`。生成方式：
   ```bash
   cd /home/rain/HeterCache/TaskData/SharedGPT-multi-turn
   bash data.sh    # → models/<model>/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
   ```
   冒烟可直接用 `demo/ShareGPT_cleaned_valid_session_data_head20.jsonl`（20 session）。
   **tokenizer 必须与 server 加载的模型一致**。
3. **依赖**：`aiohttp`（sglang 环境已自带）。

## 快速开始（冒烟）

```bash
cd /home/rain/sglang

# 1) 起实验组 server（另一个终端）
POLICY=heter_cache MAX_TOTAL_TOKENS=200000 bash Heter-Bench/scripts/serve.sh

# 2) 跑 driver（smoke 配比：四类各 10 条 session）
python Heter-Bench/scripts/bench_heap.py \
  --tasks-config Heter-Bench/mixexp/configs/mix_smoke.yaml \
  --base-url http://127.0.0.1:30000 \
  --max-inflight 8 --wait-scale 0 \
  --output Heter-Bench/result/smoke.json

# 3) 观察 server 日志里的 [HeterCache profile]（每 30s）：text 队列应有 hit/evicted/alpha
```

冒烟预期：cache 调大（无淘汰）时命中率接近上限——跨 session 共享 31-token system prompt
+ 会话内轮间共享历史。

## 混合负载（模仿 HeterCache 异构负载混合）

把多个异构负载按 HeterCache 的「比例」（`lambda_rate` Poisson 会话到达率之比）混成
一条到达流交错压测，触发 HeterCache 的多队列（text/single/random/agentic…）竞争。

```bash
cd /root/sg-unicache/Uni-Bench
# 起 server（heter_cache）
POLICY=heter_cache MAX_TOTAL_TOKENS=100000 bash scripts/serve.sh

# 跑混合负载（4 类负载 lambda_rate=1:1:1:1，见 configs/hybrid_tasks_config.yaml）
source scripts/env.sh
"$PY" scripts/bench_heap.py \
  --tasks-config configs/hybrid_tasks_config.yaml \
  --base-url http://127.0.0.1:30000 \
  --arrival-scale 1 --wait-scale 1 \
  --output result/hybrid.json
```

关键参数：

| 参数 | 含义 |
|---|---|
| `--tasks-config` | 混合负载 YAML（含每个 task 的 data_path/lambda_rate/mu/sigma/max_record） |
| `--arrival-scale` | 会话到达节奏缩放：1=HeterCache 真实 Poisson；0=全部立即到达；<1 加速 |
| 配置里 `lambda_rate` | 「混合比例」= 各 task 的 lambda_rate 之比（如 2:1） |

结果 JSON 含 `per_task` 字段：每个负载各自的命中率细分。

## 跑多臂矩阵

矩阵脚本（原 `run_all.sh` / `run_mixexp.sh` / `run_single.sh`）已随旧 driver 一并删除。
现在按臂逐次手动跑：每个臂起一次 server（`serve.sh`）→ 跑一次 `bench_heap.py` → 停 server。
`result/` 下历史结果不受影响。

## 关键参数

| 参数 | 含义 | 调参建议 |
|---|---|---|
| `--max-inflight` | 同时在飞的请求上限 | 0=不限（排队交给服务端）；设值即驱动侧限流 |
| `--max-total-tokens` | KV pool token 上限 | **关键**：调到装不下工作集，淘汰才会发生 |
| `--wait-scale` | 轮间等待缩放 | 0=快速跑；1=按 lognormal(mu,sigma) 真实等待 |
| 配置里的 `max_record` | 每个 task 用多少 session | 冒烟 10；正式 500~2000 |

## 指标解读

| 指标 | 说明 |
|---|---|
| `cache_hit_rate_pct` | 前缀缓存命中率（核心） |
| `ttft_p50/p99_ms` | 首 token 时延分位 |
| `e2e_p99_ms` | 端到端时延分位 |
| `[HeterCache profile]`（server 日志） | 各队列 hit/evicted/occ/alpha |

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

把 `serve.sh` 的 `POLICY` 换成 `lru` / `lfu` / `slru` / `heter_cache` 即可多基线对比
（这些策略都在 `RADIX_EVICTION_POLICY_CHOICES` 里，无需 `--enable-heter-cache`）。

## 四阶段负载自动预测准确率实验

`phased_load_prediction.yaml` 使用现有四类 session 数据，按以下顺序生成一条连续
到达流：

1. `mixed`：Codeparrot、AgentBank apps、ShareGPT、AgentBank alfred 为 `1:1:1:1`；
2. `struct_dominant`：Codeparrot 为主，比例 `4:1:1:1`；
3. `chat_dominant`：ShareGPT 为主，比例 `1:1:4:1`；
4. `agentic_dominant`：AgentBank alfred 为主，比例 `1:1:1:4`。

结构复用真值仅指 Codeparrot；AgentBank apps 保留为 `random`。先启动服务，再运行：

**驱动已下线**：分阶段（`--phases-config`）模式随旧 driver 一并删除，`bench_heap.py`
只支持 `--tasks-config`。要重跑这条实验，需要先给新驱动补回相位调度。

**数据源已下线**：这里原本依赖 server 侧 `SGLANG_QUEUE_OCC_LOG` 打印的 `[QueueOcc]`
日志。该采样器（策略无关的队列占用日志）已从 `radix_cache` 删除，现在只有
`[HeterCache profile]`（每 30s 一张按队列的表，含 occ / hit / evicted / target / cap）。
`eval/evaluate_load_prediction.py` 只认 `[QueueOcc]` 的行格式，喂新日志会直接报「解析不到
任何样本」——要恢复这条评测，得先把采样器加回来或改写解析口径。

实验输出的 CSV 保存真实数据集标签，脚本按 60 秒窗口、30 秒步长和 10% 主导差距
计算窗口级准确率（需要上面那份日志）：

```bash
"$PY" eval/evaluate_load_prediction.py \
  --requests-csv result/phased_load_prediction.csv \
  --occ-log result/server_heter_cache_100000.log \
  --output result/phased_load_prediction_accuracy.json
```

结果中的 `accuracy` 是排除 `mixed` 和 `unknown` 窗口后的窗口级准确率，`coverage` 是
有效窗口中具有自动预测结果的比例，`by_phase` 给出四个阶段的分项结果。
