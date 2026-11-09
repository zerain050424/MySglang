# Uni-Bench 完整 Benchmark 指南

> 目标：在 sglang 的 `SimpleHybridRadixCache`（RadixCache + 多级队列淘汰）上做
> **前缀缓存（prefix caching）命中率 + 时延** benchmark。
>
> 目录：`/root/sg-unicache/Uni-Bench`
> 硬件：NVIDIA RTX 5090 D（32GB） | 模型：`Qwen2.5-7B-Instruct`
> 运行环境 python：`/root/sl_old/bin/python`（UniCache 部署 venv）

本文覆盖完整流程：**初始化环境 → 启动服务器 → 单一负载测试 → 多负载（混合）测试**，
并在最后给出**各数据集的测试时间预估**。

---

## 目录

1. [环境初始化](#1-环境初始化)
2. [启动服务器](#2-启动服务器)
3. [单一负载测试](#3-单一负载测试)
4. [多负载（混合）测试](#4-多负载混合测试)
5. [测试时间预估](#5-测试时间预估)
6. [实验矩阵（run_all.sh）](#6-实验矩阵run_allsh)
7. [常见陷阱](#7-常见陷阱)

---

## 1. 环境初始化

### 1.1 前置检查

```bash
# GPU 应可见（单卡 RTX 5090 D）
nvidia-smi

# UniCache 部署 venv（python 3.12.3 + torch 2.13.0，与基线 pin 一致）
ls -d /root/sl_old/bin/python
/root/sl_old/bin/python -c "import sglang, aiohttp, yaml; print('deps ok')"

# 模型（tokenizer 必须与数据准备时一致，否则 token ids 错位全 miss）
ls /root/autodl-tmp/model/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors
```

### 1.2 数据下载

原始数据集下载脚本（国内镜像加速，四个任务并行 + 断点续传）：

```bash
bash /root/autodl-tmp/TaskData/download_datasets.sh /root/autodl-tmp/TaskData
```

| 数据集 | 来源 | 大小 | 用途 |
|---|---|---|---|
| ShareGPT | HuggingFace `anon8231489123/ShareGPT_Vicuna_unfiltered` | ~642 MB | 多轮文本 → MULTI_TURN_TEXT |
| CodeParrot(APPS) | `codeparrot/apps` test.jsonl | ~1.29 GB | 单轮代码 → SINGLE_TURN |
| AgentBank | `Solaris99/AgentBank`（parquet） | ~32 MB | alfred→agentic 多轮 / apps→单轮 API |
| MASHQA | Google Drive | ~1-3 GB | 单轮问答 → SINGLE_TURN |

> ToolBench 需浏览器手动下载（OneDrive），本 benchmark 默认不涉及。

### 1.3 数据准备（每个负载一次）

用 Qwen2.5-7B tokenizer 把原始数据转成 `session_data.jsonl`（逐 turn token 化）：

```bash
cd /root/sg-unicache/Uni-Bench

# ① ShareGPT —— 多轮文本（id 前缀 ShareGPT-multi-turn-*）
bash TaskData/SharedGPT-multi-turn/data.sh
#   产物: TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl

# ② Codeparrot —— 单轮（id 前缀 Code-* → SINGLE_TURN）
bash TaskData/Codeparrot/data.sh
#   产物: TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl

# ③ AgentBank —— agentic 多轮(alfred_*) + 单轮 API(apps_*)
bash TaskData/AgentBankData/data.sh
#   产物: TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl
#         TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl

# ④ MASHQA —— 单轮问答（id 前缀 QA-*）
bash TaskData/mashqa_data/data.sh
#   产物: TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl
```

> 已就绪的数据（本次服务器已生成，可跳过 1.3）见下表：

| 负载 | 文件 | 会话数 | 分类（session_id 前缀 → 队列） |
|---|---|---|---|
| ShareGPT | `.../SharedGPT-multi-turn/.../ShareGPT_cleaned_valid_deduplicated_session_data.jsonl` | 57,519 | `ShareGPT-multi-turn-*` → text |
| Codeparrot | `.../Codeparrot/.../Code_session_data.jsonl` | 4,133 | `Code-*` → single |
| AgentBank apps | `.../AgentBankData/.../AgentBank_apps_session_data.jsonl` | 4,408 | `apps_*` → random |
| AgentBank alfred | `.../AgentBankData/.../AgentBank_alfred_scale_5000_session_data.jsonl` | 3,358 | `alfred_*` → agentic |
| MASHQA | `.../mashqa_data/.../mashqa_session_data.jsonl` | 21,699 | `QA-*` → single |

---

## 2. 启动服务器

### 2.1 手动启动（推荐单次实验）

```bash
cd /root/sg-unicache/Uni-Bench

# 实验组：simple_hybrid（需要两个 flag 同时给，缺一不可）
POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 bash serve.sh

# 基线：lru（或 lfu / slru，均在 RADIX_EVICTION_POLICY_CHOICES 里）
POLICY=lru MAX_TOTAL_TOKENS=100000 bash serve.sh
```

`serve.sh` 实际执行的命令等价于：

```bash
cd /root/sg-unicache
/root/sl_old/bin/python -m sglang.launch_server \
  --model-path /root/autodl-tmp/model/Qwen2.5-7B-Instruct \
  --radix-eviction-policy simple_hybrid \
  --enable-metrics \
  --max-total-tokens 100000 \
  --port 30000 \
  --enable-simple-hybrid \
  2>&1 | tee Uni-Bench/result/server_simple_hybrid_100000.log
```

### 2.2 关键环境变量（可 source config.sh 后覆盖）

| 变量 | 默认 | 含义 |
|---|---|---|
| `MODEL` | `/root/autodl-tmp/model/Qwen2.5-7B-Instruct` | 模型路径（tokenizer 必须与数据一致） |
| `PORT` | `30000` | 服务端口 |
| `POLICY` | `lru` | 淘汰策略：`lru/lfu/slru/simple_hybrid` |
| `MAX_TOTAL_TOKENS` | `200000` | **KV pool token 上限**：调小到装不下工作集才触发淘汰，policy 差异才显现 |
| `LOG` | `result/server_${POLICY}_${MAX_TOTAL_TOKENS}.log` | server 日志 |

### 2.3 健康检查

```bash
# 服务就绪后 /health 返回 200（run_all.sh 内部也用这个探测）
curl -sf http://127.0.0.1:30000/health && echo "OK"

# 观察 simple_hybrid 的多队列统计（每 30s 打一次 [SimpleHybrid profile]）
tail -f result/server_simple_hybrid_100000.log | grep -i "SimpleHybrid"
```

停止所有 sglang server：

```bash
bash /root/sg-unicache/scripts/killall_sglang.sh
```

---

## 3. 单一负载测试

单负载模式：一个 `session_data.jsonl` 反复压测。所有 session **立即并发**（受
`--concurrency` 限制），会话内逐轮串行。

### 3.1 命令

> **两个要点**：
> 1. `--wait-scale` 只对**多轮**负载（num_turns > 1：ShareGPT / alfred）生效；单轮负载
>    （Codeparrot / apps / mashqa）num_turns=1，轮间等待永不触发，加不加无区别。
> 2. **mashqa 必须加 `--no-shuffle`**：同一文档的多个问题在文件中相邻，乱序会摧毁
>    长文档前缀共享（命中率从 ~76% 跌到 ~5%）。

```bash
cd /root/sg-unicache/Uni-Bench
source env.sh

# ============ 多轮负载（--wait-scale 生效）============

# ShareGPT 多轮文本（mu=4.15 / sigma=0.971，平均 3.16 轮）
"$PY" bench_unicache.py \
  --data-path TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 32 --num-sessions 2000 \
  --wait-scale 1 \
  --output result/bench_sharegpt.json

# AgentBank alfred 多轮 agentic（mu=1.81 / sigma=1.092，平均 42 轮，wait-scale=1 显著拉长）
"$PY" bench_unicache.py \
  --data-path TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 32 --num-sessions 2000 \
  --wait-scale 1 \
  --output result/bench_alfred.json

# ============ 单轮负载（--wait-scale 无效，可省略）============

# Codeparrot 单轮代码（num_turns=1）
"$PY" bench_unicache.py \
  --data-path TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 32 --num-sessions 2000 \
  --output result/bench_codeparrot.json

# AgentBank apps 单轮 API（num_turns=1）
"$PY" bench_unicache.py \
  --data-path TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 32 --num-sessions 2000 \
  --output result/bench_apps.json

# MASHQA 单轮问答（★ 必须 --no-shuffle：同文档问题相邻才能共享长文档前缀）
#   建议 --concurrency 4~8，减少同文档兄弟问题的并发竞争（racing）
"$PY" bench_unicache.py \
  --data-path TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 8 --num-sessions 2000 \
  --no-shuffle \
  --output result/bench_mashqa.json

# 冒烟：ShareGPT 30 会话（快速验证链路）
"$PY" bench_unicache.py \
  --data-path TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl \
  --base-url http://127.0.0.1:30000 \
  --concurrency 8 --num-sessions 30 \
  --output result/smoke_sharegpt.json
```


### 3.2 参数详解

| 参数 | 默认 | 含义 |
|---|---|---|
| `--data-path` | 必填 | session_data.jsonl 路径（UniCache 格式） |
| `--base-url` | `http://127.0.0.1:30000` | server 地址 |
| `--concurrency` | 32 | 并发会话数（会话间并发；越大竞争越激烈） |
| `--num-sessions` | -1 | 使用 session 数，-1=全部 |
| `--seed` | 42 | 打乱与轮间等待随机种子 |
| `--no-shuffle` | 关 | 保持文件原序（mashqa 同文档问题相邻；**同时启用提前终止加载，避免读全文件**） |
| `--wait-scale` | 0.0 | 轮间等待缩放：0=不等（快速跑）；1=按 lognormal(mu,sigma) 真实等待（**仅多轮 num_turns>1 生效**） |
| `--mu` / `--sigma` | None | 全局覆盖轮间 lognormal 参数；None=按负载类型取 |
| `--arrival-config` | `arrival_config.yaml` | 轮间 mu/sigma 查表（按 session_id 前缀区分负载） |
| `--timeout-s` | 1800 | 单请求超时（秒） |
| `--output` | None | 结果 JSON 输出路径 |

### 3.3 结果解读

结果 JSON（含 `per_task` 细分）：

```json
{
  "num_sessions": 30, "num_requests": 95, "num_errors": 0,
  "prompt_tokens": 72061, "cached_tokens": 40584,
  "cache_hit_rate_pct": 56.32,
  "ttft_p50_ms": 61.93, "ttft_p99_ms": 371.74,
  "e2e_mean_ms": 1440.57,
  "wall_time_s": 26.16, "request_throughput": 3.63
}
```

| 指标 | 说明 |
|---|---|
| `cache_hit_rate_pct` | 前缀缓存命中率（核心，Σcached/Σprompt） |
| `ttft_p50/p99_ms` | 首 token 时延分位 |
| `e2e_mean/p99_ms` | 端到端时延 |
| `per_task` | 混合模式下每个负载各自的命中率细分 |

---

## 4. 多负载（混合）测试

多负载模式：多个**异构负载**按 UniCache 的「比例」混成**一条到达流**交错压测，触发
SimpleHybrid 的多队列（text/single/random/agentic…）竞争同一 KV 池。

### 4.1 混合原理（对齐 UniCache `vllm/simulator/simulator.py`）

- **「比例」= 各 task 的 `lambda_rate` 之比**（Poisson 会话到达速率之比）。
  例如 `lambda_rate: 2` vs `1` → 2:1 混合。
- 每个 task 用 `rng(seed+task_id)` 打乱自身 session → `Poisson(lambda_rate)` 生成到达
  时刻 → 全部 task 的 `(到达时刻, counter)` 合并进**同一条全局到达流** → 按时序交错。
- 会话内轮间等待：`lognormal(mu, sigma)`（单轮负载 mu=0/sigma=0 不等待）。

### 4.2 命令

```bash
cd /root/sg-unicache/Uni-Bench
source env.sh

# 混合负载（默认 4 类负载 lambda_rate=1:1:1:1，见 hybrid_tasks_config.yaml）
"$PY" bench_unicache.py \
  --tasks-config hybrid_tasks_config.yaml \
  --base-url http://127.0.0.1:30000 \
  --concurrency 32 --arrival-scale 1 --wait-scale 0 \
  --output result/hybrid.json
```

### 4.3 参数详解（混合模式专属）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--tasks-config` | None | 混合负载 YAML（含每个 task 的 data_path/lambda_rate/mu/sigma/seed/max_record） |
| `--arrival-scale` | 1.0 | 会话到达时刻整体缩放：`1`=UniCache 真实 Poisson 节奏；`0`=全部立即到达（等同单负载并发）；`<1` 加速冒烟 |

`hybrid_tasks_config.yaml` 字段（逐字段对齐 UniCache `tasks_config.yaml`）：

```yaml
tasks:
  - name: ShareGPT          # 任务名（也作为 per_task 统计键）
    task_id: 2              # 参与 seed 组合（stable_seed = seed + task_id*1000）
    data_path: TaskData/.../ShareGPT_..._session_data.jsonl   # 相对本文件目录解析
    lambda_rate: 1          # ★ 混合比例：各 task 的 lambda_rate 之比
    mu: 4.15                # 轮间 lognormal mean（多轮负载非 0）
    sigma: 0.971            # 轮间 lognormal sigma
    max_trace_duration: -1  # 保留字段（对齐 UniCache 结构）
    seed: 42
    max_record: 500         # 最多加载 session 数
    shuffle: true           # 是否打乱该负载自身 session 顺序
```

### 4.4 结果解读

结果 JSON 顶部为全局指标，`per_task` 给出**每个负载各自命中率**（异构负载竞争同一
缓存时的差异化表现）：

```json
"per_task": {
  "ShareGPT":        {"num_requests": 1580, "cache_hit_rate_pct": 78.5},
  "AgentBank_alfred": {"num_requests": 21070, "cache_hit_rate_pct": 96.5},
  "Codeparrot":      {"num_requests": 1800, "cache_hit_rate_pct": 4.3},
  "AgentBank_apps":  {"num_requests": 1800, "cache_hit_rate_pct": 5.1}
}
```

> 期望：agentic 多轮（会话内历史前缀充分共享）命中率高；单轮随机负载命中率低
> （只共享 system prompt）。调小 `MAX_TOTAL_TOKENS` 后多队列竞争加剧，差异更明显。

---

## 5. 测试时间预估

### 5.1 时间模型

一次 benchmark 的 wall time ≈ 三部分叠加：

```
wall_time ≈ max(数据加载, 推理计算) + 调度等待
```

- **数据加载**：读 `session_data.jsonl`。单负载模式默认 `shuffle=True` 会**读全文件**
  再打乱截取；`--no-shuffle` 或混合模式会**提前终止**（只读 max_record 行）。
- **推理计算**：`未命中 prefill 时间 + decode 时间`，GPU 为瓶颈时由此决定。
- **调度等待**：混合模式的 Poisson 到达跨度（`max_record / lambda_rate`）与
  多轮负载的轮间 lognormal 等待（`wait_scale>0` 时才生效）。

### 5.2 吞吐假设（实测校准）

- **实测锚点（2026-10-18）**：ShareGPT 30 会话（95 请求）@ `concurrency=8` →
  `wall 26s`、命中率 56.3%。
- 未命中 prefill 吞吐 ≈ **10,000 tok/s**，decode 吞吐 ≈ **2,000 tok/s**（RTX 5090 D + Qwen2.5-7B）。
- 实际 wall ≈ 纯推理 × **1.5~2**（队列 / 时延 / radix 缓存管理开销；低并发时更大）。
- JSON 解析 ≈ 10~40 MB/s（大 token 数组文件偏慢）。

### 5.3 单负载预估（`wait_scale=0`，`concurrency=32`）

| 数据集 | 冒烟规模 | 冒烟预估 | 建议正式规模 | 正式预估 wall |
|---|---|---|---|---|
| ShareGPT（多轮 text） | 30 会话(95 请求) | **~0.5 min**（26s 实测@conc8） | 1000 会话(3160 请求) | **~12-16 min** |
| Codeparrot（单轮） | 30 会话 | **<1 min** | 2000 会话(2000 请求) | **~9-12 min** |
| AgentBank apps（单轮） | 30 会话 | **<1 min** | 2000 会话(2000 请求) | **~17-22 min** |
| AgentBank alfred（agentic 多轮） | 30 会话(1264 请求) | **~2-3 min** | 200 会话(8428 请求) | **~5-8 min** |
| MASHQA（单轮） | 30 会话 | **<1 min** | 2000 会话(2000 请求) | **~7-9 min** |

> 注意：单负载模式默认读全文件，ShareGPT(1.9GB)/alfred(2.9GB) 额外有 **~2-5 min**
> 数据加载时间；加 `--no-shuffle`（文件顺序无关时）可省掉这步。

### 5.4 混合负载预估（默认 `hybrid_tasks_config.yaml`：apps1800 + code1800 + sharegpt500 + alfred500）

| 场景 | 关键参数 | 预估 wall |
|---|---|---|
| 纯并发（最快） | `--arrival-scale 0 --wait-scale 0` | **~40-50 min**（GPU 满负荷） |
| 真实 Poisson 到达 | `--arrival-scale 1 --wait-scale 0` | **~35-45 min**（到达跨度 30min + 尾部） |
| 真实到达 + 轮间等待 | `--arrival-scale 1 --wait-scale 1` | **~2-4 h**（多轮负载轮间等待主导） |
| 冒烟 | 改 max_record 为 30，`--arrival-scale 0 --wait-scale 0` | **~1-2 min** |

到达跨度（Poisson 第 N 个到达 ≈ N/λ，各 task 并行重叠）：

| task | λ | max_record | 到达跨度 |
|---|---|---|---|
| AgentBank apps | 1 | 1800 | ~30 min |
| Codeparrot | 1 | 1800 | ~30 min |
| ShareGPT | 1 | 500 | ~8 min |
| AgentBank alfred | 1 | 500 | ~8 min |

### 5.5 影响时间的关键变量

| 变量 | 影响 | 建议 |
|---|---|---|
| `--concurrency` | 越大吞吐越高（趋近 GPU 饱和），但竞争越激烈、淘汰越频繁 | 命中率实验用 8~32 |
| `MAX_TOTAL_TOKENS` | 越小淘汰越多，多队列差异越明显；不影响时长 | 调到装不下工作集（5万~20万） |
| `--arrival-scale` | 混合模式到达节奏；1=真实 Poisson（慢），0=立即（快） | 冒烟用 0 或 0.01 |
| `--wait-scale` | 多轮负载轮间等待；1=真实（alfred 42 轮 ≈ 数小时） | 命中率实验用 0 |
| `--num-sessions` / `max_record` | 直接决定请求总量 | 冒烟 30；正式 500~2000 |
| `--no-shuffle` | 跳过全文件加载（提前终止） | 文件顺序无关时必加 |

---

## 6. 实验矩阵（run_all.sh）

自动跑 `policy × max_total_tokens` 矩阵并汇总（自动启停 server）：

```bash
cd /root/sg-unicache/Uni-Bench

# 默认：POLICIES="lru simple_hybrid" × TOKENS_LIST="50000 100000 200000"
bash run_all.sh

# 自定义
POLICIES="lru lfu slru simple_hybrid" TOKENS_LIST="50000 200000" bash run_all.sh
```

输出：`result/bench_<policy>_<tokens>.json` + 终端汇总表（hit% / ttft_p50 / ttft_p99 / reqs / err）。

> 注意：`run_all.sh` 走**单负载**路径（`--data-path $DATA`），`DATA` 在 config.sh 里配置。
> 混合负载请用第 4 节的 `--tasks-config` 手动逐 policy 跑。

---

## 7. 常见陷阱

| 陷阱 | 后果 | 规避 |
|---|---|---|
| tokenizer 与数据不一致 | token ids 错位，前缀全 miss | 数据准备与 server 均用 Qwen2.5-7B-Instruct |
| `MAX_TOTAL_TOKENS` 太大 | 无淘汰，policy 无差异 | 调小到装不下工作集 |
| `session_id` 无正确前缀 | 全落 single 队列 | 保留数据 id 前缀（`ShareGPT-multi-turn-*`/`alfred_*`/`apps_*`/`Code-*`） |
| 每 session 不同 `extra_key` | 跨 session 共享失效 | 用 `session_id`，`extra_key` 留空 |
| 会话内并发发轮次 | 命中率虚低 | driver 已保证会话内串行 |
| simple_hybrid 只给一个 flag | 启动报错 | `--radix-eviction-policy=simple_hybrid` + `--enable-simple-hybrid` 同时给 |
| 端口/GPU 被旧 server 占用 | 启动失败或 OOM | 先 `killall_sglang.sh` |
| 单负载读全文件（大文件） | 启动前卡 2~5 min | 加 `--no-shuffle` |
| 多轮负载 `wait_scale=1` | 跑数小时 | 命中率实验用 `wait_scale=0` |
