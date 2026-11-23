# Heter-Bench 服务器复现流程（AutoDL + Qwen2.5-7B-Instruct）

本目录在服务器 `/root/sg-unicache/Uni-Bench`，是对本地 HeterCache 数据准备与发送逻辑的
**复用 + 服务器适配**。所有脚本含中文注释，可直接按序执行。

> **驱动已下线（2026-11）**：本文的 `bench_hetercache.py` / `run_all.sh` 命令已随旧
> 发射器一并删除，仅作历史记录；当前驱动为 `scripts/bench_heap.py`（只支持
> `--tasks-config`，无单负载 / 相位 / burst-pause 模式）。

## 环境

| 组件 | 位置 |
|---|---|
| HeterCache 部署 venv | `/root/sl_old`（python 3.12.3，torch 2.13.0，与 9-26 基线 pin 一致） |
| 模型（tokenizer 必须一致） | `/root/autodl-tmp/model/Qwen2.5-7B-Instruct` |
| 原始数据集 | `/root/autodl-tmp/TaskData/`（下载脚本见 `download_datasets.sh`） |
| server 启动环境 | `scripts/env.sh`（venv python + ninja PATH + LD_LIBRARY_PATH） |

## 一、数据准备（一键：下载 + 处理）

### 0. 一键入口（新环境推荐）

```bash
# 下载侧依赖（无 torch/tokenizer）
pip install -U huggingface_hub

# 处理侧需要带 transformers 的 python（AgentBank 另需 pandas）。
# 不设则默认 python3；TOKENIZER 默认取 scripts/config.sh 的 MODEL（可被环境变量覆盖）。
bash TaskData/prepare_all.sh

# 常用变体：
SKIP_DOWNLOAD=1 bash TaskData/prepare_all.sh          # 原始数据已就绪，只跑处理
DATASETS="sharegpt apps" bash TaskData/prepare_all.sh # 只下指定数据集
RAW_DIR=/data/raw bash TaskData/prepare_all.sh        # 原始数据放别处
PY=/root/sl_old/bin/python bash TaskData/prepare_all.sh # 指定处理侧 python
```

`prepare_all.sh` = `download_datasets.sh`（huggingface_hub，断点续传，
默认 `HF_ENDPOINT=https://hf-mirror.com`）→ 依序跑四个 `data.sh`（sharegpt/
apps/agentbank/mashqa），任一失败即停，重复跑幂等（已存在且大小合理的原始文件跳过下载）。

原始数据落位（默认 `TaskData/raw/`）：

| 数据集 | HF 来源 | 落位 | 处理产物 |
|---|---|---|---|
| ShareGPT | `anon8231489123/ShareGPT_Vicuna_unfiltered` 单文件 | `raw/SharedGPT-multi-turn/` | `ShareGPT_cleaned_valid_deduplicated_session_data.jsonl` |
| Codeparrot(APPS) | `codeparrot/apps` 的 `test.jsonl` (~1.29GB) | `raw/Codeparrot/` | `Code_session_data.jsonl` |
| AgentBank | `Solaris99/AgentBank` 整仓 parquet | `raw/AgentBank/` | `AgentBank_apps/alfred_scale_5000_session_data.jsonl` |
| MASHQA | `VivekNaga/Medicaldata` 三个 `_full.json` | `raw/mashqa_data/` | `mashqa_session_data.jsonl` |

### 0.1 单独跑某个环节

```bash
bash TaskData/download_datasets.sh sharegpt apps agentbank mashqa   # 只下载
bash TaskData/SharedGPT-multi-turn/data.sh   # 只处理（路径已改为 env：PY/TOKENIZER/RAW_DIR）
bash TaskData/Codeparrot/data.sh
bash TaskData/AgentBankData/data.sh
bash TaskData/mashqa_data/data.sh
```

### 环境注意

- 处理侧 python 只需 `transformers`（tokenizer）+ `pandas`（AgentBank），不需要 torch/sglang；
  本机 base conda 的 huggingface-hub 1.33 与 transformers `<1.0` 冲突时，自备一个干净 venv
  并用 `PY=<venv>/bin/python` 指定。
- `TOKENIZER` 必须与 sglang server 加载的模型一致（默认同取 `scripts/config.sh` 的 MODEL），
  否则 input_ids 错位全 miss。
- 国内网络靠 `HF_ENDPOINT=https://hf-mirror.com`（下载脚本已默认）；hf_hub_download 会先
  落 hub 缓存再拷到 local_dir，两处都要有 ~4GB 余量。

### 1. ShareGPT（多轮 chat 数据；实际队列由缓存状态机动态判断）
```bash
bash TaskData/SharedGPT-multi-turn/data.sh
# 产物: TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
```

### 2. Codeparrot（结构化单轮数据；实际队列由缓存状态机动态判断）
```bash
bash TaskData/Codeparrot/data.sh
# 产物: TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl
```

### 3. AgentBank（agentic 多轮 + 单轮 API）
```bash
bash TaskData/AgentBankData/data.sh
# 产物: TaskData/AgentBankData/Qwen2.5-7B-Instruct/
#   AgentBank_alfred_scale_5000_session_data.jsonl   (alfred_* → MULTI_TURN_AGENTIC)
#   AgentBank_apps_session_data.jsonl                (apps_* 仅为数据集命名空间)
```

### 4. MASHQA（结构化单轮长共享前缀；hybrid 四负载之外备用）
```bash
bash TaskData/mashqa_data/data.sh
# 产物: TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl
```

## 二、起 server（heter_cache 实验组）

```bash
# 需要先停掉占用 GPU/端口的其他 server
POLICY=heter_cache MAX_TOTAL_TOKENS=100000 bash scripts/serve.sh
# 或 lru 基线
POLICY=lru MAX_TOTAL_TOKENS=100000 bash scripts/serve.sh
```

## 三、跑负载（bench_hetercache.py）

```bash
source scripts/env.sh
"$PY" scripts/bench_hetercache.py \
  --data-path TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl \
  --base-url http://127.0.0.1:30000 --concurrency 8 --num-sessions 30 \
  --output result/smoke.json
```

## 四、全矩阵实验

```bash
bash scripts/run_all.sh   # POLICIES="lru heter_cache" × TOKENS_LIST="50000 100000 200000"
```

## 关键设计（为什么这么改）

1. **tokenizer 必须 Qwen2.5-7B-Instruct**：本地 HeterCache 数据是 Llama-3 tokenizer 产物
   （demo head20 也是），服务器模型是 Qwen2.5，必须重生成，否则 token ids 错位全 miss。
2. **直发 input_ids + session_id 标识会话**：session_id 前缀不再决定 task_type，
   同一 session 恒定且不影响 radix tree 命名空间。
3. **会话间并发、会话内串行**：下一轮依赖上一轮 KV，串行才能让前缀缓存命中。
4. **max-total-tokens 是关键**：调小到装不下工作集才触发淘汰，policy 差异才显现。

## 已验证结果（服务器实测，2026-10-18）

| 负载 | sessions | requests | 命中率 | 说明 |
|---|---|---|---|---|
| Codeparrot（单轮随机） | 30 | 30 | 4.3% | 跨 session 只共享 31-token system prompt（预期低） |
| AgentBank alfred（agentic 多轮） | 20 | 805 | 96.5% | 会话内历史前缀充分共享（预期高） |

## 注意事项

- demo `ShareGPT_cleaned_valid_session_data_head20.jsonl` 是 Llama-3 tokenizer 产物，
  仅可用于验证链路，勿用于正式对比。
- 3GB 的 alfred session_data 用 `--num-sessions` 限制样本量。
- heter_cache 需要 `--radix-eviction-policy=heter_cache` + `--enable-heter-cache` 同时给。
