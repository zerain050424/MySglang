# Heter-Bench 服务器复现流程（AutoDL + Qwen2.5-7B-Instruct）

本目录在服务器 `/root/sg-unicache/Uni-Bench`，是对本地 HeterCache 数据准备与发送逻辑的
**复用 + 服务器适配**。所有脚本含中文注释，可直接按序执行。

> **驱动已下线（2026-11）**：本文的 `bench_hetercache.py` / `run_all.sh` 命令已随旧
> 发射器一并删除，仅作历史记录；当前驱动为 `scripts/bench_heap.py`（只支持
> `--tasks-config`，无单负载 / 相位 / burst-pause 模式）。

## 环境与工具链

仓库采用**双环境解耦**设计，彻底消除对特定云平台（如 AutoDL `/root/autodl-tmp`）的硬编码依赖：

| 组件 | 默认位置 / 说明 | 准备脚本 |
|---|---|---|
| 轻量数据处理 venv | `.venv_data`（仅需 `transformers`/`pandas`/`modelscope`，无 CUDA/Torch 依赖） | `bash scripts/setup_data_env.sh` |
| 基准模型与 Tokenizer | `models/Qwen2.5-7B-Instruct`（默认 ModelScope 国内高速免翻墙下载，支持 HF 镜像） | `bash scripts/download_model.sh` |
| 原始数据集 | `TaskData/raw/`（断点续传，国内默认走 `https://hf-mirror.com`） | `bash TaskData/download_datasets.sh` |
| SGLang 生产/评测 venv | `.venv_sglang`（完整 PyTorch + CUDA + FlashInfer + SGLang editable install） | `bash scripts/setup_sglang_env.sh` |
| Server 启动环境配置 | `scripts/env.sh`（指定运行环境 `PY`、ninja `PATH` 与 `LD_LIBRARY_PATH`） | 手动配置或 source 虚拟环境 |

---

## 零、新服务器一键全流程（推荐）

在全新服务器克隆本仓库后，只需在 `Heter-Bench/` 目录下执行一条命令，即可按序自动完成「轻量数据环境创建 → 模型下载 → 数据集下载 → 4 个负载 session_data 生成」：

```bash
cd Heter-Bench
bash scripts/init_pipeline.sh
```

> **常用控制开关（支持断点续跑）：**
> - `SOURCE=huggingface bash scripts/init_pipeline.sh`：模型下载改走 Hugging Face 镜像。
> - `SKIP_VENV=1 bash scripts/init_pipeline.sh`：跳过环境创建，直接复用当前环境。
> - `SKIP_MODEL=1 bash scripts/init_pipeline.sh`：跳过模型下载步骤。
> - `SKIP_DOWNLOAD=1 bash scripts/init_pipeline.sh`：跳过数据集下载，仅重新跑 session_data 处理。

---

## 一、分步数据准备

### 0. 准备数据环境与模型（若未使用一键流水线）

```bash
# 1. 创建轻量数据处理环境（仅需几十秒）
bash scripts/setup_data_env.sh

# 2. 下载 Qwen2.5-7B-Instruct 模型与 Tokenizer
bash scripts/download_model.sh
```

### 0.1 运行数据准备入口

```bash
# 默认使用 .venv_data/bin/python 与 models/Qwen2.5-7B-Instruct：
bash TaskData/prepare_all.sh

# 常用变体：
SKIP_DOWNLOAD=1 bash TaskData/prepare_all.sh          # 原始数据已就绪，只跑处理
DATASETS="sharegpt apps" bash TaskData/prepare_all.sh # 只下指定数据集
RAW_DIR=/data/raw bash TaskData/prepare_all.sh        # 原始数据放别处
PY=/root/sl_old/bin/python bash TaskData/prepare_all.sh # 指定自定义 python
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
