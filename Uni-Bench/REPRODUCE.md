# Uni-Bench 服务器复现流程（AutoDL + Qwen2.5-7B-Instruct）

本目录在服务器 `/root/sg-unicache/Uni-Bench`，是对本地 UniCache 数据准备与发送逻辑的
**复用 + 服务器适配**。所有脚本含中文注释，可直接按序执行。

## 环境

| 组件 | 位置 |
|---|---|
| UniCache 部署 venv | `/root/sl_old`（python 3.12.3，torch 2.13.0，与 9-26 基线 pin 一致） |
| 模型（tokenizer 必须一致） | `/root/autodl-tmp/model/Qwen2.5-7B-Instruct` |
| 原始数据集 | `/root/autodl-tmp/TaskData/`（下载脚本见 `download_datasets.sh`） |
| server 启动环境 | `env.sh`（venv python + ninja PATH + LD_LIBRARY_PATH） |

## 一、数据准备（每个负载一次）

### 1. ShareGPT（多轮文本 → MULTI_TURN_TEXT）
```bash
# 前提：/root/autodl-tmp/TaskData/SharedGPT-multi-turn/ShareGPT_V3_unfiltered_cleaned_split.json 已下载
bash TaskData/SharedGPT-multi-turn/data.sh
# 产物: TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
```

### 2. Codeparrot（单轮随机 → SINGLE_TURN，可切 apps_ 前缀归 SINGLE_TURN_RANDOM）
```bash
# 前提：/root/autodl-tmp/TaskData/Codeparrot/test.jsonl 已下载（已就绪）
bash TaskData/Codeparrot/data.sh
# 产物: TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl
```

### 3. AgentBank（agentic 多轮 + 单轮 API）
```bash
# 前提：AgentBank parquet 已下载（已就绪）
bash TaskData/AgentBankData/data.sh
# 产物: TaskData/AgentBankData/Qwen2.5-7B-Instruct/
#   AgentBank_alfred_scale_5000_session_data.jsonl   (alfred_* → MULTI_TURN_AGENTIC)
#   AgentBank_apps_session_data.jsonl                (apps_* → SINGLE_TURN_RANDOM)
```

## 二、起 server（simple_hybrid 实验组）

```bash
# 需要先停掉占用 GPU/端口的其他 server
POLICY=simple_hybrid MAX_TOTAL_TOKENS=100000 bash serve.sh
# 或 lru 基线
POLICY=lru MAX_TOTAL_TOKENS=100000 bash serve.sh
```

## 三、跑负载（bench_unicache.py）

```bash
source env.sh
"$PY" bench_unicache.py \
  --data-path TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl \
  --base-url http://127.0.0.1:30000 --concurrency 8 --num-sessions 30 \
  --output result/smoke.json
```

## 四、全矩阵实验

```bash
bash run_all.sh   # POLICIES="lru simple_hybrid" × TOKENS_LIST="50000 100000 200000"
```

## 关键设计（为什么这么改）

1. **tokenizer 必须 Qwen2.5-7B-Instruct**：本地 UniCache 数据是 Llama-3 tokenizer 产物
   （demo head20 也是），服务器模型是 Qwen2.5，必须重生成，否则 token ids 错位全 miss。
2. **直发 input_ids + session_id 打分类**：session_id 前缀决定 task_type
   （ShareGPT-multi-turn-* / alfred_* / apps_* / mind2web_*），同一 session 恒定，
   不影响 radix tree 命名空间。
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
- simple_hybrid 需要 `--radix-eviction-policy=simple_hybrid` + `--enable-simple-hybrid` 同时给。
