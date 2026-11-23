# 计划：TaskData 一键数据准备（下载 + 处理）

## Context

现状：`TaskData/` 下四个数据集各有处理脚本（`prepare_data.py` 等），`data.sh` 编排
「原始数据 → 对话 JSON → session_data.jsonl」，但有两个断点：

1. **没有下载脚本**。原服务器有个 `download_datasets.sh`，本仓库里只留下引用，
   文件本身不存在。新环境拿到的是「假设原始数据已在 `/root/autodl-tmp/TaskData`」
   的半套流程。
2. **路径全部硬编码**。各 `data.sh` 里写死了 `/root/sl_old/bin/python`、
   `/root/autodl-tmp/model/Qwen2.5-7B-Instruct`、`/root/autodl-tmp/TaskData/...`，
   换台机器就得手改四个脚本。

目标：新环境 clone 下来后，一条命令完成「下载原始数据 → 四个负载全部产出
session_data.jsonl」。

## 需要的原始数据源（已核实）

| 数据集 | 来源 | 产物 | 在 hybrid 配置里? |
|---|---|---|---|
| ShareGPT | HF `anon8231489123/ShareGPT_Vicuna_unfiltered` 的 `ShareGPT_V3_unfiltered_cleaned_split.json`（单文件直链） | ShareGPT_cleaned_valid_deduplicated_session_data.jsonl | 是 (task 2) |
| Codeparrot (APPS) | HF `codeparrot/apps` 的 `test.jsonl`（1.29GB 单文件） | Code_session_data.jsonl | 是 (task 1) |
| AgentBank | HF `Solaris99/AgentBank` 整仓 parquet（snapshot 下载） | AgentBank_apps / alfred_scale_5000 session_data | 是 (task 0, 3) |
| MASHQA | HF `VivekNaga/Medicaldata`（train/test/val `_full.json` 三个文件，LFS） | mashqa_session_data.jsonl | 否（可选） |

备用镜像：所有 HF 下载走 `HF_ENDPOINT=https://hf-mirror.com`（国内环境必需），
ShareGPT 备用源还有 Kaggle / 百度网盘，脚本里给直链即可。

## Approach

1. **新增 `TaskData/download_datasets.sh` + `download_datasets.py`**：
   - python 脚本用 `huggingface_hub`（hf_hub_download / snapshot_download），
     支持断点续传与 HF_ENDPOINT 镜像；AgentBank 整仓下载，其余按文件名下。
   - 下载目标目录 `RAW_DIR` 用环境变量控制。
2. **改四个 `data.sh`**：把硬编码路径全部改成环境变量 + 合理默认值：
   - `PY`（默认 `python3`）
   - `MODEL` / `TOKENIZER`（默认取 `scripts/config.sh` 的 `MODEL`）
   - `RAW_DIR`（默认与处理产物分开，见待定问题 1）
   - 行为不变：流程、产物文件名、id 前缀都不动。
3. **新增 `TaskData/prepare_all.sh` 一键入口**：download → 依序跑四个 data.sh，
   每个 dataset 幂等（原始文件已存在则跳过下载；产物已存在可提示跳过或重跑）。

## Files to modify

- 新增 `TaskData/download_datasets.sh`（壳）+ `TaskData/download_datasets.py`（下载逻辑）
- 新增 `TaskData/prepare_all.sh`（一键入口）
- 修改 `TaskData/{SharedGPT-multi-turn,Codeparrot,AgentBankData,mashqa_data}/data.sh`
  （只改路径/变量，不动处理逻辑）
- 更新 `docs/REPRODUCE.md` 的数据准备段落（去掉「已就绪」假设，写清一键用法）

**不改**：`tools/prepare_session_data.py`、各 `prepare_*.py`、`Alfred_scale_up.py`、
`deduplicate.py`、`merge_app_mind2web.py`、`configs/hybrid_tasks_config.yaml`。

## Reuse

- `tools/prepare_session_data.py`：统一的 session_data 转换器（transformers tokenizer），
  四个 data.sh 都已在调它，不动。
- `scripts/config.sh`：已有 `MODEL` 变量，新脚本从这里取默认值。
- 各 dataset 的 `prepare_*.py`：处理逻辑全部保留，只改外层编排。

## 决议（用户已确认）

1. **RAW_DIR 默认 `$UB/TaskData/raw/`**，支持 `RAW_DIR=...` 环境变量覆盖；子目录
   `raw/SharedGPT-multi-turn/`、`raw/Codeparrot/`、`raw/AgentBank/AgentBank/`
   （snapshot_download 展开后 data/<task>/*.parquet 在其下）、`raw/mashqa_data/`。
2. **PY 默认 `python3`；TOKENIZER 默认 source `scripts/config.sh` 拿 `MODEL`**
   （config.sh 自身也支持 `MODEL=` 环境变量覆盖，优先级：显式 env > config.sh）。
3. **MASHQA 纳入** prepare_all 流程（约 200MB，四个负载外加一个备用）。
4. **下载用 python `huggingface_hub`**（hf_hub_download / snapshot_download，
   断点续传 + `HF_ENDPOINT` 镜像支持），外层包一个 `download_datasets.sh` 薄壳。

## Steps

- [x] Step 1: 写 `TaskData/download_datasets.py`：
    - ShareGPT：`hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
      "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")` →
      `raw/SharedGPT-multi-turn/`
    - Codeparrot：`hf_hub_download("codeparrot/apps", "test.jsonl",
      repo_type="dataset")` → `raw/Codeparrot/`
    - AgentBank：`snapshot_download("Solaris99/AgentBank", ...)` → `raw/AgentBank/`
      （prepare_agentbank_data.py 期望 `AgentBank/data/<task>/*.parquet` 结构）
    - MASHQA：`hf_hub_download("VivekNaga/Medicaldata", <3 个 _full.json>,
      repo_type="dataset")` → `raw/mashqa_data/`
    - 已存在且大小合理的文件直接跳过；无 TOKENIZER/重依赖，只要 huggingface_hub
- [x] Step 2: 写 `TaskData/download_datasets.sh` 薄壳（env 解析 + export HF_ENDPOINT 默认镜像 + 调 python）
- [x] Step 3: 改四个 `data.sh`：`PY`/`TOKENIZER`/`RAW` 全部改为 env 变量 + 默认值
    （脚本顶部先 source `$UB/scripts/config.sh` 取 MODEL 默认），处理逻辑与产物不动
- [x] Step 4: 写 `TaskData/prepare_all.sh`：download → 依序跑四个 data.sh，
    `set -eu`，任一失败即停；支持 `SKIP_DOWNLOAD=1` 只跑处理
- [x] Step 5: 更新 `docs/REPRODUCE.md` 数据准备段落：一键用法 + 变量说明 +
    环境注意（transformers venv、HF_ENDPOINT）
- [x] Step 6: 验证（见下）

## Verification

- `bash -n` 全部新/改的 sh（注意 CRLF → `tr -d '\r'` 先转，见 AGENTS.md 5.4）
- 语法检查 python：`python -m py_compile`
- 幂等性：重复跑 prepare_all.sh，第二次应跳过已下载文件、产物存在时行为明确
- 本机若 transformers/tokenizer 可用，小样本跑通 Codeparrot 全链（`--max-samples`）
  产物行数、格式抽查；若碰上 base conda 的 huggingface-hub 1.33 冲突（AGENTS.md 5.1），
  在文档里写明需自备带 transformers 的 venv
