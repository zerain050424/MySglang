# Heter-Bench 数据准备指南（TaskData）

本指南介绍如何在全新服务器上完成从环境初始化、模型下载到四个基准负载 `session_data.jsonl` 生成的完整流程。

---

## 一、一键端到端打通（新服务器推荐）

在全新服务器克隆代码后，直接运行一键 Pipeline：

```bash
cd Heter-Bench
bash scripts/init_pipeline.sh
```

该脚本将自动执行以下 3 个阶段：
1. **环境准备 (`scripts/setup_data_env.sh`)**：创建轻量 Python 虚拟环境（默认 `.venv_data`），安装 `huggingface_hub`, `modelscope`, `transformers`, `pandas`。
2. **模型下载 (`scripts/download_model.sh`)**：通过 ModelScope（默认，国内高速）或 Hugging Face 镜像下载 `Qwen/Qwen2.5-7B-Instruct` 至 `models/Qwen2.5-7B-Instruct`。
3. **数据下载与处理 (`TaskData/prepare_all.sh`)**：下载 4 个数据集原始数据并生成供 `configs/hybrid_tasks_config.yaml` 消费的 `session_data.jsonl`。

流水线具备幂等性与断点续传能力，支持如下环境变量控制：
- `SOURCE=huggingface`：模型下载切换为 Hugging Face 镜像。
- `SKIP_VENV=1`：跳过虚拟环境创建步骤。
- `SKIP_MODEL=1`：跳过模型下载步骤。
- `SKIP_DOWNLOAD=1`：跳过数据集下载步骤，仅重新执行处理。

---

## 二、分步执行说明

### 1. 独立初始化轻量数据处理环境

```bash
bash scripts/setup_data_env.sh
# 环境变量支持:
#   VENV_DIR=/path/to/venv       指定虚拟环境路径（默认: .venv_data）
#   PIP_INDEX_URL=https://...    指定 PyPI 镜像（默认: 清华开源镜像站）
```

### 2. 独立下载 Qwen2.5-7B-Instruct 模型

```bash
# 默认使用 ModelScope 下载至 models/Qwen2.5-7B-Instruct
bash scripts/download_model.sh

# 使用 Hugging Face 镜像下载至自定义目录
SOURCE=huggingface MODEL=/data/models/Qwen2.5-7B-Instruct bash scripts/download_model.sh
```

### 3. 数据集下载与 session_data 处理

```bash
# 默认自动读取 scripts/config.sh 中的 MODEL 与 .venv_data/bin/python
bash TaskData/prepare_all.sh

# 常用变体:
SKIP_DOWNLOAD=1 bash TaskData/prepare_all.sh          # 仅处理，不下载
DATASETS="sharegpt apps" bash TaskData/prepare_all.sh # 仅下载特定子集
```

---

## 三、基准负载产物清单

完成上述流程后，各任务目录将产出对应的标准 `session_data.jsonl`：

| 负载类别 | 原始来源 | 处理生成产物 | 对应 Hybrid 评测任务 |
|---|---|---|---|
| Single-turn API | HF `Solaris99/AgentBank` | `TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl` | Task 0 |
| Code (APPS) | HF `codeparrot/apps` | `TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl` | Task 1 |
| Multi-turn Chat | HF `anon8231489123/ShareGPT_Vicuna_unfiltered` | `TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl` | Task 2 |
| Multi-turn Agentic | HF `Solaris99/AgentBank` | `TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl` | Task 3 |
| Medical QA (备用) | HF `VivekNaga/Medicaldata` | `TaskData/mashqa_data/Qwen2.5-7B-Instruct/mashqa_session_data.jsonl` | 备用负载 |

---

## 四、完整 SGLang 生产运行环境配置

若需在服务器上启动 `sglang serve` 实验组并跑实际 GPU 评测，可使用配套的完整环境安装脚本：

```bash
bash scripts/setup_sglang_env.sh
```
该脚本将构建包含 PyTorch (CUDA)、FlashInfer 并以 `-e` 方式安装当前仓库 SGLang 的完整虚拟环境（默认 `.venv_sglang`）。
