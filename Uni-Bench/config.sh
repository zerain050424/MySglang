# =============================================================================
# Uni-Bench 配置（被 run_all.sh source；也可手动 export 覆盖）
# 用法：source config.sh 后运行 run_all.sh
# =============================================================================

# 模型路径（tokenizer 必须与数据准备时一致，否则 token ids 错位）
export MODEL=${MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}

# sglang server 端口
export PORT=${PORT:-30000}

# session_data.jsonl 路径（三种负载切换，见下方注释）
#   - ShareGPT 多轮文本:   TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl
#   - Codeparrot 单轮随机: TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl
#   - AgentBank agentic:  TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl
#   - AgentBank 单轮 API: TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_single_api_session_data.jsonl
# 与 Qwen2.5 模型不匹配，仅用于验证链路，勿用于正式对比）
export DATA=${DATA:-/root/sg-unicache/Uni-Bench/TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl}

# 并发会话数（会话间并发，会话内串行）
export CONC=${CONC:-32}

# 混合负载配置（模仿 UniCache TaskData/Hybrid/*/tasks_config.yaml）。
# 用 bench_unicache.py --tasks-config 时生效；每个 task 的 lambda_rate 之比即混合比例。
export HYBRID_TASKS_CONFIG=${HYBRID_TASKS_CONFIG:-$BENCH_DIR/hybrid_tasks_config.yaml}
# 会话到达节奏缩放：1=UniCache 真实 Poisson 到达；0=全部立即到达（等同单负载并发）。
export ARRIVAL_SCALE=${ARRIVAL_SCALE:-1.0}

# 实验矩阵
export POLICIES=${POLICIES:-"lru simple_hybrid"}
export TOKENS_LIST=${TOKENS_LIST:-"50000 100000 200000"}   # KV pool token 上限（制造淘汰压力）

# 轮间等待缩放（inter-turn 到达间隔）：
#   0 = 不等（快速跑）
#   >0 = 按各负载类型各自的 lognormal(mu,sigma) 采样并乘此缩放。
#        mu/sigma 来自 arrival_config.yaml（模仿 UniCache TaskData/*/tasks_config.yaml），
#        由 session_id 前缀自动区分负载（text/agentic/single/random/file/image/search）。
export WAIT_SCALE=${WAIT_SCALE:-0.0}
