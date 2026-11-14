# =============================================================================
# Heter-Bench 配置（被 serve.sh source；也可手动 export 覆盖）
# 用法：source config.sh 后再跑 serve.sh
# =============================================================================

# 模型路径（tokenizer 必须与数据准备时一致，否则 token ids 错位）
export MODEL=${MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}

# sglang server 端口
export PORT=${PORT:-30000}
