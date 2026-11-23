# =============================================================================
# Heter-Bench 配置（被 serve.sh / download_model.sh / data.sh source；支持手动 export 覆盖）
# 用法：source scripts/config.sh
# =============================================================================
_HB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 模型路径（tokenizer 必须与数据准备时一致，否则 token ids 错位）
export MODEL=${MODEL:-$_HB_DIR/models/Qwen2.5-7B-Instruct}

# sglang server 端口
export PORT=${PORT:-30000}
