#!/usr/bin/env bash
# 为 mixexp 准备数据：生成去掉 Qwen 公共 system 段的副本 <name>.nosys.jsonl。
#
# 为什么要这一步：Qwen 默认 system prompt 只有 24 token，但运行时判定跨会话共享
# 时没有 token 门槛，任一单轮会话只要命中过别人拥有的这段前缀就会被标成 STRUCT。
# 结果是 random 召回 0.000、apps 全被钉在 struct。离线消融实测（同数据）：
#
#   臂 A 生产缺省         acc 0.5791   random 召回 0.000
#   臂 C 只去 Qwen system acc 0.7697   random 召回 0.993
#
# 用法：
#   bash mixexp/prepare_data.sh            # 幂等：已存在则跳过
#   FORCE=1 bash mixexp/prepare_data.sh    # 重新生成
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$HERE/.." && pwd)"
PYTHON="${PY:-python}"
TOOL="$BENCH_DIR/tools/strip_qwen_system.py"
FORCE="${FORCE:-0}"

DATA=(
  "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl"
  "TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl"
  "TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl"
  "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl"
)

[ -f "$TOOL" ] || { echo "缺少 $TOOL" >&2; exit 1; }

missing=0
for rel in "${DATA[@]}"; do
  src="$BENCH_DIR/$rel"
  dst="${src%.jsonl}.nosys.jsonl"
  if [ ! -f "$src" ]; then
    echo "!! 缺少原始数据：$rel（先在实验机上用 tools/prepare_session_data.py 生成）" >&2
    missing=1
    continue
  fi
  if [ -f "$dst" ] && [ "$FORCE" != "1" ]; then
    echo "跳过（已存在）：${dst#$BENCH_DIR/}"
    continue
  fi
  echo "剥离：${rel}"
  "$PYTHON" "$TOOL" --in "$src" --mode qwen || missing=1
done

if [ "$missing" = "1" ]; then
  echo
  echo "有数据未就绪，mixexp 无法开跑。" >&2
  exit 1
fi
echo
echo "全部就绪；mix 配置已指向 *.nosys.jsonl。"
