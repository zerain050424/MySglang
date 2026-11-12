#!/usr/bin/env bash
# 为 mixexp 准备数据，两个串行步骤：
#
#   1) strip_qwen_system.py   去掉 Qwen 公共 system 段 → <name>.nosys.jsonl
#   2) drop_short_sessions.py  多轮源再剔掉轮数 < MULTI_TURN_MIN 的会话
#                              → <name>.nosys.mt5.jsonl
#
# 为什么第 1 步：Qwen 默认 system prompt 只有 24 token，但运行时判定跨会话共享
# 时没有 token 门槛，任一单轮会话只要命中过别人拥有的这段前缀就会被标成 STRUCT。
# 结果是 random 召回 0.000、apps 全被钉在 struct。离线消融实测（同数据）：
#
#   臂 A 生产缺省         acc 0.5791   random 召回 0.000
#   臂 C 只去 Qwen system acc 0.7697   random 召回 0.993
#
# 为什么第 2 步：分类器对 ``turn_count < MULTI_TURN_MIN`` 的会话判 TRANS，于是
# ShareGPT / alfred 里那些「总共只有 1–4 轮」的会话终身都是 TRANS——它们每轮都
# 在复用自己上一轮的前缀，却被 one-shot 带优先清掉（均匀命中率损失的主要来源）。
# 剔掉它们之后，TRANS 带里只剩单轮负载和长会话的前几轮。**单轮源（apps /
# Codeparrot）不做这一步**，它们本来就该是 TRANS。
#
# 用法：
#   bash mixexp/prepare_data.sh            # 幂等：已存在则跳过
#   FORCE=1 bash mixexp/prepare_data.sh    # 重新生成
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$HERE/.." && pwd)"
PYTHON="${PY:-python}"
STRIP_TOOL="$BENCH_DIR/tools/strip_qwen_system.py"
DROP_TOOL="$BENCH_DIR/tools/drop_short_sessions.py"
FORCE="${FORCE:-0}"

DATA=(
  "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl"
  "TaskData/Codeparrot/Qwen2.5-7B-Instruct/Code_session_data.jsonl"
  "TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl"
  "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl"
)

# 只有多轮源需要「剔除短会话」：它们刷新的是 AGENTIC/CHAT 队列，
# 短会话留在里面会以 TRANS 身份抢 one-shot 带。
MULTI_TURN=(
  "TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl"
  "TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl"
)

# full 档单任务最大 max_record；剔完不够这个数就要提示（数据会静默变少）。
NEED_PER_MULTI_TURN=500

[ -f "$STRIP_TOOL" ] || { echo "缺少 $STRIP_TOOL" >&2; exit 1; }
[ -f "$DROP_TOOL" ] || { echo "缺少 $DROP_TOOL" >&2; exit 1; }

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
  "$PYTHON" "$STRIP_TOOL" --in "$src" --mode qwen || missing=1
done

if [ "$missing" = "1" ]; then
  echo
  echo "有数据未就绪，mixexp 无法开跑。" >&2
  exit 1
fi

# 阈值只从代码常量取，不在 shell 里另写一个数字。
MIN_TURNS="$("$PYTHON" "$DROP_TOOL" --print-min-turns)" || exit 1

echo
echo "剔除多轮源里轮数 < $MIN_TURNS 的会话（threshold = MULTI_TURN_MIN）"
for rel in "${MULTI_TURN[@]}"; do
  src="$BENCH_DIR/${rel%.jsonl}.nosys.jsonl"
  dst="${src%.jsonl}.mt$MIN_TURNS.jsonl"
  if [ ! -f "$src" ]; then
    echo "!! 缺少中间文件：${src#$BENCH_DIR/}" >&2
    exit 1
  fi
  if [ -f "$dst" ] && [ "$FORCE" != "1" ]; then
    echo "跳过（已存在）：${dst#$BENCH_DIR/}"
    continue
  fi
  echo "过滤：${rel}"
  "$PYTHON" "$DROP_TOOL" --in "$src" || exit 1
done

# 剔完复检：保留会话数不够 full 档的 max_record 时提前说，
# 否则要到 bench 起跑才发现「只跑到一半会话」。
for rel in "${MULTI_TURN[@]}"; do
  dst="${BENCH_DIR}/${rel%.jsonl}.nosys.mt$MIN_TURNS.jsonl"
  [ -f "$dst" ] || { echo "!! 缺少过滤产物：${dst#$BENCH_DIR/}" >&2; exit 1; }
  kept=$(wc -l < "$dst")
  echo "  $(basename "$dst")：保留 $kept 条会话"
  if [ "$kept" -lt "$NEED_PER_MULTI_TURN" ]; then
    echo "  !! 少于 full 档所需（$NEED_PER_MULTI_TURN），收缩 max_record 或补数据源" >&2
  fi
done

echo
echo "全部就绪；多轮负载的 data_path 指向 *.nosys.mt$MIN_TURNS.jsonl，单轮负载指向 *.nosys.jsonl。"
