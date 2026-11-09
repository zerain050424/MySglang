# 会话级分类准确率评测（classify/）

目标：把「会话类型分类器」（`SessionRegistry.session_type`）的准确率按**会话口径**测出来，
而不是像 `../evaluate_node_labels.py` 那样按节点口径测。

- **单元**：一条会话只贡献一个判定——它最后一轮结束时的 `session_type`。
- **真值**：会话语义（`session_id` 前缀 → 负载家族），与节点级评测共用同一张表：

  | 前缀 | 真值 | 负载 |
  | --- | --- | --- |
  | `Code-` | struct | Codeparrot（单轮，跨会话共享同一代码前缀） |
  | `apps_` | random | AgentBank apps（单轮，彼此不共享内容） |
  | `ShareGPT` | chat | ShareGPT（人类节奏多轮） |
  | `alfred_` | agentic | AgentBank alfred（带 `tool_call` 的多轮） |

- **预测**：`trans→random`、`struct→struct`、`chat→chat`、`agentic→agentic`。
  `trans` 是「未判定」，对单轮且无跨会话结构共享的会话它就是正确桶
  （与 `task_type_of_tag(QTRANS) == RANDOM_SINGLE_TURN` 一致）。
- **不复制规则**：重放直接调用生产代码 `python/sglang/srt/mem_cache/heter_utils.py`
  的 `SessionRegistry`，评测的就是正在跑的规则，而不是它的替身。

## 目录

| 文件 | 作用 |
| --- | --- |
| `session_data.py` | 数据口径：任务数据 / 真实 trace → 会话事件流；GT 映射；轮间隔模型 |
| `replay.py` | 把证据按真实时间顺序喂进 `SessionRegistry`；块粒度前缀归属索引（跨会话共享） |
| `metrics.py` | 会话级 acc / macro-F1 / P-R-F1 / 混淆矩阵 / 逐轮精度 / 收敛轮 / 可达上限 / token 级错标暴露 |
| `dataset_build.py` | CLI 参数与数据集装配 |
| `run_eval.py` | 主实验，输出 `report.md` + `metrics.json` + `sessions.csv` |
| `sweep.py` | 阈值与证据口径扫描（单变量） |
| `SHARE64_SUMMARY.md` | `min_share=64` 的三件套指标实验总结（最终判定 / 逐轮精度 / token 暴露） |
| `SYSTEM_PROMPT_ABLATION.md` | 去掉公共 Qwen system prompt 后重跑：与抬高 `min_share` 等价性验证 |
| `分类器判定逻辑.md` | 判定逻辑全流程：证据 → 状态 → 有序规则 → 队列标签（含判定表与边界情况） |
| `分类器实验.md` | 面向汇报的完整实验报告（数据准备 / 纯 CPU 模拟设计 / 三件套结果 / 局限），图在 `figs/` |
| `分类器残差分析.md` | 附录 A：去掉公共 system prompt 后仍有 12 条 chat 被判 struct 的逐条证据 |
| `run.sh` | 一键跑 4 组 |
| `tests/test_session_accuracy.py` | 自检：口径映射、trace 回填、重放语义、指标计算 |

## 两种数据来源

1. **`--source trace`（主口径）**：读 `result/phased_load_prediction_C.csv` 的真实请求时间轴。
   时间用 `ts` 的差分，正是分类器 `now - last_request_start` 的口径（不用
   `gap_since_prev_end_s`，那是 end→start）。`prior_tool_call` 用任务数据里**上一轮**的
   `tool_call` 回填，与 bench 的传递方式一致；prompt 长度用实测 `prompt_tokens`。
2. **`--source task`（扩样本 / 消融）**：读 `TaskData/*/Qwen2.5-7B-Instruct/*_session_data.jsonl`，
   轮间隔由模型给出：`per_workload`（chat=20s / agentic=2s / 单轮=0，见
   `phased_load_prediction.yaml`）、`fixed:<s>`、`lognormal:<mu>:<sigma>`。

## 跨会话结构证据怎么重放

生产里 `RadixCache._record_cross_session_hit` 只要命中「别的会话写入的缓存节点」就记一次
跨会话共享，**不设最小长度**。离线用 `PrefixIndex` 复刻这个语义：

- 按块（默认 16 token）给前缀算滚动哈希，只登记首写归属；`match` 返回「被其他会话拥有的
  最长共享前缀长度 + owner」，与逐节点命中同义；
- 只登记未见过的前缀块，内存 ≈ 不同前缀块数；
- 不模拟淘汰，因此是「共享证据」的上界；
- 默认只喂首轮 prompt（这些负载的结构性共享都发生在首轮 system prompt / 代码仓库前缀上），
  `--structure-turns all` 可放开到全部轮次。

## 复现

```bash
cd Heter-Bench/classify
/root/sl_old/bin/python run_eval.py --source trace                 # 真实 trace 主结果
/root/sl_old/bin/python run_eval.py --source task --max-per-workload 1500 --tag task
/root/sl_old/bin/python sweep.py --source trace --tag trace        # 单变量扫描
/root/sl_old/bin/python run_eval.py --source trace --strip-system-prompt qwen  # 去掉公共 system prompt
bash run.sh                                                        # 以上四组一起跑
/root/sl_old/bin/python -m pytest tests/test_session_accuracy.py -q
```

输出在 `../result/classify/`；`sessions.csv` 逐会话给出 `gt, pred, n_turns, converged_turn,
shared_len, ever_wrong, mislabel_turns, mislabel_tokens, preds`，可以直接和后续 node dump
（`stype`）按 `session_id` join。

报告里同时给三套口径，缺一不可：

1. **最终判定**：`accuracy` / `macro_f1` / `ceiling` / `achievement`——会话最后判对了吗；
2. **逐轮精度**：`curve`（`acc_at_turn`）+ `convergence`——第几轮才判对；
3. **token 级暴露**：`exposure`——进行中有多少 KV 被标错，其中 `sticky_tokens` 是
   `qstruct` 黏性错标（改判后不会自愈），其余会在后续命中时重写。

## 结果

### 1) 真实 trace：1537 会话（struct 480 / random 295 / chat 462 / agentic 300）

| 指标 | 值 |
| --- | --- |
| 会话级准确率 | **0.5791** |
| macro-F1 | 0.4961 |
| 规则可达上限 | 0.7716 |
| 达成率（acc / 上限） | **0.7505** |

| GT \ pred | struct | random | chat | agentic |
| --- | --- | --- | --- | --- |
| struct | 479 | 1 | 0 | 0 |
| random | **295** | 0 | 0 | 0 |
| chat | 351 | 0 | 111 | 0 |
| agentic | 0 | 0 | 0 | 300 |

- `agentic` 召回 1.000（300/300），`chat` 在「≥5 轮」的可达集合里 111/111 全对，
  `struct` 479/480。
- `random` 召回 **0**：apps 会话与别人共享同一个 24 token 的 Qwen system prompt，
  按生产语义（任意节点命中即跨会话共享）就被判成 `struct`。
  这不是仿真独有的：`result/node_labels_phased.jsonl` 里 163 条 apps 会话中有 15 条
  的节点带过 `qstruct`，ShareGPT 侧 379 条里 137 条也出现过 `qstruct`。
- 逐轮精度：k=1 时 0.31（单轮 struct/random 定型），k=2–4 掉到 0（多轮会话都还在
  `trans`），k≥5 直接 1.00。分类是「第 5 轮一次性定型」，不是渐进逼近。

### 2) 任务数据回放：6001 会话（每负载 1500）

acc 0.5677、macro-F1 0.4909、上限 0.8179、达成率 0.6941；
chat 召回 0.272、随机桶 0、struct 0.999、agentic 1.000。与真实 trace 同构。

### 3) 扫描：准确率由哪些规则决定（真实 trace）

| 变体 | acc | 上限 | 达成率 | struct R | random R | chat R | agentic R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 生产缺省 | 0.5791 | 0.7716 | 0.7505 | 0.998 | 0.000 | 0.240 | 1.000 |
| structure=off | 0.4593 | 0.7716 | 0.5953 | 0.000 | 1.000 | 0.240 | 1.000 |
| min_share=32 | 0.7671 | 0.7716 | 0.9941 | 0.998 | 0.980 | 0.240 | 1.000 |
| **min_share=64** | **0.7710** | 0.7716 | **0.9992** | 0.998 | 1.000 | 0.240 | 1.000 |
| min_share=128 | 0.7710 | 0.7716 | 0.9992 | 0.998 | 1.000 | 0.240 | 1.000 |
| min_share=512 | 0.7053 | 0.7716 | 0.9140 | 0.787 | 1.000 | 0.240 | 1.000 |
| gap_split=5/10/20s | 0.5791 | 0.7716 | 0.7505 | 0.998 | 0.000 | 0.240 | 1.000 |
| gap_split=30s | 0.5407 | 0.7716 | 0.7007 | 0.998 | 0.000 | 0.113 | 1.000 |
| multi_turn_min=3 | 0.6571 | 0.8497 | 0.7734 | 0.998 | 0.000 | 0.500 | 1.000 |
| multi_turn_min=6 | 0.5498 | 0.7424 | 0.7405 | 0.998 | 0.000 | 0.143 | 1.000 |
| tool_call=off | 0.5791 | 0.7716 | 0.7505 | 0.998 | 0.000 | 0.240 | 1.000 |
| growth=500 / 3000 | 0.5791 | 0.7716 | 0.7505 | 0.998 | 0.000 | 0.240 | 1.000 |

### 4) slow-gap 场景：把轮间隔整体压成 20s（只有 `--source task` 才有）

| 变体 | acc | 达成率 | agentic R |
| --- | --- | --- | --- |
| slow-gap baseline | 0.5677 | 0.6941 | 1.000 |
| slow-gap + tool_call=off | 0.3178 | 0.3885 | **0.000** |

## 结论

1. **规则在「可达集合」内几乎是满的**：agentic 100%，≥5 轮 chat 100%，struct 99.8%。
   剩下的误差几乎只有两个来源，而且都可定位。
2. **跨会话共享不设最小长度是 random 桶全军覆没的原因**。24 token 的公共 system prompt
   就足以把单轮会话钉成 `struct`；把门限抬到 **≥64 token（4 个块）** 后，random 召回
   0→1.0、struct 召回不动（Code 前缀 ~0.9k token），达成率 0.7505→**0.9992**
   （task 6100 会话：0.6941→0.9971）。抬到 512 就过头了，struct 掉到 0.79/0.48。
3. **15s 的间隔切分在真实数据上很稳**（5–30s 都只影响 chat 的尾巴）：真实 chat 中位
   start→start 间隔 ~28.5s，agentic ~5.5s，中间有很宽的安全区。
4. **`tool_call` 证据在当前 trace 里没有边际贡献**（agentic 间隔本来就短，间隔规则已足够）。
   但它是必需的保险：一旦 agentic 的间隔变慢（工具往返/人工确认，20s 场景），关掉它
   agentic 召回直接归零、达成率 0.694→0.389。保留「只传上一轮」的传递方式即可。
5. **`>4` 轮门槛是剩余误差的唯一来源**：chat 会话里 76%（trace，351/462）轮数 <5，
   按规则不可能判成 chat；`multi_turn_min=3` 能把 chat 召回抬到 0.50（达成率 0.7734），
   代价是短会话更容易被间隔/增长规则抢走。门槛是超参，不是缺陷——但报告准确率时必须
   同时给「可达上限」，否则数字会被短会话掩盖。

## 已知限制

- 前缀索引是 **16 token 块粒度**的近似（生产 `page_size=1`），只影响共享长度的量化；
  不模拟淘汰，所以「共享证据」偏多，`min_share=64` 的结论是保守方向。
- trace 只有 1537 条会话、且来自 `phased_load_prediction_C.csv` 那一次跑；更广的验证
  应该换新的 bench 输出重跑（脚本对任意同 schema 的 CSV 通用，`--trace-csv` 指定即可）。
- `apps_` / `Code-` 都是单轮，因此这两类只检验「结构证据 + 会话类型」的映射，
  不检验多轮演化；多轮演化由 ShareGPT / alfred 覆盖。
- 任务数据模式的轮间隔是模型（`per_workload` / `fixed` / `lognormal`），不是实测；
  所以主结论只引用 `--source trace`，task 模式用于扩大样本和做反事实消融。
