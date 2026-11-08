# 实验总结：`min_share=64` 的会话级分类效果（三件套指标）

**配置**：`--source trace --min-share-tokens 64`，其余全部为生产缺省
（`multi_turn_min=5`、`gap_split=15s`、`growth=1500`、`tool_call=prior`、`structure=prefix`）。

**数据**：`result/phased_load_prediction_C.csv`（真实请求时间轴），1537 条会话 / 14532 轮：
struct 480（Code，单轮共享代码前缀）、random 295（apps，单轮）、chat 462（ShareGPT）、
agentic 300（alfred，带 tool_call）。插入 KV 合计 29.9M token。

**唯一改动**：跨会话结构证据要求「共享前缀 ≥ 64 token（4 个 16-token 块）」才计入，
即只把 24-token 公共 system prompt 造成的命中过滤掉。

---

## 指标一：最终判定（会话级）

| 指标 | 值 |
| --- | --- |
| 会话级准确率 | **0.7710**（1185/1537） |
| macro-F1 | 0.7525 |
| 规则可达上限 | 0.7716 |
| 达成率（acc / 上限） | **0.9992** |

| GT \ pred | struct | random | chat | agentic | 合计 |
| --- | --- | --- | --- | --- | --- |
| struct | 479 | 1 | 0 | 0 | 480 |
| random | 0 | **295** | 0 | 0 | 295 |
| chat | 7 | 344 | 111 | 0 | 462 |
| agentic | 0 | 0 | 0 | 300 | 300 |

| 类别 | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| struct | 0.986 | 0.998 | 0.992 | 480 |
| random | 0.461 | **1.000** | 0.631 | 295 |
| chat | 1.000 | 0.240 | 0.387 | 462 |
| agentic | 1.000 | 1.000 | 1.000 | 300 |

对照生产缺省（`min_share=1`）：acc 0.5791 → **0.7710**（+0.192），
增益全部来自 random 桶 0.000 → 1.000；struct / agentic 召回不动。

---

## 指标二：多久判对（逐轮精度 + 收敛轮）

| k（只重放前 k 轮） | 会话数 | 准确率 |
| --- | --- | --- |
| 1 | 1537 | **0.5036** |
| 2 | 618 | 0.0000 |
| 3 | 531 | 0.0000 |
| 4 | 475 | 0.0000 |
| 5 | 411 | **1.0000** |
| ≥6 | 366 → 1 | 1.0000 |

判定收敛轮次：struct 1、random 1、chat 中位 1（短会话直接停在 trans）/ p90 5、agentic 5。

读法：k=1 只覆盖单轮会话（struct/random 定型），因此从 0.3116 升到 0.5036；
k=2–4 剩下的全是多轮会话，按规则只可能是 trans/struct，所以准确率为 0 是**规则使然**，
不是随机波动；第 5 轮所有多轮会话一次性定型，k≥5 全为 1.0000。

---

## 指标三：运行期 token 级错标暴露

| 类别 | 会话 | 曾判错会话 | 逐轮错标率 | token 错标率 | qstruct 黏性错标 tok（占该类 tok） |
| --- | --- | --- | --- | --- | --- |
| struct | 480 | 1（0.002） | 0.0021 | 0.0054 | 0（0.0000） |
| random | 295 | **0**（0.000） | 0.0000 | 0.0000 | 0（0.0000） |
| chat | 462 | 462（1.000） | 0.8101 | 0.6322 | 15,880（0.0164） |
| agentic | 300 | 300（1.000） | 0.0976 | 0.0072 | 0（0.0000） |
| **合计** | 1537 | 763（0.496） | **0.1643** | **0.0274** | **15,880（0.0005）** |

「黏性错标」= 会话被判成 `struct` 且真值不是 struct 的那些轮次写入的 KV。`qstruct` 是
黏性标签（`_record_node_hit` 只改写非 qstruct 节点），这部分 KV 在改判后**不会自愈**；
其余错标（trans ↔ chat/agentic）会在后续命中时被重写。

与生产缺省对比：

| 指标 | min_share=1 | **min_share=64** | 变化 |
| --- | --- | --- | --- |
| 逐轮错标率 | 0.1846 | 0.1643 | −0.020 |
| token 错标率 | 0.0311 | 0.0274 | −0.004 |
| **黏性错标 token（不可自愈）** | 927,410（3.10%） | **15,880（0.05%）** | **−98.3%（58×）** |
| ↳ random 的黏性错标 | 108,797 | 0 | 清零 |
| ↳ agentic 的黏性错标 | 205,131 | 0 | 清零 |
| ↳ chat 的黏性错标 | 613,482 | 15,880 | −97.4% |

---

## 结论

1. **最终判定基本到顶**：0.7710 与上限 0.7716 只差 1 条会话，达成率 0.9992。
   剩余误差不是分类器问题，而是 `>4` 轮门槛：chat 有 351/462（76%）会话轮数 <5，
   结构上不可能被判成 chat。
2. **真正被修掉的是「不可自愈」的错标**：黏性 token 从 3.10% 降到 0.05%。
   生产缺省下 apps 与 ShareGPT 首轮 KV 会被 24-token 公共 system prompt 钉死在 `qstruct`，
   抬高门限后这部分彻底消失；agentic 前 4 轮的黏性误标也归零。
3. **代价为零**：struct 召回 0.998 不变（Code 前缀 ~0.9k token，远高于 64），
   agentic 1.000 不变，逐轮曲线只在 k=1 变好（0.3116 → 0.5036）。
4. **仍存在的 2.74% token 错标主要来自 chat 的短会话**（chat 占 63.2%，但 chat 总量只有
   0.97M token，占全部插入量 3.2%）：它们被判成 trans（random 桶）并长期停留。
   这不是 `min_share` 能解决的，只能放宽轮数门槛或引入新信号（无 tool_call + 上下文弱增长）。
5. **边界案例**：仍有 7 条 chat 会话（1.64% chat token）因共享前缀 ≥64 token 被判成 struct。
   这类是真正的跨会话共享（同源 prompt 被复用），可视为「struct 语义成立」而非误判，
   但报告里应单列，避免把 struct 精确率读到 1.000。

## 限制

- `token 错标率` 的每轮写入量取该次真实运行的 `recompute_tokens`；这次运行用的是
  `min_share=1` 的标签（标签影响淘汰顺序），因此它是**估计值**，方向而非精确绝对值。
- 前缀索引是 16-token 块粒度、不模拟淘汰，共享证据偏多；`min_share=64` 的结论因此偏保守。
- `min_share` 目前只存在于评测侧（`classify/replay.py` 的 `PrefixIndex` 门控）。生产
  `RadixCache._record_cross_session_hit` 仍无最小长度限制，落地需要改代码并复跑一次 server
  验证（node dump 的 `stype` 可直接 join `sessions.csv`）。

## 复现

```bash
cd Uni-Bench/classify
/root/sl_old/bin/python run_eval.py --source trace --min-share-tokens 64 --tag share64
# 产物：../result/classify/report_share64.md / metrics_share64.json / sessions_share64.csv
# 对照：run_eval.py --source trace --tag baseline
```
