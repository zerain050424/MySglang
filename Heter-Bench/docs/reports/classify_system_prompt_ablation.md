# 实验总结：去掉 Qwen 公共 system prompt 后重跑（对照 `min_share=64`）

**问题**：`min_share=64` 的收益全部来自「24-token 公共 Qwen system prompt 导致的跨会话误命中」。
如果直接从 workload 里把这段 system prompt 去掉，是不是不改阈值也能得到同样结果？

**做法**：离线重放（不启动 server、不占 GPU）时，对每轮 prompt 的 **ChatML system 段**做剥离，
其余全部保持生产缺省（`multi_turn_min=5`、`gap_split=15s`、`growth=1500`、`tool_call=prior`）：

- `--strip-system-prompt qwen`：只剥 Qwen 默认 system prompt
  `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud...<|im_end|>\n`；
- `--strip-system-prompt all`：任何 system 段都剥（连 Codeparrot 自己的领域 system prompt 一起）。

剥离只作用于**结构证据**（跨会话前缀共享的 token 序列）；时间轴仍取真实 trace 的 `ts`，
每轮写入量仍取 `recompute_tokens`。

**数据**：`result/phased_load_prediction_C.csv`，1537 会话 / 14532 轮 / 29.9M token
（struct 480、random 295、chat 462、agentic 300）。

---

## 五个实验臂

| # | 臂 | system prompt | min_share | acc | 上限 | 达成率 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 生产缺省 | 原样 | 1 | 0.5791 | 0.7716 | 0.7505 |
| B | 仅抬阈值 | 原样 | 64 | **0.7710** | 0.7716 | **0.9992** |
| C | **只去 Qwen system** | 去 Qwen | **1** | **0.7697** | 0.7716 | **0.9975** |
| D | 去 Qwen + 抬阈值 | 去 Qwen | 64 | 0.7710 | 0.7716 | 0.9992 |
| E | 去全部 system | 全去 | 1 | 0.7241 | 0.7716 | 0.9385 |

分类别召回：

| 臂 | struct | random | chat | agentic | macro-F1 |
| --- | --- | --- | --- | --- | --- |
| A | 0.998 | 0.000 | 0.240 | 1.000 | 0.4961 |
| B | 0.998 | 1.000 | 0.240 | 1.000 | 0.7525 |
| C | 0.998 | 0.993 | 0.240 | 1.000 | 0.7509 |
| D | 0.998 | 1.000 | 0.240 | 1.000 | 0.7526 |
| E | **0.852** | 0.993 | 0.240 | 1.000 | 0.7201 |

---

## 三件套指标

### 指标一：最终判定
见上表。**C 与 B 只差 2 条会话**（0.7697 vs 0.7710），C 的 2 条误差是 apps 会话之间还有
一段 16-token 的公共模板；D 与 B 完全一致（0.7710 / 0.9992）。也就是说：**不去动阈值、
只去掉公共 system prompt，就能拿到同样的准确率**。

### 指标二：多久判对

| 臂 | k=1 | k=2 | k=5 | k≥5 | 收敛轮（struct/random/chat/agentic） |
| --- | --- | --- | --- | --- | --- |
| A | 0.3116 | 0.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 5 |
| B | 0.5036 | 0.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 5 |
| C | 0.5023 | 0.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 5 |
| D | 0.5036 | 0.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 5 |
| E | 0.4567 | 0.0000 | 1.0000 | 1.0000 | 1 / 1 / 1 / 5 |

k=2–4 恒为 0 是 `>4` 轮门槛本身决定的（多轮会话此时只可能是 trans/struct）；
五个臂的差别只在 k=1（单轮会话能否判对），本质就是 apps 那一桶。

### 指标三：token 级错标暴露

| 臂 | 逐轮错标率 | token 错标率 | 黏性错标 tok（占全部 tok） |
| --- | --- | --- | --- |
| A | 0.1846 | 0.0311 | 927,410（0.0310） |
| B | 0.1643 | 0.0274 | 15,880（0.0005） |
| C | 0.1644 | 0.0274 | 25,165（0.0008） |
| D | 0.1643 | 0.0274 | 14,341（0.0005） |
| E | 0.1692 | 0.0288 | 25,165（0.0008） |

分类别看 C：

| 类别 | 逐轮错标率 | token 错标率 | 黏性 tok |
| --- | --- | --- | --- |
| struct | 0.0021 | 0.0054 | 0 |
| random | 0.0068 | 0.0023 | 249（2 条会话，16-token 公共模板） |
| chat | 0.8101 | 0.6322 | 24,916（1 条 928-token 真共享 + 短会话） |
| agentic | 0.0976 | 0.0072 | 0 |

---

## 结论

1. **误判根因确认就是公共 system prompt**：去掉它以后，在**生产阈值 `min_share=1`** 下
   random 召回 0.000 → 0.993，达成率 0.7505 → **0.9975**；与抬阈值（B）只差 2 条会话。
   两种修法（改数据 / 改门限）效果等价，不是分类规则的问题。
2. **不可自愈的错标同样消失**：黏性 token 从 3.10% 降到 0.08%（C）；D 为 0.05%，
   与 B 相同。agentic 前 4 轮不再被钉在 `qstruct`。
3. **残留的 2 条 random 误差**：去 Qwen system 后，apps 会话之间仍有 16 token 的公共模板，
   只需极小的 `min_share`（≥32）就能滤掉；在 C 里这属于「阈值仍在 1」的边界效应。
4. **E 暴露了 GT 口径问题**：把 Codeparrot 的领域 system prompt 也去掉后，struct 召回
   0.998 → **0.852**。这 71 条 Code 会话彼此**没有任何真实共享内容**，它们此前被判对，
   只是因为共享了同一个 system prompt。说明 `struct` 在数据侧是**负载家族标签**，
   而不是「实际发生了复用」的观测标签——报 struct 精确率时必须说明这一点，
   否则会把「碰巧正确」算成能力。
5. chat 的 0.240 在五个臂里完全不动：它由 `>4` 轮门槛决定（351/462 会话轮数 <5），
   与 system prompt、`min_share` 都无关。

## 限制

- 只改了离线重放的**结构证据 token**，没有真的重跑 server；真实请求仍会带 system prompt，
  因此这是「若 workload 不带公共 system prompt」的反事实结果。
- 剥离后 Code 会话的 `prompt_len`（仅 task 模式）会少几十 token，对增长判据无实质影响；
  trace 模式的 `prompt_tokens` / `recompute_tokens` 仍是实测原值。
- 前缀索引仍是 16-token 块粒度、不模拟淘汰。

## 复现

```bash
cd Heter-Bench/classify
# C：去 Qwen system，生产阈值
/root/sl_old/bin/python run_eval.py --source trace --strip-system-prompt qwen --tag stripqwen_ms1
# D：去 Qwen system + 抬阈值
/root/sl_old/bin/python run_eval.py --source trace --strip-system-prompt qwen \
    --min-share-tokens 64 --tag stripqwen_ms64
# E：连 Codeparrot 的 system prompt 一起去
/root/sl_old/bin/python run_eval.py --source trace --strip-system-prompt all --tag strip_ms1
# 对照：A / B
/root/sl_old/bin/python run_eval.py --source trace --tag baseline
/root/sl_old/bin/python run_eval.py --source trace --min-share-tokens 64 --tag share64

# 产物：../result/classify/{report,metrics,sessions}_<tag>.{md,json,csv}
```
