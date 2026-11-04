# Autoresearch: 两带 SLRU 的模板保留追上上游 SLRU

## Objective

框架：`SimpleHybridStrategy`（`evict_policy.py`）+ `SlruEvictPolicy`（`victim_override.py`）实现的
**两带**淘汰：一次性内容进 one-shot 带优先清空，跨会话共享的模板进 protected 段。

现状问题：在 D-SLRU 负载上，**模板完整命中率**明显低于上游 `SLRUStrategy`
（`--radix-eviction-policy slru`，按 `hit_count>=2` 晋升）：

| 绝对池 | A0 LRU | A1 上游 SLRU | A3 两带 |
|---|---|---|---|
| 150k | 78.3% | **88.5%** | 78.2% |
| 250k | 80.9% | **92.4%** | 75.1% |

目标：**在不改变大框架的前提下**，让 A3 的模板完整命中率达到上游水平（≥88.5%），
且不能靠牺牲其他指标换取。

## Metrics

- **Primary**: `template_intact_min`（%，higher better）
  = min over 池 {150k, 250k} of 「A3 的模板会话请求中 `cached_tokens >= 2304` 的比例」。
  2304 = 模板长度（prefix_depth 2048 + variant 256）。取两档最小值，避免只在宽松池子上变好。
- **Secondary**:
  - `gap_min_vs_upstream` = min(A3 − A1)（pp，负值=落后）
  - `intact_min_A1` / `intact_min_A0`（参照，应保持稳定：88.5 / 78.3）
  - `tpl_tok_min`（模板 token 级命中，防止只看整段命中而忽略部分命中）
  - `priv_tok_250k`（#### 私有上下文命中，防止把模板保护建立在牺牲会话内复用上）
  - `hit_250k`（总命中率，最终目标不能倒挂）
  - `picks_total`（`slru_picks`，观测 SLRU 是否被行使）

## How to Run

`./.auto/measure.sh` — 把 `victim_override.py` / `evict_policy.py` / `uni_utils.py` 同步到实验机，
在 `result/data/dslru3`（D-SLRU trace：64 模板 × 5 会话 × 3 轮 + 800 篇 2048 token 扫描文档）
上跑 A0/A1/A3，输出 `METRIC name=value` 行。单次约 75 s。

`./.auto/checks.sh` — 本地语义单测（`test_slru_one_shot.py`、`test_evict_loop.py`、`wtl` smoke）。

## Files in Scope

- `python/sglang/srt/mem_cache/victim_override.py` — `SlruEvictPolicy`：
  `on_node_hit` / `on_node_inserted` / `applies_to` / `pick` / 账本。
- `python/sglang/srt/mem_cache/evict_policy.py` — `is_one_shot` / `one_shot_key` /
  `victim_key` / `SlruParams` 读取。
- `python/sglang/srt/mem_cache/uni_utils.py` — `SlruParams`（阈值、可新增参数）、
  `SESSION_SHARE_MAX`。
- 对应单测（语义变更时同步更新，但**不得为了让测试通过而弱化断言**）。

## Off Limits

- `simple_hybrid_radix_cache.py` 的淘汰循环结构（`if one_shot: ... elif leaves:`）——大框架不动。
- `radix_cache.py` 的树结构、`_insert_observers` / `_hit_observers` 分发。
- 负载生成器 `wtl/synth.py` 与实验口径（`--pool-tokens`、seed、trace）——**不能改负载来提分**。
- 不能针对指标作弊：不得按 session id 前缀 / 固定 node key 长度 / trace 特定数字做特判。

## Constraints

- `./.auto/checks.sh` 必须通过。
- 不引入新依赖；改动只允许发生在"策略本身"（账本证据、排序键、准入规则），
  不得把淘汰决策转移到别的模块。
- 每次改动要能写出一句"为什么它应该更好"的机制解释。

## What's Been Tried

（空白，等待第一轮）

## 已知机制事实（诊断基线）

- `share_sessions`：**每个不同会话 +1，创建者在插入时即计 1**，同会话由 `session_bloom` 去重。
  因此 `share_min=2` = 创建者 + 1 次外部命中。
- 上游的等价条件是 `hit_count>=2`，**同一会话自己的第二次读也算**。所以在
  「每个会话 3 轮、模板每块被不同会话复用」的负载里，上游在模板创建者的 turn 2 就晋升，
  而两带要等到**下一个会话**才晋升——这段空窗里模板仍属一次性内容。
- 一次性带排序是 `(-end_offset, last_access)`：**模板段（2304）比 2048 token 的扫描文档更深**，
  所以在带内模板比扫描文档更早被选中。
- `slru_picks` 在实践里恒为 0：只要请求流持续产生一次性内容，带就不会被清空，
  `elif leaves:` 分支不可达 → `pick()` 不参与，保护全靠"排除出带"。
