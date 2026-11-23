# exp_rethink · exp3 离线复盘的改动与遗留问题

> 【消融实验专题 · 离线复盘】exp3 核心瓶颈消融（one-shot 供给枯竭、MULTI_TURN_MIN 调整与 chat 队挤压）
> 时间：2026-11-16 · 分支 `W1`（基线 `461517c`）
> 载体：`sim/`（CPU 离线仿真，跑的是 `python/sglang/srt/mem_cache/` 的真实代码，非替身）
> 数据：`Heter-Bench/TaskData/exp3/{scan,struct,chat,agentic}`，四 cell 各 ≈24.5M prompt token

---

## 0. TL;DR

| 结论 | 证据 |
|---|---|
| 改前 heter **四个 cell 全输给 LRU**（−1.3 ~ −2.7pp） | v1 vs LRU |
| 根因不是 VD 排序，而是 **one-shot 带供给枯竭**（90s 豁免 > 池周转 160s） | 带内 token 中位 27 / 叶子 31,055；39.7% 决策无候选 |
| `floor_tokens` 与「级联删除」在这批数据上**基本无影响**（±0.1pp） | 消融 A / C |
| `ONE_SHOT_IDLE_S 90→20` 是最大杠杆：scan +2.65pp，agentic 淘汰 0.567M→0.01M | 消融 B |
| `MULTI_TURN_MIN 5→2` 让 ShareGPT 全线回升 +4.0~12.9pp，**scan 反超 LRU +2.07pp** | v2 → v3 |
| 剩余缺口换到了 **chat 队被 VD「赢者通吃」削掉 76% 的淘汰量** | VD 探针（0.56 vs 0.75 密度差 → 19,972 vs 5,575 次）|

---

## 1. 本次改动（4 项 + 3 处配套）

### 1.1 `heter_cache/utils.py` — `DEFAULT_VALUE_FLOOR_TOKENS: 2048 → 0`

`VD_q = H_q / max(occ_q, floor)` 里的统计下限。2048 会把小队列系统性算成低价值：
scan cell 实测 agentic 驻留中位 **530 token**、真实密度 **3.2**，除掉 floor 后只剩 **0.684**，
与 struct 的 0.68 打平/略低 → 优先被削 → occ 更小 → 分母恒被 floor 顶住 → 继续被削。
`__post_init__` 校验由 `<=0` 放宽为 `<0`；`QueueValueParams.densities()` 增加除零保护（occ=0 返回 0）。

### 1.2 `heter_cache/utils.py` — `ONE_SHOT_IDLE_S: 90 → 20`

90s 的原依据是「覆盖多轮重访间隔 p95（中位 30s、p90 44s、最大 63s）」，但它让单次流量的
豁免窗口远长于池周转：apps 流入 **1225 token/s**、池 198k → 周转 ≈160s，于是任一时刻
apps 的整份足迹（≈110k，池的 55%）都还没到 90s，**永远不进带**。

### 1.3 `heter_cache/radix_cache.py` — 放弃级联删除

`_evict_one()` 由「沿 one-shot 私有祖先链向上级联」改为 `return self._release_node(node)`。
历史代价（v1 实测）：one_shot 淘汰 14.57M 里只有 **2.67M** 是被选中的节点本身，
其余 **11.9M** 全是级联祖先，并连带删掉 **2.88M** 个 struct 标签 token——级联不参与价值判断，
却让释放量达到被选节点的 5.5 倍，决策与结果无法归因。

### 1.4 `heter_cache/utils.py` — `MULTI_TURN_MIN: 5 → 2`

`_classify` 在 `turn_count < MULTI_TURN_MIN` 时一律判 TRANS/STRUCT，而 `is_one_shot()` 的
第 4 条正是 `session_type in (TRANS, STRUCT)`。门槛 5 意味着多轮会话**前 4 轮**插入的
每轮新增尾部都可以进 one-shot 带，而 turn k 的尾部正是 turn k+1 要命中的那截
（`.auto/ideas.md` 记录的 M1「早期轮死亡螺旋」）。降到 2 后只有第 1 轮暴露，
而第 1 轮 prompt 只占 chat cell token 的 **1.2%**（543 会话共 40,344 token）。

### 1.5 配套同步

| 文件 | 改动 |
|---|---|
| `heter_cache/value.py` | `densities()` 除零保护 + docstring 说明 floor=0 语义 |
| `sim/runner.py` / `sim/__main__.py` / `sim/clock.py` | `value_floor_tokens` 默认 0.0；自检 3.3 段改为按常量取边界（不再硬编码 89.9/90.0） |
| `test/manual/test_evict_loop.py` | `test_aged_struct_prefix_cascade` → `..._not_cascaded`，断言改为「前缀存活、只删两个子节点」 |
| `test/manual/test_queue_selector.py` | 非法参数用例 `floor_tokens: 0.0` → `-1.0`（0 已合法） |
| `test/manual/test_share_one_shot.py` | ① 修 HEAD 上既有的 `NameError`（`main()` 引用了 41726c7 里已改名的函数）；② `[10]` 起点改为 `980 + IDLE/2` |
| `test/manual/test_session_type_classifier.py` | 「4 轮恒 TRANS」→「1 轮恒 TRANS，第 2 轮按间隔/工具调用分类」 |

---

## 2. 实验设置（四 cell 一致，保证可比）

| 项 | 值 |
|---|---|
| 数据 | `TaskData/exp3/<cell>/tasks.yaml`（`build_exp3_data.py --total-tokens 25e6`）|
| 容量 | `capacity=198000`（= exp3 在线「结论档」`MAX_TOTAL_TOKENS`；工作集 17.4M，淘汰必然发生）|
| 页/并发 | `page_size=1`、`max_inflight=0`、`arrival_scale=wait_scale=1.0`、`seed=0` |
| 服务模型 | `prefill 8319.6 tok/s`、`ttft 截距 -82.8ms → 夹到 0`、`decode 40 tok/s` |
| 单轮最大 prompt | 13,592 token（< cap，故 `oversize="error"`，全程 `skipped=0`）|
| 耗时 | 每 cell 26–50s 墙钟，RSS ≤1.6GB |

---

## 3. 结果

### 3.1 总体命中率（四 cell）

| cell | 主导类 | LRU | v1（原始） | v2（1.1–1.3） | **v3（+1.4）** | v3 − LRU |
|---|---|---|---|---|---|---|
| `scan` | apps | 28.58% | 26.26% | 28.99% | **30.65%** | **+2.07pp** |
| `struct` | struct | 79.21% | 77.93% | 76.67% | 77.20% | −2.02pp |
| `chat` | chat | 50.13% | 47.45% | 42.58% | 46.14% | −3.99pp |
| `agentic` | agentic | 78.52% | 76.38% | 75.04% | 75.73% | −2.79pp |
| 合并（仅参考，会话重叠） | — | 59.25% | 57.14% | 55.95% | **57.56%** | −1.69pp |

### 3.2 分负载（v3 vs LRU）

| cell / 策略 | AgentBank_apps | Struct | ShareGPT | AgentBank_alfred |
|---|---|---|---|---|
| scan / LRU | 0.001 | 0.727 | 0.456 | 0.969 |
| **scan / v3** | 0.001 | **0.770** | **0.556** | **0.978** |
| struct / LRU | 0.006 | 0.968 | 0.628 | 0.978 |
| **struct / v3** | 0.006 | 0.968 | 0.480 | 0.976 |
| chat / LRU | 0.005 | 0.729 | 0.462 | 0.971 |
| **chat / v3** | 0.005 | **0.756** | 0.388 | 0.977 |
| agentic / LRU | 0.006 | 0.886 | 0.613 | 0.971 |
| **agentic / v3** | 0.006 | 0.786 | 0.477 | 0.977 |

### 3.3 天花板（scan cell，`capacity=24.6M` ≈ 全流量、淘汰 0）

| 类 | 份额 | 天花板 | v3 | v3/天花板 |
|---|---|---|---|---|
| apps | 60.0% | 0.003 | 0.001 | 33% |
| struct | 13.6% | 0.964 | 0.770 | 80% |
| chat | 13.6% | 0.740 | 0.556 | 75% |
| agentic | 12.8% | 0.978 | 0.978 | **100%** |
| 总体 | 100% | **35.89%** | 30.65% | **85%** |

> 关键口径：scan cell 的总体命中率上限只有 35.9%，因为 60% 的 token 是单次、互不共享的 apps。
> 评价「抗扫描」应看 struct/agentic 的命中率，而不是总体。

---

## 4. 归因消融（每次只改一项，scan / chat）

| 变体 | scan | chat |
|---|---|---|
| v1 基线（floor 2048 / idle 90 / 级联） | 26.26% | 47.45% |
| A：只 `floor=0` | 26.34%（+0.08） | 48.82%（+1.37） |
| B：只 `idle=20` | **28.91%（+2.65）** | **42.34%（−5.11）** |
| C：只去级联 | 26.25%（−0.01） | 47.32%（−0.13） |
| 三项全开（v2） | 28.99% | 42.58% |

### 机制结论（有意修正早前的判断）

1. **瓶颈是 one-shot 带的「供给」，不是 VD 排序**。A 变体把 agentic 淘汰从 0.567M 只压到
   0.55M、命中 0.778→0.782：因为 Stage 1 按 VD 升序只在**有叶子**的队列里取，
   `struct` 占用 73k 但叶子常为几十个，`chat` 更少 —— 能淘汰的叶子实际只有 apps 与
   agentic 的链尾，agentic 是「唯一可选项」，排第几无所谓。
2. **`idle=20` 才是杠杆**：它把 Stage 0 重新填满（scan 的 one_shot 淘汰占比 70.6% → **93.6%**），
   Stage 1 几乎不再触发，agentic 被削 0.567M → **0.01M**，命中 0.778 → **0.978**（= 天花板）。
3. **级联删除对命中率中性**（±0.1pp），但让释放量从「5.5× 放大」回到 1:1，可归因、可复现。
4. **M1 早期轮死亡螺旋成立**：门槛 5→2 后 ShareGPT 在四个 cell 回升 **+12.9 / +4.0 / +6.1 / +5.8 pp**。

---

## 5. 仍然存在的问题

### P1 · chat 队被 VD「赢者通吃」（影响最大，≈4pp）

chat cell（v3）实测：`value:chat` 削掉 **7.27M**，占总淘汰 19.4M 的 **37.5%**；
Stage 1 决策探针（每次选点时的现场）：

| 被选中队列 | 次数 | VD(struct/chat/agentic) | occ(struct/chat/agentic) |
|---|---|---|---|
| **chat** | **19,972** | 0.75 / **0.56** / 0.90 | 136k / 28k / 0.6k |
| struct | 5,575 | 0.61 / 0.62 / 0.74 | 135k / 29k / 0.3k |
| agentic | 849 | 0.92 / 0.74 / 0.71 | 140k / 18k / 1.2k |

**密度只差 25%（0.56 vs 0.75），淘汰量差 3.6 倍**：`queue_order` 是纯排序、无份额，
每轮重排都把缺口判给垫底队列。chat 因此长期停在工作集的很小子集上（ShareGPT 0.388 vs 天花板 0.740）。

### P2 · chat cell 仍低于 LRU 4pp（46.14% vs 50.13%）

P1 的直接后果。struct cell 的 ShareGPT 差距最大（0.480 vs LRU 0.628），因为该 cell 里 chat 只占
13.6% token，LRU 能把整条链留住，而 VD 每轮都在削它。

### P3 · 第 1 轮仍然暴露在 one-shot 带里

`MULTI_TURN_MIN=2` 只保护第 2 轮之后；第 1 轮链在 `turn_count=1` 时仍是 TRANS。
当前损失有界（第 1 轮约占 chat token 1.2%），但 agentic 与 struct 的冷启动仍可能有同类问题未量化。

### P4 · 20s 与轮间分布不匹配（chat 的既有风险已部分兑现）

chat 轮间空窗 `lognormal(mu=4.15, sigma=0.971)`：中位 **63.6s**、p10 18.4s、
`P(gap>20s)=88.4%`。20s 窗口下大多数轮间都会跨过窗口，靠「第 2 轮已分类为 CHAT」兜住；
一旦分类尚未成立（第 1 轮）就会被清。40s 是下一个待测点。

### P5 · `struct` 队被削的量反而在变大

chat cell（v3）：`struct` 队被削 **7.84M**（v1 是 11.49M，v2 13.61M，趋势向下但仍高于 chat 7.54M 的
结构合理性待核）。struct 命中率 0.756（LRU 0.729）尚可，但 `value:struct` 只有 1.81M ——
说明 struct 的淘汰大多发生在 Stage 0（被当成 one-shot）或 fallback，缺少价值判断。

### P6 · LRU 臂逐请求不可复现（评估口径限制）

同参数两次 run：`hit_rate / hit_tokens / evicted_tokens` 逐位相同，但 per-request jsonl 有
**193/46,119 行**不同，且只差 `evicted` 与 `used` 两列（命中逐行相同）。
原因：上游 `RadixCache.evict()` 用 `list(self.evictable_leaves)`（对象集合，迭代序依赖地址）建堆，
而 LRU 优先级 `last_access_time` 在同一虚拟时刻大量并列——28,723 次 evict 里 1,799 次（6.3%）有并列。
**heter 臂不受影响**（`selector.pick_victim` 有确定性排序，两次 run 指纹相同）。
→ 对比请用 `hit_rate` / `per_task` 聚合值，不要把 `fingerprint()` 当门禁。

### P7 · LRU 模式下 `per_type` 无意义

`record.session_type` 在 `begin()` 读取，而 `observe_request` 只在 heter 分支调用；
`_adopt` 里的 `touch/attach` 会建默认状态，于是只出现 `none`（每会话首请求）与 `trans`（其后请求）。
按类型分析只能看 `per_task`。

### P8 · 测试基线（既有，与本次改动无关）

`test/manual/test_share_one_shot.py` 仍有 **10 项失败**（`node_share / record_hit` 口径漂移），
已在干净 worktree（`/tmp/w1_head`，仅补同一处 NameError）复现同样 10 项。
其余可跑套件全绿：`test_queue_selector / test_evict_loop / test_reuse_count / test_session_type_classifier(16 passed) / python -m sim(OK)`。

---

## 6. 下一步候选（按预期收益）

1. **软仲裁 / 滞回**（改 `heter_cache/selector.py::queue_order`）
   纯排序 → 按 `1/VD` 分摊缺口，或只有 VD 低于「上一受害者 × (1−ε)」才切换。
   直接对准 P1（25% 密度差 → 3.6× 淘汰量）。
2. **chat 加 Stage 0.5**（改 `pick_victim`）
   若 chat 叶子中存在 `ReuseScoreEvict.score < p0` 的节点，直接削它，绕开队列级 VD 均值。
   需要先只读采样 score 分布定 p0。
3. **窗口 20 → 40s** 复测（P3/P4）：预计 scan 保住大头，chat 回收第 1 轮暴露。
4. **P5 的归因**：stat struct 的淘汰是否大多走 Stage 0/fallback（缺价值判断）。

---

## 7. 复现与产物

```bash
cd /root/sg-unicache
export PYTHONPATH=$PWD/python:$PWD PYTHONDONTWRITEBYTECODE=1 STRAT_LOG_LEVEL=quiet CAP=198000
# 四 cell（heter v3 口径）
for c in scan struct chat agentic; do CELL=$c /root/sl_old/bin/python -u /root/sim_runs/sim_heter_cell.py; done
# LRU 基线
for c in scan struct chat agentic; do CELL=$c /root/sl_old/bin/python -u /root/sim_runs/sim_lru_cell.py; done
```

| 产物 | 路径 |
|---|---|
| LRU 四 cell | `Heter-Bench/mixexp/result/exp3/sim_lru_<cell>/lru_cap198000.{json,jsonl}` |
| v1 / v2 / v3 | `.../sim_heter_<cell>/`、`.../sim_heter_v2_<cell>/`、`.../sim_heter_v3_<cell>/` |
| 数据 | `Heter-Bench/TaskData/exp3/{scan,struct,chat,agentic}/`（920MB，**不入库**）|
| 日志 | `/root/sim_runs/*.log` |

> 注：`mixexp/result/` 与 `TaskData/exp3/` 均为本地产物，未提交；复跑命令会自动重建。
