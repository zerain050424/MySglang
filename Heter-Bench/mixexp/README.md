# mixexp · 混合负载实验文档索引与专题字典

本目录包含四类异构负载混合（及主导类轮换）场景下的所有基准实验设计、仿真推演、策略优化与消融复盘文档。

各文档内容与历史实测数字均原样保留，通过以下四大模块化专题字典进行分类导航与索引查阅。

---

## 快速导航目录

- [一、长尾优化专题 (Tail Stress & TTFT)](#一长尾优化专题-tail-stress--ttft)
- [二、漂移测试专题 (Drift3 Phase Shift)](#二漂移测试专题-drift3-phase-shift)
- [三、消融实验专题 (Exp3 Ablations & Evolution)](#三消融实验专题-exp3-ablations--evolution)
- [四、上限诊断专题 (Oracle & Upper Bound)](#四上限诊断专题-oracle--upper-bound)
- [五、实验运行脚本导航 (scripts/sim & scripts/real)](#五实验运行脚本导航-scriptssim--scriptsreal)
- [六、基础实验环境与度量口径（附录）](#六基础实验环境与度量口径附录)

---

## 一、长尾优化专题 (Tail Stress & TTFT)

> **专题目标**：将淘汰器的优化目标从单纯的“命中率最大”转变为“保护长尾 TTFT、防止整链击穿”。

| 文档 | 状态定位 | 核心发现与要点 | 关键指标 / 推荐配置 |
|---|---|---|---|
| [`best_configs.md`](best_configs.md) | **✅ 现行最优推荐** | 汇总 TTFT 优先部署配置与 heter 保守改进档位。 | **chain_lru 全局版**：tail-stress p90 404ms（LRU 920ms，降 56%），四 cell 加权 66.25% 全胜 heter。<br>`{floor_tokens: 8192, weight: 0.02, gate: "multiturn", tail_first: true}` |
| [`design.md`](design.md) | **✅ 核心策略设计** | 淘汰目标转向防整链击穿；设计后悔弃权门（档 A）与全弃权全局定价（档 B）。 | 淘汰时不剥穿私有前缀，避免单次 miss 导致巨额 prefill 重算税；档 B 实测 p90 581ms、加权 64.42%。 |
| [`tail_stress_design.md`](tail_stress_design.md) | **✅ 压力负载设计** | 构造 hot（频繁刷 recency）、warm（长轮私有前缀）、cold（大模板共享）三类会话。 | 专门暴露 LRU 策略因整链淘汰导致的 P90 TTFT 极端长尾问题（cap=198k，1700 请求）。 |
| [`chain_shape_report.md`](chain_shape_report.md) | **✅ 机制推演报告** | 按“下游依赖体量”选受害者：浅段优先保护、优先砍链尾；全线方案横向对拍。 | 全线实测汇总表：OPT (p90 184ms) > chain_lru 全局 (404ms) > heter 档 B (581ms) > LRU (920ms)。 |

---

## 二、漂移测试专题 (Drift3 Phase Shift)

> **专题目标**：评估负载构成在时间轴上发生相位交替主导（Phase Shift）时，各策略的动态适应速度、稳态恢复时间（T90）与抗冲击性。

| 文档 | 状态定位 | 核心发现与要点 | 关键指标 / 实验结论 |
|---|---|---|---|
| [`drift3_design.md`](drift3_design.md) | **✅ 漂移负载设计** | 设计 struct → chat → agentic 依次主导的 3 相位漂移，单轮 trans 作为可调背景流量注入。 | 门禁通过（gate 6/6 PASS）；验证各会话类型的队列分流纯度（trans 99%, struct 94%, chat 86%, agentic 93%）。 |
| [`drift3_report.md`](drift3_report.md) | **✅ 漂移评测报告** | 评测构成漂移下 LRU / heter_cache / OPT 的命中率曲线与稳态恢复时间。 | **背景流量越强，heter 收益单调放大**（+0.13 → +1.49 → +3.95pp）；**漂移本身存在过渡成本**；时间轴逆序路径依赖极小（≤0.2pp）。 |

---

## 三、消融实验专题 (Exp3 Ablations & Evolution)

> **专题目标**：对 exp3 基线进行深度离线复盘，定位淘汰策略初期落后于 LRU 的根因，逐项进行参数与判据消融。

| 文档 | 状态定位 | 核心发现与要点 | 关键指标 / 结论 |
|---|---|---|---|
| [`exp_rethink.md`](exp_rethink.md) | **📜 核心瓶颈消融** | 深入复盘 heter 四 cell 全输 LRU 的根因；消融 `ONE_SHOT_IDLE_S` 与 `MULTI_TURN_MIN`。 | 根因是 **one-shot 供给枯竭**（90s 豁免 > 池周转 160s）；下调豁免窗口（90s→20s）大幅释放候选；`MULTI_TURN_MIN 5→2` 让 scan 负载反超 LRU +2.07pp。 |
| [`scheme_ab_report.md`](scheme_ab_report.md) | **📜 方案实测与修复** | 方案 A（会话生命周期老化 session_life）与方案 B（按跨会话共享数分流）实测；修复淘汰侧 regret stamp 记账。 | 接回淘汰侧 stamp，恢复 regret 精确率测量；session_life 在长间隔场景下更贴合实际会话生命周期。 |
| [`exp3_cells_ttl300.md`](exp3_cells_ttl300.md) | **📜 参数微调复跑** | 验证 chat 队列老化 TTL 从 180s 调至 300s 后的 4 cell 对照。 | 实测命中 drift3 报告中 TTL 扫描倒 U 曲线平台顶；LRU 与 OPT 逐位相同，验证了仿真确定性与改动隔离。 |

---

## 四、上限诊断专题 (Oracle & Upper Bound)

> **专题目标**：探索更激进的优化方向（如按算力代价加权的目标函数），测量与理论物理天花板的差距。

| 文档 | 状态定位 | 核心发现与要点 | 关键指标 / 结论 |
|---|---|---|---|
| [`cost_oracle_gap.md`](cost_oracle_gap.md) | **ℹ️ 探索诊断报告** | 评估“代价加权命中率”（不同长度前缀具有非线性 prefill 重算代价）的优化天花板。 | **结论：TTFT 目标值不值得投**。加权 OPT 与朴素 OPT 相比，算力节省率仅微涨 0.2~0.4pp，单纯按代价加权带来的收益边际极窄。 |

---

## 五、实验运行脚本导航 (`scripts/sim` & `scripts/real`)

本目录将所有实验驱动划分为**模拟器（纯 CPU 离线仿真）**与**真机（实机 GPU SGLang 评测）**两套独立运行入口：

### 5.1 模拟器实验脚本 (`scripts/sim/`)

| 脚本 | 职责 | 一键命令 |
|---|---|---|
| [`scripts/sim/sim_exp3_cells.py`](scripts/sim/sim_exp3_cells.py) | exp3 四 cell（scan/struct/chat/agentic）策略对比引擎驱动 | `bash Heter-Bench/mixexp/scripts/sim/run_exp3.sh` |
| [`scripts/sim/drift3_run.py`](scripts/sim/drift3_run.py) | drift3 构成漂移实验驱动（支持 `--gate`、稳态与逆序对照） | `bash Heter-Bench/mixexp/scripts/sim/run_drift3.sh` |
| [`scripts/sim/run_tail_stress.sh`](scripts/sim/run_tail_stress.sh) | tail-stress 长尾 TTFT 压力测试（调用 `tools/build_tail_stress.py`） | `bash Heter-Bench/mixexp/scripts/sim/run_tail_stress.sh` |

### 5.2 真机实验脚本 (`scripts/real/`)

| 脚本 | 职责 | 一键命令 |
|---|---|---|
| [`scripts/real/run_exp3_real.sh`](scripts/real/run_exp3_real.sh) | exp3 真机全自动对比（启停服务 + `bench_heap.py` + `analyze.py` 汇总） | `CELL=chat bash Heter-Bench/mixexp/scripts/real/run_exp3_real.sh` |
| [`scripts/real/run_opt_real.sh`](scripts/real/run_opt_real.sh) | OPT 上界真机全流程（Capture -> UseMap -> Replay） | `CELL=chat bash Heter-Bench/mixexp/scripts/real/run_opt_real.sh` |
| [`scripts/real/build_use_map_from_bench.py`](scripts/real/build_use_map_from_bench.py) | 离线生成时间轴重锚定的 OPT `use_map` JSON | `python mixexp/scripts/real/build_use_map_from_bench.py --cell chat` |
| [`scripts/real/analyze.py`](scripts/real/analyze.py) | 真机压测结果 JSON 聚合与公平/命中/TTFT 判定表生成 | `python mixexp/scripts/real/analyze.py --dir mixexp/result/exp3/chat --scale full` |

---

## 六、基础实验环境与度量口径（附录）

### 6.1 实验负载与数据准备

四个 cell，每个 cell 里四类负载同时在场，其中一类按 **prompt token** 占优（≥60%），用于检验不同压力源下多队列隔离的效果：

| cell | 主导类 | 考察目标（该 cell 制造的偏倚） |
|---|---|---|
| `scan` | apps（单轮冲刷） | 大流量、零复用冲刷下，可复用前缀能否保留 |
| `struct` | struct（跨会话共享前缀） | 公共前缀是否被误驱逐 |
| `chat` | chat（多轮人工会话） | 轮间间隔长（T50≈118s）时，早期轮次能否活到下次访问 |
| `agentic` | agentic（Agent 会话） | 长会话（单条 ≈89k token）、短间隔（T50≈6.1s）、上下文快速增长 |

#### 数据准备命令
```bash
cd Heter-Bench

# 步 1+2：剥离 Qwen system 段 + 剔除过短会话（幂等）
bash mixexp/prepare_data.sh

# 步 3：构造四个主导 cell（每 cell 25M token 预算）
# 产物：TaskData/exp3/<cell>/tasks.yaml + 四份 *_exp3.nosys.jsonl + manifest.json
python mixexp/tools/build_exp3_data.py --mix all --total-tokens 25e6
```

### 6.2 OPT 离线最优上界测算

OPT = Belady 离线最优：淘汰下次使用时刻最远的块（`evict_policy.OPTStrategy`）。它的输入是 use map（`{block_hash: [使用时刻]}`）。在真实 server 上跑 OPT 的流程为：**先采一次到达时刻，离线造 use map，再带 map 重放**。

1. **采一次到达时刻（capture run）**：
   使用 `POLICY=lru` 在目标 cell 跑一次基线测试，逐请求 CSV 记录 `(task, session_id, turn, ts)`。
2. **生成 use map（重锚定时间轴）**：
   ```bash
   # 设定重放起点 R（epoch 秒，留出模型加载余量）
   R=$(( $(date +%s) + 600 ))
   python mixexp/scripts/real/build_use_map_from_bench.py --cell chat --replay-start $R
   ```
3. **带 map 起服务并重放**：
   ```bash
   POLICY=opt MAX_TOTAL_TOKENS=198000 \
     POLICY_CONFIG='{"use_map_path":"TaskData/exp3/chat/use_map_p1.json","block_size":1}' \
     bash scripts/serve.sh
   ```

### 6.3 Regret Rate（淘汰后悔率）账本

Regret 账本（`Heter-Bench/profile/regret_ledger.py`）衡量每次逐出决策造成的重算损失：
- 一次 `_pick_victim` + 级联 = 一个 stamp，`span` 为腾出的 token 数。
- 断点页若在保留期内回访，说明该次淘汰产生了“后悔”；损失计入该决策源。
- 开启方式：服务启动时设置 `HETER_REGRET=1` 与 `HETER_REGRET_OUT=...`。
