# W-TinyLRU 实验套件（Uni-Bench/W-TinyLRU）

实现文档：`python/sglang/srt/mem_cache/TAG_LRU.md`（标签版 TinyLRU 的规则/参数/淘汰顺序/实测）
**本目录不修改任何框架代码**：观测全部通过子类 / 运行时 monkeypatch / 运行时属性完成。

```
W-TinyLRU/
├── README.md                  ← 本文件
├── wtl/
│   ├── arms.py                实验臂 A0–A5（单一事实来源）
│   ├── simclock.py            事件驱动仿真时钟（替换 sglang 模块内的 time）
│   ├── trace.py               session_data.jsonl 读取 + 到达调度（与 bench_unicache 同语义）
│   ├── synth.py               合成「共享前缀强度阶梯」trace（场景 A）
│   ├── prefix_stats.py        共享前缀 LCP 统计（M1 的基准）
│   ├── backends.py            RealBackend（真 RadixCache）/ FakeBackend（镜像 LRU，smoke）
│   ├── metrics.py             保留度 / 后悔率 / 保护浪费率
│   ├── replay.py              离线回放驱动（L1）
│   ├── run_matrix.py          离线矩阵并发执行器（可续跑）
│   ├── struct_profile.py      真机 struct 状态 JSONL 埋点
│   ├── serve.py               python -m wtl.serve（带埋点的 launch_server）
│   ├── analyze.py             表 + 配对 bootstrap + 图
│   └── gate.py                Phase-0 门控（G1–G5）
├── scripts/                   run_offline.sh / run_online.sh / run_all.sh / run_gate.sh
├── tests/test_smoke.py        无 GPU 链路测试（8 项）
└── result/                    输出（offline / online / analysis / gate.json）
```

---

## 1. 环境

```bash
# 实验机（与 Uni-Bench 相同）
export PYTHONPATH=/path/to/MySglang/python:$PYTHONPATH
export PY=/root/sl_old/bin/python          # 或 source ../env.sh

# 本机（没有 sglang/torch 也能验证链路）
python tests/test_smoke.py
```

`RealBackend` 需要能 `import sglang.srt.mem_cache.*` 且 `torch` 可用（CPU 即可，**不需要 GPU、不需要模型权重**）。
大文件（AgentBank/ShareGPT 的 jsonl 有 2~3GB）用 `--read-limit N` 只读前 N 行再打乱，
否则全量读入会吃掉几十 GB 内存；该设置会写进结果 meta，报告里要注明采样子集。
导入失败时会给出明确提示，`--backend auto` 自动退回 `FakeBackend` 并在结果里标 `"backend":"fake"`；
`analyze.py` **拒绝**把 fake 结果算进主表与判据。

---

## 2. 快速开始

```bash
cd Uni-Bench/W-TinyLRU

# 0) 链路自检（无 GPU）
python tests/test_smoke.py

# 1) Phase-0 门控：不过门控不要跑正式矩阵
python -m wtl.gate --backend auto

# 2) 真实负载的共享前缀统计（task 名要与 trace 里的 task 一致）
python -m wtl.prefix_stats \
  --data-path ../TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl --task AgentBank_alfred \
  --data-path ../TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl --task ShareGPT \
  --num-sessions 500 --out result/prefix_stats.json
#    注意：--task 的命名必须和 trace 里的 task 一致（单负载=WORKLOADS[name]["task"]，
#    混合负载=tasks_config 的 name）。自定义 jsonl 用 --task-name 指定。

# 3) 离线矩阵（L1，主力）
bash scripts/run_offline.sh --arms all --workloads sharegpt,alfred,codeparrot \
     --ratios 0.3,0.6,1.0 --seeds 42,43,44 --jobs 12 --prefix-stats result/prefix_stats.json

# 4) 合成前缀阶梯（场景 A：深度 × 广度）
bash scripts/run_offline.sh --mode synth --arms A0,A1,A2,A3 \
     --synth-depths 512,2048,8192 --synth-breadths 3,10 --ratios 0.3,0.6

# 5) 真机 TTFT（L2，需 GPU + 模型）
ARMS="A0 A2 A3" MODES="hybrid sharegpt" TOKENS_LIST="100000" SEEDS="42" \
  bash scripts/run_online.sh

# 6) 分析
python -m wtl.analyze --offline result/offline --online result/online \
  --prefix-stats result/prefix_stats.json \
  --struct 'result/online/struct_*.jsonl' --out result/analysis
# -> result/analysis/summary.md / results_offline.csv / paired.csv / fig1-3.png
```

单次回放：

```bash
python -m wtl.replay --arm A3 --workload alfred --num-sessions 300 --seed 42 \
  --pool-ratio 0.3 --out result/offline/single.json

# 自定义 jsonl（合成 trace 或新数据集）
python -m wtl.replay --arm A3 --data-path data/synth_p2048_b10/session_data.jsonl \
  --synth-meta data/synth_p2048_b10/meta.json --task-name synth \
  --pool-ratio 0.3 --out result/offline/synth_A3.json
# 打印：hit%、各层保留度（full/cover）、evictions、struct 快照
```

---

## 3. 三个核心口径

### 3.1 压力点：`--pool-ratio` = 池容量 / 工作集

工作集 = **不淘汰时 radix 树的 token 占用**（用镜像 radix 树以无限池跑一遍得到，
缓存在 `result/calib.json`）。这样换模型 / 换数据集仍然可比，也正好是设计文档 §5.3 的口径。
`--pool-tokens` 可直接指定绝对容量。

### 3.2 保留度（M1）

对每个请求 r 与其 task 的分层前缀长度 `L_k`（来自 `prefix_stats` 或合成 trace 的 meta）：

```
全保留 I_k(r) = 1[cached_tokens_r >= L_k]          → full_rate
覆盖度 C_k(r) = min(cached_tokens, L_k) / L_k      → cover
```

离线与在线**同一套判据**：在线只要逐请求 CSV 里有 `cached_tokens` 即可算，无需服务端埋点。

### 3.3 后悔率 / 保护浪费率（M4）

- **后悔率** = 被淘汰、且在 `--regret-window` 秒内又被请求的 token 数 ÷ 总 prompt token；
- **保护浪费率** = protected 区被淘汰的 token 中，之后再也没有被命中的比例。

实现是 **token 序列滚动哈希 + 绝对深度**（节点级精确，不是块近似）：
淘汰时对「根到该节点结束」的 token 序列取哈希，请求到达时对 prompt 逐个位置取哈希比对。
两者是一对孪生指标：好策略应压低后悔率、不抬高浪费率。

---

## 4. 观测了哪些机制证据

| 证据 | 来源 | 用途 |
|---|---|---|
| `victim_src`（`one_shot` / `global` / `qstruct`） | `SimpleHybridStrategy._victim_sources` | **gate**：qstruct=0 说明策略从未行使淘汰权，结论作废 |
| `promote / counted / dedup` | `TagLruPolicy.snapshot()` | 证明「第二个会话命中 → 晋升 QSTRUCT」真的在发生 |
| `evicted_by_tag`（qtrans/qchat/qagent/qstruct） | `SimpleHybridStrategy._evicted_by_tag` | 一次性带占比、受保护前缀被淘汰的 token 量 |
| `tag` 快照时间序列 | `wtl.struct_profile`（真机 JSONL） | 看晋升速率、去重率是否随负载漂移 |
| `evict_us_p50/p99` | 同上（包在 `evict()` 外计时） | 策略 CPU 开销护栏 |
| `protected_cascade_deleted_tokens` | `_delete_leaf(is_victim=False)` + `region==PROTECTED` | §7.4 最大正确性风险的可观测证据（门控 G5） |

---

## 4.0 Q_struct 队内策略：标签版 TinyLRU（`tag_lru`）

**分类即晋升**：STRUCT（模板/共享前缀型）会话下，`share_sessions >= struct_share_min`
的节点当场升为 `QSTRUCT`（类内 LRU 淘汰）；会话内热点按 `local_hot` 处置
（默认 `QCHAT`，**不**进 one-shot 硬优先带）；真一次性内容留 `QTRANS`。
其他会话类型沿用分类器映射，不参与晋升。

| 臂 | 配置 | 用途 |
|---|---|---|
| A0 | `lru` | 上游基线 |
| A1 | `slru` | 上游分段 LRU 参照 |
| A2 | `simple_hybrid` + `tiny_lru=off` | 隔离框架增益（多队列 + one-shot 带 + 分类器标签） |
| **A3** | `tag_lru`（默认） | 主对象 |
| A4 | `tag_lru` + `local_hot=one_shot` | 消融：会话内热点也进 one-shot 带（规则原样） |
| A5 | `tag_lru` + `struct_share_min=3` | 消融：要 3 个会话共享才晋升 |

实测（离线 L1，300 会话，pool/工作集=0.3）：

| 配置 | ShareGPT 命中率 | 会话内历史保留 | 合成 p2048/b10 |
|---|---|---|---|
| LRU（A0） | 66.90% | 0.736 | 81.09% |
| `off`（A2） | 49.26% | 0.551 | 80.37% |
| `local_hot=one_shot`（A4） | 54.61% | 0.738 | 75.67% |
| **默认（A3）** | **78.10%** | **0.892** | **82.46%** |

设计说明见 `python/sglang/srt/mem_cache/TAG_LRU.md`（含参数、淘汰顺序、死链闸、限制）。
早期「三区 + 频数准入 + aging」的实现已从本分支移除；移除理由与实测见该文档 §1。

## 4.1 离线确定性平票（`wtl/compat.py`）

`RadixCache.evict` 用 `min(leaves, key=...)` 在 `set` 上选候选。仿真时钟下同一请求命中的
节点时间戳完全相同，平票极多，`min` 的结果就取决于 set 遍历顺序（对象地址）——
同一配置两次进程内回放的淘汰序列会分叉（门控 G1 实测出现）。

`install_stable_ties`（**仅离线回放安装**，`--unstable-ties` 可关）给排序键追加
`(路径深度, 路径 token 哈希)` 作为确定性后缀，只改变「平票时选谁」，不改变策略语义。
在线服务不安装（真实时钟下平票少，且不应改变生产行为）。

## 5. 不改框架代码是怎么做到的

| 需求 | 做法 | 文件 |
|---|---|---|
| 淘汰/删除事件 | 子类覆写 `_evict_one` / `_delete_leaf`（只读） | `backends.py` |
| 节点路径哈希/深度 | 给 `TreeNode` 挂 `_wtl_pd/_wtl_ph` 记忆化属性（框架不读） | `backends.py` |
| 仿真时间 | 替换 `sglang.srt.mem_cache.{radix_cache,evict_policy,...}` 模块级 `time` | `simclock.py` |
| 真机 struct 快照 | 运行时包装 `SimpleHybridRadixCache.evict`，退出时 atexit flush | `struct_profile.py` |
| 服务启动 | 复制 `launch_server.__main__` 的调用序列 + 先装埋点 | `serve.py` |

框架侧唯一「被用到」的私有接口是 `evictable_size_` / `_profile_token_counts` / `session_registry`
（只读）。若这些名字变化，`struct_profile` 会记录 `null` 而不是崩溃。

---

## 6. 输出 schema

离线（`wtl.replay/1`）：

```jsonc
{
  "schema": "wtl.replay/1", "backend": "real",
  "arm": "A3", "policy": "simple_hybrid", "config": {...},
  "pool_tokens": 51234, "pool_ratio": 0.3, "calib_peak": 170780,
  "seed": 42, "wait_scale": 1.0, "arrival_scale": 1.0, "trace_fp": "…",
  "layers_by_task": {"ShareGPT": [512, 1024, 2048]},
  "summary": {
    "num_requests": 1500, "hit_pct": 43.2, "evictions": 812, "oom": 0, "peak_used": 51300,
    "per_task": {"ShareGPT": {"hit_pct": 55.1, ...}},
    "retention": {"ShareGPT": {"2048": {"n": 900, "full_rate": 0.91, "cover": 0.95}}},
    "regret": {"regret_rate": 0.021, "protected_waste_rate": 0.08,
               "protected_cascade_deleted_tokens": 0},
    "struct": {"window": 12, "probation": 44, "protected": 130, "parked": 9,
               "admit": 21, "reject": 7, "ages": 2, "forced": 0},
    "victim_src": {"one_shot": 120000, "qstruct": 34000}
  },
  "rows": [{"ts":..,"task":..,"sid":..,"turn":..,"prompt_tokens":..,
            "cached_tokens":..,"total_tokens":..,"evicted":..,"oom":0}]
}
```

在线：`bench_<arm>_<mode>_<tokens>k_s<seed>.json` + 同名 `.csv`（逐请求）
+ `struct_<tag>.jsonl`（每 30s 一条 struct 快照）+ `manifest.jsonl`。
分析脚本从文件名解析 arm/mode/tokens/seed，所以命名不要改。

---

## 7. 已知 deviation（写进报告 limitation）

1. **不建模 lock_ref**：真机运行中的请求会锁住自己的前缀；离线端淘汰只发生在下一次分配前，
   因此离线命中率可能略高（同一请求内部不会自淘汰）。
2. **不建模 chunked prefill / 批处理**：只建模「到达顺序 + KV 池容量」。
3. **服务时间**：在线端下一轮要等上一轮完成才开始；离线默认 `service_s=0`
   （用 `--service-ms-per-1k` 可近似补上），复用间隔略短于真机。
4. **hit_count 双计**：离线每请求只做一次 `insert`（对齐 `cache_finished_req` 的 `_adopt`），
   而真机在 chunked 预填路径上可能再 `+1`。这会影响 SLRU/LFU 的段位判定，故 **A1 的解释边界**
   要写明（见设计文档 §6 的 limitation）。
5. **fake 后端**只用于链路验证：`analyze` 会排除，`gate` 会把 G2/G3/G4 记 N/A。
6. **真实 TaskData 的跨会话共享前缀极短**：`prefix_stats` 实测 turn_1 的成对 LCP 中位数
   ——Codeparrot 317、AgentBank_apps 24、ShareGPT 24、AgentBank_alfred 24。
   因此 M1 在真实数据上退化为**会话内历史保留度**（`history_retention`：
   本轮是否仍命中上一轮完整 prompt），跨会话分层前缀只在合成梯子上有意义。
7. **one-shot 硬优先带会吃掉绝大部分淘汰压力**：合成 trace 里只要掺入一次性文档，
   实测 `victim_src` 里 `one_shot` 占 90%+，struct 队列永远等不到「成员变成可淘汰叶子」，
   策略相当于空转。门控 G2 的 trace 因此**不放一次性文档**。
8. **频数准入（`admit/reject`）在真实负载上近乎不可达**：window 里的成员一旦成为叶子
   往往立刻被淘汰，很难同时积累到 `window_entries=64` 以上；A3 与 A6（w=16）在实测中
   结果完全相同。于是 G2 改为「状态机在管节点 + struct 策略真的接管过淘汰
   （`qstruct>0` 或 `forced>0`）」，不再要求 `reject>0`（设计文档 §8.5 的判据需要修正）。

---

## 8. 与设计文档的对应

| 设计文档 | 实现 |
|---|---|
| §2.1 M1 保留度 | `metrics.RetentionTracker` + `prefix_stats` + `analyze` |
| §2.2 M2 命中率 | 客户端 `cached/prompt`；服务端 `sglang:cache_hit_rate` 交叉校验 |
| §2.3 M3 TTFT | `bench_unicache` 已有；`analyze` 做请求级配对与分解列 |
| §2.4 M4 机制（含 gate） | `struct` / `victim_src` / `qstruct_pct` / G2 |
| §2.4 后悔率/浪费率 | `RegretTracker`（token 级精确） |
| §4.1 prefix_stats | `wtl/prefix_stats.py` |
| §4.2 离线回放器 | `wtl/replay.py` + `wtl/backends.py` |
| §4.3 SGLANG_STRUCT_PROFILE | `wtl/struct_profile.py`（env 开关，零开销关闭） |
| §4.4 分析脚本 | `wtl/analyze.py` |
| §5.1 场景 A | `wtl/synth.py` + `--mode synth` |
| §6 消融矩阵 | `wtl/arms.py`（A0–A5）+ `scripts/run_online.sh` 的 `arm_policy` |
| §8 判定 | `analyze.judge`：M1/M2/M3/M4-gate/M5 逐条 PASS/FAIL |
| §10 前置检查 | `wtl/gate.py --arm A3`（G1–G5；G2 看 tag.promote 与 evicted_by_tag） |

---

## 9. 常见问题

**Q：`--backend real` 报无法导入 sglang？**
检查 `PYTHONPATH` 是否包含仓库的 `python/` 目录；本机若没有匹配的 torch/transformers，
用 `--backend fake` 只验证链路，正式结果回到实验机跑。

**Q：跑得太慢？**
* `SimpleHybridRadixCache.evict` 每次都会调 `_profile_token_counts()`（O(树大小)），
  这是框架行为，不是本套件加的；用 `--num-sessions 200~400` 控制树规模；
* `run_matrix --jobs 12` 并发；`--regret-window 0` 关掉后悔率统计可以再快一截；
* 工作集校准按 `(workload, seed, sessions, wait_scale…)` 缓存，矩阵里只算一次。

**Q：判据全是 N/A？**
缺少对应数据：M1 需要 trace 里存在 ≥2k 的共享前缀层；M3/M5 需要 `--online` 的逐请求 CSV。

**Q：可以只跑某个压力点做配对吗？**
`analyze --ratio-focus 0.3` 只对该压力点做配对检验。
