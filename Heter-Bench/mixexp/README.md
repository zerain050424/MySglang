# mixexp · 混合负载实验

这套实验要回答一个问题：**在负载异构、而且组成会随时间变化的场景下，HeterCache 那套「三队列 + 队内策略 + 配额仲裁」是不是真的强过单策略（LRU / LFU / SLRU / T-LRU），以及每个部件各自贡献了多少。**

实验沿两条主线展开：

- **规模主线** `smoke → quick → mid → full`：先用小规模把整条链路走通，再逐步放大到能出结论的规模；
- **组成漂移主线** A → B → C → A 四段相位：专门检验「负载组成变了之后，缓存还能不能命中」。

---

## 零、先准备数据（这一步不做，后面全是空跑）

Qwen 的默认 system prompt 只有 24 个 token，而运行期判断「跨会话共享」时**没有 token 数量门槛**：
只要某个单轮会话碰巧命中过别人持有的这段前缀，就会被记成 `STRUCT`。标签分布会因此严重失衡。

同一份数据上的离线消融实验（`docs/reports/classify_system_prompt_ablation.md`）：

| 对照组 | system prompt | 准确率 | random 召回 | chat 召回 |
|---|---|---|---|---|
| A | 原样保留（生产缺省） | 0.5791 | **0.000** | 0.240 |
| C | **只去掉 Qwen system** | **0.7697** | **0.993** | 0.240 |

换句话说，不剥掉这一段，`random` 队列在实验里根本不会出现，`apps` 会整批被算进 `struct`。
所以本套件一律读去掉该段之后的副本：

```bash
bash mixexp/prepare_data.sh     # 生成 *.nosys.jsonl（可重复执行）
```

`configs/` 下所有 `mix_*.yaml` 与 `drift_phases.yaml` 的 `data_path` 都已经指向 `*.nosys.jsonl`。
`run_mixexp.sh` 会在启动 server 之前先核对这些文件是否存在，缺了就立刻停下，并提示上面这条命令。

两点提醒：

- **Codeparrot 不会被剥。** 它自带领域 system prompt，不属于 Qwen 默认段，`--mode qwen` 会原样保留，
  因此它生成出的 `Code_session_data.nosys.jsonl` 与源文件内容一致。若某个方案需要「彻底无 system 前缀」，
  得用 `--mode all` 单独处理。
- **`chat` 召回停留在 0.240 与 system prompt 无关。** 76% 的 ShareGPT 会话不到 5 轮，
  结构上本来就不可能判成 CHAT，剥前缀解决不了这个问题（详见第六节）。

---

## 一、规模主线

规模只体现为一件事：`configs/mix_<scale>.yaml` 里各负载的 `max_record`（每类取多少条会话）。
四类负载的到达权重 `lambda_rate` 始终为 1，也就是**混合比例不随规模变化**——否则不同档之间无法互相比照。

| 档位 | 会话数（apps / code / chat / agent） | 池容量 | 并发 | wait-scale | 用途 |
|---|---|---|---|---|---|
| `smoke` | 10 / 10 / 10 / 10 = 40 | 30k | 8 | 0.1 | 确认链路通、各方案都能起来、指标在动 |
| `quick` | 50 / 50 / 25 / 25 = 150 | 60k | 16 | 0.3 | 看趋势，单个方案几分钟 |
| `mid` | 400 / 400 / 150 / 150 = 1100 | 120k | 24 | 1.0 | **出趋势性结论的主力档** |
| `full` | 1800 / 1800 / 500 / 500 = 4600 | 200k | 32 | 1.0 | 最终数字 |

三个配套配置参数（池容量、并发、`wait-scale`）必须跟着规模同向放大。
只加会话数、不动池容量，工作集压不满池子，所有方案会跑出一样的结果——这是这类实验最常见的白跑原因，
`analyze.py` 的第 5 条验收标准专门检查这一点。

`wait-scale` 压缩的是轮与轮之间的等待时间，它只影响跑多久，同时也会改变淘汰概率。
所以 **`smoke` / `quick` 只能用来验证流程，结论必须取自 `mid` / `full`**。

---

## 二、方案清单（`arms.tsv`）

| 方案 | 策略 | 角色 |
|---|---|---|
| `lru` | lru | 最弱基线：全局只有一个顺序 |
| `lfu` | lfu | 反向基线：长会话不断累积频次，陈旧前缀赖着不走 |
| `slru` | slru | 中档基线：只按命中次数分段，没有队列概念 |
| `tlru` | tlru | **最需要战胜的对手**：上游同方向的工作，按「尾部是否还影响 TTFT」决定逐出 |
| `hc_full` | heter_cache | 完整方案：三队列 + 队内策略 + 配额仲裁 |
| `hc_norr` | heter_cache | 消融：关掉会话轮询（agentic 队内退化为无策略） |
| `hc_nomt` | heter_cache | 消融：关掉多轮改写器（chat 队内退化为无策略） |
| `hc_flat` | heter_cache | 消融：不分队列，全按 agentic 归类 |
| `hc_caps_eq` | heter_cache | 消融：配额均分 1/3，不向负载倾斜 |

仅仅「赢了 LRU」说服力有限。真正能支撑方案成立的是**五个 heter_cache 方案之间的对比**：
关掉任一部件，结果是否变差。少了这一步，就排除不掉「只是恰好和 LRU 走了不同的路」这种可能。

`tlru` 必须带上 `--radix-eviction-policy-config`，否则上游会因为
`threshold <= next_prompt_estimate` 直接拒绝启动（默认值 0/0）。`arms.tsv` 里已经写好。

---

## 三、组成漂移（`--mode drift`）

`configs/drift_phases.yaml` 排了四段，每段 120 秒，各由一类负载主导：

```
code_dominant → agent_dominant → chat_dominant → code_return
```

最后一段刻意回到第一段的组成。全局单一顺序的策略碰上这种序列，段 1 攒下的代码前缀早被段 2、段 3 冲干净了，
段 4 回来时命中率会明显塌下去；多队列方案如果真做到了隔离，段 4 应该还能命中相当一部分。

这一档没有规模梯度，放大时把 `drift_phases.yaml` 里 `max_record` 的几个数字一起按倍数调即可。

---

## 四、怎么跑

```bash
cd Heter-Bench

# 1) 先确认命令拼得对（不起服务）
bash mixexp/run_mixexp.sh --scale smoke --dry-run

# 2) 冒烟：9 个方案全跑一遍，并自动汇总
bash mixexp/run_mixexp.sh --scale smoke

# 3) 只跑关键三个，快速看趋势
bash mixexp/run_mixexp.sh --scale quick --arms lru,tlru,hc_full

# 4) 组成漂移
bash mixexp/run_mixexp.sh --scale mid --mode drift --arms lru,slru,tlru,hc_full

# 5) 全量（耗时很长，建议放进 tmux）
bash mixexp/run_mixexp.sh --scale full
```

常用参数（其余见 `--help`）：

| 参数 | 含义 |
|---|---|
| `--scale` | 规模档，`smoke` / `quick` / `mid` / `full` |
| `--mode` | `hybrid`（默认，固定混合）或 `drift`（四段漂移） |
| `--arms` | 逗号分隔的方案名，缺省 `all` |
| `--reps` | 重复次数，结果文件加 `_rN` 后缀，汇总时按方案取均值 |
| `--tokens` / `--conc` / `--wait-scale` | 覆盖该档默认的池容量、并发、轮间压缩 |
| `--out` | 结果目录，缺省 `result/mixexp` |

每个方案独占一次 server：起 server → 等 `/health` → 跑 bench → 停 server。

只汇总、不重跑：

```bash
python mixexp/analyze.py --dir result/mixexp --scale mid --tokens 120000
```

产物：

```
result/mixexp/<scale>_<arm>.json         单次运行的 bench 结果
result/mixexp/server_<scale>_<arm>.log   server 日志（含 [HeterCache profile] 段）
```

---

## 五、验收标准

`analyze.py` 会打印一张 markdown 表，并逐条给出判定：

| # | 标准 | 含义 |
|---|---|---|
| 1 | 主标准 | `hc_full` 的总命中率严格高于所有基线 |
| 2 | 公平标准 | `hc_full` 的**最弱一类**命中率不低于表现最好的基线 |
| 3 | 时延标准 | `hc_full` 的 `ttft_p99` 不劣于基线的 1.10 倍 |
| 4 | 消融标准 | `hc_full` 不弱于任何一个 `hc_*` 消融方案 |
| 5 | 有效性 | 总 `prompt_tokens` 确实超过池容量，也就是真的发生了淘汰 |

第 2 条要防的是「靠饿死某一类负载去换总命中率」。表里的**类间差距**那一列就是给这条用的：
差距越大，说明有一类正在被系统性地牺牲。

第 1 条（主标准）与第 5 条（有效性）是硬条件：任一条不满足，`analyze.py` 就以非零退出码结束，
方便串进脚本；报告本身照常打印完整。

---

## 五·补、exp2：池容量 0.85 + 多轮子集 + agentic 配额 0.6

exp1（结果归档在 `result/exp1/`）暴露出三个问题，exp2 逐条对应修改：

| 问题 | exp1 实测 | exp2 改法 |
|---|---|---|
| 池容量被钳制 | 请求 200 000，实际只有 **35 975** | `--mem-fraction-static 0.85` → 实测 **99 036** |
| chat 队列空转 | 占用均值 167 / 上限 10 792 | ShareGPT 全换 `num_turns > 5` 子集（9 995 会话，均值 8.1 轮） |
| agentic 长期超配额 | 占用 31 127 / 上限 17 988 | `caps.agentic` 0.5 → **0.6**，struct 0.2 → 0.1 |

`0.9` 试过，KV 池能到 120 633 token，但 CUDA graph 捕获阶段 OOM，服务起不来；0.85 是这台 4090 上的可用上限。

```bash
bash mixexp/prepare_data.sh                    # 生成 *.nosys.jsonl 与 >5 轮子集（幂等）
bash mixexp/run_exp2.sh --dry-run              # 先看命令
bash mixexp/run_exp2.sh --arms lru,hc_full_v2  # 建议先跑两个方案
bash mixexp/run_exp2.sh                        # 三个方案：lru / hc_full / hc_full_v2
```

- 方案清单：`arms_exp2.tsv`（`hc_full` 用旧配额跑新数据，`hc_full_v2` 用新配额，可分离"数据变了"与"配额变了"两种影响）
- 配置：`configs/exp2/mix_full.yaml`
- 输出：`mixexp/result/exp2/`，与 `result/exp1/` 互不覆盖
- 归档：exp1 的产物在 `mixexp/result/exp1/`，附 `SUMMARY.md`

---

## 六、已知限制（先说清楚，免得误读结论）

1. **`per_task` 是数据集维度，不是队列维度。** 队列由运行期的状态机动态判定
   （`ShareGPT_*` → chat、`alfred_*` → agentic 等），而 bench 只按 task 名统计。
   所以「最弱一类」实际是「最弱数据集」的替身，不能当成精确的队列指标。
   要看真正的队列数字，去 server 日志里的 `[HeterCache profile]`。

2. **本仓库不含数据集。** `TaskData/` 下只有生成脚本，`*_session_data.jsonl` 需要在实验机上
   先用 `tools/prepare_session_data.py` 生成，再经 `mixexp/prepare_data.sh` 剥掉 Qwen 公共 system 段。
   在本地直接跑，会报找不到文件。

3. **这套脚本在本机跑不了**：它需要 GPU、模型和数据。提交前做过的校验只有静态层面——
   `bash -n`、YAML 结构校验加调度构建、`HeterCacheStrategy` 参数构造、`analyze.py` 用合成数据自测、以及 `--dry-run`。

4. **单次运行方差不可忽略。** 同一方案只跑一次，数字里带着噪声。要写进结论，至少 `--reps 3` 起。

---

## 七、CPU 回放器（离线测算命中率）

真机一轮 full 要 3 小时以上（lru 11,270 s、hc_full 11,330 s），九个方案扫一遍接近 30 小时。
`W-TinyLRU/wtl/replay.py --backend real` 可以在 **CPU 上跑同一份 `HeterCacheRadixCache`
与同一套策略代码**（不加载模型、不起 server、不占 GPU），把 full 档 2.47 万请求 /
5,100 万 prompt token 压到 **约 55 秒**，加速约 200 倍。

```bash
# 单个实验臂（策略与配置直接从 arms.tsv 取）
cd Heter-Bench/W-TinyLRU
PYTHONPATH=/root/sg-unicache/python:/root/sl_old/lib/python3.12/site-packages \
python -m wtl.replay --arm lru --policy lru \
  --tasks-config ../mixexp/configs/mix_full.yaml --num-sessions -1 \
  --pool-tokens 35975 --inflight-reserve 3000 --concurrency 32 \
  --inflight-ms 1414 --backend real --quiet --out /tmp/sim.json
```

### 必须对齐的三件事

离线/在线差一个数量级的原因全在调度模型，不在缓存：

| 参数 | 含义 | 真机来源 |
|---|---|---|
| `--concurrency` | **并发会话上限**（一条会话跑完 40 轮才让位） | `bench_hetercache --concurrency` |
| `--inflight-reserve` | allocator 里被在途请求占掉、不归缓存的 token | 拟合值约 94 token/并发槽位 |
| `--inflight-ms` | 单请求在途时长；期间其命中前缀 `lock_ref` 不可淘汰 | 结果里的 `e2e_mean_ms` |

另外 `--wait-scale` / `--arrival-scale` 必须与真机那一轮完全一致（smoke 档真机用 0.1）。

### 标定

```bash
python mixexp/calibrate_sim.py --scale full --arms lru,hc_full --threshold 2
```

脚本会从 `server_<scale>_<arm>.log` 自动取 allocator 实际容量（配置的 200k 会被显存裁剪到 35,975），
从结果 JSON 取并发数与 e2e，逐 task 对比命中率，超阈值以非零码退出。

| 实验臂 | 模拟 | 真机 | Δ 合计 | 最大单 task 偏差 | 判定 |
|---|---|---|---|---|---|
| `lru` | 59.25% | 59.17% | **+0.08pp** | 0.49pp | 通过 |
| `hc_full` | 32.72% | 48.29% | −15.57pp | 16.59pp | **未通过** |

`lru` 已经可以替代真机做规模/容量敏感性分析。`heter_cache` 系列还不能用：
缺口集中在 agentic（−16.6pp），机制上指向**在途请求的 KV 生命周期**——真机每轮
prefill/decode 会反复分配与释放请求自己的 KV，离线端只有「插入/淘汰」两条路径。
下一步是把在途请求的 decode 段也纳入模型（或在 `evict` 里显式扣除在途额度）。

### 标定过程中修掉的两个回放器缺陷

这两个缺陷都会让离线命中率虚高，且都不会报错：

1. **命中前缀未上锁**：`match_prefix` 之后直接 `evict + insert`，被命中的节点可能在
   淘汰阶段被删掉，随后 `insert` 用已释放的池索引重建节点 → 树里同一段索引出现在
   多个节点上（实测树内 token 数涨到池容量的 4 倍，`hc_full` 虚高到 85%）。
   修法：`acquire` 期间 `inc_lock_ref` / `dec_lock_ref`。
2. **会话账本未 touch**：`session_registry.touch`（重设链长 L、缺口 d 清零）只在
   `RadixCache._adopt` 里调用，而回放走的是 `cache.insert`。缺了它，RR 的 `loss_cap`
   永远读到 L=0，所有会话都被判「已达上限」→ 淘汰全部落到 `quota:fallback`
   （实测 fallback 从 2.4% 涨到 68%）。修法：插入前显式 `touch`。
