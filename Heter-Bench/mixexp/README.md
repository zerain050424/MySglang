# mixexp · 混合负载实验套件

用来回答一个问题：**在异构负载混合、且组成会变的场景下，HeterCache 的三队列 + 队内策略 + 配额仲裁，是否比单策略（LRU/LFU/SLRU/T-LRU）更好，以及每个部件各自贡献多少。**

设计上刻意做成两把梯子：

- **规模阶梯**：`smoke → quick → mid → full`，先验证链路，再出结论；
- **组成漂移**：四段相位 A→B→C→A，测「组成变了之后还能不能命中」。

---

## 零、准备数据（必须先做）

Qwen 的默认 system prompt 只有 24 token，但运行时判定「跨会话共享」时**没有 token 门槛**：
任一单轮会话只要命中过别人拥有的这段前缀，就会被标成 `STRUCT`。后果是标签分布严重偏斜。

同一份数据上的离线消融（`docs/reports/classify_system_prompt_ablation.md`）：

| 臂 | system prompt | acc | random 召回 | chat 召回 |
|---|---|---|---|---|
| A | 原样（生产缺省） | 0.5791 | **0.000** | 0.240 |
| C | **只去 Qwen system** | **0.7697** | **0.993** | 0.240 |

也就是说：不剥这一段，实验里根本看不到 `random` 队列，`apps` 全被算进 `struct`。
本套件因此统一读去掉该段之后的副本：

```bash
bash mixexp/prepare_data.sh     # 生成 *.nosys.jsonl（幂等）
```

`mix_*.yaml` 与 `drift_phases.yaml` 的 `data_path` 已全部指向 `*.nosys.jsonl`。
`run_mixexp.sh` 会在起 server 之前预检这些文件，缺失直接退出并提示上面这条命令。

注意 `chat` 召回仍是 0.240——那是另一个原因（76% 的 ShareGPT 会话轮数 <5，
结构上不可能判成 CHAT），剥 system prompt 解决不了。见第六节。

---

## 一、规模阶梯

规模只由 `configs/mix_<scale>.yaml` 里的 `max_record` 控制（各负载取多少条会话），
比例恒为 1:1:1:1 不变——**否则不同规模之间不可比**。

| scale | 会话数（apps/code/chat/agent） | 池容量 | 并发 | wait-scale | 用途 |
|---|---|---|---|---|---|
| `smoke` | 10/10/10/10 = 40 | 30k | 8 | 0.1 | 验证链路、各臂能起来、指标在动 |
| `quick` | 50/50/25/25 = 150 | 60k | 16 | 0.3 | 看趋势，单臂几分钟 |
| `mid` | 400/400/150/150 = 1100 | 120k | 24 | 1.0 | **出趋势性结论的主力档** |
| `full` | 1800/1800/500/500 = 4600 | 200k | 32 | 1.0 | 最终数字 |

三个旋钮必须同向放大。只放大会话数而不动池容量，工作集压不满池子，所有臂都会打平——
这是这套实验最常见的无效原因，`analyze.py` 会显式检查（判据 5）。

`wait-scale` 压缩轮间等待，只影响运行时长，会改变淘汰概率，
所以 **`smoke` / `quick` 只用于验证流程，结论必须取自 `mid` / `full`**。

---

## 二、实验臂（`arms.tsv`）

| arm | 策略 | 用途 |
|---|---|---|
| `lru` | lru | 最弱基线：全局单一顺序 |
| `lfu` | lfu | 反向基线：长会话累积高频，陈旧前缀赖着不走 |
| `slru` | slru | 中档基线：只按命中次数分段，无队列概念 |
| `tlru` | tlru | **最需要打败的对手**：上游同方向工作，按尾部是否还影响 TTFT 逐出 |
| `hc_full` | heter_cache | 完整方案 |
| `hc_norr` | heter_cache | 消融：关会话轮询 |
| `hc_nomt` | heter_cache | 消融：关多轮改写器 |
| `hc_flat` | heter_cache | 消融：不分队列，全部按 agentic 归类 |
| `hc_caps_eq` | heter_cache | 消融：配额均分 1/3，不按负载倾斜 |

单看「打赢 LRU」说服力有限。真正能证明方案成立的是后四行：
**关掉任一部件，结果是否变差**。没有这一步，无法排除「只是碰巧和 LRU 不同」。

`tlru` 必须带 `--radix-eviction-policy-config`，否则上游会因
`threshold <= next_prompt_estimate` 直接拒绝启动（默认值 0/0）。

---

## 三、组成漂移（`--mode drift`）

`configs/drift_phases.yaml` 排四段，每段 120 s，被一类负载主导：

```
code_dominant → agent_dominant → chat_dominant → code_return
```

最后一段刻意回到第一段的组成。全局单策略在这种序列下，段 1 的代码前缀早被段 2、3 冲干净，
段 4 回归时命中率会明显塌；多队列方案如果真能隔离，段 4 应该还能命中相当一部分。

这一档不提供规模阶梯（`drift_phases.yaml` 里两个数字一起乘即可放大）。

---

## 四、怎么跑

```bash
cd Heter-Bench

# 1) 先看命令对不对（不起服务）
bash mixexp/run_mixexp.sh --scale smoke --dry-run

# 2) 冒烟：全部 9 个臂 + 自动汇总
bash mixexp/run_mixexp.sh --scale smoke

# 3) 只跑关键三臂，快速看趋势
bash mixexp/run_mixexp.sh --scale quick --arms lru,tlru,hc_full

# 4) 组成漂移
bash mixexp/run_mixexp.sh --scale mid --mode drift --arms lru,slru,tlru,hc_full

# 5) 全量（耗时长，建议 tmux）
bash mixexp/run_mixexp.sh --scale full
```

只汇总、不重跑：

```bash
python mixexp/analyze.py --dir result/mixexp --scale mid --tokens 120000
```

输出：

```
result/mixexp/<scale>_<arm>.json        每次运行的 bench 结果
result/mixexp/server_<scale>_<arm>.log  server 日志（含 [HeterCache profile]）
```

---

## 五、判据

`analyze.py` 打印 markdown 表，并逐条判定：

| # | 判据 | 含义 |
|---|---|---|
| 1 | 主判据 | `hc_full` 总命中率严格高于所有 baseline |
| 2 | 公平判据 | `hc_full` 的**最弱类**命中率不低于最好的 baseline |
| 3 | 时延判据 | `hc_full` 的 `ttft_p99` 不劣于 baseline 的 1.10 倍 |
| 4 | 消融判据 | `hc_full` 不弱于任一 `hc_*` 消融臂 |
| 5 | 有效性 | 总 `prompt_tokens` 超过池容量，确实发生了淘汰 |

判据 2 是防止「靠饿死一类换总命中率」——表里的**类间差距**列就是给这个用的：
差距越大，说明有一类在被系统性牺牲。

---

## 五·补、exp3：四类混合负载（独立 struct + 大流量冲刷 + 周期性突发）

exp3 换了一套负载：四类，各类彼此独立。会话结构上有三处与 exp1/exp2 不同，
分别对应缓存行为里最容易分化的三个条件。

### 负载组成

| 类别 | 数据源 | 会话数 | 轮数结构 | token 占比 |
|---|---|---|---|---|
| chat | ShareGPT `gt5` | 800 | 6–8 轮为主，20% 取 10–15 轮 | ~22% |
| agentic | AgentBank_alfred | 96 | 30 条 60+ 轮长尾，其余 6–10 轮 | ~35% |
| **apps** | AgentBank_apps + 合成冲刷会话 | 15 000 | 单轮，彼此无共享 | ~22% |
| **struct** | 独立构造（长前缀链） | 2 600 | 单轮，链内共用 2 000 token 前缀 | ~22% |
| 合计 | | 18 496 | | ≈25 M prompt token |

apps 是主导冲刷的流量：15 000 条单轮请求占请求总数 **61%**，全部彼此无共享，
每条请求都要真实占一段 capacity，因此对 struct / chat 的前缀形成持续挤压。
真实 apps 会话只有 5 390 条，超出部分由 `build_exp3_data.py` 按**真实 prompt 长度分布**
补合成会话（正文为独立随机 token，互不重复），所以冲刷强度不受原始数据量限制：

```bash
# 冲刷更强 / 更弱，只改这一个数
python mixexp/tools/build_exp3_data.py --only apps --apps-sessions 25000
```

改完同步 `mix_full.yaml` 里 `AgentBank_apps` 的 `max_record` 与 `lambda_rate`
（`lambda_rate = 会话数 / 12000`）。

三处结构差异：

1. **struct 独立成一类**。struct 不与其他负载混用：chat / agentic / apps
   保持原始文本，不拼任何公共前缀。struct 由 `build_exp3_data.py` 单独构造——
   8 条前缀链，每条链上的 325 条会话逐轮共用同一段 2 000 token 的前缀，私有正文
   每会话唯一，所以除前缀外没有别的共享节点。
   **要调 struct 的占比，只动它自己的两个旋钮**：前缀长度与会话数，

   ```bash
   # 前缀拉到 4 000 token → struct 占比约翻倍（不动其他四类）
   python mixexp/tools/build_exp3_data.py --only struct --struct-prefix-tokens 4000
   # 或者只加会话数
   python mixexp/tools/build_exp3_data.py --only struct --struct-sessions 4000
   ```

   改完记得同步 `configs/exp3/mix_full.yaml` 里 `Struct` 的 `max_record`（该值等于
   会话数）。
2. **长尾会话**。agentic 里 30 条会话跑 60 轮以上，其余只有 6–10 轮；chat 有两成是
   10–15 轮。会话长度呈长尾，短会话与大会话争夺同一份容量。
3. **周期性突发**。相位排程为 40 个 300 秒周期（合计 3.3 h），每周期 280 秒基线 +
   20 秒突发，突发窗口内到达率放大 10 倍。

另外，`lambda_rate` 按 `会话数 / 12000 s` 逐类设定，四类同时收尾，混合组成不随时间漂移
（此前一律填 1 会让长尾任务拖在后面，后段组成偏向单类）。

### 运行参数

| 项 | 值 | 说明 |
|---|---|---|
| 池容量 | `--mem-fraction-static 0.85` + `--kv-cache-dtype fp8_e4m3` | 约 198 k token；bf16 只有 99 k，0.9 会 OOM |
| 并发 | 32 | |
| wait-scale | 1.0 | 轮间不压缩，保持复用间隔与 `reuse_config` 标定一致 |
| 配额 | struct 0.20 / agentic 0.50 / chat 0.30 | `arms_exp3.tsv`；apps 走一次性队列，占剩余 |
| 方案 | `lru`、`hc_full`（`hc_norr` 备用） | 同参数成对，缺一不可 |

### 怎么跑

```bash
cd Heter-Bench

# 1) 构造数据（幂等；产出 TaskData/exp3/*.nosys.jsonl 与 manifest.json）
bash mixexp/run_exp3.sh --build-data --dry-run     # 只打印计划
bash mixexp/run_exp3.sh --build-data               # 真跑前先构造

# 2) 稳态：单方案约 3.3 h，两个方案约 6.6 h（建议 tmux）
tmux new -s exp3
bash mixexp/run_exp3.sh

# 3) 相位模式（含 20 秒突发窗口）
bash mixexp/run_exp3.sh --mode drift

# 4) 只汇总
python mixexp/analyze.py --dir mixexp/result/exp3 --scale full --tokens 198000
```

数据构造细节：

```bash
python mixexp/tools/build_exp3_data.py --dry-run          # 看各类别规模与 struct 前缀
python mixexp/tools/build_exp3_data.py --only struct --dry-run
python mixexp/tools/build_exp3_data.py --only apps --apps-sessions 500 --out-dir /tmp/demo
python mixexp/tools/tests/test_build_exp3_data.py         # 自测：分桶/struct 共享与隔离/字段保留
```

脚本会打印每类的会话数、请求数、轮数区间与 token 占比，并写 `manifest.json`。
若某一桶的会话不足，会用剩余会话补足目标条数。

### 跑起来先看这四个数

| 指标 | 目标区间 | 偏离怎么办 |
|---|---|---|
| `token usage` 峰值 | 0.60–0.90 | 长期 ≥0.95 → 减 `max_record` 或加大池；长期 ≤0.75 → 提高 `--conc` |
| apps 占比（请求数） | 55–65% | 低于 50% 说明冲刷不够，调 `--apps-sessions` |
| chat 命中率 | 25–40% | 明显偏低时先查 `MULTI_TURN_MIN` 与数据轮数是否匹配 |
| `share≥2` 节点占可调度比例 | 15% 以上 | 偏低说明 struct 没被识别成结构性节点，检查 `manifest.json` 里 struct 的 `prefix_tokens` 与变体数 |
| 突发后 30 秒的命中率 | 与基线相差 ≤20pp | 掉得更多说明容量不足，按第一条调整 |

`analyze.py` 的五项判据照旧适用；`--tokens` 请传实际池容量（启动日志里的
`max_total_num_tokens`），不是命令行的 200000。

---

## 六、已知限制（写在前面，避免误读）

1. **`per_task` 是数据集维度，不是队列维度。** 队列由运行期状态机动态判定
   （`ShareGPT_*` → chat、`alfred_*` → agentic 等），bench 只按 task 名统计。
   所以「最弱类」是「最弱数据集」的代理，不是精确的队列指标。要看真队列数字，
   去 server 日志里的 `[HeterCache profile]`。
2. **本仓库不含数据集**。`TaskData/` 下只有生成脚本，`*_session_data.jsonl`
   需要在实验机上由 `tools/prepare_session_data.py` 生成，再经
   `mixexp/prepare_data.sh` 剥掉 Qwen 公共 system 段；本地跑会直接报找不到文件。
3. **本机无法执行**：这套脚本需要 GPU + 模型 + 数据。提交前只做了静态校验
   （`bash -n`、YAML schema + 调度构建、`HeterCacheStrategy` 参数构造、`analyze.py`
   合成数据自测、`--dry-run`）。
4. **单次运行方差**：同一臂单跑一次的数字含噪声。要写进结论，`--reps 3` 起。
