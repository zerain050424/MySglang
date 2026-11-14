# mixexp · 混合负载实验套件

回答一个问题：**异构负载混合、且主导类会轮换时，HeterCache 的三队列 + 队内策略 +
配额仲裁，是否比单策略（LRU / LFU / SLRU / T-LRU）更好，以及每个部件各自贡献多少。**

实验主体是**四个单类主导的 cell**。每个 cell 里四类负载同时在场，但只有一类
按 token 占优——这样每个 cell 都在问同一个问题的不同切面：当压力主要来自这一类时，
多队列隔离有没有用？

| cell | 主导类 | 考察什么 |
|---|---|---|
| `scan` | 一次性扫描流（apps） | 大流量、零复用的冲刷下，能复用的前缀能不能活下来 |
| `struct` | 结构复用（struct） | 跨会话共享的公共前缀是否被饿死 |
| `chat` | 多轮人工会话（chat） | 轮间间隔长的会话，早期轮次能否撑到重访 |
| `agentic` | Agent 会话（agentic） | 长会话、短间隔、上下文暴涨下的会话内复用 |

---

## 零、

**主导 = 该类 prompt token 占四类 token 总和的份额 ≥ 60%，与会话数无关。**

这条必须写死，因为两者的排序经常相反：

| 例子（实测量级） | 会话数 | 单条 token | token 占比 |
|---|---|---|---|
| apps 冲刷流 | 15 000 | ~400 | ~22% |
| agentic 长会话 | 96 | ~89 000 | ~35% |

agentic 的会话数只有 apps 的 0.6%，token 却更多。所以这套实验里：

- **不按会话数排序**，也不用每类 N 条这种配额去凑份额；
- 份额在数据构造时按 token 反解（`build_exp3_data.py`），写进 `manifest.json`；
- 跑前由 bench 复核数据组成，跑后由 `analyze.py` 用实测 `prompt_tokens` 复核——数据份额与实际跑出来的份额要一致，否则结论无效。

配角份额默认均分：`(1-60%)/3 ≈ 13.3%` 每类。允许实际略高（最后一条会话会越界），
所以主导类实测通常落在 60%~70%。

---

## 一、四类负载

| 类 | 源 | 轮数结构 | 是否合成 | 跑前校验 |
|---|---|---|---|---|
| `apps` | AgentBank_apps 单轮 | 1 轮 | 真实会话用尽后按真实长度分布补**合成冲刷会话**（互不共享） | — |
| `struct` | 独立构造 | 1 轮 | 8 条前缀链 × 2000 token，链内共用前缀、私有正文唯一 | — |
| `chat` | ShareGPT `mt5` | 6–8 轮为主，20% 取 10–15 轮 | 否 | `min_turns: 5` |
| `agentic` | AgentBank_alfred | 30% 为 60+ 轮长尾，其余 6–10 轮 | 否 | `min_turns: 5` |
---

## 二、准备数据（三步，串行）

```bash
# 0) 原始 session_data.jsonl 需先由 tools/prepare_session_data.py 在实验机上生成
bash mixexp/prepare_data.sh          # 步 1 + 步 2（幂等；FORCE=1 重生成）
bash mixexp/run_exp3.sh --build-data # 步 3：构造四个 cell（也可单独调 build_exp3_data.py）
```
---

## 三、规模

exp3 的规模由两个旋钮控制（都是环境变量），池容量与并发另算：

| 档 | `TOTAL_TOKENS` | 池容量 | 并发 | 用途 |
|---|---|---|---|---|
| 冒烟 | `3e6` | 60k | 8 | 验证链路、四类都在动 |
| 趋势 | `10e6` | 120k | 16 | 看方向 |
| 结论 | `25e6` | ~198k（fp8 0.85） | 32 | 出数字 |

```bash
TOTAL_TOKENS=3e6 CONC=8 bash mixexp/run_exp3.sh --build-data --mix scan
```
---

## 四、实验臂（`arms_exp3.tsv`）

| arm | 策略 | 用途 |
|---|---|---|
| `lru` | lru | 最弱基线：全局单一最近性顺序 |
| `hc_full` | heter_cache | 完整方案：三队列 + 队内策略 + 配额仲裁 |
| `hc_norr` | heter_cache | 消融：关会话轮询（agentic 队内退化） |
| `hc_demandcap` | heter_cache | 消融：配额按 mid 实测需求对齐（0.05/0.92/0.03），归因 cap 错配 |

`hc_full` 的配额是 `caps = struct 0.20 / agentic 0.50 / chat 0.30`（合计 1.0），
`reservations = 0.05 / 0.20 / 0.10`。`random`（one-shot）队列**不在配额账本里**：
它整行都是 one-shot，在 Stage 0 就被清掉，所以单轮冲刷流量不受 caps 约束。

> 这套 caps 是定性配的（agentic 长会话最需要保前缀），**没有按四个新 cell 重新标定**。
> 想让结论更强，先跑一轮看 server 日志里 `[HeterCache profile]` 的 per-queue 占用与
> target，再决定要不要给每个 cell 单独配 caps。

---

## 五、判据

`analyze.py` 读完结果逐条判定（硬判据不过时以非零退出）：

| # | 判据 | 含义 |
|---|---|---|
| 1 | 主判据 | `hc_full` 总命中率严格高于所有 baseline |
| 2 | 公平判据 | `hc_full` 的**最弱类**命中率不低于最好的 baseline（不许靠饿死一类换总分数） |
| 3 | 时延判据 | `hc_full` 的 `ttft_p99` 不劣于 baseline 的 1.10 倍 |
| 4 | 消融判据 | `hc_full` 不弱于任一 `hc_*` 消融臂 |
| 5 | 有效性 | 总 `prompt_tokens` 超过池容量，确实发生了淘汰 |

跨 cell 还有一个额外判据，人工核对：

**6. 份额判据** —— 实测 token 份额与 `manifest.json` 一致（±2pp）。
不一致说明到达节奏被打散（例如中途 server 重启、`--wait-scale` 被改），
这一轮的「单类主导」前提就不成立了。

---

## 六、怎么跑

```bash
cd Heter-Bench

# 1) 只看命令（不起服务）
bash mixexp/run_exp3.sh --mix all --dry-run

# 2) 全流程：准备数据 → 构造四个 cell → 每 cell 跑 lru,hc_full
tmux new -s exp3
bash mixexp/run_exp3.sh --build-data

# 3) 只跑一个 cell（已有的数据不会再构造）
bash mixexp/run_exp3.sh --mix agentic

# 4) 指定臂 / 透传参数
bash mixexp/run_exp3.sh --mix chat --arms lru,hc_full,hc_norr --conc 48

# 5) 只汇总
python mixexp/analyze.py --dir mixexp/result/exp3/agentic --scale full --tokens 198000
```

产物按 cell 分开，互不覆盖：

```
mixexp/result/exp3/<cell>/full_<arm>.json         每次运行的结果
mixexp/result/exp3/<cell>/server_full_<arm>.log   server 日志（含 [HeterCache profile]）
```

旧的单组成（22/35/22/22）仍然保留为 `--mix legacy`，产物落在
`mixexp/result/exp3/`、配置读 `configs/exp3/mix_full.yaml`；相位模式
`--mode drift` **只**支持 legacy（相位文件是按那份组成写的）。

---

## 七、跑起来先看这几个数

| 指标 | 位置 | 目标 | 偏离怎么办 |
|---|---|---|---|
| 主导类 token 份额 | `manifest.json` / 结果 JSON | ≥60%，且与数据一致 | 不一致 → 查到达节奏与 `lambda_rate` |
| `token usage` 峰值 | server 日志 | 0.60–0.90 | 长期 ≥0.95 → 降 `TOTAL_TOKENS` 或加大池；长期 ≤0.75 → 提高 `--conc` |
| 主导类请求数占比 | 结果 JSON `per_task` | 与 token 份额同向，但**不必相等** | 差得远是正常的（会话长度差异），别拿请求数当份额 |
| 共用前缀存活 | `[HeterCache profile]` 里 `share≥2` | `struct` cell 明显高于其他 cell | 偏低 → 查 struct 的 `prefix_tokens` 与变体数 |
| `chat` 命中率 | `per_task.ShareGPT` | 25–40% | 明显偏低 → 查 `MULTI_TURN_MIN` 与数据轮数是否匹配 |
| `scan` cell 的 `apps` 命中率 | `per_task.AgentBank_apps` | 接近 0 | 明显 >0 → 说明有复用漏进了冲刷流，查合成会话是否重复 |

---

## 八、排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `缺少输入数据：...nosys.mt5.jsonl` | 步 2 没跑 | `bash mixexp/prepare_data.sh` |
| `缺少数据文件` + 提示 prepare_data | 配置里的 data_path 还不存在 | 同上，再 `--build-data` |
| `[chat] 主导类 chat 只能做到 55.3%` | 数据源不够撑这个份额 | 降 `--dominant-share` / `--total-tokens`，或补数据源 |
| `task 'ShareGPT': 4/10 条会话轮数 < 5` | 数据没过滤（旧文件） | `FORCE=1 bash mixexp/prepare_data.sh` |
| `session_data.jsonl 期望 {session_id: session}` | 输入格式不对（如直接喂 raw 对话） | 先用 `tools/prepare_session_data.py` 生成 |
| `全部会话都被剔除：这看起来是单轮数据集` | 把单轮源（apps / Code）喂给了过滤器 | 过滤器只用于多轮源 |
| `phase 'x' needs N sessions from 'y', but only M are available` | 相位模式的条数超过数据池 | 降相位 `max_record`，或放大数据 |

---