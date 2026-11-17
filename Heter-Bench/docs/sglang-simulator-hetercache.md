# 用 SGLang Simulator 跑 HeterCache 离线仿真

**结论先行：能跑，而且跑的是真 Scheduler + 真缓存装配链。** 但有两件事必须自己补：
CPU-only 环境要补一套依赖和一个 `sgl_kernel` 替身；trace 里必须带 `session_id`，
否则策略会静默失准——缓存照常淘汰，日志照常打印，但队列语义是空的。

本文所有数字都是在 `~/venvs/sim`（Python 3.13.9）上实跑得到的，不是估计值。

---

## 1. 这套东西到底在跑什么

`tools/sglang-simulator` 不重写缓存：它用 hook 把 `ModelRunner` 的 forward 换成延迟
预测器，`Scheduler`、`PrefillAdder`、`RadixCache` 装配链都是仓库里那份真代码。

实测启动日志（`--enable-heter-cache` 已带）：

```
Tree cache initialized: source=default impl=HeterCacheRadixCache hybrid_swa=False ...
[hetercache.strategy] INFO [Profile] [HeterCache profile]
```

补丁记录见提交 `2e2728a`：`simulation/sglang/radix_cache.py` 的
`C_RadixCacheClockHook` 把 `CacheInitParams.now` 接到 `StateManager.get_global_clock()`。
不接这根线，HeterCache 的秒级判据（老化 TTL、`ONE_SHOT_IDLE_S`、承压窗口）在
OFFLINE 模式下永不触发。

---

## 2. 补的第一件事：CPU-only 环境

```bash
python -m venv ~/venvs/sim
~/venvs/sim/bin/pip install -U pip setuptools wheel

# torch 走 CPU 源，2.13.0 是 sglang pyproject 的 pin
~/venvs/sim/bin/pip install torch==2.13.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

# triton 必须单独装：CPU torch 不带它，而 sglang 的 utils/common.py 在模块顶层 import
~/venvs/sim/bin/pip install triton

# transformers 必须 5.x（sglang 的 configs 用 PreTrainedConfig，4.x 没有）
~/venvs/sim/bin/pip install "transformers==5.12.1" "tokenizers==0.22.2" "compressed-tensors==0.18.0"

# sglang 的其余纯 Python 依赖（不装 GPU 专用那几个）
~/venvs/sim/bin/pip install aiohttp fastapi uvicorn pydantic msgspec orjson numpy scipy \
  requests tqdm psutil prometheus-client pyzmq python-multipart setproctitle xxhash zstandard \
  uvloop watchfiles packaging pybase64 partial_json_parser interegular distro easydict einops \
  ninja py-spy build sentencepiece tiktoken timm gguf blobfile \
  IPython torch_memory_saver mistral_common llguidance xgrammar outlines openai anthropic \
  modelscope soundfile datasets

# aiconfigurator 是 simulation/utils.py 的模块级硬依赖，必须 --no-deps，
# 否则它会把 torch 顶掉
~/venvs/sim/bin/pip install --no-deps "aiconfigurator==0.10.0"
~/venvs/sim/bin/pip install --no-deps "sglang-kernel==0.4.7"

# simulator 本体
~/venvs/sim/bin/pip install -e tools/sglang-simulator --no-deps
```

实测版本组合（能跑通的那一套）：

| 包 | 版本 |
|---|---|
| torch | 2.13.0+cpu |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.33.0 |
| triton | 3.9.0 |
| numpy | 2.5.4 |
| aiconfigurator | 0.10.0 |
| sglang-kernel | 0.4.7 |

三个已经踩过的坑：

1. **不要用 `--system-site-packages` 建 venv。** 继承 base 后，`torchvision` 会因为
   版本不匹配在 import 时抛 `partially initialized module`，且 `gradio`/`unsloth`
   的版本冲突会干扰 resolver。
2. **`compressed-tensors` 会拉 torch。** 先装 torch 2.13.0，再装它，否则进程内 torch
   会被升到 2.15 而和 sglang 的 pin 脱节。
3. **`sglang-kernel` 的模块名是 `sgl_kernel`**，`import sglang_kernel` 会失败。

---

## 3. 补的第二件事：`sgl_kernel` 替身

真 `sgl_kernel` 在纯 CPU 环境整包 import 不成功（`__init__.py` 顺序 import 一批 CUDA
扩展，缺 `libnvrtc.so.13` 就抛）。而 `mem_cache/memory_pool.py` 在模块顶层通过
`sglang.kernels.ops.kvcache.cache_move` 间接 import 它，于是 `build_kv_cache` 整条
装配链在 CPU 上断掉。

替身在 `tools/sglang-simulator/cpu_shim/`，靠 `PYTHONPATH` 前置遮蔽真包。
**它只保证 import 能过，任何符号被真正调用都会抛 `RuntimeError`**——不做静默返回，
否则「跑通了」和「根本没算」外观一致。

覆盖边界（核对过 `python/sglang/srt` 全树）：

- 模块级 `from sgl_kernel import ...` 只在 flashmla / xpu / npu / minicpm 这些
  attention backend 里。仿真固定用 `torch_native`，不走这些模块；
- 唯一 CPU 侧真实调用点是 `copy_all_layer_kv_cache_cpu`（裸地址搬 KV 行，没法用
  torch 等价实现）。CPU 仿真不触发 HiCache 多池搬运，正常路径调不到。

启动时看到的 `Ignore import error when loading ... '_OpNamespace' 'sgl_kernel' object
has no attribute ...` 是 sglang 自己扫描模型时的预期告警，不影响。

---

## 4. 补的第三件事：trace 里必须带 `session_id`

HeterCache 在 `radix_cache.py` 用 `getattr(req, "session_id", None)` 认会话。拿不到
就没有会话账本：队列分类、one-shot 判定、跨会话共享计数全部失效，**而且不报错**。

好消息：**不需要改任何代码。** 链路是这样的：

```
trace 行里的 "session_id"（顶层字段）
  -> autobench.py `_RESERVED_FIELDS` 不含它 -> DatasetRow.extra_request_body
  -> benchmark/serving.py  payload.update(extra_request_body)
  -> GenerateReqInput.session_id（io_struct.py）
  -> TokenizedGenerateReqInput.session_id
  -> req.session_id -> SessionRegistry
```

注意 `benchmark/simulator/bench_runner.py`（直接 Python 驱动那条路）没有这条通路，
所以走 HTTP serving（`bench_serving.py`）而不是 `SGLBenchmarkRunner`。

### 实跑对照（同一份 trace，只删掉 `session_id` 字段）

下表的「有」列取自一次代表性运行。同一条 trace 重跑，`share_counted` 会在 64~65
之间波动（`share_dedup` 稳定 108）——这正好印证第 8 节第 3 条：模拟器时钟带真实
抖动。所以只看数量级和 `victim_src` 的构成，不要逐位比数字。

| 指标 | 有 `session_id` | 无 `session_id` |
|---|---|---|
| `share: share_counted` | **65** | **0** |
| `share: share_dedup` | 108 | 0 |
| `victim_src` | `value:agentic=67636 value:struct=15461` | `value:fallback=69998` + 少量 value |
| agentic `occ` / `vd` | 3206 / 0.0104 | 1403 / 9.7682 |
| struct `occ` / `vd` | 4228(-3607) / 0.2487 | 1673 / 1.0372 |
| random `occ` | 76 | **4541**（会话归属丢失后落进来的） |

两次都是 124 请求 / 92753 input tokens / 仿真时长 36.31s / 全部成功。
**benchmark 指标几乎一样，profile 却有本质区别**——这就是「能跑通」和「测的是你的
设计」之间的差距。没有 `session_id` 时 7 万 token 走 `value:fallback`，正是
AGENTS.md 记的那条静默退化：VD 排名恒空、只能走兜底。

---

## 5. 数据准备

本地没有现成 ShareGPT 数据时，从 HF 拉一份（642MB）再截小样本：

```bash
curl -sSL -o /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
```

三步转换（tokenizer 必须和 server 的一致，否则前缀全 miss）：

```bash
TOK=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)

# 1) role 规范化（human->user, gpt->assistant）并过滤非法多轮
cd Heter-Bench/TaskData/SharedGPT-multi-turn
~/venvs/sim/bin/python clean_ShareGPT_V3.py \
  --raw_data_path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --clean_data_path /tmp/simgpt/cleaned.json

# 2) 逐 turn token 化（--max-records 截小样本；本机实测 40 条 = 124 请求）
~/venvs/sim/bin/python Heter-Bench/tools/prepare_session_data.py \
  --raw_data_path /tmp/simgpt/cleaned.json \
  --save_data_path /tmp/simgpt/session_data.jsonl \
  --tokenizer_path "$TOK" --max-records 40

# 3) 摊平成 Autobench trace，并把 session_id 写到每行顶层
~/venvs/sim/bin/python Heter-Bench/tools/make_simulator_trace.py \
  --session-data /tmp/simgpt/session_data.jsonl \
  --out /tmp/simgpt/trace.jsonl \
  --turn-gap-ms 4000 --session-stagger-ms 300 --output-cap 64
# -> 124 个请求 / 40 个会话 / 时间跨度 36.2s
```

`--turn-gap-ms` 不是随便取的：同一会话后一轮必须等前一轮**落树**才发，否则命中必然
miss。`--output-cap` 是因为真实 response 常常几百 token，仿真不需要那么长。

必须跳过第 1 步会全部失败——`prepare_session_data.py` 直接拿 `from` 值当 role，
ShareGPT 原文是 `human`/`gpt`，Qwen 模板不认。

---

## 6. 跑

```bash
cd /home/rain/MySglang
bash Heter-Bench/scripts/run_sglang_simulator_heter.sh \
  --trace /tmp/simgpt/trace.jsonl \
  --out   /tmp/simgpt/run-sid \
  --port  30083
```

脚本做的事：起 server（带 `--enable-heter-cache --radix-eviction-policy heter_cache`）
→ 等 `/model_info` 就绪 → 校验日志里 `impl=HeterCacheRadixCache` → 跑 `bench_serving.py`
→ 把 `[HeterCache profile]` 打到终端，产物落 `--out` 目录。

可调的环境变量：`SIM_PY`（venv python）、`MODEL`、`TOKENIZER`、`SIM_CFG`、
`NUM_PROMPTS`。额外的 sglang 参数用 `--` 透传。

**不要传 `--max-concurrency`。** OFFLINE 模式要把整条 trace 一次性交给 scheduler，
由模拟器按逻辑时钟排程；压并发会让它等不齐请求，卡在
`Received 4 requests so far`（实测踩过，`max-concurrency=4` 时 900s 超时）。

---

## 7. 结果怎么读

```
  queue              hit     evicted             occ        sched       vd
  agentic            46        67636             3206        3206   0.0104
  struct           1230        15461       4228(-3607)        4228   0.2487
  random              0            0               76           -        -
  pool=7510 schedulable=3903 one_shot_evicted=0
  share: min_share=2 share_counted=65 share_dedup=108 aged_evicted=0
  victim_src: value:agentic=67636 value:struct=15461
  reuse[chat]: picks=0 abstains=0
```

| 字段 | 含义 |
|---|---|
| `hit` / `evicted` | 该队列累计命中 / 淘汰 token，与 `occ` 同口径 |
| `occ` | 该队列全部驻留 token（含 response 段）；`(-N)` 后缀是其中 one-shot 部分 |
| `sched` | 账本口径的可调度占用（排除 one-shot，含 locked） |
| `vd` | 价值密度；淘汰按 vd 升序 |
| `pool` / `schedulable` | `Σocc`，以及 `Σocc - Σone_shot` |
| `share_counted` / `share_dedup` | 计入共享的节点数 / 去重后的共享数。**这是判断 `session_id` 是否到位的指纹** |
| `victim_src` | 淘汰来源：`one_shot` 带 / `value:<queue>` / `value:fallback` |
| `reuse[q]` | 队内改写器的 picks / abstains |

---

## 8. 已知边界（别过度解读）

1. **`[HeterCache profile]` 只在 evict 路径里打，间隔是 `PROFILE_INTERVAL = 30.0`
   模拟秒。** trace 的时间跨度不到 30s 就一条都不出。实测 36.2s 的 trace 只出 1 条。
2. **空闲时间不推进模拟时钟。** 时钟按 iteration 和预测延迟走，请求之间的空档会被
   跳过。所以秒级老化判据基本不会触发——实测两条 profile 都是 `aged_evicted=0`。
   想验证 TTL，得靠 `sim/` 那条路（纯虚拟时钟、可以任意 `advance_to`）。
3. **不确定可复现。** `StateManager` 的 `cpu_overhead` 取真实 `time.time()` 差值，
   `add()` 里还有 `sleep(0.05)`。同一条 trace 跑两次，老化边界可能落在不同请求上。
   做 A/B 对比要跑多次看分布。
4. **`hit_count` 会偏高。** 真 scheduler 的 chunked prefill 插完块会走
   `cache_unfinished_req → match_prefix` 回扫，多涨一截。one-shot 判据只看
   `reuse_count` 不受影响；依赖 `hit_count` 的其它口径会偏。
5. **容量必须显式给。** 传了 `--max-total-tokens` 才会按它定池
   （`C_KVCacheConfiguratorHook.override_profile_available_bytes`）；不传就按 CPU
   可用内存估算，承压判据的基准跟着漂，VD 不可比。
6. **漏 `--enable-heter-cache` 是静默的。** 会退到 `UnifiedRadixCache`，而时钟 hook
   不覆盖它，秒级判据全失效且不报错。脚本里已经加了 `impl=` 校验。
7. **它不建模 UnifiedRadixCache 的时钟**，所以别拿 HeterCache 和 Unified 在这上面
   做时钟相关对比。

---

## 9. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: transformers.configuration_utils: PreTrainedConfig` | transformers 是 4.x | 装 5.x |
| `ImportError: huggingface-hub>=0.34,<1.0` | hub 版本与 transformers 不配 | 让 pip 按 transformers 解析 |
| `No module named 'triton'` | CPU torch 不带 triton | `pip install triton` |
| `No module named 'compressed_tensors'` | 被 resolver 跳过 | 先装 torch 再单独装它 |
| `libnvrtc.so.13` / `common_ops` | 真 `sgl_kernel` 在 CPU 上不可用 | 前置 `tools/sglang-simulator/cpu_shim` 到 `PYTHONPATH` |
| `Received N requests so far` 卡住 | 传了 `--max-concurrency` | 去掉它 |
| 服务 502 / 迟迟不就绪 | 首次启动要 ~17s，`COMPILE` 慢 | 等 `/model_info` 返回 200 |
| `impl=UnifiedRadixCache` | 漏 `--enable-heter-cache` | 加上，并 `--radix-eviction-policy heter_cache` |
| `share_counted=0`、`victim_src` 全是 `value:fallback` | trace 没有 `session_id` | 用 `make_simulator_trace.py` 重新生成 |
| 一条 profile 都没有 | 时间跨度 < 30 模拟秒 | 调大 `--turn-gap-ms` |

---

## 10. 和 `sim/` 的分工

两条路不是替代关系：

| | `tools/sglang-simulator` | `sim/` |
|---|---|---|
| 调度 | 真 Scheduler + 真 `PrefillAdder` | 复刻准入子集 |
| 缓存 | 真 `HeterCacheRadixCache` | 同一份 |
| 时间 | 模拟器时钟（iteration 级，带真实抖动） | 纯虚拟时钟，可 `advance_to` |
| 起得来 | 要补依赖 + 替身 | 本机直接跑 |
| 适合 | 端到端命中率/延迟、真调度下的策略行为 | 策略口径指标、TTL/老化类判据、快速对照 |

要验证「策略逻辑对不对」，用 `sim/` 快；要验证「在真调度器下的表现」，用这条路。
两边都能跑之后，可以拿同一条 trace 做交叉对拍，但只期望同量级，不期望逐 token
相同（`sim/README.md` 的「边界」一节已写明）。
