# 池文件清单（TaskData/burst/）

| 文件 | 用途 | N | L@turn10 范围 / 均值 |
|---|---|---|---|
| `idle_8.jsonl` / `idle_8_s1n3.jsonl` | s1n3 / n4 的 idle 组 | 8 | 5,750–6,272 / 6,001 |
| `active_3.jsonl` / `active_3_s1n3.jsonl` | s1n3 的 active 组 | 3 | 2,208–2,758 / 2,570 |
| `active_4_n4.jsonl` | n4 的 active 组 | 4 | 2,236–2,469 / 2,361 |
| `idle_8_n8.jsonl` | n8 的 idle 组 | 8 | 5,529–6,429 / 5,806 |
| `active_8_n8.jsonl` | n8 的 active 组 | 8 | 2,217–2,743 / 2,512 |
| `idle_8_s20251029.jsonl` | seed2 的 idle 组 | 8 | 5,558–8,348 / 6,188 |
| `active_3_s20251029.jsonl` | seed2 的 active 组 | 3 | 2,468–2,684 / 2,608 |
| `idle_8_s20251030.jsonl` | seed3 的 idle 组 | 8 | 5,558–6,149 / 5,776 |
| `active_3_s20251030.jsonl` | seed3 的 active 组 | 3 | 2,212–2,767 / 2,574 |

生成命令（可复现）：

```bash
/root/sl_old/bin/python RR/scripts/build_burst_pool.py \
    --seed <20251027|20251029|20251030> --n-active <3|4|8> --n-idle 8 --pause-s 60 --name <tag>
```

切分规则：APIGenMT 会话中，第 10 轮链长 ∈ [5.5k, 9.5k] → idle 组；∈ [1.5k, 2.8k] → active 组；
两组都截断到 12 轮（10 轮 burst + pause + 1 轮恢复）。
