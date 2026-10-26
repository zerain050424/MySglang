"""冷启动 prefill 标定：flush_cache 后逐长度测 TTFT，线性拟合出 prefill 速率。

用法: python calibrate.py --base-url http://127.0.0.1:30000 --output result/tailshot/calib_lru.json
"""
import argparse, asyncio, json, random, time
from pathlib import Path
import aiohttp


async def one(http, url, input_ids):
    t0 = time.perf_counter()
    ttft = None
    body = {"input_ids": input_ids, "max_new_tokens": 1, "stream": True}
    async with http.post(url, json=body,
                         timeout=aiohttp.ClientTimeout(total=600)) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.strip()
            if line.startswith(b"data:") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.perf_counter() - t0
    return ttft, time.perf_counter() - t0


async def main_async(args):
    url = args.base_url.rstrip("/") + "/generate"
    flush = args.base_url.rstrip("/") + "/flush_cache"
    rng = random.Random(1234)
    rows = []
    async with aiohttp.ClientSession(read_bufsize=10 * 1024**2) as http:
        # 预热：首次大 prefill 会触发 kernel 编译/autotune（实测 ~10s 偏高），
        # 先丢弃一次长请求，避免污染线性拟合。
        async with http.post(flush) as r:
            await r.read()
        warm = [rng.randrange(1000, 150000) for _ in range(max(args.lengths))]
        await one(http, url, warm)
        print("  [warmup] discarded", flush=True)
        for L in args.lengths:
            for rep in range(args.reps):
                async with http.post(flush) as r:  # 清缓存：强制冷启动
                    await r.read()
                await asyncio.sleep(1.0)
                ids = [rng.randrange(1000, 150000) for _ in range(L)]
                t0 = time.perf_counter()
                ttft, e2e = await one(http, url, ids)
                rows.append({"len": L, "ttft_ms": round(ttft * 1000, 2),
                             "e2e_ms": round(e2e * 1000, 2)})
                print(f"  L={L:>6} ttft={ttft*1000:8.1f} ms", flush=True)

    # 线性拟合 ttft = a + len / rate（最小二乘）
    n = len(rows)
    sx = sum(r["len"] for r in rows)
    sy = sum(r["ttft_ms"] for r in rows)
    sxx = sum(r["len"] ** 2 for r in rows)
    sxy = sum(r["len"] * r["ttft_ms"] for r in rows)
    denom = n * sxx - sx * sx
    slope_ms_per_tok = (n * sxy - sx * sy) / denom if denom else 0.0
    intercept = (sy - slope_ms_per_tok * sx) / n
    rate_tok_per_s = 1000.0 / slope_ms_per_tok if slope_ms_per_tok > 0 else None
    out = {
        "rows": rows,
        "prefill_rate_tok_per_s": round(rate_tok_per_s, 1) if rate_tok_per_s else None,
        "ttft_ms_per_token": round(slope_ms_per_tok, 4),
        "ttft_intercept_ms": round(intercept, 2),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] prefill_rate ≈ {rate_tok_per_s:.0f} tok/s "
          f"(intercept {intercept:.1f} ms) -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--output", default="result/tailshot/calib.json")
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[1000, 2000, 4000, 8000, 16000, 24000])
    p.add_argument("--reps", type=int, default=2)
    asyncio.run(main_async(p.parse_args()))
