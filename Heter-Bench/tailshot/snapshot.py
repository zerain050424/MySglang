"""Tailshot 矩阵快照：完成的 run 逐个出受害者画像 + 当前运行状态。"""
import csv, json, glob, os, statistics as S, re, time

BASE = "/root/sg-unicache/Uni-Bench/result/tailshot"
MATRIX = f"{BASE}/matrix.log"

log = open(MATRIX).read() if os.path.exists(MATRIX) else ""
done = re.findall(r"=== \[(\S+)\] bench done ([\d:]+)", log)
cur = re.findall(r"=== \[(\S+)\] bench start ([\d:]+)", log)
cur = cur[-1] if cur else None
active = cur[0] if cur and (not done or cur[0] not in [d[0] for d in done]) else None

print(f"== 矩阵进度: {len(done)}/8 完成 | 运行中: {active or '—'} | {time.strftime('%H:%M')} ==")
for tag, t in done:
    print(f"  ✓ {tag} @ {t}")

for f in sorted(glob.glob(f"{BASE}/*_F*.csv") + glob.glob(f"{BASE}/*_control.csv")):
    tag = os.path.basename(f)[:-4]
    rows = list(csv.DictReader(open(f)))
    vic = [r for r in rows if r["task"] == "AgentBank_alfred"]
    chn = [r for r in rows if r["task"] != "AgentBank_alfred"]
    big = [r for r in vic if int(r["prompt_tokens"]) > 500]  # 排除首轮冷启动
    if not big:
        print(f"\n-- {tag}: victims 无有效轮次"); continue
    rec = [1 - int(r["cached_tokens"]) / int(r["prompt_tokens"]) for r in big]
    tt = sorted(float(r["ttft_ms"]) for r in big)
    gaps = [float(r["gap_since_prev_end_s"]) for r in big if r["gap_since_prev_end_s"]]
    churn_tok = sum(int(r["prompt_tokens"]) - int(r["cached_tokens"]) for r in chn)
    p99 = tt[min(int(0.99 * (len(tt) - 1)), len(tt) - 1)]
    print(f"\n-- {tag}")
    print(f"   victims(n={len(big)}): 重算率 med={S.median(rec)*100:5.1f}%  max={max(rec)*100:5.1f}%")
    print(f"   TTFT: p50={S.median(tt):7.0f}ms  p99={p99:7.0f}ms  max={max(tt):7.0f}ms")
    if gaps:
        print(f"   轮间隔 med={S.median(gaps):.1f}s | 搅拌新token注入≈{churn_tok:,}")
