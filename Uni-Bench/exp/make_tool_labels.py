#!/usr/bin/env python3
"""给 session_data 的每一轮打 `tools` 标签，供 gap_mode=tool_aware 使用。

判定规则：
  1) response_text 是 {"name": ..., "arguments": ...} 形式的工具调用 → 取 name
     （APIGen-MT / 大多数 function-calling 数据集）
  2) alfred 的 "Action: xxx"（环境动作，真实环境里 ~0.1s）→ "env_action"
  3) 其余 → []（没有工具调用 → gap 走「人工等待」桶）

用法：
  python exp/make_tool_labels.py --src in.jsonl [--dst out.jsonl] --mode auto
  python exp/make_tool_labels.py --src in.jsonl.gz --dst out.jsonl --mode apigen --limit 200
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]+)")


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def tool_of(text: str, mode: str) -> list[str]:
    if not text:
        return []
    if mode in ("auto", "apigen"):
        t = text.strip()
        # 有些样本尾部带 <|im_end|> 之类
        brace = t.find("{")
        if brace >= 0:
            try:
                obj = json.loads(t[brace : t.rfind("}") + 1])
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    return [str(obj["name"])]
            except Exception:
                pass
    if mode in ("auto", "alfred"):
        m = ACTION_RE.search(text)
        if m:
            return ["env_action"]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None, help="缺省覆盖 --src（gz 源则写同名 .jsonl）")
    ap.add_argument("--mode", choices=["auto", "apigen", "alfred"], default="auto")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.with_suffix(".jsonl")
    n_sess = n_turn = n_tool = 0
    names: dict[str, int] = {}

    with _open(src) as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            sid, sess = next(iter(json.loads(line).items()))
            for t in range(1, sess["num_turns"] + 1):
                turn = sess[f"turn_{t}"]
                tl = tool_of(turn.get("response_text", ""), args.mode)
                turn["tools"] = tl
                n_turn += 1
                if tl:
                    n_tool += 1
                    for x in tl:
                        names[x] = names.get(x, 0) + 1
            g.write(json.dumps({sid: sess}, ensure_ascii=False) + "\n")
            n_sess += 1
            if args.limit > 0 and n_sess >= args.limit:
                break

    top = sorted(names.items(), key=lambda x: -x[1])[:8]
    print(f"{src.name} → {dst}")
    print(f"  会话 {n_sess}  轮 {n_turn}  其中有工具 {n_tool} ({n_tool/max(n_turn,1)*100:.1f}%)")
    print(f"  工具 top8: {top}")


if __name__ == "__main__":
    main()
