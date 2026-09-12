#!/usr/bin/env python3
"""体检 SFT 数据在**真实训练配置**下的规模与截断路径（只读、CPU、不碰 GPU）。

为什么需要它：SFT 的预算以前是"估"的，而估错了两处——40k 条 × 2 epoch 被写成
"10 万条"（实际 8 万样本），监督 token 写成 21.2M（实测 14.5M，高估 46%）。
这类数字直接决定"对 47.9M 模型算不算过量"，值得用真数据量一遍而不是拍脑袋。

它按 `InstructionDataset` 的真实逻辑跑一遍，统计：
  - fit              整段（含历史轮）在 max_len 内，原样使用
  - prompt_trimmed   超长，但截掉前面的 prompt 后最后一轮回复完整保留
  - supervised_tail  连回复本身都超长 → 退化为"监督尾部"（等价续写目标）
  - zero_supervision 一条监督 token 都没有（**期望为 0**，非 0 说明截断策略退化）

用法：
  python3 scripts/audit_sft_lengths.py                      # 默认 sft_data_zh 前 4 万条 / ctx512
  python3 scripts/audit_sft_lengths.py --jsonl dataset/sft/sft_cot_hard_80k.jsonl --max-lines 80000
  python3 scripts/audit_sft_lengths.py --json /tmp/sft_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from minigpt.data.sft_dataset import InstructionDataset, assistant_spans  # noqa: E402


def measure(jsonl: str, tokenizer_dir: str, max_lines: int, max_len: int, progress: bool = True) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    ds = InstructionDataset(jsonl, tok, max_len=max_len, max_lines=max_lines)
    n = len(ds)

    # 包一层拿到**截断前**的完整长度：process() 只返回截断后的结果，
    # 直接从结果反推分不清"恰好 512"和"被截到 512"。
    rec: dict = {}
    orig = tok.apply_chat_template

    def wrapped(*a, **k):
        out = orig(*a, **k)
        ids = out if isinstance(out, list) else out["input_ids"]
        rec["full"] = len(ids)
        return out

    tok.apply_chat_template = wrapped

    paths = {"fit": 0, "prompt_trimmed": 0, "supervised_tail": 0, "zero_supervision": 0}
    sup_tokens = 0
    full_lens: list[int] = []
    t0 = time.time()
    for i in range(n):
        ids, supervise_all = ds[i]
        full = rec.get("full", len(ids))
        full_lens.append(full)
        if full <= max_len:
            paths["fit"] += 1
        elif supervise_all:
            paths["supervised_tail"] += 1
        else:
            paths["prompt_trimmed"] += 1
        if supervise_all:
            sup = len(ids)
        else:
            spans = assistant_spans(ids, tok)
            sup = len(ids) - spans[-1][0] if spans else 0
        if sup <= 0:
            paths["zero_supervision"] += 1
        sup_tokens += max(0, sup)
        if progress and (i + 1) % 10000 == 0:
            print(f"[audit_sft] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)

    full_lens.sort()
    return {
        "jsonl": jsonl, "tokenizer_dir": tokenizer_dir,
        "samples": n, "max_len": max_len,
        "paths": paths,
        "path_pct": {k: round(v / n * 100, 2) for k, v in paths.items()},
        "supervised_tokens_1epoch": sup_tokens,
        "supervised_tokens_2epoch": sup_tokens * 2,
        "over_max_len_pct": round(sum(1 for x in full_lens if x > max_len) / n * 100, 2),
        "full_len_avg": round(sum(full_lens) / n, 1),
        "full_len_p50": full_lens[n // 2],
        "full_len_p90": full_lens[int(n * 0.9)],
        "full_len_p99": full_lens[int(n * 0.99)],
        "full_len_max": full_lens[-1],
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="dataset/sft/sft_data_zh.jsonl")
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_v3")
    ap.add_argument("--max-lines", type=int, default=40000,
                    help="只统计前 N 条（0=全文件）；默认 40000 = 下游 SFT 阶段实际用量")
    ap.add_argument("--max-len", type=int, default=512, help="与 sft_trainer 的 --data_max_len 一致")
    ap.add_argument("--json", default=None, help="同时把结果写入该路径")
    args = ap.parse_args()

    out = measure(args.jsonl, args.tokenizer_dir, args.max_lines, args.max_len)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[audit_sft] json -> {args.json}")
    if out["paths"]["zero_supervision"]:
        print(f"[audit_sft] ⚠️ 有 {out['paths']['zero_supervision']} 条零监督样本，"
              f"说明截断策略在这些样本上退化了")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
