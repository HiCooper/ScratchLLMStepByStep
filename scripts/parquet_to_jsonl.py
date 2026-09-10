#!/usr/bin/env python3
"""把 IndustryCorpus2（parquet，text 字段）转成训练用 jsonl，并可混入通用语料。

用途：
  - 领域增量预训练（推荐）：产出 {"text": ...} jsonl → scripts/build_pretrain_bin.py → 续训
  - 之后可再基于它构造指令数据做 SFT

示例：
  # 1) 转换（只取 text，过滤短/超长与低质量，混入 15% 通用语料防遗忘）
  python scripts/parquet_to_jsonl.py \
      --src dataset/IndustryCorpus2_computer_programming_code_high \
      --out dataset/domain/code_corpus.jsonl \
      --min-chars 300 --max-chars 8000 --max-line-length 500 --min-quality 3.0 \
      --mix-jsonl dataset/pretrain_t2t_mini.jsonl --mix-ratio 0.15 --mix-limit 200000
  # 2) 建 bin（自动选 uint16/uint32 + meta）
  python scripts/build_pretrain_bin.py build --corpus-jsonl dataset/domain/code_corpus.jsonl \
      --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/code_domain.bin --max-lines 0
  # 3) 续训（低 lr、防遗忘；示例从 pretrain_v2_full 续跑 1 epoch）
  #   见 skills/minigpt-train/SKILL.md「领域增量预训练」或 AGENTS.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys


def iter_parquet_texts(src: str, content_key: str, filters: dict):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"未找到 parquet 文件: {src}")
    for path in files:
        pf = pq.ParquetFile(path)
        cols = [content_key]
        for c in ("quality_score", "max_line_length"):
            if c in pf.schema.names and c not in cols:
                cols.append(c)
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            for row in batch.to_pylist():
                text = row.get(content_key) or ""
                if len(text) < filters["min_chars"] or len(text) > filters["max_chars"]:
                    continue
                q = row.get("quality_score")
                if q is not None and float(q) < filters["min_quality"]:
                    continue
                mll = row.get("max_line_length")
                if mll is not None and float(mll) > filters["max_line_length"]:
                    continue
                yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="parquet 根目录（递归查找 *.parquet）")
    ap.add_argument("--out", required=True, help="输出 jsonl")
    ap.add_argument("--content-key", default="text")
    ap.add_argument("--limit", type=int, default=0, help="最多写入多少条（0=全部）")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--min-quality", type=float, default=0.0, help="quality_score 下限（0=不过滤）")
    ap.add_argument("--max-line-length", type=float, default=1e9, help="max_line_length 上限")
    ap.add_argument("--mix-jsonl", default="", help="混入的通用语料 jsonl（防灾难性遗忘）")
    ap.add_argument("--mix-ratio", type=float, default=0.15, help="混入比例（按最终总条数计）")
    ap.add_argument("--mix-limit", type=int, default=0, help="从 mix-jsonl 最多读取多少条")
    ap.add_argument("--mix-content-key", default="text")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tokenizer", default="models/tokenizer_v3", help="用于估算 tokens（可空）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    filters = {"min_chars": args.min_chars, "max_chars": args.max_chars,
               "min_quality": args.min_quality, "max_line_length": args.max_line_length}

    domain, stats = [], {"chars": 0}
    for text in iter_parquet_texts(args.src, args.content_key, filters):
        domain.append(text)
        stats["chars"] += len(text)
        if args.limit and len(domain) >= args.limit:
            break

    mix = []
    if args.mix_jsonl and args.mix_ratio > 0:
        want = int(len(domain) * args.mix_ratio / max(1e-9, (1 - args.mix_ratio)))
        rng = random.Random(args.seed)
        with open(args.mix_jsonl, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.mix_limit and i >= args.mix_limit:
                    break
                try:
                    t = json.loads(line).get(args.mix_content_key, "")
                except json.JSONDecodeError:
                    continue
                if t:
                    mix.append(t)
        rng.shuffle(mix)
        mix = mix[:want]
        stats["chars"] += sum(len(x) for x in mix)

    rows = [(t, "domain") for t in domain] + [(t, "general") for t in mix]
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for text, _ in rows:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    est_tokens = None
    if args.tokenizer and os.path.isdir(args.tokenizer):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            sample = [t for t, _ in rows[:200]]
            n = sum(len(tok(t)["input_ids"]) for t in sample)
            c = sum(len(t) for t in sample)
            est_tokens = int(stats["chars"] * (n / max(c, 1)))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] token 估算失败: {exc}", file=sys.stderr)

    print(f"[domain2jsonl] domain={len(domain)} general(mix)={len(mix)} total={len(rows)} -> {args.out}")
    print(f"[domain2jsonl] chars≈{stats['chars']/1e6:.1f}M, bytes≈{stats['chars']*1.5/1e6:.0f}MB(估)"
          + (f", tokens≈{est_tokens/1e6:.1f}M（按 {args.tokenizer} 抽样估算）" if est_tokens else ""))
    print(f"[domain2jsonl] 下一步: python scripts/build_pretrain_bin.py build "
          f"--corpus-jsonl {args.out} --tokenizer-dir models/tokenizer_v3 "
          f"--out-bin dataset/bins/{os.path.splitext(os.path.basename(args.out))[0]}.bin --max-lines 0")


if __name__ == "__main__":
    main()
