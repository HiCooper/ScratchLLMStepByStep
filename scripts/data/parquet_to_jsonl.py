#!/usr/bin/env python3
"""把 IndustryCorpus2（parquet，text 字段）转成训练用 jsonl，并可混入通用语料。

用途：
  - 领域增量预训练（推荐）：产出 {"text": ...} jsonl → scripts/data/build_pretrain_bin.py → 续训
  - 之后可再基于它构造指令数据做 SFT

示例：
  # 1) 转换（只取 text，过滤短/超长与低质量，混入 15% 通用语料防遗忘）
  python scripts/data/parquet_to_jsonl.py \
      --src dataset/IndustryCorpus2_computer_programming_code_high \
      --out dataset/domain/code_corpus.dedup.jsonl \
      --min-chars 300 --max-chars 8000 --max-line-length 500 --min-quality 3.0 \
      --mix-jsonl dataset/pretrain_t2t.jsonl --mix-ratio 0.15 --mix-limit 200000
  # 2) 建 bin（自动选 uint16/uint32 + meta）
  python scripts/data/build_pretrain_bin.py build --corpus-jsonl dataset/domain/code_corpus.dedup.jsonl \
      --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/code_domain_dedup.bin --max-lines 0
  # 3) 续训（低 lr、防遗忘；示例从 pretrain_v2_full 续跑 1 epoch）
  #   见 skills/minigpt-train/SKILL.md「领域增量预训练」或 AGENTS.md
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys


def hidden_parquet(src: str) -> list:
    """列出藏在**隐藏目录**里的 parquet（ModelScope 下载中断会留下 `._____temp/`）。

    `glob("**/*.parquet")` 按 POSIX 惯例**不匹配**以 `.` 开头的目录，于是这些文件会被
    静默跳过：转换"成功"了，语料却少了一批（实测 37GB 数学语料里有 8 个分片停在
    `._____temp`，且**全部是 footer 截断的半截文件**，连读都读不了）。
    """
    out = []
    for d in glob.glob(os.path.join(src, "**", ".*"), recursive=True):
        if os.path.isdir(d):
            out += glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)
    return sorted(out)


def check_parquet_readable(files: list) -> list:
    """只读 footer 逐个验活，返回 [(path, 错误)]。

    必须**在开始转换之前**全量验一遍：转换是边读边写输出 jsonl，若中途某个分片损坏，
    会留下一个"看起来正常、实际被截断"的 jsonl，而调用方很可能直接拿它去建 bin。
    """
    import pyarrow.parquet as pq

    bad = []
    for p in files:
        try:
            pq.ParquetFile(p).metadata       # 只读 footer，不扫描数据
        except Exception as exc:             # noqa: BLE001
            bad.append((p, f"{type(exc).__name__}: {str(exc)[:70]}"))
    return bad


def iter_parquet_texts(src: str, content_key: str, filters: dict, shuffle_seed: int | None = None,
                       max_files: int = 0, per_shard_chars: float = 0):
    """产出 `(text, source)`；`source` 是分片相对路径的第一级目录（如 chinese/english）。

    `shuffle_seed` 非空时打乱**分片读取顺序**：否则"按预算提前停止"等于只取前几个分片，
    对按语言/领域分目录的语料就是彻底的领域偏差（本仓库的数学语料 chinese 只有 1 个分片，
    顺序读会先读它，正是最坏情况）。
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"未找到 parquet 文件: {src}")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(files)
    if max_files:
        files = files[:max_files]

    hidden = hidden_parquet(src)
    if hidden:
        print(f"[parquet] ⚠️  发现 {len(hidden)} 个 parquet 位于隐藏目录（通常是 ModelScope "
              f"的 ._____temp 半截下载），glob 不会匹配它们，本次转换将**跳过**：")
        for p in hidden[:5]:
            print(f"           {os.path.relpath(p, src)}")
        if len(hidden) > 5:
            print(f"           ... 其余 {len(hidden) - 5} 个")
        print(f"[parquet]    若要包含它们，先续传并确认可解析（scripts/data/audit_dataset.py 会验活）")

    bad = check_parquet_readable(files)
    if bad:
        print(f"[parquet] ❌ {len(bad)}/{len(files)} 个分片无法解析（读取前先失败，避免产出截断的 jsonl）：")
        for p, err in bad[:8]:
            print(f"           {os.path.relpath(p, src)}: {err}")
        raise SystemExit(2)

    truncate = int(filters.get("truncate_chars") or 0)
    max_chars = int(filters["max_chars"]) or 10 ** 12      # 0 = 不限（否则 truncate 无法替代丢弃）
    for path in files:
        source = os.path.relpath(path, src)
        shard_chars = 0
        pf = pq.ParquetFile(path)
        cols = [content_key]
        for c in ("quality_score", "max_line_length"):
            if c in pf.schema.names and c not in cols:
                cols.append(c)
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            for row in batch.to_pylist():
                text = row.get(content_key) or ""
                # 过滤按**原始长度**判（截断是"保留这篇文档"，不该让它因截断而落选）
                if len(text) < filters["min_chars"] or len(text) > max_chars:
                    continue
                q = row.get("quality_score")
                if q is not None and float(q) < filters["min_quality"]:
                    continue
                mll = row.get("max_line_length")
                if mll is not None and float(mll) > filters["max_line_length"]:
                    continue
                if truncate and len(text) > truncate:
                    text = text[:truncate]
                yield text, source
                shard_chars += len(text)
                if per_shard_chars and shard_chars >= per_shard_chars:
                    break
            if per_shard_chars and shard_chars >= per_shard_chars:
                break


def sample_jsonl_texts(path: str, content_key: str, limit: int, seed: int) -> list:
    """从**整个** jsonl 里均匀抽 `limit` 条文本（蓄水池抽样，单遍、O(limit) 内存）。

    真实问题：旧实现顺序读前 `limit` 行当混料池。本仓库的 `pretrain_t2t.jsonl` 是
    **按领域排序**的（前 120 万行中文短文本，最后约 7 万行英文长文本），于是
    `run_code_domain.sh --mix-limit 300000` 抽到的防遗忘语料 **100% 来自中文头部、0% 来自英文尾部**
    （抽样核对：14,232 条混料全部命中前 30 万行）——这正好和"领域续训后通用 ppl +41%"的实测吻合。

    `limit<=0` 返回空列表（调用方必须传正数池大小；main() 里已保证）。
    """
    rng = random.Random(seed)
    pool: list = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                t = json.loads(line).get(content_key, "")
            except json.JSONDecodeError:
                continue
            if not t:
                continue
            n += 1
            if len(pool) < limit:
                pool.append(t)
            else:
                j = rng.randrange(n)          # 蓄水池：以 limit/n 的概率替换
                if j < limit:
                    pool[j] = t
    return pool


def report_composition(domain: list, sources, args) -> None:
    """打印领域语料的**组成画像**（来源分片 / 语种 / 长度分档）。

    为什么必须有：分布问题的根源就是"取到了哪一部分"不可见——旧实现按分片顺序读、
    按预算提前停，实际拿到的可能只是某一个语言/质量档；不打印出来就无法在训练前发现。
    """
    n = len(domain)
    if not n:
        print("[parquet] ⚠️  领域语料为空：检查 --min-chars/--max-chars/--min-quality 过滤条件")
        return
    chars = sum(len(t) for t in domain)
    print(f"[parquet] 领域语料: {n:,} 条 / {chars/1e6:.1f}M 字符 / "
          f"来源分片 {dict(sources.most_common(6))}"
          + (f"（共 {len(sources)} 个）" if len(sources) > 6 else ""))
    step = max(1, n // 2000)                       # 抽样 2000 条估语种，避免全量扫描
    samp = domain[::step]
    zh = sum(1 for t in samp
             if sum(1 for c in t[:2000] if "\u4e00" <= c <= "\u9fff")
             > sum(1 for c in t[:2000] if c.isascii() and c.isalpha()))
    print(f"[parquet] 语种（抽样 {len(samp)} 条）: 中文为主 {zh/len(samp):.0%} / "
          f"英文为主 {1 - zh/len(samp):.0%}")
    buckets = [(0, 1000, "<1k"), (1000, 8000, "1k–8k"), (8000, 32000, "8k–32k"),
               (32000, 10**12, ">32k")]
    hist = []
    for lo, hi, name in buckets:
        c = sum(1 for t in samp if lo <= len(t) < hi)
        hist.append(f"{name} {c/len(samp):.0%}")
    print(f"[parquet] 长度分布: " + " | ".join(hist) +
          (f"（已截断到 {args.truncate_chars}）" if args.truncate_chars else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="parquet 根目录（递归查找 *.parquet）")
    ap.add_argument("--out", required=True, help="输出 jsonl")
    ap.add_argument("--content-key", default="text")
    ap.add_argument("--limit", type=int, default=0, help="最多写入多少条（0=全部）")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="超长文档丢弃阈值（0=不限；长文档语料建议配 --truncate-chars 用截断代替丢弃）")
    ap.add_argument("--min-quality", type=float, default=0.0, help="quality_score 下限（0=不过滤）")
    ap.add_argument("--max-line-length", type=float, default=1e9, help="max_line_length 上限")
    ap.add_argument("--mix-jsonl", default="", help="混入的通用语料 jsonl（防灾难性遗忘）")
    ap.add_argument("--mix-ratio", type=float, default=0.15,
                    help="混入比例（默认按**字符/token 量**计，即分布口径；--mix-by lines 可回到旧的按条数）")
    ap.add_argument("--mix-by", choices=["chars", "lines"], default="chars",
                    help="混料比例口径：chars=按字符（默认，推荐）；lines=按条数（旧行为，"
                         "对长短差异大的语料会严重低估混料量）")
    ap.add_argument("--mix-limit", type=int, default=0,
                    help="混料池大小（**全文件均匀抽样**，不是前 N 行；0=直接抽所需条数）")
    ap.add_argument("--target-chars", type=int, default=0,
                    help="领域语料字符预算（0=全部）。配合 --shuffle-files 用，避免预算截断变成领域截断")
    ap.add_argument("--truncate-chars", type=int, default=0,
                    help="超长文档**截断**到 N 字符（0=不截断；通常配 --max-chars 0 使用）")
    ap.add_argument("--shards", type=int, default=0,
                    help="最多用几个分片（0=自动按预算均摊到 min(8,分片数) 个；"
                         "只用 1 个分片会把预算全花在单源上，是本仓库明令避免的偏差）")
    ap.add_argument("--shuffle-files", action="store_true",
                    help="打乱分片读取顺序（seed 控制）：预算/limit 截断时避免只取前几个分片")
    ap.add_argument("--mix-content-key", default="text")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tokenizer", default="models/tokenizer_v3", help="用于估算 tokens（可空）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    filters = {"min_chars": args.min_chars, "max_chars": args.max_chars,
               "min_quality": args.min_quality, "max_line_length": args.max_line_length,
               "truncate_chars": args.truncate_chars}

    n_files = len(glob.glob(os.path.join(args.src, "**", "*.parquet"), recursive=True)) or 1
    max_files = args.shards or (min(8, n_files) if args.target_chars else 0)
    per_shard = (args.target_chars / max_files) if (args.target_chars and max_files) else 0
    if args.target_chars:
        print(f"[parquet] 使用 {max_files or n_files}/{n_files} 个分片，每片预算 {per_shard/1e6:.1f}M 字符")

    domain, stats = [], {"chars": 0}
    sources = collections.Counter()
    for text, source in iter_parquet_texts(
            args.src, args.content_key, filters,
            shuffle_seed=args.seed if args.shuffle_files else None,
            max_files=max_files, per_shard_chars=per_shard):
        domain.append(text)
        stats["chars"] += len(text)
        sources[source.split(os.sep)[0]] += 1
        if args.limit and len(domain) >= args.limit:
            break
        if args.target_chars and stats["chars"] >= args.target_chars:
            break
    report_composition(domain, sources, args)

    mix = []
    if args.mix_jsonl and args.mix_ratio > 0:
        if args.mix_by == "chars":
            # 按**字符/token 量**配比：分布控制看的是这个口径。
            # 真实事故：旧实现按**条数**算，而通用语料条目短（均值 274–385 字符）、领域语料
            # 条目长（代码 2421、数学 28651），于是"混 15%"在字符口径下只有 **1.6%** —— 这正是
            # 当年"15% 混料没防住灾难性遗忘"的真正原因（见 SKILL.md §7.6）。
            want_chars = stats["chars"] * args.mix_ratio / max(1e-9, 1 - args.mix_ratio)
            pool = args.mix_limit or max(1000, int(want_chars / 200))
            mix = sample_jsonl_texts(args.mix_jsonl, args.mix_content_key, pool, args.seed)
            random.Random(args.seed).shuffle(mix)
            out, acc = [], 0
            for t in mix:
                if acc >= want_chars:
                    break
                out.append(t)
                acc += len(t)
            mix = out
        else:
            want = int(len(domain) * args.mix_ratio / max(1e-9, (1 - args.mix_ratio)))
            pool = args.mix_limit if args.mix_limit > 0 else want
            mix = sample_jsonl_texts(args.mix_jsonl, args.mix_content_key, pool, args.seed)
            random.Random(args.seed).shuffle(mix)
            mix = mix[:want]
        mix_chars = sum(len(x) for x in mix)
        stats["chars"] += mix_chars
        print(f"[mix] 从 {args.mix_jsonl} **全文件均匀**抽 {len(mix):,} 条 / "
              f"{mix_chars/1e6:.1f}M 字符 = 最终**字符**的 "
              f"{mix_chars/max(1, stats['chars']):.1%}（口径 {args.mix_by}）")

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
    print(f"[domain2jsonl] 下一步: python scripts/data/build_pretrain_bin.py build "
          f"--corpus-jsonl {args.out} --tokenizer-dir models/tokenizer_v3 "
          f"--out-bin dataset/bins/{os.path.splitext(os.path.basename(args.out))[0]}.bin --max-lines 0")


if __name__ == "__main__":
    main()
