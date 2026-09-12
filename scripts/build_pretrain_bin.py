"""数据管线（生产）：jsonl -> .bin + .meta.json，并支持 info 校验。

用法：
    python scripts/build_pretrain_bin.py build \
        --corpus-jsonl dataset/pretrain_t2t.jsonl \
        --tokenizer-dir models/tokenizer_v3 \
        --out-bin dataset/bins/pretrain_qwen.bin \
        --max-lines 300000
    python scripts/build_pretrain_bin.py info --bin dataset/bins/pretrain_qwen.bin

并行：`--nproc N` 按**行首对齐**把语料切成 N 段并行 tokenize，再按序拼接
（实测 3.24G 字符的通用语料单进程约 65 分钟；本机 6 核可压到 ~12 分钟）。
并行与串行产物**逐字节一致**（`tests/test_build_bin_parallel.py` 会验证），
meta 里多两个信息字段（parts / parallel_bounds）。
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from transformers import AutoTokenizer  # noqa: E402

from minigpt.data.pretrain_dataset import (  # noqa: E402
    load_bin_meta, tokenize_jsonl_to_bin,
)


def line_aligned_bounds(path: str, n: int) -> list:
    """把文件按字节切成 n 段，每段起点对齐到行首，返回 n+1 个单调边界。

    切分点落在行中间时丢掉半行（`readline()` 到下一个行首），保证
    "第 i 段 = 起始偏移落在 [bounds[i], bounds[i+1]) 的行"，既不重复也不遗漏。
    """
    size = os.path.getsize(path)
    bounds = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()
            pos = f.tell()
            if pos > bounds[-1]:
                bounds.append(pos)
    if bounds[-1] != size:
        bounds.append(size)
    return bounds


def _tokenize_range(job):
    """worker：tokenize 一个字节区间，产出独立的 .part 文件 + 自己的 meta。"""
    src, out, tok_dir, content_key, start, end = job
    tok = AutoTokenizer.from_pretrained(tok_dir)
    meta_path = out + ".meta.json"
    tokenize_jsonl_to_bin(src, out, tok, content_key=content_key,
                          meta_path=meta_path, byte_range=(start, end))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return {"part": out, "meta_file": meta_path,
            "lines": meta["lines"], "tokens": meta["tokens"], "bad": meta.get("bad_lines", 0),
            "dtype": meta["dtype"], "vocab_size": meta["vocab_size"],
            "eos_id": meta["eos_id"], "tokenizer": meta.get("tokenizer", "")}


def cmd_build(args):
    if args.nproc <= 1:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        tokenize_jsonl_to_bin(
            args.corpus_jsonl, args.out_bin, tokenizer,
            content_key=args.content_key, max_lines=args.max_lines,
        )
        return
    if args.max_lines:
        raise SystemExit("--nproc>1 与 --max-lines 互斥（后者是全局行数上限，无法在分段内判定）")

    bounds = line_aligned_bounds(args.corpus_jsonl, args.nproc)
    parts = [f"{args.out_bin}.part{i}" for i in range(len(bounds) - 1)]
    jobs = [(args.corpus_jsonl, parts[i], args.tokenizer_dir, args.content_key,
             bounds[i], bounds[i + 1]) for i in range(len(parts))]
    print(f"[build] 并行 {len(jobs)} 进程；分段边界（按行首对齐）: {bounds}")
    with multiprocessing.Pool(len(jobs)) as pool:
        results = pool.map(_tokenize_range, jobs)

    total = {"lines": 0, "tokens": 0, "bad": 0}
    with open(args.out_bin, "wb") as out:
        for r in results:                      # 必须按分段顺序拼接
            with open(r["part"], "rb") as pf:
                shutil.copyfileobj(pf, out, length=1 << 22)
            for k in total:
                total[k] += r[k] or 0
    for r in results:
        os.remove(r["part"])
        os.remove(r["meta_file"])

    head = results[0]
    meta = {"dtype": head["dtype"], "lines": total["lines"], "tokens": total["tokens"],
            "vocab_size": head["vocab_size"], "eos_id": head["eos_id"],
            "content_key": args.content_key, "tokenizer": head["tokenizer"],
            "bad_lines": total["bad"], "parts": len(results), "parallel_bounds": bounds}
    meta_path = os.path.splitext(args.out_bin)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[build] lines={total['lines']} tokens={total['tokens']} dtype={head['dtype']} "
          f"bad_lines={total['bad']} -> {args.out_bin}")
    print(f"[build] meta -> {meta_path}")


def cmd_info(args):
    import numpy as np  # noqa: PLC0415
    meta, meta_file = load_bin_meta(args.bin)
    print(f"bin: {args.bin}")
    print(f"meta file: {meta_file} (exists={os.path.exists(meta_file)})")
    if meta:
        for k, v in meta.items():
            print(f"  {k}: {v}")
    # 用 memmap 只映射、不整份读入内存（全量 uint16 约 500MB、uint32 约 1GB）
    arr = np.memmap(args.bin, dtype=(meta or {}).get("dtype", "uint16"), mode="r")
    print(f"  actual_tokens: {arr.size}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus-jsonl", required=True)
    b.add_argument("--tokenizer-dir", required=True)
    b.add_argument("--out-bin", required=True)
    b.add_argument("--content-key", default="text")
    b.add_argument("--max-lines", type=int, default=0, help="0=全量（与 --nproc>1 互斥）")
    b.add_argument("--nproc", type=int, default=1,
                   help="并行分词进程数（>1 时按行首切分成 N 段并行，产物与串行逐字节一致）")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info")
    i.add_argument("--bin", required=True)
    i.set_defaults(func=cmd_info)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
