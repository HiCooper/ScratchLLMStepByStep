"""数据管线（生产）：jsonl -> .bin + .meta.json，并支持 info 校验。

用法：
    python scripts/build_pretrain_bin.py build \
        --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
        --tokenizer-dir models/tokenizer_v3 \
        --out-bin dataset/bins/pretrain_qwen.bin \
        --max-lines 300000
    python scripts/build_pretrain_bin.py info --bin dataset/bins/pretrain_qwen.bin
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from transformers import AutoTokenizer  # noqa: E402

from minigpt.data.pretrain_dataset import (  # noqa: E402
    load_bin_meta, tokenize_jsonl_to_bin,
)


def cmd_build(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenize_jsonl_to_bin(
        args.corpus_jsonl, args.out_bin, tokenizer,
        content_key=args.content_key, max_lines=args.max_lines,
    )


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
    b.add_argument("--max-lines", type=int, default=0, help="0=全量")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info")
    i.add_argument("--bin", required=True)
    i.set_defaults(func=cmd_info)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
