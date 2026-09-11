"""预训练评估（生产）：对验证集窗口计算 loss / perplexity，落盘 JSON 指标。

用法：
    python scripts/evaluate_pretrain.py \
        --checkpoint models/checkpoints/pretrain_v2_full/final.pt \
        --tokenizer-dir models/tokenizer_v3 \
        --bin dataset/bins/pretrain_v3_full.bin \
        --split val --batch-size 8 --output metrics_eval.json

口径说明（重要）
    --split val（默认）：评估 .bin 尾部连续、对齐文档边界的**验证集**，与训练时
        split_train_eval_blocks 完全同一口径，可用于比较泛化能力。
    --split all / --max-rows N：评估 bin 的前 N 个窗口。若该 bin 参与过训练，
        前 N 个窗口几乎必然落在训练集里，得到的是**训练 loss**，不能作为泛化证据。
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from minigpt.data.pretrain_dataset import (TokenBinDataset,  # noqa: E402
                                          split_train_eval_blocks)
from minigpt.model.checkpoint import build_model_from_checkpoint
from minigpt.model.transformer import GPTConfig, MiniGPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_v3")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--bin-meta", default="", help=".meta.json 路径（空=与 bin 同名）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0, help="0=全部窗口")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="在 .bin 的哪个切分上评估。val=尾部连续且对齐文档边界的验证集"
                         "（与训练时 split_train_eval_blocks 同一口径）；all=整个 bin")
    ap.add_argument("--eval-ratio", type=float, default=0.002,
                    help="--split val/train 时的验证比例，需与训练时一致")
    ap.add_argument("--eval-blocks", type=int, default=32,
                    help="验证集分块数，需与训练时一致（1=只取一段）")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 优先用 checkpoint 自带 config，缺失则按权重形状反推（兼容 best.pt/周期 checkpoint 等老产物）
    model, ckpt, kws = build_model_from_checkpoint(args.checkpoint, tokenizer=tokenizer, device=device)
    gpt = model.config

    full_ds = TokenBinDataset(args.bin, gpt.context_length, args.bin_meta)
    split_info = {"split": args.split}
    if args.split == "all":
        ds = full_ds
        if args.max_rows:
            # 真实事故：对训练用过的 bin 取最前面 N 个窗口评估，得到的其实是**训练集
            # loss**（README 里 v2 的"37.80→23.66"、领域的"−64%"都因此不可信）。
            print("[eval] ⚠️ --split all 且指定了 --max-rows：若该 bin 参与过训练，"
                  "前 N 个窗口几乎必然在训练集内，指标会被严重高估。\n"
                  "        要评估泛化请用 --split val（尾部连续 + 文档边界对齐）。")
    else:
        eos_id = (full_ds.meta or {}).get("eos_id")
        if eos_id is None:
            eos_id = tokenizer.eos_token_id
        train_ds, val_ds, split_info = split_train_eval_blocks(
            full_ds, eval_ratio=args.eval_ratio, eos_id=eos_id,
            n_blocks=max(1, args.eval_blocks))
        ds = val_ds if args.split == "val" else train_ds
    if args.max_rows and args.max_rows < len(ds):
        ds = Subset(ds, range(args.max_rows))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    total, nb_tokens = 0.0, 0
    t0 = time.time()
    with torch.no_grad():
        for X, Y in loader:
            logits = model(X.to(device))
            loss = F.cross_entropy(logits.flatten(0, 1), Y.to(device).flatten(),
                                   reduction="sum")
            total += loss.item()
            nb_tokens += Y.numel()
    mean = total / max(1, nb_tokens)
    metrics = {
        "rows": len(ds),
        "tokens": nb_tokens,
        "eval_loss": mean,
        "perplexity": math.exp(min(mean, 80.0)),
        "checkpoint": args.checkpoint,
        "bin": args.bin,
        "bin_vocab": (full_ds.meta or {}).get("vocab_size"),
        "split": split_info,
        "max_rows": args.max_rows,
        "eval_seconds": round(time.time() - t0, 2),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[eval] saved -> {args.output}")


if __name__ == "__main__":
    main()
