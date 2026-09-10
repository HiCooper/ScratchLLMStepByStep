"""预训练评估（生产）：对验证集窗口计算 loss / perplexity，落盘 JSON 指标。

用法：
    python scripts/evaluate_pretrain.py \
        --checkpoint models/checkpoints/minigpt_pretrain/checkpoint-2000.pth \
        --tokenizer-dir models/tokenizer_qwen2 \
        --bin dataset/bins/pretrain_qwen.bin \
        --batch-size 8 --max-rows 0 --output metrics_eval.json
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

from minigpt.data.pretrain_dataset import TokenBinDataset  # noqa: E402
from minigpt.model.checkpoint import build_model_from_checkpoint
from minigpt.model.transformer import GPTConfig, MiniGPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_qwen2")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0, help="0=全部窗口")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 优先用 checkpoint 自带 config，缺失则按权重形状反推（兼容 best.pt/周期 checkpoint 等老产物）
    model, ckpt, kws = build_model_from_checkpoint(args.checkpoint, tokenizer=tokenizer, device=device)
    gpt = model.config

    ds = TokenBinDataset(args.bin, gpt.context_length)
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
        "rows": min(len(ds), (args.max_rows or len(ds))),
        "tokens": nb_tokens,
        "eval_loss": mean,
        "perplexity": math.exp(min(mean, 80.0)),
        "checkpoint": args.checkpoint,
        "bin": args.bin,
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
