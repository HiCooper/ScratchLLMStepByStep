"""单卡预训练验证脚本（对应 notebook 09/10 的预训练流程）。

用途：在单张 GPU 上端到端跑通「数据加载 -> 前向 -> 反向 -> 参数更新 -> checkpoint -> 生成」，
用于验证 minigpt 包的训练链路是否可用（非完整训练，默认只跑 1 个 epoch 的小样本）。

用法：
    python scripts/validate_pretrain.py \
        --tokenizer models/tokenizer_v3 \
        --bin dataset/bins/pretrain_v4_full.bin \
        --output models/checkpoints \
        --epochs 1 --batch-size 2 --grad-accum 4
"""
import argparse
import os
import sys
import time

# 允许直接 `python scripts/validate_pretrain.py` 运行（与仓库内其他脚本一致）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import AutoTokenizer

from minigpt.config import TrainConfig
from minigpt.data.pretrain_dataset import PretrainBinaryDataset, split_dataset
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="单卡预训练验证")
    parser.add_argument("--tokenizer", required=True, help="分词器目录")
    parser.add_argument("--bin", required=True, help="预训练 .bin 文件（texts_to_bin 生成）")
    parser.add_argument("--output", required=True, help="checkpoint 输出目录")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0, help="0 表示跑满 epochs")
    args = parser.parse_args()

    tc = TrainConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    # flash_attn 需要 Ampere+；旧卡(Turing)请保持 False，走标准注意力
    config = GPTConfig(flash_attn=False, context_length=args.context_length)
    model = MiniGPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params/1e6:.2f}M, context_length: {config.context_length}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=tc.weight_decay)

    ds = PretrainBinaryDataset(args.bin, config.context_length)
    train_set, eval_set = split_dataset(ds[:], args.train_ratio)
    print(f"train_set: {len(train_set)}, eval_set: {len(eval_set)}")

    train_args = {
        "train_batch_size": args.batch_size,
        "eval_strategy": "step",
        "eval_steps": args.eval_steps,
        "warmup_steps": 10,
        "save_strategy": "step",
        "save_steps": args.save_steps,
        "num_train_epochs": args.epochs,
        "output_dir": args.output,
        "last_checkpoint_path": None,
        "use_mixed_precision": True,
        # RTX 2060 (Turing) 不支持 bf16，用 fp16 + GradScaler
        "mixed_precision_dtype": "float16",
        "gradient_accumulation_steps": args.grad_accum,
    }

    trainer = Trainer(model, optimizer, train_args, device="cuda", verbose=True)
    trainer.set_seed(123)
    trainer.set_dataset(train_set, eval_set)

    start = time.time()
    trainer.train()
    print(f"train use time: {(time.time()-start)/60:.2f}min")

    # 训练完测试生成（trainer.predict 内部会解包 DDP，单卡/多卡均可）
    if trainer.is_main_process:
        input_text = "秋天来了，"
        generated = trainer.predict(tokenizer, input_text, max_length=60)
        print(f"\n[生成测试] 输入: {input_text}\n[生成测试] 输出: {generated}")


if __name__ == "__main__":
    main()
