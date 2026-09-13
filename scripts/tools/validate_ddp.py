"""验证 DDP 多卡训练链路（检查分布式代码路径）。

用 torchrun 启动：
    # 单卡单进程：验证 DDP 代码路径本身（init_process_group / DDP 包装 / 采样器 / barrier）
    torchrun --nproc_per_node 1 scripts/tools/validate_ddp.py --tokenizer models/tokenizer_v3 --bin dataset/debug/pretrain_head.bin

    # 多卡：每进程独占一张卡，验证跨进程梯度同步
    torchrun --nproc_per_node 2 scripts/tools/validate_ddp.py --tokenizer models/tokenizer_v3 --bin dataset/debug/pretrain_head.bin

注意：跨进程梯度同步需要 >=2 张真实 GPU（NCCL 需要多卡环境）；单卡只能验证到
「DDP 代码路径本身能跑通」这一层（nproc=1）。

训练结束后每个进程会打印「参数和」；若 DDP 梯度同步正确，所有进程打印的数值应完全一致。

默认用小模型，便于快速验证。
"""
import argparse
import os
import sys

# 允许直接 `python scripts/tools/validate_ddp.py` / torchrun 运行（与仓库内其他脚本一致）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
from transformers import AutoTokenizer

from minigpt.data.pretrain_dataset import PretrainBinaryDataset, split_dataset
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="DDP 训练链路验证")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--output", default="models/checkpoints_ddp")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    config = GPTConfig(
        flash_attn=False,
        context_length=args.context_length,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    model = MiniGPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    ds = PretrainBinaryDataset(args.bin, config.context_length)
    train_set, eval_set = split_dataset(ds[:], 0.95)

    train_args = {
        "train_batch_size": args.batch_size,
        "eval_steps": 10,
        "warmup_steps": 5,
        "save_strategy": "no",
        "num_train_epochs": args.epochs,
        "output_dir": args.output,
        "last_checkpoint_path": None,
        "use_mixed_precision": True,
        "mixed_precision_dtype": "float16",
        "gradient_accumulation_steps": args.grad_accum,
    }

    # device 由 Trainer 的 _init_distributed_mode 在 DDP 模式下按 local_rank 覆盖
    trainer = Trainer(model, optimizer, train_args, device="cuda", verbose=True)
    trainer.set_seed(123)
    trainer.set_dataset(train_set, eval_set)
    trainer.train()

    # 训练后打印参数和：DDP 梯度同步正确时，各进程应打印完全相同的值
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    param_sum = sum(p.sum().item() for p in model.parameters())
    print(f"[rank {trainer.rank}] param_sum after training: {param_sum:.6f}")


if __name__ == "__main__":
    main()
