"""预训练入口（生产）：python minigpt/train/pretrainer.py 或 torchrun --nproc_per_node N ...

配置来源（优先级从低到高）：minigpt/config.py 默认值 -> --config-file json -> CLI 扁平参数
（如 --model_emb_dim 384 --data_tokenized_bin ... --train_batch_size 8 --paths_output_dir ...）。

产物（--paths_output_dir 下）：
    checkpoint-{step}.pth  按 save_steps 周期保存（含 RNG/scaler）
    final.pt               训练结束主进程最终保存
    tensorboard/           SummaryWriter 标量日志
    metrics.json           最终验证指标
    config.json            本次运行的完整配置快照
    sample.txt             训练后主进程采样输出
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from minigpt.config import (ModelConfig, TrainConfig, DataConfig, PathConfig,
                            add_cli_overrides, as_nested_dict, build_run_config,
                            dump_run_config, ROOT)
from minigpt.data.pretrain_dataset import TokenBinDataset
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.trainer import Trainer
from torch.utils.data import random_split


def parse_args():
    ap = argparse.ArgumentParser(description="MiniGPT 预训练（生产）")
    ap.add_argument("--config-file", default=None, help="JSON 配置快照（可用 dump 出的 config.json）")
    for section, cls in (("model", ModelConfig), ("data", DataConfig),
                         ("train", TrainConfig), ("paths", PathConfig)):
        add_cli_overrides(ap, section, cls)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = build_run_config(args, args.config_file)
    tc, mc, dc, pc = cfg.train, cfg.model, cfg.data, cfg.paths

    rank0 = int(os.environ.get("RANK", -1)) in (-1, 0)
    torch.manual_seed(tc.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(pc.output_dir, exist_ok=True)
    dump_run_config(cfg, os.path.join(pc.output_dir, "config.json"))
    if rank0:
        print(f"[pretrainer] output_dir={pc.output_dir} device={device}")

    # 词表由 tokenizer 推导
    tokenizer = AutoTokenizer.from_pretrained(dc.tokenizer_dir)
    vocab_size = mc.vocab_size or len(tokenizer)
    gpt_cfg = GPTConfig(
        emb_dim=mc.emb_dim, n_layers=mc.n_layers, n_heads=mc.n_heads,
        context_length=mc.context_length, vocab_size=vocab_size,
        drop_rate=mc.drop_rate, qkv_bias=mc.qkv_bias,
        tie_word_embeddings=mc.tie_word_embeddings,
        use_swiglu=mc.use_swiglu, use_checkpoint=mc.use_checkpoint,
        flash_attn=mc.flash_attn,
    )
    model = MiniGPT(gpt_cfg)

    # 数据
    ds = TokenBinDataset(dc.tokenized_bin, mc.context_length, dc.bin_meta)
    eval_len = min(int(len(ds) * dc.eval_ratio), 512)
    train_len = len(ds) - eval_len
    generator = torch.Generator().manual_seed(tc.seed)
    train_set, eval_set = random_split(ds, [train_len, eval_len],
                                       generator=generator)
    if rank0:
        print(f"[pretrainer] ds_rows={len(ds)} train={len(train_set)} eval={len(eval_set)} "
              f"vocab={vocab_size} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate,
                                  weight_decay=tc.weight_decay)
    train_args = {
        "train_batch_size": tc.batch_size,
        "eval_strategy": "step",
        "eval_steps": tc.eval_steps,
        "warmup_steps": tc.warmup_steps,
        "save_strategy": "step",
        "save_steps": tc.save_steps,
        "num_train_epochs": tc.epochs,
        "max_steps": tc.max_steps,
        "gradient_accumulation_steps": tc.grad_accumulation_steps,
        "grad_clip": tc.grad_clip,
        "output_dir": pc.output_dir,
        "last_checkpoint_path": pc.last_checkpoint_path,
        "use_mixed_precision": tc.mixed_precision_dtype != "none",
        "mixed_precision_dtype": tc.mixed_precision_dtype
        if tc.mixed_precision_dtype in ("float16", "bfloat16") else "float16",
        "num_workers": tc.num_workers,
    }
    trainer = Trainer(model, optimizer, train_args, device=device, verbose=rank0)
    if rank0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            trainer.set_writer(SummaryWriter(os.path.join(pc.output_dir, "tensorboard")))
        except Exception as exc:  # noqa: BLE001
            print(f"[pretrainer] tensorboard disabled: {exc}")
    trainer.set_seed(tc.seed)
    trainer.set_dataset(train_set, eval_set)
    trainer.train()

    # 给 final.pt 补充 config，使推理/评估脚本可直接读取
    final_path = os.path.join(pc.output_dir, "final.pt")
    if rank0 and os.path.exists(final_path):
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        ck["config"] = gpt_cfg.to_dict()
        torch.save(ck, final_path)
        print(f"[pretrainer] config injected -> {final_path}")

    # 主进程落盘指标与采样
    if rank0:
        metrics = getattr(trainer, "final_metrics", {}) or {}
        metrics["config"] = as_nested_dict(cfg)
        with open(os.path.join(pc.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[pretrainer] metrics -> {os.path.join(pc.output_dir, 'metrics.json')}")
        try:
            text = "库里在第三节上篮时被防守球员犯规，但裁判并未理会"
            resp = trainer.predict(tokenizer, text, 80)
            with open(os.path.join(pc.output_dir, "sample.txt"), "w", encoding="utf-8") as f:
                f.write(f"输入: {text}\n生成: {resp}\n")
            print(f"[pretrainer] sample:\n输入: {text}\n生成: {resp}")
        except Exception as exc:  # noqa: BLE001
            print(f"[pretrainer] sample generation failed: {exc}")


if __name__ == "__main__":
    main()
