"""SFT 指令微调入口（生产）：在预训练 checkpoint 基础上用指令数据微调。

用法：
    python3 -m minigpt.train.sft_trainer \
        --pretrain models/checkpoints/pretrain_qwen_v1/final.pt \
        --tokenizer-dir models/tokenizer_qwen2 \
        --sft-jsonl dataset/sft/sft_data_zh.jsonl \
        --paths_output_dir models/checkpoints/sft_qwen_v1 \
        --train_epochs 1 --train_batch_size 4 --train_max_steps 4000 \
        --data_max_lines 30000 --data_max_len 512

产物（--paths_output_dir 下）：checkpoint-{step}.pth / final.pt / metrics.json /
tensorboard/ / sample.txt（微调后若干指令的生成结果）。
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from minigpt.config import (DataConfig, ModelConfig, TrainConfig, PathConfig,
                            add_cli_overrides, build_run_config, dump_run_config)
from minigpt.data.sft_dataset import (InstructionDataset, create_batch_collator,
                                      split_dataset)
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.trainer import Trainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default=None)
    ap.add_argument("--pretrain", default="", help="预训练 checkpoint（final.pt/checkpoint-N.pth），缺省随机初始化")
    ap.add_argument("--sft-jsonl", default=None)
    ap.add_argument("--data_max_len", type=int, default=512)
    for section, cls in (("model", ModelConfig), ("data", DataConfig),
                         ("train", TrainConfig), ("paths", PathConfig)):
        add_cli_overrides(ap, section, cls)
    return ap.parse_args()


def build_chat(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    cfg = build_run_config(args, args.config_file)
    mc, dc, tc, pc = cfg.model, cfg.data, cfg.train, cfg.paths
    rank0 = int(os.environ.get("RANK", -1)) in (-1, 0)
    os.makedirs(pc.output_dir, exist_ok=True)
    dump_run_config(cfg, os.path.join(pc.output_dir, "config.json"))

    tokenizer = AutoTokenizer.from_pretrained(dc.tokenizer_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.unk_token = tokenizer.eos_token

    sft_jsonl = args.sft_jsonl or dc.sft_dataset
    max_len = args.data_max_len

    # 架构：优先继承预训练 checkpoint 的 config，未提供时用 CLI/默认模型配置
    if args.pretrain:
        base = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        cfg_dict = base.get("config") or {}
    else:
        base, cfg_dict = None, {}
    kws = {k: cfg_dict[k] for k in
           ("emb_dim", "n_layers", "n_heads", "context_length", "drop_rate",
            "qkv_bias", "flash_attn", "tie_word_embeddings", "use_swiglu",
            "use_checkpoint") if k in cfg_dict}
    for k, v in (("emb_dim", mc.emb_dim), ("n_layers", mc.n_layers),
                 ("n_heads", mc.n_heads), ("context_length", mc.context_length)):
        kws.setdefault(k, v)
    kws["vocab_size"] = len(tokenizer)
    gpt = GPTConfig(**kws)
    model = MiniGPT(gpt)
    if base is not None:
        model.load_state_dict(base["model_state"])
    if rank0:
        print(f"[sft] vocab={len(tokenizer)} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
              f"arch={kws['emb_dim']}/{kws['n_layers']}/{kws['n_heads']}/ctx{kws['context_length']}")

    torch.manual_seed(tc.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate,
                                  weight_decay=tc.weight_decay)

    sft_max_lines = args.data_max_lines if args.data_max_lines else dc.max_lines
    ds = InstructionDataset(sft_jsonl, tokenizer, max_len=max_len,
                            max_lines=sft_max_lines)
    print(f"[sft] dataset lines={len(ds)} max_len={max_len}") if rank0 else None
    train_set, eval_set, test_set = split_dataset(ds, 0.98, 0.01)
    collator = create_batch_collator(tokenizer)

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
    }
    trainer = Trainer(model, optimizer, train_args, device=device, verbose=rank0)
    if rank0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            trainer.set_writer(SummaryWriter(os.path.join(pc.output_dir, "tensorboard")))
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] tensorboard disabled: {exc}")
    trainer.set_seed(tc.seed)
    trainer.set_dataset(train_set, eval_set, collator)
    trainer.train()

    if rank0:
        final_path = os.path.join(pc.output_dir, "final.pt")
        if os.path.exists(final_path):
            ck = torch.load(final_path, map_location="cpu", weights_only=False)
            ck["config"] = gpt.to_dict()
            torch.save(ck, final_path)
        metrics = dict(getattr(trainer, "final_metrics", {}) or {})
        with open(os.path.join(pc.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        # 微调后采样若干指令
        model.eval()
        lines = []
        for text in ["什么是AI？", "用一句话解释机器学习。", "写一首描写秋天的五言绝句。"]:
            prompt = build_chat(tokenizer, text)
            ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
            out = model.generate(ids, 100, tokenizer.eos_token_id, use_kv_cache=False)
            resp = tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=False).strip()
            lines.append(f"用户: {text}\n模型: {resp}\n")
            print(f"用户: {text}\n模型: {resp}\n", flush=True)
        with open(os.path.join(pc.output_dir, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
