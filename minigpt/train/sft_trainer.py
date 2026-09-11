"""SFT 指令微调入口（生产）：在预训练 checkpoint 基础上用指令数据微调。

用法：
    python3 -m minigpt.train.sft_trainer \
        --pretrain models/checkpoints/pretrain_qwen_v1/final.pt \
        --data_tokenizer_dir models/tokenizer_v3 \
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
                                      resolve_stop_token_ids, split_dataset)
from minigpt.model.checkpoint import model_kwargs_from_checkpoint
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

    # sft_dataset 定义在 PathConfig 上（旧写法 dc.sft_dataset 在不传 --sft-jsonl 时
    # 会直接 AttributeError：DataConfig 里没有这个字段）
    sft_jsonl = args.sft_jsonl or pc.sft_dataset
    max_len = args.data_max_len

    # 架构：优先继承预训练 checkpoint 的 config；缺失时按权重形状反推；最后才用 CLI/默认配置
    if args.pretrain:
        base = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        kws, inferred = model_kwargs_from_checkpoint(base, vocab_size=len(tokenizer))
        if inferred and rank0:
            print(f"[sft] {args.pretrain} 未携带 config，已按权重形状推断结构：{kws}")
    else:
        base, kws = None, {}
    for k, v in (("emb_dim", mc.emb_dim), ("n_layers", mc.n_layers),
                 ("n_heads", mc.n_heads), ("context_length", mc.context_length)):
        kws.setdefault(k, v)
    kws["vocab_size"] = len(tokenizer)
    gpt = GPTConfig(**kws)
    model = MiniGPT(gpt)
    if base is not None:
        try:
            model.load_state_dict(base["model_state"])
        except RuntimeError as exc:
            raise SystemExit(
                f"加载 {args.pretrain} 权重失败：{exc}\n"
                f"推断结构 {kws}；请确认 --data_tokenizer_dir 与 checkpoint 词表一致。"
            ) from exc
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
    if len(train_set) == 0 or len(eval_set) == 0:
        raise RuntimeError("SFT 数据过少（train/eval 为空），请调大 --data_max_lines 或换更大的数据文件")
    collator = create_batch_collator(tokenizer)

    train_args = {
        "train_batch_size": tc.batch_size,
        "eval_strategy": "step",
        "eval_steps": tc.eval_steps,
        "warmup_steps": tc.warmup_steps,
        "save_strategy": "step",
        "save_steps": tc.save_steps,
        "save_best": tc.save_best,
        "reset_step": tc.reset_step,
        "extra_steps": tc.extra_steps,
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
            from minigpt.train.metrics import MetricsLogger
            writer = SummaryWriter(os.path.join(pc.output_dir, "tensorboard"))
            trainer.set_writer(writer)
            trainer.tokenizer = tokenizer
            trainer.metrics = MetricsLogger(
                writer, model, tokenizer=tokenizer, device=device,
                log_hist_every=tc.log_hist_every, log_hist_max_numel=tc.log_hist_max_numel,
                log_embedding_every=tc.log_embedding_every,
                projector_max_tokens=tc.projector_max_tokens,
                log_attention_every=tc.log_attention_every, log_graph=tc.log_graph,
                log_samples_every=tc.log_samples_every,
                sample_max_new_tokens=tc.sample_max_new_tokens,
                sample_prompts=[x for x in (tc.sample_prompts or "").split("|") if x],
            )
            print(f"[sft] tensorboard metrics enabled")
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] tensorboard disabled: {exc}")
    trainer.set_seed(tc.seed)
    # 每个 checkpoint（含 best.pt / final.pt）都带模型 config，推理与评测脚本才能正确重建结构
    trainer.extra_ckpt = gpt.to_dict()
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
            # chat 模型的回合结束符是 <|im_end|>，未必等于 eos_token_id（qwen2 下不同）
            out = model.generate(ids, 100, resolve_stop_token_ids(tokenizer), use_kv_cache=False)
            resp = tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
            lines.append(f"用户: {text}\n模型: {resp}\n")
            print(f"用户: {text}\n模型: {resp}\n", flush=True)
        with open(os.path.join(pc.output_dir, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
