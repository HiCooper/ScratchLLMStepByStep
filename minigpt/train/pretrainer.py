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
from minigpt.data.pretrain_dataset import (TokenBinDataset, split_train_eval_blocks,
                                           validate_bin_tokenizer)
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.optim import build_optimizer, param_group_summary
from minigpt.train.train_args import as_train_args
from minigpt.train.trainer import Trainer


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
        qkv_merged=mc.qkv_merged,
        tie_word_embeddings=mc.tie_word_embeddings,
        use_swiglu=mc.use_swiglu, use_checkpoint=mc.use_checkpoint,
        flash_attn=mc.flash_attn,
        norm_type=mc.norm_type, ffn_hidden_dim=mc.ffn_hidden_dim,
        lm_head_bias=mc.lm_head_bias, rope_theta=mc.rope_theta,
    )
    model = MiniGPT(gpt_cfg)

    # 数据
    ds = TokenBinDataset(dc.tokenized_bin, mc.context_length, dc.bin_meta)
    # 词表一致性校验：meta.vocab_size 必须与 tokenizer 相符（否则 embedding 静默错配）
    bin_info = validate_bin_tokenizer(dc.tokenized_bin, tokenizer, dc.bin_meta)
    # 验证集 = 均匀分布的若干块，每块双端对齐文档边界：
    #   - 随机抽窗口会让同一文档前半进训练集、后半进验证集（文档中位仅 ~151 token，
    #     窗口 512 平均横跨 2.6 篇文档），eval_loss 偏乐观；
    #   - 只取尾部连续块则失去代表性（本语料按领域排序，尾部整段是英文选择题）。
    eos_id = (ds.meta or {}).get("eos_id")
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    train_set, eval_set, split_info = split_train_eval_blocks(
        ds, eval_ratio=dc.eval_ratio, eos_id=eos_id, n_blocks=dc.eval_blocks)
    if rank0:
        print(f"[pretrainer] ds_rows={len(ds)} train={len(train_set)} eval={len(eval_set)} "
              f"vocab={vocab_size} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"[pretrainer] split={split_info['split']} blocks={split_info['n_blocks']} "
              f"eval_rows={split_info['eval_rows']} actual_eval_ratio={split_info['actual_eval_ratio']} "
              f"bin_vocab={bin_info.get('bin_vocab')}")

    # weight decay 只作用于 >=2 维且非 bias 的参数（LayerNorm/RMSNorm 的 scale/shift 与
    # 所有 bias 显式 0 衰减），避免把归一化尺度往 0 拉
    optimizer = build_optimizer(model, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    if rank0:
        print(f"[pretrainer] optimizer_groups={param_group_summary(optimizer)}")
    # TrainConfig/PathConfig -> Trainer 的翻译只有一份实现（minigpt/train/train_args.py）：
    # 两个入口以前各自手写映射，已经漂移到"sft 侧漏传 torch_compile/num_workers 而 flag 照样出现在 --help"。
    train_args = as_train_args(tc, pc)
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
            print(f"[pretrainer] tensorboard metrics enabled "
                  f"(hist/{tc.log_hist_every} emb/{tc.log_embedding_every} "
                  f"attn/{tc.log_attention_every} graph/{tc.log_graph})")
        except Exception as exc:  # noqa: BLE001
            print(f"[pretrainer] tensorboard disabled: {exc}")
    trainer.set_seed(tc.seed)
    trainer.extra_ckpt = gpt_cfg.to_dict()   # 周期 checkpoint 与 final.pt 都带 config
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
        # 记录切分口径与语料元信息：不同切分/不同 bin 的 loss 不可直接比较
        metrics["data_split"] = split_info
        metrics["bin_meta"] = bin_info
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
