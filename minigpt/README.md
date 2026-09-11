# minigpt 包使用说明

模型、数据管线、训练器、指标记录、推理与评估的实现都在这里。所有入口共享 `minigpt/config.py` 的集中配置，
支持扁平 CLI 覆盖；产物带配置快照，可续训、可复现。项目级信息（notebook 章节、数据集下载、模型规模估算、实测结果）
见 [根 README](../README.md)。

```
minigpt/
├── config.py               # RunConfig（model/data/train/paths）+ CLI 覆盖 + 布尔解析
├── model/
│   ├── attention.py        # 多头注意力 / FlashAttention 版（可选捕获注意力权重）
│   ├── transformer.py      # LayerNorm/RMSNorm / FFN / GPTConfig / MiniGPT（含 _init_weights 与采样解码）
│   ├── checkpoint.py       # 重建模型（config 优先→形状反推）+ 宽容权重加载器
│   └── generation.py       # 停止串 / 思考模式（两阶段 CoT）生成工具
├── data/
│   ├── pretrain_dataset.py # jsonl→.bin(+meta) / TokenBinDataset(memmap) / 分块验证集切分 / 词表校验
│   └── sft_dataset.py      # InstructionDataset / collate（逐轮 assistant 掩码 + 停止符解析）
└── train/
    ├── trainer.py          # 训练循环：AMP / 梯度累积 / eval / save / resume / DDP / compile 编排
    ├── schedule.py         # LR 调度（warmup+cosine）与 loss 记账（纯函数）
    ├── ddp_utils.py        # DDP 包装/解包/跨 rank 平均
    ├── checkpoint_io.py    # checkpoint 原子读写 + RNG/缩放器恢复
    ├── optim.py            # 优化器构造（weight decay 分组）
    ├── metrics.py          # TensorBoard 指标
    ├── pretrainer.py       # 预训练入口（支持 DDP）
    ├── sft_trainer.py      # SFT 入口
    └── pretrainer_single.py # 单卡教学版训练器（notebook 09/10 使用）
```

## 1. 快速开始

```bash
python3 -m minigpt.train.pretrainer --paths_output_dir models/checkpoints/pretrain_v3 \
    --train_batch_size 8 --train_torch_compile True --train_max_steps 200000
NPROC=2 bash scripts/pretrain_start.sh --paths_output_dir models/checkpoints/pretrain_ddp   # 多卡

python3 -m minigpt.train.sft_trainer \
    --pretrain models/checkpoints/pretrain_v3/final.pt \
    --sft-jsonl dataset/sft/sft_data_zh.jsonl --data_max_lines 60000 \
    --paths_output_dir models/checkpoints/sft_v3_chat

python scripts/generate.py --checkpoint models/checkpoints/sft_v3_chat/final.pt \
    --tokenizer-dir models/tokenizer_v3 --chat --prompt "什么是AI？"
python scripts/evaluate_pretrain.py --checkpoint models/checkpoints/pretrain_v3/final.pt \
    --tokenizer-dir models/tokenizer_v3 --bin dataset/bins/pretrain_v3_full.bin --split val
```

## 2. 配置

| dataclass | 关键字段 |
|---|---|
| `ModelConfig` | `emb_dim / n_layers / n_heads / context_length / vocab_size(0=由 tokenizer 推导) / drop_rate / qkv_bias / qkv_merged / flash_attn / tie_word_embeddings / use_swiglu / use_checkpoint / norm_type / ffn_hidden_dim / lm_head_bias / rope_theta` |
| `DataConfig` | `tokenizer_dir / corpus_jsonl / content_key / max_lines / tokenized_bin / bin_meta / eval_ratio / eval_blocks` |
| `TrainConfig` | `epochs / learning_rate / batch_size / weight_decay / grad_accumulation_steps / max_steps / reset_step / extra_steps / warmup_steps / eval_steps / save_steps / save_best / grad_clip / mixed_precision_dtype / seed / num_workers / torch_compile / ddp_timeout_seconds / deterministic_cudnn` + 指标项（§4） |
| `PathConfig` | `output_dir / last_checkpoint_path / sft_dataset` |

扁平覆盖：`--model_emb_dim 512 --train_batch_size 8 --paths_output_dir ...`；布尔必须带值（`True/False/1/0/yes/no`）。
运行结束把完整配置写入 `output_dir/config.json`，可用 `--config-file <该文件>` 整份读回（CLI 优先级更高）。

## 3. 训练产物

```
output_dir/
├── config.json             # 本次运行完整配置
├── checkpoint-{step}.pth   # 周期存档：model/optimizer/scaler/RNG/config
├── best.pt                 # eval_loss 历史最优（--train_save_best False 可关；下游/评测优先用它）
├── final.pt                # 最终存档（同样含 config，可直接推理）
├── tensorboard/            # 指标事件（§4）
├── metrics.json            # 最终指标 + data_split（评估口径）+ bin_meta
└── sample.txt              # 训练后采样
```

续训：`--paths_last_checkpoint_path output_dir/checkpoint-2000.pth` —— 自动恢复优化器、混合精度缩放器与随机数状态。
数据顺序由独立 generator 按 `seed+epoch` 播种（与全局 RNG 解耦），因此续训跳过的正是已训过的同一批样本。

> 小模型在 SFT/CoT 后期几乎必然过拟合（实测 CoT easy：`best_eval_loss=1.611` vs `final=1.740`），
> 所以下游训练/评测/发布**优先用 `best.pt`**。每个存档含优化器+RNG（单文件 ~585MB），用
> `scripts/checkpoint_janitor.sh` 控盘；写入是原子的（tmp→fsync→rename），中断不会破坏上一个可用存档。

## 4. 训练过程指标（TensorBoard）

```bash
tensorboard --logdir models/checkpoints/pretrain_v3/tensorboard --port 6006   # http://localhost:6006
```

| 面板 | 标签 | 时机 / 默认间隔 |
|---|---|---|
| 标量 | `train/loss`、`train/lr`、`train/grad_norm`、`train/tokens_per_sec` | 每个更新步 |
| | `eval/loss`、`eval/perplexity`、`train/loss_eval_window`、`eval/final_loss` | 每 `eval_steps` / 训练结束 |
| 直方图 | `weights/*`、`grads/*`、`buffers/*` | `log_hist_every`（默认 500，`0` 关） |
| 图像 | `attention/layer0_head0`、`curves/loss` | `log_attention_every`（默认 1000，`0` 关） |
| 模型图 | GRAPH 面板 | 首步一次（`--train_log_graph True`） |
| 嵌入投影 | `token_embedding` + `/metadata` | `log_embedding_every`（默认 2000，`0` 关） |
| 文本 | `samples/generation/text_summary` | `log_samples_every`（`0`=仅结束时） |

相关配置（CLI 前缀 `--train_`）：`log_hist_every / log_hist_max_numel(超大张量跳过) / log_embedding_every /
projector_max_tokens / log_attention_every / log_graph / log_samples_every / sample_max_new_tokens / sample_prompts`。

- 实现：`MetricsLogger` 由两个入口在主进程创建并挂到 `Trainer.metrics`，回调 `on_train_step / on_eval / on_final`。
- 直方图钩子在 `zero_grad` **之前**触发（否则 `grads/*` 恒为空）；嵌入投影按 projector 插件规范直接写 `TensorProto`
  （绕开 torch 2.13 + tensorboard 2.21 下 `add_embedding` 静默失效）。
- 默认间隔下对吞吐影响 < 2%；追极致吞吐把 `log_hist_every` 调大或设 0。

## 5. 推理与评估

```bash
# 采样参数：do_sample / temperature / top_k / top_p / repetition_penalty / seed
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 \
    --prompt "..." [--prompt-file f.txt | --interactive] [--chat] \
    --max-new-tokens 128 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 \
    --repeat-penalty 1.2 --seed 0 [--output-file out.txt]
```
- 加 `--chat` 会套 chat 模板（`<|im_start|>user…<|im_start|>assistant`）；不加则是文本续写。
- 特殊 token 默认隐藏：`<|im_end|>` 是 tokenizer_v3 的 eos(id=2)，模型输出它即**提前停止**（回合正常收尾）；`--keep-special-tokens` 看原始序列。

```bash
# 验证集 loss / perplexity
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --bin <bin> \
    --tokenizer-dir models/tokenizer_v3 --batch-size 8 --split val --output metrics_eval.json
```
> **口径要点**：`--split val`（默认）= 均匀分块 + 双端对齐文档边界的留出集，与训练时 `split_train_eval_blocks`
> 同一口径，切分范围记录在输出 `data_split` 里，可跨 checkpoint 比较。
> ⚠️ `--split all --max-rows N` 取的是 `.bin` **最前面**的窗口：对参与过训练的 bin 而言那是训练集 loss，
> 且本语料按领域排序，头/尾 ppl 实测可差 3 倍以上，不同 N 之间也不可比。

## 6. 思考模式（CoT）

"思考模式"不是模型里的开关，而是**数据 + 损失掩码 + 推理协议**三件套。协议（`model/generation.py`）：

```
<|im_start|>assistant
让我逐步分析：
1. …
最终答案：<答案><|im_end|>
```

```bash
# 1) 造数据（自带 --mix-general 混通用指令；保证 train/eval 题面零重合）
python scripts/build_cot_sft.py --profile easy --easy-max 20 \
    --out-train dataset/sft/sft_cot_easy_disjoint60k.jsonl \
    --out-eval  dataset/sft/cot_eval_easy_disjoint.jsonl --n-train 60000 --n-eval 200

# 2) 训练（只对 assistant 段算 loss，逐轮掩码）
python3 -m minigpt.train.sft_trainer --pretrain models/checkpoints/pretrain_v3/final.pt \
    --sft-jsonl dataset/sft/sft_cot_easy_disjoint60k.jsonl --data_max_lines 60000 \
    --train_learning_rate 2e-5 --train_epochs 2 --paths_output_dir models/checkpoints/sft_cot_easy

# 3) 推理（--thinking；single=一次生成"思考+答案"最稳，two-phase=先思考再作答）
python3 scripts/generate.py --checkpoint models/checkpoints/sft_cot_easy/final.pt \
    --tokenizer-dir models/tokenizer_v3 --chat --thinking --thinking-strategy single \
    --prompt "请计算 27 ÷ 3 等于多少？" [--hide-thinking] [--thinking-max-tokens 120]

# 4) 评测（留出集 + 贪心；算术评测必须 --repetition-penalty 1.0）
python3 scripts/eval_thinking.py --checkpoint models/checkpoints/sft_cot_easy/final.pt \
    --eval-jsonl dataset/sft/cot_eval_easy_disjoint.jsonl --n 60 \
    --strategies plain,single,two-phase --repetition-penalty 1.0 --output result.json
```

Python 侧调用 `minigpt.model.generation.generate_with_thinking(...)`，返回 `thinking / answer / display / full / marker_hit`。

**实测结论**（RTX2060，60 题，贪心）：思考模式确实有效（easy 0% → 90% → 100%），但 **hard 是容量墙**——
47.9M 参数在多位数乘加上算不对，需放大模型或走「工具调用范式」。
⚠️ **easy 的 100% 是记忆而非泛化**：easy 题目空间只有 1063 个唯一题目，旧留出集 156 条 100% 落在训练集内
（hard 33.5%）。现在 `--easy-max` 可扩大题目空间，并用 `*_disjoint.jsonl` 做零重合验收——**该验收结果尚未产出**。

## 7. 训练看板与续训

```bash
python scripts/train_dashboard.py --serve --port 8099 --refresh 5   # 浏览器（含 grad_norm 红虚线）
python scripts/train_dashboard.py --once                            # 终端一次性
python scripts/train_dashboard.py --plot out.png                    # 导出损失走势图
```
与 TensorBoard 的分工：看板面向"长训实时盯盘"（进度/ETA/吞吐/是否卡住），TensorBoard 面向细粒度诊断（直方图/注意力/投影）。

续训要点：
- `--paths_last_checkpoint_path <ckpt>` 恢复 model/optimizer/scaler/RNG/step/best 记录；
  增量领域续训必须 `--train_reset_step True`（否则"从 211k 步基座再训 N 步"会被当成绝对步数而直接判定训完）。
- 长训用 `scripts/train_pretrain_resilient.sh`（崩溃自动续跑）+ `scripts/checkpoint_janitor.sh`（空间守护）。

## 8. 测试与环境

```bash
pytest -q                 # 189 项，CPU 即可运行，不需要数据与 GPU
```
- RTX 20 系（Turing）不支持 FlashAttention-2：保持 `--model_flash_attn False`（默认）。
- 换 tokenizer 只改 `--data_tokenizer_dir`（词表自动跟随，并与 `.bin` 的 `meta.vocab_size` 做强校验）；
  本仓库默认 32k 中文 BPE `models/tokenizer_v3`——大词表（如 151k）在 6GB 卡上吞吐仅 ~2k tok/s。
- `torch.compile` 在**固定序列长度**下收益最大（预训练 +66%）；SFT 变长 padding 会反复重编译，故默认不启用。
