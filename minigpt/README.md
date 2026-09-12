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

## 1. 模型架构

decoder-only Transformer，**pre-norm + RoPE + 输入输出嵌入共享**（GPT-2 与 LLaMA 的混合体）。
默认 `512/10/8 + ctx512 + 32k 词表 = 47.89M 参数`，实现在 `model/attention.py` + `model/transformer.py`。

```
tokens (B, S)
  │
  └─ token_emb ─ dropout ─┐
                          │   pos_cis[position_ids]  ← RoPE 旋转因子（绝对位置）
       ┌──────────────────┴─────────────────────────────────┐
       │  × n_layers (10)                                   │
       │    x = x + Wo(Attn(Norm1(x)))  ← 因果注意力 + 残差  │
       │    x = x + FFN(Norm2(x))       ← GELU 4d / SwiGLU  │
       └──────────────────┬─────────────────────────────────┘
  ┌───────────────────────┘
  └─ final_norm ─ out_head ─ logits (B, S, V=32000)
       └─ out_head 与 token_emb 共享同一份权重（tie_word_embeddings=True）
```

### 1.1 参数量分布（默认 512/10/8，实测）

| 组件 | 形状 / 取值 | 参数量 | 占比 |
|---|---|---|---|
| `token_emb` | 32000 × 512 | 16,384,000 | **34.2%** |
| `decode_layers` × 10 | 每层 3,150,848 | 31,508,480 | 65.8% |
| ↳ `atten`（Wq/Wk/Wv/Wo + Wo.bias） | 4×512² + 512 | 1,049,088 | 2.2% |
| ↳ `ffn`（GELU，中间维 2048） | 2×512·2048 + 2048 + 512 | 2,099,712 | 4.4% |
| ↳ `layernorm1/2`（scale + shift） | 2 × 2×512 | 2,048 | ~0% |
| `final_norm` | 2×512 | 1,024 | ~0% |
| `out_head` | 与 `token_emb` 共享存储 | 0（复用） | — |
| **合计** | | **47,893,504（47.89M）** | |

`d_model=512, n_heads=8 → head_dim=64`（RoPE 按相邻两维配对，故 head_dim 必须为偶数）；
`drop_rate` 默认 **0.0**（预训练与 SFT 预设都不开，需要正则化时再显式调大）。
参数量估算公式只有一份实现：`minigpt.config.estimate_params`。

### 1.2 结构细节（与主流实现的对齐点）

| 位置 | 做法 | 说明 |
|---|---|---|
| 归一化 | **pre-norm** `x + 子层(Norm(x))`，末端 `final_norm` | 默认 LayerNorm；`norm_type=rmsnorm` 可切 |
| 位置编码 | **RoPE**（`rope_theta=10000`）只作用于 Q/K，按**绝对位置**取 | 增量推理只取当前 token 的位置，不重置 |
| 因果掩码 | 上三角 `-inf` 的加性掩码直接加到得分上 | 判据是"**本次前向有没有 past_kv**"：prefill 也必须加因果掩码，否则整段 prompt 被双向处理并污染 KV cache |
| KV cache | 缓存沿 seq 维 `cat`，增量步只送最新 1 个 token | 与全量前向数值一致（实测 max\|diff\| = 9e-8） |
| padding 掩码 | `(mask==0) × finfo.min` 加到得分；长度必须 = 已缓存 + 本步 | 无 cache 时输入窗口与掩码**都取尾部** |
| 前馈 | 默认 GELU + 4d（2048）；`use_swiglu` 时 8/3·d 对齐 64 | SwiGLU 总参数与 GELU 基本持平 |
| bias | QKV 无 bias；`Wo` 与 GELU-FFN **有** bias；`out_head` 无 bias | |
| 初始化 | `N(0, 0.02)`；残差出口（`Wo`、FFN 末投影）再乘 `1/sqrt(2·n_layers)` | 修好之前 `nn.Embedding` 落在 N(0,1)，首步 loss 从应有的 ~10.4 变成 ~397 |
| 采样 | temperature → top-k → top-p → 重复惩罚；`eos_token_id` 可传**列表** | chat 模型需同时接受 `<|im_end|>` 与 `<|endoftext|>` |

### 1.3 架构开关（改结构即与旧 checkpoint 不兼容；checkpoint 自带 `config`，加载时自动对齐）

| 开关 | 默认 | 打开后 |
|---|---|---|
| `tie_word_embeddings` | **True** | 输入/输出共享权重；32k 词表下省 16.4M 参数（34%） |
| `use_swiglu` | False | SwiGLU 门控，中间维 8/3·d 对齐 64（LLaMA/Mistral/Qwen 主流） |
| `norm_type` | `layernorm` | `rmsnorm`：省均值项与 shift，参数更少 |
| `qkv_merged` | False | 单次矩阵乘产出 3·d |
| `ffn_hidden_dim` | `0`（自动） | 手动指定前馈中间维 |
| `lm_head_bias` | False | 输出头加 bias |
| `rope_theta` | `10000` | 换 RoPE 基频（**注意：只有基频，没有 rope_scaling**） |
| `flash_attn` | False | FlashAttention-2，**仅 Ampere(sm_80)+**；RTX 20 系（Turing）必须 False |
| `use_checkpoint` | False | FFN 激活重计算（注意力部分未覆盖） |

### 1.4 已知取舍（改架构时的候选方向）

- **无 GQA/MQA**：KV cache 与头数等量，长上下文推理显存偏大。
- **注意力未走 SDPA**：`q @ k.T` + softmax 会物化 (B,H,S,S) 得分矩阵——b8/h8/ctx512/fp16 单层 ≈ 33.5MB，
  10 层 ≈ 335MB；ctx1024 预设下单层 ≈ 200MB。数值等价的 `F.scaled_dot_product_attention` 替换已验证
  （训练/padding/KV-cache 三条路径 max\|diff\| ≈ 5e-7），尚未切换。
- **RoPE 用复数实现**（`view_as_complex`）：torchinductor 不为复数算子生成代码，`torch.compile` 下这段会回退。
- **前向里有一次 D2H 同步**：`position_ids.max().item()` 造成 1 个 graph break（每步一次）。

## 2. 快速开始

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
    --tokenizer-dir models/tokenizer_v3 --bin dataset/bins/pretrain_v4_full.bin --split val
```

## 3. 配置

| dataclass | 关键字段 |
|---|---|
| `ModelConfig` | `emb_dim / n_layers / n_heads / context_length / vocab_size(0=由 tokenizer 推导) / drop_rate / qkv_bias / qkv_merged / flash_attn / tie_word_embeddings / use_swiglu / use_checkpoint / norm_type / ffn_hidden_dim / lm_head_bias / rope_theta` |
| `DataConfig` | `tokenizer_dir / corpus_jsonl / content_key / max_lines / tokenized_bin / bin_meta / eval_ratio / eval_blocks` |
| `TrainConfig` | `epochs / learning_rate / batch_size / weight_decay / grad_accumulation_steps / max_steps / reset_step / extra_steps / warmup_steps / eval_steps / save_steps / save_best / grad_clip / mixed_precision_dtype / seed / num_workers / torch_compile / ddp_timeout_seconds / deterministic_cudnn` + 指标项（§5） |
| `PathConfig` | `output_dir / last_checkpoint_path / sft_dataset` |

扁平覆盖：`--model_emb_dim 512 --train_batch_size 8 --paths_output_dir ...`；布尔必须带值（`True/False/1/0/yes/no`）。
运行结束把完整配置写入 `output_dir/config.json`，可用 `--config-file <该文件>` 整份读回（CLI 优先级更高）。

## 4. 训练产物

```
output_dir/
├── config.json             # 本次运行完整配置
├── checkpoint-{step}.pth   # 周期存档：model/optimizer/scaler/RNG/config
├── best.pt                 # eval_loss 历史最优（--train_save_best False 可关；下游/评测优先用它）
├── final.pt                # 最终存档（同样含 config，可直接推理）
├── tensorboard/            # 指标事件（§5）
├── metrics.json            # 最终指标 + data_split（评估口径）+ bin_meta
└── sample.txt              # 训练后采样
```

续训：`--paths_last_checkpoint_path output_dir/checkpoint-2000.pth` —— 自动恢复优化器、混合精度缩放器与随机数状态。
数据顺序由独立 generator 按 `seed+epoch` 播种（与全局 RNG 解耦），因此续训跳过的正是已训过的同一批样本。

> 小模型在 SFT/CoT 后期几乎必然过拟合（实测 CoT easy：`best_eval_loss=1.611` vs `final=1.740`），
> 所以下游训练/评测/发布**优先用 `best.pt`**。每个存档含优化器+RNG（单文件 ~585MB），用
> `scripts/checkpoint_janitor.sh` 控盘；写入是原子的（tmp→fsync→rename），中断不会破坏上一个可用存档。

## 5. 训练过程指标（TensorBoard）

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

## 6. 推理与评估

```bash
# 采样参数：do_sample / temperature / top_k / top_p / repetition_penalty / seed
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 \
    --prompt "..." [--prompt-file f.txt | --interactive] [--chat] \
    --max-new-tokens 128 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 \
    --repeat-penalty 1.2 --seed 0 [--output-file out.txt]
```
- 加 `--chat` 会套 chat 模板（`<|im_start|>user…<|im_start|>assistant`）；不加则是文本续写。
- 特殊 token 默认隐藏：`<|im_end|>` 是 tokenizer_v3 的 eos(id=2)，模型输出它即**提前停止**（回合正常收尾）；`--keep-special-tokens` 看原始序列。
- 加载**没有 `config` 字段**的老 checkpoint（如早期 `best.pt`）时，结构按权重形状反推；
  其中 `n_heads` 无法反推（Q/K/V/O 都是 emb×emb，猜错不会报错、只会**静默**改变输出），
  因此会打印 UserWarning。此时用 `--n-heads N` 显式确认（`generate.py` / `evaluate_pretrain.py` /
  `eval_thinking.py` / `chat_probe.py` / `sft_trainer.py` 均支持）；当前训练入口写出的 checkpoint
  一律自带 `config`，不受影响。

```bash
# 验证集 loss / perplexity
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --bin <bin> \
    --tokenizer-dir models/tokenizer_v3 --batch-size 8 --split val --output metrics_eval.json
```
> **口径要点**（`--split val` 的定义、为什么禁止 `--split all --max-rows`、实测差异表）见
> `../skills/minigpt-train/references/training-recipes.md §8`（唯一详述处）。

## 7. 思考模式（CoT）

"思考模式"不是模型里的开关，而是**数据 + 损失掩码 + 推理协议**三件套。协议（`model/generation.py`）：

```
<|im_start|>assistant
让我逐步分析：
1. …
最终答案：<答案><|im_end|>
```

四步命令（造数据 → 训练 → 推理 → 评测）见 **`../skills/minigpt-train/SKILL.md §5`**（唯一详述处）。

Python 侧调用 `minigpt.model.generation.generate_with_thinking(...)`，返回 `thinking / answer / display / full / marker_hit`。

**实测结论**（数字与结果表见根 `README.md`）：思考模式本身有效，但 **hard 是容量墙**——47.9M 参数在多位数
乘加上算不对，加数据无用，要么放大模型、要么改成"模型只出算式、Python 结算"的工具调用范式。
⚠️ easy 档的"100%"曾是**记忆**（题面空间仅 1,063 个，旧留出集 100% 落在训练集内）；
现在用 `--easy-max` 扩大题面空间 + `*_disjoint.jsonl` 零重合验收，规则见
`../skills/minigpt-train/references/data-distribution.md §9.1`。

## 8. 训练看板与续训

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

## 9. 测试与环境

```bash
pytest tests/             # 214 passed + 1 skipped，CPU 即可运行，不需要数据与 GPU
```
- RTX 20 系（Turing）不支持 FlashAttention-2：保持 `--model_flash_attn False`（默认）。
- 换 tokenizer 只改 `--data_tokenizer_dir`（词表自动跟随，并与 `.bin` 的 `meta.vocab_size` 做强校验）；
  本仓库默认 32k 中文 BPE `models/tokenizer_v3`——大词表（如 151k）在 6GB 卡上吞吐仅 ~2k tok/s。
- `torch.compile` 在**固定序列长度**下收益最大（预训练 +66%）；SFT 变长 padding 会反复重编译，故默认不启用。
