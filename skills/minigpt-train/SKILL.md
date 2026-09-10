---
name: minigpt-train
description: 在 ScratchLLMStepByStep 仓库中端到端训练/微调/评估 MiniGPT 的可执行 skill：环境体检、语料与分词器准备、预训练（单卡/多卡、断点续训、torch.compile）、SFT 指令微调、CoT 思考模式、推理采样与指标对比，并给出配置选择与排障指引，供 agent 无人值守执行。
whenToUse: 当用户要求训练/继续训练/优化 MiniGPT、准备训练语料或分词器、评测模型（loss/perplexity/思考模式准确率）、排查训练 OOM/速度/磁盘/续训问题，或需要把本仓库交给 agent 自动跑通训练链路时使用。
metadata:
  repo: ScratchLLMStepByStep
  entry: scripts/preflight.sh
  pipeline: scripts/pipeline.sh
---

# MiniGPT 训练 Skill

本 skill 让 agent **无需人工介入**即可在本仓库完成：环境体检 → 语料/分词器准备 → 预训练 → SFT → CoT（思考模式）→ 评测与采样报告。

> 硬性约定：任何**长任务**（训练/微调/评测大于 2 分钟）必须用 `setsid nohup ... &` 后台运行并写日志，禁止前台阻塞；
> 任何写操作前先看 `scripts/preflight.sh` 的体检结果；训练产物一律留在 `models/checkpoints/<run>/`，语料留在 `dataset/`。

## 0. 30 秒上手（agent 最短路径）

```bash
cd "$(git rev-parse --show-toplevel)"

# 1) 环境体检（只读，输出人类可读摘要 + models/checkpoints/preflight.json）
bash skills/minigpt-train/scripts/preflight.sh

# 2) 一键流水线（默认 check：只体检；smoke：约 5 分钟小模型冒烟；full：完整预训练+SFT+CoT+评测）
bash skills/minigpt-train/scripts/pipeline.sh --smoke     # 建议先冒烟
bash skills/minigpt-train/scripts/pipeline.sh --full      # 按 10 小时预算最大化训练
```

`pipeline.sh --full` 内部做的事情等价于：

```bash
# a. 数据（幂等：已存在则跳过）
python scripts/build_pretrain_bin.py build \
  --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
  --tokenizer-dir models/tokenizer_v3 \
  --out-bin dataset/bins/pretrain_v3_full.bin --max-lines 0

# b. 预训练（可断点续训 + 自愈守护 + 空间守护；torch.compile 提速 +66%）
setsid nohup bash scripts/train_pretrain_resilient.sh   > models/checkpoints/pretrain_v2_full.watchdog.log 2>&1 &
setsid nohup bash scripts/checkpoint_janitor.sh 60 2 120 > /tmp/janitor.log 2>&1 &

# c. 下游（final.pt 一出现就自动 SFT→CoT→评测）
setsid nohup bash scripts/wait_and_run_downstream.sh     > models/checkpoints/wait_downstream.log 2>&1 &

# d. 看板
setsid nohup python3 scripts/train_dashboard.py --serve --port 8099 --refresh 5 --grad-clip 1.0 > /tmp/dashboard.log 2>&1 &
```

## 1. 仓库关键路径

| 路径 | 说明 |
|---|---|
| `minigpt/config.py` | 全部超参与路径（Model/Data/Train/Path），支持扁平 CLI 覆盖：`--model_emb_dim 512 --train_batch_size 8 --paths_output_dir ...` |
| `minigpt/model/transformer.py` | `GPTConfig / TransformerBlock / MiniGPT`（含采样解码：temperature/top-k/top-p/repetition_penalty） |
| `minigpt/model/generation.py` | 带停止序列的生成 + 思考模式（`strategy=single|two-phase`） |
| `minigpt/data/pretrain_dataset.py` | `tokenize_jsonl_to_bin`（uint16/uint32 自适应 + `.meta.json`）、`TokenBinDataset` |
| `minigpt/data/sft_dataset.py` | 指令数据 `InstructionDataset / collate / split_dataset` |
| `minigpt/train/{trainer,pretrainer,sft_trainer,metrics}.py` | 训练器（AMP/梯度累积/lr 调度/eval/save/resume/DDP/compile）、预训练与 SFT 入口、TensorBoard 指标 |
| `scripts/build_pretrain_bin.py` | 语料 → `.bin`（`build` / `info` 子命令） |
| `scripts/generate.py` | 推理 CLI（`--chat --thinking --thinking-strategy single`） |
| `scripts/evaluate_pretrain.py` | 验证集 loss / perplexity（`--max-rows` 控制耗时） |
| `scripts/eval_thinking.py` | 思考模式对比评测（plain/single/two-phase，`--repetition-penalty 1.0`） |
| `scripts/build_cot_sft.py` | CoT 数据（`--profile easy|hard`，自带可验证答案） |
| `scripts/train_dashboard.py` | 看板（网页 8099 / 终端 `--once` / PNG `--plot`） |
| `scripts/train_pretrain_resilient.sh` | 自愈长训守护（崩溃自动从最新 checkpoint 续训） |
| `scripts/checkpoint_janitor.sh` | 每目录只保留最近 N 个 checkpoint，防止磁盘写满 |
| `scripts/run_downstream.sh`、`wait_and_run_downstream.sh` | SFT→CoT→评测自动接力 |
| `tests/` | `pytest tests/ -q`（当前 24 passed, 1 skipped） |
| `minigpt/README.md` | 工程细节：指标面板、吞吐、看板、思考模式、实测结果 |

## 2. 标准训练配方（本机实测：RTX2060 6GB / 6 核 / 15GB）

| 配置 | 词表 | 单步 | 吞吐 | 适用 |
|---|---|---|---|---|
| `512/10/8, ctx512, bs8, fp16, compile` | 32k | 0.14–0.18s | **22–30k tok/s** | 本仓库默认（推荐） |
| 同上但 `--train_torch_compile False` | 32k | 0.23s | 17.9k tok/s | 调试/排障 |
| `384/8/8` + Qwen2.5 151k 词表 | 151k | ~2.0s | ~2.0k tok/s | 6GB 卡上不可行（一 epoch >10h），仅大显存使用 |

预训练（可直接复制）：

```bash
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.pretrainer \
  --data_tokenizer_dir models/tokenizer_v3 \
  --data_tokenized_bin dataset/bins/pretrain_v3_full.bin \
  --data_eval_ratio 0.001 \
  --model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 \
  --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 200 \
  --train_eval_steps 2000 --train_save_steps 4000 --train_max_steps 211000 \
  --train_epochs 7 --train_torch_compile True \
  --paths_last_checkpoint_path models/checkpoints/pretrain_v2_full/checkpoint-108000.pth \
  --paths_output_dir models/checkpoints/pretrain_v2_full \
  > models/checkpoints/pretrain_v2_full.log 2>&1 < /dev/null &
```

多卡：`NPROC=<卡数> bash scripts/pretrain_start.sh --paths_output_dir models/checkpoints/pretrain_ddp`
（DDP 由 `Trainer` 自动处理；batch/梯度累积按显存线性缩放：`--train_batch_size`、`--train_grad_accumulation_steps`）。

## 3. 配置选择决策树（agent 自行决策）

1. **时间预算 T 小时 → 步数**：`可用步数 ≈ T*3600 / 实测s_per_step`；再按 `tokens = 步数*bs*(ctx-1)` 判断是否达到 ~10×参数量（Chinchilla 量级）。
2. **显存不足（OOM）**：依次降 `--train_batch_size`（8→4→2）→ 升 `--train_grad_accumulation_steps` 保持有效 batch → 降 `--model_context_length`（512→384）→ 降 `--model_emb_dim`（512→384）。
3. **太慢**：确认 `--train_torch_compile True`（固定形状才有效）；`--train_eval_steps` 调大（如 2000）；暂停不必要的指标（`--train_log_hist_every 0 --train_log_embedding_every 0 --train_log_attention_every 0`）。
4. **需要更强基座**：优先**加数据/加 epoch**（当前 2.475 亿 tokens 全量语料）；再考虑 `--model_emb_dim 768 --model_n_layers 12 --model_n_heads 12`（约 120M，需梯度累积，仍可单卡）。
5. **上游词表切换**：只需改 `--data_tokenizer_dir`，模型 `vocab_size` 自动跟随；**注意词表 >65535 时 .bin 必须是 uint32**（管线自动判断并写 meta）。
6. **思考模式**：先在 `easy` 档验证（`--profile easy`，30 分钟级可达 90% 准确率），再上 `hard`（多位数，容量受限，需更大模型或工具调用范式）。

## 4. 训练监控（agent 必须主动汇报）

```bash
python3 scripts/train_dashboard.py --once                 # 终端快照（进度/损失/速率/ETA/GPU）
python3 scripts/train_dashboard.py --serve --port 8099    # 网页看板（5s 刷新，含 loss/grad 走势图）
python3 scripts/train_dashboard.py --plot models/checkpoints/loss_curve.png   # 导出走势图
tensorboard --logdir models/checkpoints/pretrain_v2_full/tensorboard --port 6006  # 直方图/图像/模型图/嵌入投影
tail -f models/checkpoints/pretrain_v2_full.log           # 评估点日志
```

判断健康：`eval_loss` 持续下降；`grad_norm` 在裁剪阈值（`--train_grad_clip 1.0`，看板红虚线）附近震荡而非持续升高；
GPU 利用率 >75%；`checkpoint-*.pth` 按 `--train_save_steps` 周期产出。

## 5. 下游：SFT 与思考模式（CoT）

```bash
# 5.1 指令 SFT（chat）
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.sft_trainer \
  --pretrain models/checkpoints/pretrain_v2_full/final.pt --data_tokenizer_dir models/tokenizer_v3 \
  --sft-jsonl dataset/sft/sft_data_zh.jsonl --data_max_lines 60000 --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1e-5 --train_epochs 2 \
  --paths_output_dir models/checkpoints/sft_v2_chat > models/checkpoints/sft_v2_chat.log 2>&1 &

# 5.2 CoT 数据 + 训练（easy 验证机制；hard 挑战容量）
python scripts/build_cot_sft.py --profile easy --n-train 60000 --n-eval 200 \
  --out-train dataset/sft/sft_cot_easy_60k.jsonl --out-eval dataset/sft/cot_eval_easy_zh.jsonl
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.sft_trainer \
  --pretrain models/checkpoints/sft_v2_chat/final.pt --data_tokenizer_dir models/tokenizer_v3 \
  --sft-jsonl dataset/sft/sft_cot_easy_60k.jsonl --data_max_lines 60000 --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1.5e-5 --train_epochs 2 \
  --paths_output_dir models/checkpoints/sft_v2_cot_easy > models/checkpoints/sft_v2_cot_easy.log 2>&1 &
```

## 6. 推理与评测（产出对比报告）

```bash
# 采样（普通 / 思考模式 / 隐藏思考过程）
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 --chat \
  --prompt "什么是AI？" --max-new-tokens 120 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 --chat --thinking \
  --thinking-strategy single --prompt "请计算 27 ÷ 3 等于多少？" [--hide-thinking]

# 验证集 loss / perplexity（务必带 --max-rows，否则会扫全量窗口）
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 \
  --bin dataset/bins/pretrain_v3_full.bin --max-rows 512 --output models/checkpoints/ppl.json

# 思考模式准确率（贪心，务必 --repetition-penalty 1.0）
python scripts/eval_thinking.py --checkpoint <ckpt> --eval-jsonl dataset/sft/cot_eval_easy_zh.jsonl \
  --n 60 --strategies plain,single,two-phase --repetition-penalty 1.0 --output models/checkpoints/eval_thinking.json
```

汇报模板（agent 完成时输出）：
```
预训练：step X/Y, eval_loss A → B, ppl ..., 产物 models/checkpoints/<run>/final.pt
SFT：loss ..., 产物 ...
思考模式：easy 准确率 plain/single/two-phase = x%/y%/z%（对照基座 a%）
样例：<3 条输入/输出>
```

## 7. 已知坑（详见 references/troubleshooting.md）

- 词表 >65535 必须 uint32（否则 `OverflowError`）；`TokenBinDataset` 依赖 `.meta.json`。
- 续训必须带 `--paths_last_checkpoint_path`；RNG 状态在加载时会被 `.cpu()` 修正（否则 `set_rng_state` 报错）。
- `torch.compile` 只对固定形状（预训练）收益大；SFT 变长序列可能反复重编译。
- 算术评测 `--repetition-penalty` 必须为 1.0，否则重复数字被惩罚，准确率虚低。
- 磁盘：长训练必须同时跑 `scripts/checkpoint_janitor.sh`；WSL 下 C 盘易被 vhdx 撑满。
- `nohup` 起的长任务要用 `setsid` 脱离工具进程组，否则可能被误杀；用 `pgrep -af` 校验存活。

## 8. 参考

- `references/training-recipes.md`：超参、吞吐、规模与预算换算、多卡配置。
- `references/troubleshooting.md`：OOM/速度/续训/数据/显存/磁盘/指标异常处置。
- `minigpt/README.md`：指标面板、看板、思考模式与实测结果。
