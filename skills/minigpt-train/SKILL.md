---
name: minigpt-train
description: 在 ScratchLLMStepByStep 仓库中端到端训练/微调/评估 MiniGPT 的可执行 skill：自动体检并按硬件（无 GPU/单卡小显存/中大型显存/多卡）选择模型与训练参数，完成语料与分词器准备、预训练（断点续训、torch.compile、DDP）、SFT、CoT 思考模式、推理采样与指标对比，并给出排障与汇报模板，供 agent 无人值守执行。
whenToUse: 当用户要求训练/继续训练/优化 MiniGPT、准备训练语料或分词器、评测模型（loss/perplexity/思考模式准确率）、排查 OOM/速度/续训/磁盘问题，或把本仓库交给 agent 自动跑训练链路时使用。
metadata:
  repo: ScratchLLMStepByStep
  entry: scripts/preflight.sh
  pipeline: scripts/pipeline.sh
---

# MiniGPT 训练 Skill

本 skill 让 agent **无需人工介入**完成：环境体检 → **按硬件自动选预设** → 语料/分词器准备 → 预训练 → SFT → CoT → 评测与采样报告。

> 硬性约定：长任务（>2 分钟）必须 `setsid nohup ... > <log> 2>&1 < /dev/null &`；改动代码后跑 `pytest tests/ -q`；产物只写 `models/`、`dataset/`（已 gitignore）。

## 0. 最短路径

```bash
cd "$(git rev-parse --show-toplevel)"

# 1) 环境体检 + 打印本机推荐预设（只读）
bash skills/minigpt-train/scripts/preflight.sh

# 2) 看将要执行的计划（不启动）
bash skills/minigpt-train/scripts/pipeline.sh --full --hours 10 --dry-run

# 3) 冒烟（无 GPU 也能跑）→ 完整训练
bash skills/minigpt-train/scripts/pipeline.sh --smoke
bash skills/minigpt-train/scripts/pipeline.sh --full --hours 10
```

`preflight.sh` 会在末尾给出 `[PRESET] <名字> …`，同时写入 `models/checkpoints/preflight.json` 的 `preset` 字段
（包含模型尺寸、batch、ctx、AMP、compile、nproc、预期吞吐、预估步长），`pipeline.sh` 直接消费它。

## 1. 环境自适应（**先判断硬件，再决定参数**）

| 预设 | 触发条件 | 模型 | batch×ctx | 精度 | compile | nproc | 实测/预估吞吐 |
|---|---|---|---|---|---|---|---|
| `cpu` | 无 CUDA GPU（含 MPS 回退） | 128/2/4 | 8×128 | fp32 | ✗ | 1 | **实测 0.12–0.17k tok/s**（6 核） |
| `gpu-tiny` | 单卡 <5.5GB | 384/8/8 | 4×384 | fp16 | ✓ | 1 | ~10–18k tok/s |
| `gpu-small` | 单卡 5.5–8GB | **512/10/8** | 8×512 | fp16 | ✓ | 1 | **22–30k tok/s**（RTX2060 实测） |
| `gpu-mid` | 单卡 8–16GB | 512/10/8 | 12×1024 | fp16 | ✓ | 1 | ~30–50k tok/s |
| `gpu-large` | 单卡 ≥16GB | 768/12/12 | 16×1024 | fp16 | ✓ | 1 | ~60k+ tok/s |
| `multi-gpu` | ≥2 张 GPU | 按单卡显存档 | 每卡 8–16 | fp16 | ✓ | N | ≈单卡×N×0.85（NCCL 开销） |

### 1.1 没有 GPU 怎么办（agent 必须这样处理）
1. **明确降级预期**：CPU 只能做「机制验证」，不要承诺可读模型。实测本机 6 核约 **120–170 tok/s**，
   1 小时仅 ~0.5M tokens（GPU 是它的 100 倍以上）。
2. 用 CPU 预设（自动检测即为 `cpu`，也可显式 `--preset cpu`）：
   - 模型 128/2/4、ctx 128、batch 8、`--train_mixed_precision_dtype none`、`--train_torch_compile False`；
   - 语料只取前 2 万行（`pipeline.sh --full --preset cpu` 会自动建 `dataset/bins/pretrain_v3_20k.bin`）；
   - 目标步数按预算换算（见 §1.4），建议 200–2000 步，先跑 `--smoke` 验证链路。
3. 可选优化：提高 CPU 线程（`OMP_NUM_THREADS=物理核数`）、把 `--train_num_workers 2` 用于数据加载；
   不要开 `torch.compile`（CPU 收益不稳定）。
4. 汇报时给出「CPU 环境、仅验证链路」的结论，并建议用户改用 CUDA 机器或云 GPU 继续。

### 1.2 只有一张小显存卡（≤6GB，本机情况）
- 用 `gpu-small` 预设；OOM 时按序：`batch 8→4`（同时 `--train_grad_accumulation_steps 2` 保持有效 batch）→ `ctx 512→384` → `emb 512→384`。
- 必须开 `--train_torch_compile True`（固定形状 +66% 吞吐）；`--train_eval_steps 2000`、`--train_save_steps 4000` 控制额外开销。
- 关键前提：**词表用 32k（`models/tokenizer_v3`）**。151k 词表（Qwen2.5）在本级显存只有 ~2k tok/s，不可用。

### 1.3 有多张 GPU 怎么办
```bash
# 自动选择 multi-gpu 预设（lr 自动 ×sqrt(N)）
bash skills/minigpt-train/scripts/pipeline.sh --full --hours 10
# 或显式指定卡数
NPROC=4 bash scripts/pretrain_start.sh --paths_output_dir models/checkpoints/pretrain_ddp
```
规则（agent 自行按此换算并写进汇报）：
- **每卡 batch 不变**，有效 batch = `batch × N × grad_accumulation_steps`；
- **学习率按 sqrt(N) 放大**（`pipeline.sh` 自动做；手动时 `lr = 6e-4 × √N`）；
- **目标步数按有效 batch 重算**：`steps ≈ 预算秒数 / 单步秒数`，其中单步 tokens = `batch×N×ctx`；
  同样的 tokens 预算下，N 卡时步数应缩小到 1/N 左右；
- 数据由 `DistributedSampler` 自动分片；日志/checkpoint/评测只在 rank0 产出（无需额外处理）；
- 启动必须用 `torchrun`（`train_pretrain_resilient.sh` 在 `NPROC>1` 时自动改用 torchrun）；
- 若 NCCL 报错：`export NCCL_DEBUG=WARN`、确认 `CUDA_VISIBLE_DEVICES` 与 `--nproc_per_node` 一致；
- `torch.compile` 与 DDP 可同时用（先 DDP 再 compile，代码已按此顺序）。

### 1.4 时间预算 → 目标步数（通用公式）
```
目标步数 = 预算秒数 / 单步耗时（preflight 给出 assumed_s_per_step，实测后可用真实值替换）
训练 tokens ≈ 目标步数 × batch × nproc × (ctx-1)
参考：达到 10–20×参数量 的 tokens 才算“训练充分”
```
`pipeline.sh --full --hours <H>` 会按预设的 `assumed_s_per_step` 自动算出并写入 `TARGET_STEPS`。

### 1.5 其它加速器
- **Apple MPS**：本仓库 AMP/device 逻辑只适配 CUDA/CPU，检测到 MPS 会提示并回退 `cpu` 预设。
- **XPU/ROCm**：同一原因，需先适配 `torch.amp` 与 device 选择，默认按 CPU 处理。

## 2. 仓库关键路径

| 路径 | 说明 |
|---|---|
| `minigpt/config.py` | 全部超参与路径（Model/Data/Train/Path），扁平 CLI 覆盖：`--model_emb_dim 512 --train_batch_size 8 --paths_output_dir ...` |
| `minigpt/model/transformer.py` | `GPTConfig / TransformerBlock / MiniGPT`（采样解码：temperature/top-k/top-p/repetition_penalty） |
| `minigpt/model/generation.py` | 带停止序列的生成 + 思考模式（`strategy=single|two-phase`） |
| `minigpt/data/pretrain_dataset.py` | `tokenize_jsonl_to_bin`（uint16/uint32 自适应 + `.meta.json`）、`TokenBinDataset` |
| `minigpt/data/sft_dataset.py` | `InstructionDataset / collate / split_dataset` |
| `minigpt/train/{trainer,pretrainer,sft_trainer,metrics}.py` | 训练器（AMP/梯度累积/lr 调度/eval/save/resume/DDP/compile/num_workers）、入口、TensorBoard 指标 |
| `scripts/build_pretrain_bin.py` | 语料 → `.bin`（`build` / `info`） |
| `scripts/generate.py` | 推理 CLI（`--chat --thinking --thinking-strategy single`） |
| `scripts/evaluate_pretrain.py` | loss / perplexity（`--max-rows` 控耗时） |
| `scripts/eval_thinking.py` | 思考模式对比（plain/single/two-phase，`--repetition-penalty 1.0`） |
| `scripts/build_cot_sft.py` | CoT 数据（`--profile easy|hard`，自带答案） |
| `scripts/train_dashboard.py` | 看板（网页 8099 / `--once` / `--plot`；默认**自动跟随当前正在跑的 run**，含 SFT 阶段） |
| `scripts/report_training.py` | 汇总全部 run/评测 → Markdown 报告（ppl 对比、思考模式准确率、样例；`--json` 供 agent 解析） |
| `scripts/train_pretrain_resilient.sh` | 自愈长训（崩溃自动续训；环境变量可覆盖 OUT_DIR/DATA_BIN/PRESET_ARGS/TARGET_STEPS/NPROC） |
| `scripts/checkpoint_janitor.sh` | 每目录保留最近 N 个 checkpoint（防磁盘写满） |
| `scripts/run_downstream.sh`、`wait_and_run_downstream.sh` | SFT→CoT→评测自动接力 |
| `tests/` | `pytest tests/ -q`（24 passed, 1 skipped） |
| `minigpt/README.md` | 指标面板、吞吐、看板、思考模式与实测结果 |

## 3. 各阶段标准命令

```bash
# 数据（幂等；自动选 uint16/uint32 并写 meta）
python scripts/build_pretrain_bin.py build --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
  --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/pretrain_v3_full.bin --max-lines 0

# 预训练（把 --model_*/--train_* 换成 §1 表格中对应预设）
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.pretrainer \
  --data_tokenizer_dir models/tokenizer_v3 --data_tokenized_bin dataset/bins/pretrain_v3_full.bin \
  --data_eval_ratio 0.001 \
  --model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 \
  --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 200 \
  --train_eval_steps 2000 --train_save_steps 4000 --train_max_steps 211000 \
  --train_epochs 7 --train_torch_compile True \
  --paths_last_checkpoint_path <latest checkpoint-*.pth> \
  --paths_output_dir models/checkpoints/pretrain_v2_full > models/checkpoints/pretrain_v2_full.log 2>&1 &

# SFT / CoT / 评测 / 推理 见 §5、§6
```

## 4. 监控（agent 必须定期执行并汇报）

```bash
python3 scripts/train_dashboard.py --once                  # 进度/损失/速率/ETA/GPU（终端）
python3 scripts/train_dashboard.py --serve --port 8099     # 网页看板（5s 刷新，loss/grad 走势图）
tensorboard --logdir <run>/tensorboard --port 6006         # 直方图/图像/模型图/嵌入投影
tail -n 20 <run>.log                                       # 每个 eval 一行
```
健康判据：eval_loss 下降；grad_norm 在裁剪阈值（默认 1.0，看板红虚线）附近震荡；GPU 利用率 >75%（CPU/MPS 例外）。

## 5. SFT 与思考模式（CoT）

```bash
# 5.1 指令 SFT
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.sft_trainer \
  --pretrain <pretrain>/final.pt --data_tokenizer_dir models/tokenizer_v3 \
  --sft-jsonl dataset/sft/sft_data_zh.jsonl --data_max_lines 60000 --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1e-5 --train_epochs 2 \
  --paths_output_dir models/checkpoints/sft_v2_chat > models/checkpoints/sft_v2_chat.log 2>&1 &

# 5.2 CoT（先在 easy 档验证机制，再上 hard）
python scripts/build_cot_sft.py --profile easy --n-train 60000 --n-eval 200 \
  --out-train dataset/sft/sft_cot_easy_60k.jsonl --out-eval dataset/sft/cot_eval_easy_zh.jsonl
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.sft_trainer \
  --pretrain models/checkpoints/sft_v2_chat/final.pt --data_tokenizer_dir models/tokenizer_v3 \
  --sft-jsonl dataset/sft/sft_cot_easy_60k.jsonl --data_max_lines 60000 --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1.5e-5 --train_epochs 2 \
  --paths_output_dir models/checkpoints/sft_v2_cot_easy > models/checkpoints/sft_v2_cot_easy.log 2>&1 &
```

## 6. 推理与评测（输出对比报告）

```bash
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 --chat \
  --prompt "什么是AI？" --max-new-tokens 120 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 --chat --thinking \
  --thinking-strategy single --prompt "请计算 27 ÷ 3 等于多少？" [--hide-thinking]
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 \
  --bin <bin> --max-rows 512 --output models/checkpoints/ppl.json
python scripts/eval_thinking.py --checkpoint <ckpt> --eval-jsonl dataset/sft/cot_eval_easy_zh.jsonl \
  --n 60 --strategies plain,single,two-phase --repetition-penalty 1.0 --output models/checkpoints/eval_thinking.json
```

生成汇总报告（下游跑完后执行，产物 `models/checkpoints/TRAINING_REPORT.md`）：
```bash
python3 scripts/report_training.py --json /tmp/report.json
```

汇报模板：
```
环境：<CPU/GPU 型号×数量/显存> → 采用预设 <name>（模型/batch/ctx/AMP/compile/nproc）
预训练：step X/Y, eval_loss A→B, 吞吐 Zk tok/s, ETA, 产物路径
SFT/CoT：loss 与产物；思考模式 easy/hard 准确率（对照基座）
样例：3 条输入/输出
```

## 7. 已知坑（详见 references/troubleshooting.md）

- 词表 >65535 必须 uint32（否则 `OverflowError`）；`TokenBinDataset` 依赖 `.meta.json`。
- 续训必须带 `--paths_last_checkpoint_path`；RNG 张量加载时会 `.cpu()` 修正。
- `torch.compile` 只对固定形状（预训练）收益大；SFT 变长序列可能反复重编译。
- 算术评测 `--repetition-penalty` 必须 1.0。
- 长训练必须跑 `scripts/checkpoint_janitor.sh`；WSL 下注意 C 盘 vhdx 增长。
- 长任务用 `setsid` 脱离工具进程组；用 `pgrep -af` 与日志校验存活。

## 7.5 领域增量预训练（如新下载的行业/代码语料）

外部语料（parquet/jsonl）通常是**预训练语料**而不是指令数据，正确姿势是「增量预训练 → 再 SFT」，
否则直接 SFT 会把模型训成续写器、破坏对话能力。

```bash
# 1) parquet → jsonl（过滤 + 混入通用语料防遗忘 + token 估算）
python scripts/parquet_to_jsonl.py   --src dataset/IndustryCorpus2_computer_programming_code_high   --out dataset/domain/code_corpus.jsonl   --min-chars 300 --max-chars 8000 --max-line-length 500 --min-quality 3.0   --mix-jsonl dataset/pretrain_t2t_mini.jsonl --mix-ratio 0.15 --mix-limit 200000

# 2) 建 bin（自动 uint16/uint32 + meta）
python scripts/build_pretrain_bin.py build --corpus-jsonl dataset/domain/code_corpus.jsonl   --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/code_domain.bin --max-lines 0

# 3) 从现有基座增量续训（低 lr，1 epoch；混入语料已在步骤 1 完成）
OUT_DIR=models/checkpoints/pretrain_domain_code DATA_BIN=dataset/bins/code_domain.bin \
TARGET_STEPS=<按预算换算> PRESET_ARGS="--model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 \
  --model_context_length 512 --train_batch_size 8 --train_learning_rate 1e-4 --train_warmup_steps 100 \
  --train_eval_steps 1000 --train_save_steps 4000 --train_epochs 2 --train_torch_compile True" \
FALLBACK_CKPT=models/checkpoints/pretrain_v2_full/final.pt \
setsid nohup bash scripts/train_pretrain_resilient.sh > models/checkpoints/pretrain_domain_code.watchdog.log 2>&1 &

# 4) 领域基座再跑一次 SFT/CoT（复用 §5 命令，--pretrain 指向新的 final.pt）
```
要点：混入通用语料 10–20% 防灾难性遗忘；lr 用预训练的 1/5~1/10；长文档被 ctx 切窗属正常；
过滤 `quality_score`/`max_line_length` 可显著提纯；若目标是"思考模式"，数学语料（IndustryCorpus2_mathematics_statistics_high）
比代码更适合，可从中抽取题目构造 CoT 指令数据。

## 8. 参考

- `references/training-recipes.md`：环境预设矩阵、吞吐表、预算↔tokens 换算、多卡公式、CPU 实测。
- `references/troubleshooting.md`：OOM/速度/续训/数据/磁盘/指标异常处置。
- `minigpt/README.md`：指标面板、看板、思考模式与实测结果。
