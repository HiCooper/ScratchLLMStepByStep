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


## 1. 环境自适应（**先判断硬件，再决定参数**）

**不要手挑参数**：`preflight.sh` 会把硬件判定写成 `models/checkpoints/preflight.json: preset`，
`pipeline.sh` 自动消费；也可 `--preset cpu|gpu-tiny|gpu-small|gpu-mid|gpu-large|multi-gpu` 手动覆盖。
矩阵（判定条件 / 模型 / batch×ctx / 精度 / compile / 吞吐）与 JSON 字段含义见
**`references/training-recipes.md §0`**（唯一详述处）。下面是三条例外情况的处置：

### 1.1 没有 GPU 怎么办（agent 必须这样处理）
1. **明确降级预期**：CPU 只能做「机制验证」，不要承诺可读模型。实测本机 6 核约 **120–170 tok/s**，
   1 小时仅 ~0.5M tokens（GPU 是它的 100 倍以上）。
2. 用 CPU 预设（自动检测即为 `cpu`，也可显式 `--preset cpu`）：
   - 模型 128/2/4、ctx 128、batch 8、`--train_mixed_precision_dtype none`、`--train_torch_compile False`；
   - 语料只取前 2 万行（`pipeline.sh --full --preset cpu` 会按需建 `dataset/bins/pretrain_cpu20k.bin`，不必手工准备）；
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
| `scripts/data/build_pretrain_bin.py` | 语料 → `.bin`（`build` / `info`） |
| `scripts/data/audit_dataset.py` | 数据集体检（bin↔meta↔分词器对账、切分预览、SFT 泄漏/CoT 重合、token 预算；有 BLOCKER 返回 1） |
| `scripts/data/audit_sft_lengths.py` | SFT 真实规模（监督 token、截断路径分布、零监督样本；CPU 约 30s） |
| `scripts/data/mix_corpora.py` / `parquet_to_jsonl.py` / `clean_long_docs.py` | 语料侧：按**字符**配比混合 / parquet→jsonl（`--mix-by chars`、`--shards`）/ 长文档结构化切块 |
| `scripts/generate.py` | 推理 CLI（`--chat --thinking --thinking-strategy single`） |
| `scripts/evaluate_pretrain.py` | loss / perplexity（**默认 `--split val`** 走与训练同一口径的验证集） |
| `scripts/eval_thinking.py` | 思考模式对比（plain/single/two-phase + **分档准确率** `by_difficulty`，必须 `--repetition-penalty 1.0`） |
| `scripts/chat_probe.py` | 51 场景对话质检（greedy + sampled） |
| `scripts/data/build_cot_sft.py` | CoT 数据（`--profile easy` / `hard`，自带答案） |
| `scripts/train_dashboard.py` | 看板（网页 8099 / `--once` / `--plot`；默认**自动跟随当前正在跑的 run**，含 SFT 阶段） |
| `scripts/watch_training.sh` | 终端版看板（SSH 用；不传参自动选最新 run） |
| `scripts/report_training.py` | 汇总全部 run/评测 → Markdown 报告（曲线+噪声、ppl 对比、思考模式准确率与分档、样例、历史基线；`--json` 供 agent 解析） |
| `scripts/verify_v5_delivery.py` | **交付自检**（42+ 项：架构/步数/ppl 口径/报告章节/分档/样例/`.bin` 对账，缺件 exit 1；`--allow-partial` 看进度） |
| `scripts/train_pretrain_resilient.sh` | 自愈长训（崩溃自动续训；环境变量可覆盖 OUT_DIR/DATA_BIN/PRESET_ARGS/TARGET_STEPS/NPROC） |
| `scripts/checkpoint_janitor.sh` | 每目录保留最近 N 个 checkpoint（防磁盘写满） |
| `scripts/run_downstream.sh` | **通用**下游：SFT → CoT easy → CoT hard → 评测（`BASE`/`TAG` 可覆盖，自动优先 `best.pt`） |
| `scripts/run_downstream_evals.sh` | **只跑评测/出样**（4a~4d），可单独重跑；`TAG=... bash ...` |
| `scripts/wait_and_run_downstream.sh` | **通用接力**：等 `$OUT_DIR/final.pt` → 跑上面的通用下游（`pipeline.sh` 自动拉起；`OUT_DIR` 必须传） |
| `scripts/watch_v5_chain.sh` | **v5 生产接力**：等 final.pt → `run_v5_downstream.sh`，并**二级监督**训练守护（双缺席 3 分钟自动拉起） |
| `scripts/resume_v5_chain.sh` | **一键恢复全链路**（幂等，已在跑的自动跳过）：训练守护 + janitor + watcher + 看板。**机器重启后必须跑它**——进程内监督扛得住"进程被杀"，扛不住整机重启（实测教训） |
| 其余（按需） | `download_data.sh`、`train_tokenizer.py`、`check_env.py`/`estimate_resources.py`/`validate_pretrain.py`/`validate_ddp.py`（体检与校验）、`run_code_domain.sh`/`run_domain_compare.sh`/`run_capacity_ablation.sh`（领域与容量消融配方） |
| `tests/` | `pytest tests/ -q`（279 passed；依赖真实 `dataset/` 的用例会自动 skip） |
| `minigpt/README.md` | 指标面板、吞吐、看板、思考模式与实测结果 |

## 3. 各阶段标准命令

```bash
# 造 bin（幂等；自动选 uint16/uint32 并写 meta）。已有成品则跳过 → dataset/README.md
python scripts/data/build_pretrain_bin.py build --corpus-jsonl dataset/pretrain_t2t.jsonl \
  --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/pretrain_v5_full.bin --nproc 6

# 预训练基座（--model_*/--train_* 用 §1 预设；下面是 gpu-small 档）
# 长任务必须 setsid nohup 起来，别前台阻塞
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.pretrainer \
  --data_tokenizer_dir models/tokenizer_v3 \
  --data_tokenized_bin dataset/bins/pretrain_v5_full.bin --data_eval_ratio 0.002 \
  --model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 \
  --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 2000 \
  --train_eval_steps 2000 --train_save_steps 4000 --train_epochs 1 --train_torch_compile True \
  --paths_output_dir models/checkpoints/pretrain_v5 > models/checkpoints/pretrain_v5.log 2>&1 &

# 续训加 --paths_last_checkpoint_path <ckpt>；SFT / CoT / 评测 / 推理见 §5、§6
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
# easy 必须带 --easy-max 50：默认 9 的题池只有 1064 个唯一题面，60000 条训练集里同一题平均重复 56×，
# 且切不出真正不相交的留出集（实测旧留出集与现用训练集题面级重合 77/200 = 38.5%）。
python scripts/data/build_cot_sft.py --profile easy --easy-max 50 --n-train 60000 --n-eval 200 \
  --out-train dataset/sft/sft_cot_easy_em50_60k.jsonl --out-eval dataset/sft/cot_eval_easy_em50_disjoint.jsonl
setsid nohup env PYTHONUNBUFFERED=1 python3 -m minigpt.train.sft_trainer \
  --pretrain models/checkpoints/sft_v2_chat/final.pt --data_tokenizer_dir models/tokenizer_v3 \
  --sft-jsonl dataset/sft/sft_cot_easy_em50_60k.jsonl --data_max_lines 60000 --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1.5e-5 --train_epochs 2 \
  --paths_output_dir models/checkpoints/sft_v2_cot_easy > models/checkpoints/sft_v2_cot_easy.log 2>&1 &
```

## 6. 推理与评测（输出对比报告）

```bash
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 --chat \
  --prompt "什么是AI？" --max-new-tokens 120 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 --chat --thinking \
  --thinking-strategy single --prompt "请计算 27 ÷ 3 等于多少？" [--hide-thinking]
# --split val = 训练时同一口径的验证集（均匀分块 + 双端对齐文档边界），可直接比较泛化
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --tokenizer-dir models/tokenizer_v3 \
  --bin <bin> --split val --output models/checkpoints/ppl.json
python scripts/eval_thinking.py --checkpoint <ckpt> --eval-jsonl dataset/sft/cot_eval_easy_em50_disjoint.jsonl \
  --n 60 --strategies plain,single,two-phase --repetition-penalty 1.0 --output models/checkpoints/eval_thinking.json
```

生成汇总报告（下游跑完后执行，产物 `models/checkpoints/TRAINING_REPORT.md`）：
```bash
python3 scripts/report_training.py --json /tmp/report.json
```

**交付自检**（收尾必跑：一条命令核对每个阶段的产物是否齐全、数字是否合理）：
```bash
python3 scripts/verify_v5_delivery.py                 # 缺件 exit 1；训练途中加 --allow-partial 看进度
```
它检查的不只是"文件在不在"：架构是否等于预设、步数是否到位、`perplexity == exp(eval_loss)`、
ppl 是否用 `--split val` 评在同一个 bin 上、报告是否含 eval_loss 曲线与历史基线章节、
`.bin` 的 token/vocab 与 `metrics.json` 里记录的是否对得上。

汇报模板：
```
环境：<CPU/GPU 型号×数量/显存> → 采用预设 <name>（模型/batch/ctx/AMP/compile/nproc）
预训练：step X/Y, eval_loss A→B, 吞吐 Zk tok/s, ETA, 产物路径
SFT/CoT：loss 与产物；思考模式 easy/hard 准确率（对照基座）
样例：3 条输入/输出
```

## 7. 已知坑

**症状 → 原因 → 处置的完整清单见 `references/troubleshooting.md`**（OOM / uint16 溢出 / 续训 RNG /
进程被杀 / 吞吐骤降 / 磁盘写满 / 指标异常）。这里只列两条属于"训练策略"而非"故障"的：

- **优先用 `best.pt` 而不是 `final.pt`**：小模型 SFT/CoT 后期必然过拟合（实测 eval 1.611 → 1.740）。
  训练器在 eval 创新低时写 `best.pt`（`--train_save_best False` 可关），下游训练/评测/发布都优先取它，
  只有确实不存在 `best.pt`（老产物）才回退 `final.pt`。
- 领域续训的 `--train_reset_step` 坑（会导致"几十秒训完"）见 §7.5。

## 7.5 领域续训与数据分布

**领域语料是预训练语料、不是指令数据**：正确姿势是「增量续训 → 再 SFT」；直接拿去 SFT 会把模型训成续写器。

三条必须遵守的抽样/配比规则（原理与实测证据见 **`references/data-distribution.md`**）：

1. **整文件均匀抽样**，禁止"读前 N 行"——语料按领域排序，取前缀只拿得到一个领域；
2. **配比按字符/token**、均值用全文件扫描后的精确值——历史上的"混 15%"实际只有 **1.18%**；
3. **长文档先按结构切块**再入池——中位 28,651 字符的论文一篇要占 25–35 个窗口。

```bash
# 数据源、规模、生成/重配命令  → dataset/README.md（唯一详述处）
# 训练命令：基座 / 领域续训(70:30) / 退火档(50:50) / 双口径验收 → references/data-distribution.md §8
python scripts/data/audit_dataset.py     # 开训前完整体检（别加 --quick）；有 BLOCKER 返回 1
```

**验收（缺一不可）**：领域 ppl 明显下降 **且** 通用 ppl 上升 ≤5%（`bash scripts/experiments/run_domain_compare.sh`）。
反例：`lr=1e-4 / 1 epoch / 名义 mix 15%` 的代码续训 → 领域 ppl −64%，但通用 ppl **+41%**，
下游对话 SFT eval_loss 2.78→2.91。不达标就降 lr（1e-5~3e-5）、提高混料比（≥30%）或减少步数。

**最大的坑**：`--train_max_steps` 是**绝对步数**。从 211k 步基座"再训 26,847 步"若直接传
`--train_max_steps 26847`，加载后 `step ≥ max_steps` 会**立刻判定训完**（几十秒产出 final.pt，实际一步没训）。
必须 `--train_reset_step True`（`train_pretrain_resilient.sh` 的 `RESET_STEP=auto` 已默认处理）；
启动后核对日志出现 `reset_step=True：步数 211000 -> 0`。

## 8. 参考

| 想知道什么 | 看哪里 |
|---|---|
| 环境预设矩阵、吞吐、预算↔tokens、多卡公式 | `references/training-recipes.md` |
| 症状→原因→处置（OOM/慢/续训/磁盘/指标） | `references/troubleshooting.md` |
| 数据分布的原理、实测证据、配比口径、验收 | `references/data-distribution.md` |
| **各阶段数据源、规模、生成命令** | `../../dataset/README.md` |
| 包 API、配置项、训练产物、指标面板 | `../../minigpt/README.md` |
