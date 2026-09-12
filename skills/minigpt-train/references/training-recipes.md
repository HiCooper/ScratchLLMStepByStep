# MiniGPT 训练配方与容量换算

## 0. 环境预设矩阵（preflight 自动判定，pipeline 自动消费）

`bash skills/minigpt-train/scripts/preflight.sh` 会检测硬件并写入 `models/checkpoints/preflight.json`：

| 预设 | 判定条件 | 模型(emb/layers/heads) | batch×ctx | 精度 | compile | nproc | 预期吞吐 |
|---|---|---|---|---|---|---|---|
| `cpu` | 无 CUDA（MPS 亦回退） | 128/2/4 | 8×128 | fp32 | ✗ | 1 | **0.12–0.17k tok/s**（6 核实测） |
| `gpu-tiny` | 单卡 <5.5GB | 384/8/8 | 4×384 | fp16 | ✓ | 1 | 10–18k tok/s |
| `gpu-small` | 5.5–8GB | 512/10/8 | 8×512 | fp16 | ✓ | 1 | **22–30k tok/s**（RTX2060） |
| `gpu-mid` | 8–16GB | 512/10/8 | 12×1024 | fp16 | ✓ | 1 | 30–50k tok/s |
| `gpu-large` | ≥16GB | 768/12/12 | 16×1024 | fp16 | ✓ | 1 | 60k+ tok/s |
| `multi-gpu` | ≥2 张 GPU | 按单卡显存分档 | 每卡 8–16 | fp16 | ✓ | N | 单卡×N×0.85 |

JSON 字段：`preset.name / reason / gpu_count / vram_gb / model{...} / train{...} / launch.nproc / expect / assumed_s_per_step / tokens_per_step`。
`pipeline.sh --full --hours H` 用 `assumed_s_per_step` 换算 `TARGET_STEPS`，并把 `model`/`train` 展开成 CLI 参数。

手动覆盖：`--preset cpu|gpu-tiny|gpu-small|gpu-mid|gpu-large|multi-gpu`。

## 1. CPU-only（无 GPU）实操

实测（6 核，另有一张卡在跑训练，属偏保守数据）：

| 配置 | 单步 | 吞吐 |
|---|---|---|
| 128/2/4, ctx128, bs4 | 2.98 s | **0.17k tok/s** |
| 256/4/4, ctx256, bs2 | 4.14 s | 0.12k tok/s |

建议流程（agent 直接照做）：
```bash
bash skills/minigpt-train/scripts/pipeline.sh --smoke --preset cpu          # 30 步验证链路
bash skills/minigpt-train/scripts/pipeline.sh --full  --preset cpu --hours 1 \
     --corpus-lines 20000                                                   # 2 万行小语料 + 约千步
```
- 目标：验证「数据→训练→续训→推理」链路可用，**不要承诺生成质量**；
- 提升手段：`OMP_NUM_THREADS=<物理核>`、`--train_num_workers 2`（数据加载多进程）、缩小 ctx；
- 想得到可用模型：换 CUDA 机器（同预算下算力约 100×），或云端 GPU 跑 `gpu-mid/gpu-large/multi-gpu` 预设。

## 2. 单卡档位（实测参考，bs8/ctx512，fp16 + GradScaler）

| 模型 | 词表 | 单步 | 吞吐 | 备注 |
|---|---|---|---|---|
| 512/10/8（47.9M） | 32k | 0.138s（compile）/ 0.229s（eager） | 29.7k / 17.9k tok/s | 本仓库生产配置 |
| 384/8/8（~30M） | 32k | ~0.09s（compile 估计） | ~35k tok/s | 更省显存 |
| 768/12/12（~120M） | 32k | batch≤4 + 累积 | 10–14k tok/s | 6GB 可跑但慢 |
| 任意小型 | 151k（Qwen2.5） | ~2.0s | **~2.0k tok/s** | 6GB 不可行，仅大显存 |

> 结论：小显存优先「32k 词表 + 512/10/8 + fp16 + torch.compile」，用数据量换质量。

## 3. 多卡（DDP）公式与操作

- 有效 batch = `train_batch_size × N × grad_accumulation_steps`（`train_batch_size` 是**每卡**大小）。
- 学习率：`lr = 单卡基准 × √N`（AdamW 常用；`pipeline.sh` 已对 `multi-gpu` 预设自动计算）。
- 目标步数：同样 tokens 预算下，`steps ≈ 单卡步数 / (N × 加速比)`；`tokens = steps × batch × N × (ctx-1)`。
- 数据：`DistributedSampler` 自动分片（`Trainer` 内实现），无需手工切分数据集。
- 日志/checkpoint/评测：只在 rank0（`is_main_process`）产出；看板与 `tail <run>.log` 照常可用。
- 启动（`NPROC>1` 时守护脚本自动改用 torchrun）：
```bash
NPROC=4 bash scripts/pretrain_start.sh --paths_output_dir models/checkpoints/pretrain_ddp
# 或经 skill：
bash skills/minigpt-train/scripts/pipeline.sh --full --hours 10        # 自动 multi-gpu 预设
```
- 常见问题：`NCCL_DEBUG=WARN` 看通信错误；`CUDA_VISIBLE_DEVICES` 与 `--nproc_per_node` 必须一致；
  显存不足时先降**每卡 batch**再加 `grad_accumulation_steps`；compile 与 DDP 同时用时顺序为「先 DDP 再 compile」（代码已保证）。

## 4. 预算换算

```
目标步数   = 预算秒数 / 实测s_per_step      （preflight.assumed_s_per_step 为估算值）
看到 tokens = 目标步数 × batch × nproc × (ctx-1)
Chinchilla 参考 = 10~20 × 参数量
```
本机参考（512/10/8, compile, bs8, ctx512 ≈ 4096 tok/步, 0.17s/步）：
- 1 小时 ≈ 21k 步 ≈ **8,650 万 tokens**
- 10 小时 ≈ 21 万步 ≈ **8.6 亿 tokens**（≈18×参数）
- 全量语料 `pretrain_v4_full.bin` = **2.475 亿 tokens**（127 万行），1 epoch ≈ 6.0 万步 ≈ 2.8 小时

## 5. 超参经验值

| 项 | 预训练 | SFT | CoT-SFT |
|---|---|---|---|
| lr | 6e-4 ~ 1e-3（cosine，warmup 200~500） | 1e-5 ~ 2e-5 | 1e-5 ~ 1.5e-5 |
| batch | 8（ctx512，单卡） | 8 | 8 |
| epochs | 1~5（受 tokens 预算限制） | 1~2 | 2~3 |
| ctx | 512 | 512（max_len 截断） | 512 |
| eval/save | 2000 / 4000 | 500 / 4000 | 500 / 4000 |
| 其它 | `--train_torch_compile True` | 变长不启用 compile | 变长不启用 |

## 6. 续训与守护（环境变量可覆盖）

- 续训：`--paths_last_checkpoint_path <ckpt>`（checkpoint 内含 optimizer/scaler/RNG）。
- 自愈：`scripts/train_pretrain_resilient.sh`，可用环境变量注入：
  `OUT_DIR / LOG / FALLBACK_CKPT / DATA_BIN / TOKENIZER_DIR / PRESET_ARGS / TARGET_STEPS / NPROC / MAX_RESTARTS`。
- 空间：`scripts/checkpoint_janitor.sh 60 2 120`（每目录留最近 2 个，单个 ~585MB）。
- 多卡：把 `NPROC` 设为卡数即可（脚本自动 torchrun）。

## 7. 词表/数据管线要点

- 数据：`python scripts/build_pretrain_bin.py build --corpus-jsonl ... --tokenizer-dir ... --out-bin ... --max-lines N`
  自动按词表选择 uint16/uint32 并写 `.meta.json`（`TokenBinDataset` 依赖它）。
- 换 tokenizer：只改 `--data_tokenizer_dir`，模型词表自动跟随；**词表 >65535 必须 uint32**。
- 语料顺序即切片顺序；混合多来源请先 shuffle 行顺序。

## 8. 评测协议（唯一详述处）

```bash
# 语言建模：--split val 与训练同一口径（均匀分块 + 双端对齐文档边界）
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --bin dataset/bins/pretrain_v5_full.bin --split val
# 思考模式：算术评测必须 --repetition-penalty 1.0（否则惩罚重复的数字）
python scripts/eval_thinking.py --checkpoint <ckpt> --eval-jsonl dataset/sft/cot_eval_easy_em50_disjoint.jsonl \
  --n 60 --strategies plain,single,two-phase --repetition-penalty 1.0
```

| 口径 | 命令 | 用途 |
|---|---|---|
| **验证集（推荐）** | `evaluate_pretrain.py --split val` | 均匀分块 + 双端对齐文档边界，497 窗，与训练**文档级零重叠**；可比较泛化 |
| 全量 bin | `--split all` | 只适合**未参与训练**的 bin（如纯领域留出 bin） |
| 禁止 | `--split all --max-rows N` | 取的是 bin 最前面的窗口 = 训练集 loss；本语料按领域排序，头尾 ppl 可差 3 倍以上 |

同一 checkpoint 的实测差异（`pretrain_v2_full`）：

| 口径 | ppl |
|---|---|
| 头部 64 窗（旧 README 口径） | 23.40 |
| 尾部 64 窗（全英文选择题） | 6.78 |
| **全量 blocked val（497 窗）** | **15.82** |

结论：绝对数值高度依赖"取哪些窗口"，只有固定 `--split val` 才可跨 checkpoint 比较；
切分范围记录在每个 run 的 `metrics.json:data_split` 里，可复现。

## 容量消融（isotoken）

```bash
DRY_RUN=1 bash scripts/run_capacity_ablation.sh     # 无 GPU 也能检查命令拼装
setsid nohup env TAG=cap768 TOKENS=8.6e8 BS=16 ACCUM=1 \
  bash scripts/run_capacity_ablation.sh > models/checkpoints/cap768.log 2>&1 &
```

- 512/10/8 (47.9M) 基线：`pretrain_v2_full`，8.6 亿 tokens / 211k 步
- 768/12/12 (109.6M)：同 8.6 亿 tokens → 52,490 步 @ bs16×ctx1024（单卡 ≥16GB，bf16）
- 显存不足时保持**有效 batch 不变**：`BS=4 ACCUM=4`
- 判据：hard CoT 零重合留出集（`cot_eval_hard_disjoint.jsonl`）上的准确率是否显著提升

> 本仓库开发环境无 GPU，该消融只完成代码/预设/数据侧准备，训练与评测**尚未执行**。
