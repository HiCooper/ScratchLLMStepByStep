
<h1 align="center"> ScratchLLMStepByStep</h1>

从零手写并训练一个大语言模型的教程 + 生产级训练工程：从训练分词器开始，逐步实现 attention / transformer / GPT，
再做预训练、SFT 指令微调与思考模式（CoT），最终得到一个可对话的小模型。

适合具备 Python 基础、想深入理解 LLM 原理与工程实现的读者。

## 💥 章节结构

15 个 notebook 按 `01_`~`15_` 顺序阅读，覆盖「分词器 → 模型结构 → 预训练 → SFT → 推理」全流程：

| 阶段 | # | Notebook |
|---|---|---|
| 分词与模型结构 | 01 | [带你从零训练 tokenizer](./notebooks/01_分词器训练.ipynb) |
| | 02 | [词嵌入和位置嵌入](./notebooks/02_模型结构之词嵌入和位置编码.ipynb) |
| | 03 | [从零认识自注意力](./notebooks/03_模型结构之自注意力.ipynb) |
| | 04 | [实现因果注意力机制](./notebooks/04_模型结构之因果注意力.ipynb) |
| | 05 | [实现多头注意力](./notebooks/05_模型结构之多头注意力.ipynb) |
| | 06 | [构建 TransformerBlock](./notebooks/06_模型结构之TransformerBlock.ipynb) |
| | 07 | [构建 MiniGPT](./notebooks/07_模型结构之MiniGPT.ipynb) |
| 预训练 | 08 | [预训练之高效数据加载](./notebooks/08_预训练之高效数据加载.ipynb) |
| | 09 | [预训练之从零起步](./notebooks/09_预训练之从零起步.ipynb) |
| | 10 | [预训练之运算加速](./notebooks/10_预训练之运算加速.ipynb) |
| | 11 | [预训练之多卡并行](./notebooks/11_预训练之多卡并行.ipynb) |
| SFT | 12 | [SFT 之分类微调](./notebooks/12_SFT之分类微调.ipynb) |
| | 13 | [SFT 指令微调之数据处理](./notebooks/13_SFT指令微调之数据处理.ipynb) |
| | 14 | [SFT 之指令微调训练](./notebooks/14_SFT指令微调之训练.ipynb) |
| 推理 | 15 | [模型推理之选词算法](./notebooks/15_模型推理之选词算法.ipynb) |

> notebook 会通过 `%run minigpt/...` 引用包内实现，并内联展示部分中间版本（例如
> `pretrainer_single.py` 的单卡 `Trainer`），便于对照"教学版 vs 生产版"的差异。

## 💥 数据集

统一从 **ModelScope** 下载（国内直连），分词器训练与预训练共用 [`gongjy/minimind_dataset`](https://www.modelscope.cn/datasets/gongjy/minimind_dataset) 的 `pretrain_t2t.jsonl`：

```bash
bash scripts/download_data.sh          # 下载到 dataset/
```

| 用途 | 文件 | 大小 |
|---|---|---|
| 分词器训练 + 预训练（通用） | `pretrain_t2t.jsonl` | ~7.9GB（846.9 万行 / 3.24G 字符 ≈ 16.9 亿 tokens） |
| 预训练（结构化 20%） | `chinese_cosmopedia`（教科书体） | 8/64 分片 |
| SFT | `sft_data_zh.jsonl` | 10 万条 |

> **各训练阶段用到哪些数据源、规模多大、怎么生成** → 见 [`dataset/README.md`](./dataset/README.md)（唯一详述处）。

> ⚠️ 该语料是**长段落**文本，字节级 BPE 预分词会把整段中文当成一个超长「词」，全量训练分词器会 OOM；
> 用 `scripts/train_tokenizer.py --max-lines 150000`（约 150MB）即可。

## 💥 运行环境

Ubuntu 18.04 / Python 3.10 / PyTorch 2.4+ / CUDA 12.1（模型结构部分不依赖 GPU；预训练与 SFT 建议 6GB 以上显存）。
依赖见 [`requirements.txt`](./requirements.txt)（单一事实来源是 [`pyproject.toml`](./pyproject.toml)，推荐 `pip install -e ".[notebook,dev]"`）。

## 💥 模型规模与资源估算

默认架构：GELU 前馈 + 独立 QKV + **共享输入/输出嵌入**（`tie_word_embeddings=True`），vocab=32000。
下表按**不共享**口径给出（便于与历史数据对照）：`参数量 ≈ 2·vocab·emb + n_layers·12·emb²`。
fp16 + AdamW 训练时按 **16 字节/参数**估显存（fp16 权重 2 + 梯度 2 + fp32 主权重 4 + 动量 4 + 方差 4），另需留激活余量。

| 档位 | emb/layers/heads | 参数量 | 权重+优化器 | 起步 GPU | 1 epoch（2 亿 token）参考时长 |
|---|---|---|---|---|---|
| 微型 | 256/6/8 | 21M | ~0.34GB | 4GB | — |
| 小型 | 512/8/8 | 58M | ~0.93GB | 6GB | — |
| 中型（教程默认） | 768/12/12 | 134M | ~2.15GB | 8GB | RTX2060 ~20h / 3090 ~2h |
| 大型 | 1024/24/16 | 368M | ~5.9GB | 16GB | — |
| 超大 | 1280/32/20 | 711M | ~11.4GB | 24GB | — |

> 表内「超大」是相对**消费级 GPU 从零训练**的语境（711M 绝对尺度上仍属入门级；主流开源模型 7B 起跳）。
> 交互式估算：`python scripts/estimate_resources.py`。

## 💥 目录结构

```
├── AGENTS.md / pyproject.toml / requirements.txt   # agent 入口、打包与依赖、pytest 配置
├── notebooks/                 # 15 个课程 notebook
├── minigpt/                   # 可复用代码包（详见 minigpt/README.md）
│   ├── config.py              # RunConfig（model/data/train/paths）+ CLI 覆盖 + 布尔解析
│   ├── model/                 # attention / transformer / checkpoint（重建+宽容加载）/ generation
│   ├── data/                  # pretrain_dataset(.bin memmap + 分块验证集切分) / sft_dataset(逐轮掩码)
│   └── train/                 # trainer + schedule/ddp_utils/checkpoint_io/optim + metrics + 两个入口
├── scripts/                   # 数据 / 训练 / 评测 / 监控
│   ├── download_data.sh / train_tokenizer.py / parquet_to_jsonl.py / build_pretrain_bin.py
│   ├── build_cot_sft.py       # 生成 SFT/CoT 数据（保证 train/eval 零重合）
│   ├── evaluate_pretrain.py / eval_thinking.py / generate.py
│   ├── train_dashboard.py / report_training.py / chat_probe.py / check_env.py / estimate_resources.py
│   ├── validate_pretrain.py / validate_ddp.py           # 单卡与 DDP 链路验证
│   ├── train_pretrain_resilient.sh / checkpoint_janitor.sh / pretrain_start.sh
│   └── run_downstream*.sh / run_code_domain.sh / run_domain_compare.sh / run_capacity_ablation.sh
├── skills/minigpt-train/      # agent skill：SKILL.md + preflight 体检 + pipeline 一键流水线 + references/
├── tests/                     # pytest 单元测试（200 项，CPU 即可运行，不需要数据与 GPU）
├── eval_sets/                 # chat_probe 的场景集与探针结论（随仓库跟踪的评测固件）
├── dataset/ models/           # 语料/派生 .bin/分词器/训练产物（.gitignore，不进仓库）
└── img/                       # notebook 与文档的图片素材
```

> 上图是**概览**；`minigpt/` 下每个模块的逐文件说明见 [`minigpt/README.md`](./minigpt/README.md)，
> `tests/` 的逐文件说明见该目录下的模块 docstring。

## 💥 工程化要点

超参与路径集中在 [`minigpt/config.py`](./minigpt/config.py)，其默认值即生产基线
（**512/10/8 + ctx512 + `tokenizer_v3`(32k) + fp16 + 嵌入共享**）；任意入口支持扁平 CLI 覆盖
（`--model_emb_dim` / `--data_tokenized_bin` / `--train_batch_size` / `--paths_output_dir`），
布尔参数写成 `--train_torch_compile True|False`（必须带值），也可 `--config-file <config.json>` 整份读回（CLI 优先级更高）。

| 开关 | 位置 | 说明 |
|---|---|---|
| `use_swiglu` / `qkv_merged` / `tie_word_embeddings` | GPTConfig | SwiGLU 门控 / 合并 QKV / 嵌入共享（**改结构，与旧 checkpoint 不兼容**） |
| `norm_type` / `ffn_hidden_dim` / `lm_head_bias` / `rope_theta` | GPTConfig | RMSNorm 或 LayerNorm / 前馈中间维 / 输出头 bias / RoPE 基频 |
| `use_checkpoint` / `flash_attn` | GPTConfig | FFN 激活重计算 / FlashAttention（**仅 Ampere sm_80+**，Turing 卡必须 False） |
| `mixed_precision_dtype` / `gradient_accumulation_steps` / `torch_compile` | train_args | fp16/bf16 / 小显存换大 batch / 固定形状提速 +66% |
| `seed` / `deterministic_cudnn` / `ddp_timeout_seconds` | train_args | 随机种子 / 严格复现（牺牲吞吐）/ NCCL 超时 |

训练/推理链路的可靠性约定：数据顺序由独立 generator 按 `seed+epoch` 播种（与全局 RNG 解耦，断点续训跳过的是同一批样本）、
checkpoint 原子写（tmp→fsync→rename，失败不破坏上一个可用存档）、验证集按「均匀分块 + 双端对齐文档边界」切分并记录进
`metrics.json:data_split`、权重衰减只作用于 ≥2 维且非 bias 的参数。

## 💥 快速开始

```bash
# 0) 安装
git clone https://github.com/golfxiao/ScratchLLMStepByStep.git && cd ScratchLLMStepByStep
pip install -e ".[notebook,dev]"

# 1) 环境体检（只读）：依赖/GPU/显存/磁盘/语料/分词器/bin 词表一致性/单测
bash skills/minigpt-train/scripts/preflight.sh

# 2) 一键流水线（幂等；--smoke 约 5 分钟验证链路，--full 完整训练+SFT+CoT+评测）
bash skills/minigpt-train/scripts/pipeline.sh --smoke
bash skills/minigpt-train/scripts/pipeline.sh --full --hours 10
```

手动逐步执行（数据 → 预训练 → SFT → 推理 → 评测）的完整命令：

| 步骤 | 唯一详述处 |
|---|---|
| 造语料 / 建 bin | [`dataset/README.md`](./dataset/README.md) |
| 预训练、领域续训、退火、双口径验收 | [`skills/minigpt-train/references/data-distribution.md`](./skills/minigpt-train/references/data-distribution.md) §8 |
| SFT / CoT / 推理 / 评测（agent 流程 + 监控） | [`skills/minigpt-train/SKILL.md`](./skills/minigpt-train/SKILL.md) |
| 包 API、配置项、指标面板 | [`minigpt/README.md`](./minigpt/README.md) |

产物：`checkpoint-{step}.pth`（含 optimizer/scaler/RNG/config）、`best.pt`（eval 最优）、`final.pt`、
`tensorboard/`、`metrics.json`、`sample.txt`。续训：`--paths_last_checkpoint_path .../checkpoint-N.pth`。

> 包级细节（配置字段、产物清单、TensorBoard 指标面板、采样参数、CoT 工作流、看板、续训注意事项）见
> **[`minigpt/README.md`](./minigpt/README.md)**。

## 🏋️ 实测训练结果（单卡 RTX 2060 6GB，2026-09）

> 📌 下列数字是历史实测记录；对应的 checkpoint 与 eval JSON 已清理，**不可再复现，也不指向现存文件**。
> 复现方式见上面的快速开始（同一 bin + `--split val`）。

### 分词器取舍（吞吐驱动）

| tokenizer | vocab | 单步 | 吞吐 | 结论 |
|---|---|---|---|---|
| Qwen2.5 | 151,665 | 2.01s | 2.0k tok/s | 6GB 卡一 epoch 需 ~10h，不可行 |
| 重训中文 BPE `models/tokenizer_v3` | 32,000 | 0.21s | 19.3k tok/s | ✅ 采用 |

词表由 tokenizer 推导，换 `--data_tokenizer_dir` 即可（大显存机器可换回 151k 词表，但会把 70%+ 参数花在 embedding 上：
512/10/8 从 47.9M 涨到 109.3M）。

### 基座对比（同一 `.bin`）

| 基座 | 训练步数 | 训练 tokens | 通用 ppl（旧口径¹） | **通用 ppl（新协议²）** | **代码领域 ppl（新协议²）** |
|---|---|---|---|---|---|
| `pretrain_v1_512` | 49,509 | 0.68 亿 | 37.80 | **33.56** | — |
| `pretrain_v2_full` | 211,000 | 8.6 亿 | 23.66 | **15.82** | 84.56 |
| `pretrain_domain_code` | +26,847（领域续训） | +1.1 亿 | 33.43 | **22.93** | **28.02** |

¹ 旧口径 = `--max-rows 512` 取 `.bin` **最前面**的窗口；对参与过训练的 bin 而言那是训练集 loss，不是泛化指标。
² 新协议 = `--split val`（均匀分块 + 双端对齐文档边界，497 窗，与训练集**文档级零重叠**）。

**结论**：10 小时把同口径 ppl 从 **33.56 降到 15.82**（规模换质量）；领域续训把代码 ppl **84.56 → 28.02（−67%）**，
代价是通用 ppl **15.82 → 22.93（+45%）**——15% 混料不足以抵消 lr=1e-4 一步到位的偏移，属典型部分灾难性遗忘。
**领域自适应必须双口径验收**（`bash scripts/run_domain_compare.sh`），并按需降 lr（1e-5~3e-5）、提高混料比（30%+）或减少步数。

### 思考模式（CoT）与 SFT

| run | 数据 | 步数 | eval_loss |
|---|---|---|---|
| `sft_v2_chat` | `sft_data_zh.jsonl` 60k × 2ep | 14,700 | 2.7795 |
| `sft_v2_cot_easy` | `sft_cot_easy_disjoint60k.jsonl` × 2ep | 14,700 | 1.7397（best 1.6109） |
| `sft_v2_cot_hard` | `sft_cot_hard_80k.jsonl` × 3ep | 29,400 | 1.3565 |

结论：**思考模式本身有效**（easy 从 0% → 90% → 100%），**hard 是容量墙**——47.9M 参数在多位数乘加上算不对，
需放大模型或走「工具调用范式」（模型只生成算式，由 Python 结算）。⚠️ **easy 的 100% 是记忆而非泛化**（原因见下），
需在零重合留出集上重测；完整命令、留出集与结果表见 [`minigpt/README.md` §7](./minigpt/README.md)。

### ⚠️ CoT 评测集口径修正（重要）

`build_cot_sft.py` 旧实现靠"换 seed"避免训练/评测重合，但 **easy profile 的操作数只有 1..9、全部唯一题目仅 1063 个**，
而训练集有数万行——实测旧 `cot_eval_easy_disjoint.jsonl` 的 156 条题目 **100%** 出现在 60k 训练集里（hard 33.5%）。
现已改为"先去重建池、再切出留出集"并落盘自检零重合，同时提供两份可用于验收的**零重合**留出集：
`dataset/sft/cot_eval_{easy,hard}_disjoint.jsonl`（`run_downstream_evals.sh` 已优先使用）。
easy 侧配套训练集需用 `build_cot_sft.py --profile easy --easy-max 20` 重新生成后再 SFT。

### 下一步实验：容量 vs 步数（isotoken 消融）

```bash
DRY_RUN=1 bash scripts/run_capacity_ablation.sh     # 只打印命令，无 GPU 也能检查
setsid nohup env TAG=cap768 TOKENS=8.6e8 BS=16 ACCUM=1 \
  bash scripts/run_capacity_ablation.sh > models/checkpoints/cap768.log 2>&1 &
```
768/12/12（109.6M）在**同样 8.6 亿 tokens** 下对比 512/10/8（47.9M）：若 hard CoT 显著提升则"容量墙"成立，
应继续扩宽/加深而非加步数；若持平则瓶颈在数据/监督方式。需单卡 ≥16GB（本仓库开发环境无 GPU，**该实验尚未执行**）。

## 💥 测试

```bash
pytest tests/             # 231 passed + 1 skipped，CPU 即可，无需数据与 GPU
```

覆盖注意力/模型结构/初始化/数据管线与切分/SFT 掩码/训练器（LR 调度、记账、续训语义、原子写、权重衰减分组）/
checkpoint 反推与宽容加载/评测脚本口径等。端到端验证（需 GPU）：`scripts/validate_pretrain.py`、`scripts/validate_ddp.py`。
环境排查：`python scripts/check_env.py`。

## 🤖 交给 Agent 自动训练（Skill）

仓库提供可被 agent 自动发现的 skill **`skills/minigpt-train/`**：agent 会读 `SKILL.md` 了解流程与决策树 →
跑 `preflight.sh` 体检 → 按预算执行 `pipeline.sh --smoke|--full`（自愈守护训练 + 空间守护 + 自动 SFT/CoT/评测 + 看板 `:8099`）
→ 参考 `references/` 的超参/吞吐换算与排障手册。

环境自适应：无 GPU → `cpu`（仅机制验证）；单卡 5.5–8GB → `gpu-small`（512/10/8、bs8、fp16+compile，22–30k tok/s）；
更大显存 → `gpu-mid`/`gpu-large`；多卡 → `multi-gpu`（torchrun DDP，lr×√N，步数按有效 batch 重算）。
可用 `--preset` 覆盖、`--hours` 指定预算自动换算步数、`--dry-run` 先看计划。

## 📄 License

[Apache-2.0](./LICENSE)
