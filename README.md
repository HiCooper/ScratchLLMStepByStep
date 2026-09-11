
<h1 align="center"> ScratchLLMStepByStep</h1>

欢迎来到这套全面的从零开始编写并训练大语言模型的教程！本项目旨在为对语言模型和深度学习感兴趣的开发者提供一套系统的、易于理解的学习资源。通过本系列教程，您将逐步了解并掌握大语言模型的基本概念、核心算法及其实现细节。

本教程将会带你从分词器训练开始，一步一步编写和实现自己的attention、transformer以及gptmodel，并对这个模型进行预训练、监督微调(SFT)，最终训练出一个可以进行对话聊天的大语言模型。

## 💥 目标受众

本教程适合具有以下背景的读者：
- 具备基本的编程知识，尤其是Python
- 对机器学习和深度学习有一定的了解
- 希望深入理解语言模型的工作原理和实现方法

## 💥 章节结构

- [带你从零训练tokenizer](./notebooks/01_分词器训练.ipynb)
- [词嵌入和位置嵌入](./notebooks/02_模型结构之词嵌入和位置编码.ipynb)
- [从零认识自注意力](./notebooks/03_模型结构之自注意力.ipynb)
- [实现因果注意力机制](./notebooks/04_模型结构之因果注意力.ipynb)
- [实现多头注意力](./notebooks/05_模型结构之多头注意力.ipynb)
- [构建TransformerBlock](./notebooks/06_模型结构之TransformerBlock.ipynb)
- [构建MiniGPT](./notebooks/07_模型结构之MiniGPT.ipynb)
- [预训练之高效数据加载](./notebooks/08_预训练之高效数据加载.ipynb)
- [预训练之从零起步](./notebooks/09_预训练之从零起步.ipynb)
- [预训练之运算加速](./notebooks/10_预训练之运算加速.ipynb)
- [预训练之多卡并行](./notebooks/11_预训练之多卡并行.ipynb)
- [SFT之分类微调](./notebooks/12_SFT之分类微调.ipynb)
- [SFT指令微调之数据处理](./notebooks/13_SFT指令微调之数据处理.ipynb)
- [SFT之指令微调训练](./notebooks/14_SFT指令微调之训练.ipynb)
- [模型推理之选词算法](./notebooks/15_模型推理之选词算法.ipynb)

## 💥 数据集

训练所需数据统一从 **ModelScope（魔搭）** 下载（国内可直连，避免 HuggingFace 网络问题）。本教程的分词器训练与预训练共用同一个文件 [`gongjy/minimind_dataset`](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)（即 HuggingFace `jingyaogong/minimind_dataset` 的镜像）里的 `pretrain_t2t_mini.jsonl`：

| 用途 | 文件 | 大小 | 说明 |
|---|---|---|---|
| 分词器训练 + 预训练 | `pretrain_t2t_mini.jsonl` | ~1.2GB | 中英混合文本，每行 `{"text": "..."}`（替代原 33GB 的 mobvoi 通用语料） |
| SFT | `sft_data_zh.jsonl` | — | 见下方链接 |

**一键下载**（下载到 `dataset/`）：

```bash
bash scripts/download_data.sh
```

手动下载（三选一）：
- 网页：打开 [gongjy/minimind_dataset 文件列表](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)，点击文件右侧「下载」按钮
- 命令行：`pip install modelscope && modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl`
- [SFT 数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl)

> 说明：原预训练数据 mobvoi 通用语料高达 33GB，体积大且难以获取；现改用同源的 `pretrain_t2t_mini.jsonl`（~1.2GB），其 `text` 字段格式与教程代码完全一致，`texts_to_bin` 时仍用 `content_key="text"`，无需改动任何代码逻辑。

> ⚠️ 分词器训练的注意点：`pretrain_t2t_mini.jsonl` 是**长段落**文本，字节级 BPE 预分词会把整段中文当成一个超长「词」，导致训练内存暴涨（全量 1.2GB 在 16GB 内存的机器上会 OOM）。分词器训练只需有代表性的语料子集即可，推荐用 `scripts/train_tokenizer.py --max-lines 150000`（约 150MB）训练，避免内存爆炸。

## 💥 运行环境

仅是我个人的软硬件环境配置，自行酌情更改：

* Ubuntu == 18.04
* Python == 3.10
* Pytorch == 2.4.0
* CUDA == 12.1

前面编写模型结构的部分对GPU不是强依赖，后面预训练、SFT需要使用GPU进行训练，并且尽量是多块GPU（个人使用的4块24G的GPU进行训练）。

有两篇配套的环境搭建教程可以作为参考：
- [conda&pytorch环境搭建笔记](https://golfxiao.blog.csdn.net/article/details/140819506)
- [cuda安装笔记](https://golfxiao.blog.csdn.net/article/details/140877932)

依赖安装见 [`requirements.txt`](./requirements.txt)。

## 💥 模型规模与资源估算

训练前可用下面两张表粗略评估资源需求（默认架构：GELU 前馈 + 独立 QKV + **共享输入/输出嵌入**，vocab=32000。
注意：`config.py` 的 `ModelConfig.tie_word_embeddings` 默认 **True**，因此默认参数量比"不共享输出头"少 `vocab×emb_dim`；
下表按**不共享**口径给出，便于与历史数据对照）。

**参数量估算公式**：

```
参数量 ≈ 2 × vocab × emb_dim + n_layers × 12 × emb_dim²
```

**显存估算**：fp16 混合精度 + AdamW 训练时，权重/梯度/优化器状态合计约 **16 字节/参数**（fp16 模型 2 + fp16 梯度 2 + fp32 主权重 4 + 动量 4 + 方差 4）；实际还需再加激活显存（随 batch×seq 增加），故起步 GPU 要留余量。

| 档位 | emb_dim | n_layers | n_heads | 参数量 | 权重+优化器显存 | 起步 GPU |
|---|---|---|---|---|---|---|
| 微型 | 256 | 6 | 8 | 21M | ~0.34GB | 4GB |
| 小型 | 512 | 8 | 8 | 58M | ~0.93GB | 6GB |
| 中型（教程默认） | 768 | 12 | 12 | 134M | ~2.15GB | 8GB（6GB 可小 batch） |
| 大型 | 1024 | 24 | 16 | 368M | ~5.9GB | 16GB |
| 超大 | 1280 | 32 | 20 | 711M | ~11.4GB | 24GB |

> 注：开启前沿开关（`use_swiglu`/`qkv_merged`/`tie_word_embeddings`）会改变参数量——SwiGLU 每层多 4×emb² 参数，权重共享省掉输出头 vocab×emb；本表档位下 SwiGLU 的增量通常大于权重共享的节省，参数量略有上升。

> ⚠️ 口径说明：本表的「微型/大型/超大」是相对**从零训练、消费级 GPU**的教程语境而言；绝对尺度上 711M 仍属「边缘/入门」级（`M`=百万、`B`=十亿，相差 1000 倍）。主流开源模型从 **7B 起跳**（如 Qwen3-8B 为 8B、emb 4096、36 层、GQA），前沿 MoE 模型（DeepSeek-V3 671B/激活 37B、Kimi K2 1T/激活 32B）总参数更达千亿~万亿级。

**训练时长参考**（以教程默认 134M 模型、1.2GB 数据 ≈ 2 亿 token 的 1 个 epoch 为例，fp16）：

| 单卡 GPU | 相对吞吐(约) | 1 epoch 参考时长 |
|---|---|---|
| RTX 2060 6GB | 1× | ~20 小时 |
| RTX 3060 / 4060 | ~3× | ~7 小时 |
| RTX 3090 / 4090 | ~10× | ~2 小时 |

> 以上时长基于本机 RTX 2060 实测（134M、fp16 ≈ 2700 token/s）按算力线性外推，仅作量级参考。完整预训练需多个 epoch（教程的 450K step ≈ 20 epoch），总时长按 epoch 数线性放大；换更大数据则再按比例放大。

> 也可用 `python scripts/estimate_resources.py` 交互式估算——输入配置与目标 GPU，自动输出参数量/显存/起步 GPU/训练时长。

## 💥 目录结构

```
├── AGENTS.md                  # agent 入口：最短路径、环境预设、训练相关事实、硬性约定
├── pyproject.toml             # 打包/依赖/pytest 配置（pip install -e . 后可直接 import minigpt）
├── requirements.txt           # 与 pyproject.toml 等价的依赖清单（给 pip install -r 的用户，勿单独改）
├── notebooks/                 # 15 个课程 notebook，01_~15_ 前缀按课程顺序排序
├── minigpt/                   # 可复用代码包
│   ├── README.md              # 包使用说明：指标面板、推理/评估、吞吐、续训注意事项
│   ├── config.py              # RunConfig（model/data/train/paths 四段）+ CLI 覆盖 + 布尔解析
│   ├── model/
│   │   ├── attention.py       # 自注意力/多头注意力/FlashAttention/RoPE（KV cache 的因果判据）
│   │   ├── transformer.py     # LayerNorm/RMSNorm/FFN/TransformerBlock/GPTConfig/MiniGPT(+_init_weights)
│   │   ├── checkpoint.py      # 重建模型（config 优先→形状反推）+ 宽容权重加载器
│   │   └── generation.py      # 停止串/思考模式（两阶段 CoT）生成工具
│   ├── data/
│   │   ├── pretrain_dataset.py # .bin(memmap/uint16-uint32) + 分块验证集切分 + 词表一致性校验
│   │   └── sft_dataset.py     # SFT 数据集/逐轮 assistant 掩码/停止符解析
│   └── train/
│       ├── trainer.py         # 训练循环：AMP/梯度累积/eval/save/resume/DDP/compile 编排
│       ├── schedule.py        # LR 调度（warmup+cosine）与 loss 记账（纯函数，可单测）
│       ├── ddp_utils.py       # DDP 包装/解包/跨 rank 平均（纯工具，可在 CPU 单测）
│       ├── checkpoint_io.py   # checkpoint payload 读写 + RNG/缩放器恢复
│       ├── metrics.py         # TensorBoard 面板（标量/直方图/图像/投影/样本）
│       ├── pretrainer.py      # 预训练入口（生产，支持 DDP）
│       ├── sft_trainer.py     # SFT/指令微调入口
│       └── pretrainer_single.py # 单卡教学版训练器（notebook 09/10 使用）
├── scripts/                   # 数据 / 训练 / 评测 / 监控
│   ├── download_data.sh       # 从 ModelScope 下载 pretrain_t2t_mini.jsonl
│   ├── train_tokenizer.py     # 训练 BPE 分词器（对应 notebook 01，支持 --max-lines 子集）
│   ├── parquet_to_jsonl.py    # 领域 parquet 语料转 jsonl（可混入通用语料防遗忘）
│   ├── build_pretrain_bin.py  # jsonl -> .bin + .meta.json（build / info，词表 >65535 自动 uint32）
│   ├── build_cot_sft.py       # 生成 SFT / CoT 数据（--profile easy|hard，保证 train/eval 零重合）
│   ├── evaluate_pretrain.py   # loss/perplexity（默认 --split val，与训练同一口径）
│   ├── eval_thinking.py       # 思考模式评测（plain/single/two-phase）
│   ├── generate.py            # 推理入口（--chat / --thinking）
│   ├── train_dashboard.py     # 训练看板（--once 终端 / --serve 浏览器）
│   ├── report_training.py     # 汇总 metrics.json 生成 TRAINING_REPORT.md（含"评估口径"列）
│   ├── chat_probe.py          # 对话能力探针（多场景 × 贪心/采样 + 自动质检）
│   ├── check_env.py           # 打印环境信息(Python/PyTorch/CUDA/GPU/CPU/包版本)
│   ├── estimate_resources.py  # 估算参数量/显存/起步 GPU/训练时长
│   ├── validate_pretrain.py   # 单卡端到端预训练验证（对应 notebook 09/10）
│   ├── validate_ddp.py        # DDP 链路验证（nproc=1 单卡可跑，nproc=2 需真多卡）
│   ├── train_pretrain_resilient.sh  # 崩溃自动续训的长训守护（推荐入口）
│   ├── run_capacity_ablation.sh     # isotoken 容量消融：768/12/12 vs 512/10/8
│   ├── checkpoint_janitor.sh  # checkpoint 空间守护（每目录保留最近 N 个）
│   ├── run_downstream.sh / run_downstream_evals.sh  # SFT + 下游评测流水线
│   ├── run_code_domain.sh / run_domain_compare.sh   # 领域增量续训与双口径 ppl 对比
│   ├── wait_and_run_downstream.sh / watch_training.sh
│   └── pretrain_start.sh      # torchrun DDP 多卡启动
├── skills/minigpt-train/      # agent skill（DSH 会扫描 skills/）
│   ├── SKILL.md               # 完整流程、配置决策树、监控与汇报模板
│   ├── references/            # training-recipes.md（超参/吞吐/预算换算）、troubleshooting.md
│   └── scripts/               # preflight.sh / preflight.py 环境体检、pipeline.sh 一键流水线
├── tests/                     # 单元测试(pytest，CPU 即可运行，不需要数据与 GPU)
│   ├── conftest.py            # 共享 fixture(小模型/迷你分词器) + sys.path 注入
│   ├── test_attention.py      # 因果掩码/RoPE/padding 掩码
│   ├── test_transformer.py    # 前向/pos_cis 扩展/生成 + KV-cache 与全量前向等价性
│   ├── test_init.py           # 权重初始化、norm_type/SwiGLU 宽度/head bias/rope_theta
│   ├── test_ddp_unwrap.py     # DDP+torch.compile 解包与 checkpoint 键名
│   ├── test_train_modules.py  # schedule / ddp_utils / checkpoint_io
│   ├── test_trainer.py        # 梯度范数/累积/LR 调度/eval 触发/loss 记账/续训语义
│   ├── test_metrics.py        # TensorBoard 面板（含 grads/* 与 tokens_per_sec 回归）
│   ├── test_sft_masking.py    # 逐轮 assistant 掩码/padding/停止符
│   ├── test_data.py / test_data_bin.py  # 文本与二进制数据集、memmap、分块切分、词表校验
│   ├── test_config.py         # CLI 覆盖/布尔解析/config.json 往返/默认值一致性
│   ├── test_checkpoint_loader.py     # 结构反推与宽容加载
│   ├── test_build_cot_sft.py  # CoT 训练/评测零重合
│   ├── test_eval_scripts.py   # 评测脚本端到端 + --split 口径
│   ├── test_generation.py / test_thinking.py / test_tokenizer.py
│   ├── test_report.py / test_parquet_to_jsonl.py / test_chat_probe.py
│   └── test_scripts_syntax.py # 所有 shell/python 脚本语法可编译
├── dataset/                   # 语料与派生 .bin（.gitignore，不进仓库）
├── models/                    # 分词器与训练产物 checkpoint（.gitignore，不进仓库）
└── img/                       # 图片素材
```

> 各包目录下的 `__init__.py` 仅为包标记（空文件），未在上图逐个列出。

> ⚠️ **评测口径**：`scripts/evaluate_pretrain.py` 默认 `--split val`，即在 `.bin` 上按「均匀分块 + 双端对齐文档边界」切出的验证集（与训练时 `split_train_eval_blocks` 同一口径，见 `metrics.json:data_split`）。
> 不要用 `--split all --max-rows N` 去评估**参与过训练**的 bin：那取的是文件最前面的窗口，得到的其实是训练集 loss，且本语料按领域排序、头尾分布差异极大（实测同一 checkpoint 头/尾 ppl 可差 3 倍以上），不同 `--max-rows` 之间也不可比。

> 涉及 `minigpt` 包的 notebook 开头都有一个「设置」单元格，会自动切到项目根目录，
> 因此无论从哪个目录启动 Jupyter，`%run minigpt/...` 都能正常找到代码。
>
> 生产脚本建议 `pip install -e .`（见 `pyproject.toml`）；不安装时在仓库根目录加 `PYTHONPATH=.` 亦可。

## 💥 工程化说明

正式训练脚本的超参与路径已统一收敛到 [`minigpt/config.py`](./minigpt/config.py)，`config.py` 的默认值即文档所述的生产基线（**512/10/8 + ctx512 + `tokenizer_v3`(32k) + fp16 + 嵌入共享**）：

```bash
pip install -e .            # 或 PYTHONPATH=. python -m minigpt.train.pretrainer ...
python3 -m minigpt.train.pretrainer --paths_output_dir models/checkpoints/xxx --train_max_steps 200000
```

CLI 覆盖使用扁平前缀（`--model_emb_dim` / `--data_tokenized_bin` / `--train_batch_size` / `--paths_output_dir`），布尔参数写成 `--train_torch_compile True|False`（**必须带值**，`false/0/no/off` 均可解析）；
也可以 `--config-file <dump 出的 config.json>` 读回整份配置，CLI 优先级更高。

模型与训练器都支持一些**前沿开关**（在 `GPTConfig` / `train_args` 中；`tie_word_embeddings` 与 `qkv_merged` 见下，其余默认关闭以兼容旧 checkpoint，开启即为前沿配置）：

| 开关 | 位置 | 说明 |
|---|---|---|
| `use_swiglu` | GPTConfig | 前馈层用 SwiGLU 门控代替 GELU |
| `qkv_merged` | GPTConfig | Q/K/V 投影合并为单个 `Linear(dim, 3*dim)` |
| `tie_word_embeddings` | GPTConfig | 词嵌入与输出头共享权重 |
| `use_checkpoint` | GPTConfig | 训练时对 FFN 做激活重计算省显存 |
| `flash_attn` | GPTConfig | 用 FlashAttention 加速注意力 |
| `mixed_precision_dtype` | train_args | `float16` / `bfloat16`（bf16 无需 GradScaler） |
| `gradient_accumulation_steps` | train_args | 梯度累积，用小显存训练大 batch |

> 注意：开启 `use_swiglu`/`qkv_merged`/`tie_word_embeddings` 会改变模型结构，与旧版（GELU/独立 QKV/不共享权重）训练出的 checkpoint 不兼容；加载旧 checkpoint 请保持这些开关关闭。

> ⚠️ FlashAttention 硬件约束：`flash_attn=True` 依赖 flash-attn 库的 FlashAttention-2 内核，**只支持 Ampere（sm_80）及以上 GPU**（如 A100、RTX 30/40/50 系列）。旧卡（如 Turing 的 RTX 2060/1660、Pascal/V100 等）无法运行，会直接报错；这类显卡请保持 `flash_attn=False` 走标准注意力实现。

## 💥 测试

**单元测试**（`tests/`，CPU 即可运行，无需数据与 GPU）：

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -q
```

覆盖注意力、模型结构、数据、训练器、分词器等核心逻辑，作为后续迭代的回归测试。

**端到端验证**（需 GPU + 数据，非单元测试）：
- `scripts/validate_pretrain.py`：单卡端到端预训练验证
- `scripts/validate_ddp.py`：DDP 链路验证（`nproc=1` 单卡可跑，`nproc=2` 需真多卡）

**环境检查**：`python scripts/check_env.py` 打印 Python/PyTorch/CUDA/GPU/CPU/内存及主要包版本，便于排查硬件/环境问题。

## 💥 如何开始？
1. 克隆本项目到本地：
```
git clone https://github.com/golfxiao/ScratchLLMStepByStep.git
```
2. 下载上面列出的依赖数据集，将notebook中用到的数据集地址修改成你本地地址
3. 按照顺序阅读每个Notebook，并运行其中的代码。
4. 根据需要修改和实验代码，以加深对相关概念的理解。

---

## 🔧 生产级训练/推理（非 notebook demo）

仓库除教学 notebook 外，提供一套按生产规范组织的训练/推理/验证链路（配置集中化、CLI 覆盖、
DDP/torchrun、AMP+梯度累积、checkpoint/RNG 恢复、tensorboard、采样推理、pytest）。
packages 级说明与**训练过程指标（标量/直方图/图像/模型图/嵌入投影/文本）**、**思考模式（CoT）** 详见 [`minigpt/README.md`](./minigpt/README.md)。

### 配置
- `minigpt/config.py`：`ModelConfig / DataConfig / TrainConfig / PathConfig` 默认值全部基于仓库根自动定位；
  任意入口都接受扁平 CLI 覆盖：`--model_emb_dim 384 --train_batch_size 8 --paths_output_dir ...`；
  运行配置会以 `config.json` 快照写入输出目录，可复现。

### 数据管线（jsonl → .bin）
```bash
python scripts/build_pretrain_bin.py build \
    --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
    --tokenizer-dir models/tokenizer_v3 \
    --out-bin dataset/bins/pretrain_v3_full.bin
python scripts/build_pretrain_bin.py info --bin dataset/bins/pretrain_v3_full.bin
```
自动按词表选择 uint16/uint32 并生成 `.meta.json`；`TokenBinDataset` 按元数据自动识别。

### 预训练
```bash
# 单卡
python3 -m minigpt.train.pretrainer \
    --model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 \
    --train_batch_size 8 --train_warmup_steps 300 --train_eval_steps 2000 --train_save_steps 4000 \
    --train_torch_compile True \
    --paths_output_dir models/checkpoints/pretrain_v3
# 多卡 DDP
NPROC=2 bash scripts/pretrain_start.sh --paths_output_dir models/checkpoints/pretrain_ddp
```
产物：`checkpoint-{step}.pth`（含 optimizer/RNG/scaler）、`final.pt`（含 config，供推理）、
`tensorboard/`、`metrics.json`、`sample.txt`。续训：`--paths_last_checkpoint_path .../checkpoint-N.pth`。

### 采样推理
```bash
python scripts/generate.py --checkpoint models/checkpoints/pretrain_v3/final.pt \
    --tokenizer-dir models/tokenizer_v3 --prompt "什么是AI？" --chat \
    --do-sample --temperature 0.8 --top-k 50 --top-p 0.9 --repeat-penalty 1.1 \
    --max-new-tokens 200
# 或 --interactive / --prompt-file
```

### 评估与测试
```bash
# --split val：与训练同一口径的验证集（均匀分块 + 双端对齐文档边界）
python scripts/evaluate_pretrain.py --checkpoint models/checkpoints/pretrain_v2_full/final.pt \
    --tokenizer-dir models/tokenizer_v3 --bin dataset/bins/pretrain_v3_full.bin --split val
pytest tests/ -q        # 数据管线 / 采样 / config / 模型 / trainer 单元测试
```

### 领域增量预训练与报告

外部行业语料（parquet/jsonl）应走「增量预训练 → 再 SFT」，而不是直接 SFT：

```bash
# 1) 转换 + 过滤 + 混入 15% 通用语料（防灾难性遗忘）+ token 估算
python scripts/parquet_to_jsonl.py --src dataset/IndustryCorpus2_computer_programming_code_high \
  --out dataset/domain/code_corpus.jsonl --min-chars 300 --max-chars 8000 --max-line-length 500 \
  --min-quality 3.0 --mix-jsonl dataset/pretrain_t2t_mini.jsonl --mix-ratio 0.15 --mix-limit 300000
# 2) 建 bin
python scripts/build_pretrain_bin.py build --corpus-jsonl dataset/domain/code_corpus.jsonl \
  --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/code_domain.bin --max-lines 0
# 3) 一键全自动：等 GPU 空闲 → 增量续训(lr=1e-4) → 复跑 SFT/CoT(TAG=domain) → 生成报告
setsid nohup bash scripts/run_code_domain.sh > models/checkpoints/domain_pipeline.log 2>&1 < /dev/null &
```
产物：`models/checkpoints/pretrain_domain_code/`、`sft_domain_*/`、`eval_domain_cot_*.json`、
`ppl_*.json`（多基座同切分对比），报告汇总在 `models/checkpoints/TRAINING_REPORT.md`。

### 说明
- `models/tokenizer_v3`：项目**唯一**使用的分词器（32,000 词表中文 BPE，由 `scripts/train_tokenizer.py`
  在 notebook 01 训练得到，自带 `<|im_start|>/<|im_end|>` chat 模板）。训练侧由它推导词表大小，
  并与 `.bin` 的 `meta.vocab_size` 做一致性校验；默认启用输入/输出嵌入权重共享（`--model_tie_word_embeddings`）。
  （早期实验用的 `tokenizer_qwen2`（151k 词表）已随其产物一并清理：151k 词表会把 70%+ 参数花在
  embedding 上，实测 512/10/8 下从 47.9M 涨到 109.3M，不划算。）
- RTX 20 系（Turing）不支持 FlashAttention-2，默认 `flash_attn=False`。

## 🏋️ 实测训练结果（单卡 RTX 2060 6GB，2026-09）

### 分词器取舍（实测吞吐驱动）
本机实测不同词表下 `512/10/8/ctx512` 的步耗与吞吐：

| tokenizer | vocab | 单步(s) | 吞吐(tok/s) | 说明 |
|---|---|---|---|---|
| Qwen2.5（modelscope 下载） | 151,665 | ~2.01 | ~2.0k | 6GB 卡一 epoch(~77M tokens) 需 ~10h，不可行 |
| 本仓库重训中文 BPE（`models/tokenizer_v3`，400k 行语料） | 32,000 | ~0.21 | ~19.3k | 与 notebook 词表 32000 兼容，采用 ✅ |

结论：模型词表由 tokenizer 推导（config 可切回 Qwen2.5 在任何更大显存机器上使用）。

### 产物
- 预训练 v1：`models/checkpoints/pretrain_v1_512/`
  - 语料：600,000 行 → `dataset/bins/pretrain_v3_600k.bin`（67.7M tokens，uint16+meta）
  - 模型：384→实际 512/10/8，47.9M 参数，ctx512，嵌入共享，fp16+scaler，3 epochs / 49,509 步
  - 结果：eval_loss ≈ 3.51（perplexity ≈ 33.3）；周期 checkpoint-{9000,18000,27000,36000,45000}.pth、
    `final.pt`（含 config）、tensorboard/、metrics.json、sample.txt
- 预训练 v2（10 小时满负荷）：`models/checkpoints/pretrain_v2_full/`
  - 语料：全量 1,269,916 行 → `dataset/bins/pretrain_v3_full.bin`（**247.5M tokens**，uint16+meta）
  - 从 v1 续训 6 个 epoch / **211,000 步**（≈ 8.6 亿训练 tokens），cosine 退火 + grad clip 1.0 + `torch.compile`
  - 结果：eval_loss **2.8188**（perplexity **16.76**），训练 loss 1.396
- SFT：`models/checkpoints/sft_v1_512/`（v1 基座）

### 基座对比（历史口径：`--max-rows 512` 取 .bin 最前面的窗口）

> ⚠️ **口径警告（2025-09 修订）**：下表是历史数据，用 `evaluate_pretrain.py --max-rows 512` 测的，那取的是 `.bin` **最前面**的 512 个窗口——对参与过训练的 bin 而言就是**训练集 loss**，不是泛化指标；而且本语料按领域排序（头部中文创作/指令、尾部英文选择题），实测同一 checkpoint 取头部 vs 取验证集 ppl 可差 3 倍以上（v2：头部 23.40 / 尾部 6.78 / 全局分块验证集见下表）。
> **同一批窗口内的横向对比仍然可用**（三个基座测的是同一批窗口），但绝对数值不代表泛化能力。
> 新版协议：`scripts/evaluate_pretrain.py --split val`（均匀分块 + 双端对齐文档边界，与训练同一口径，切分范围记录在 `metrics.json:data_split`）。

| 基座 | 训练步数 | 训练 tokens | 通用 eval_loss（旧口径） | 通用 ppl（旧口径） | 代码领域 ppl（旧口径） | **通用 ppl（新协议）** | **代码领域 ppl（新协议）** |
|---|---|---|---|---|---|---|---|
| `pretrain_v1_512` | 49,509 | 0.68 亿 | 3.6323 | 37.80 | — | **33.56** | — |
| `pretrain_v2_full` | 211,000 | 8.6 亿 | **3.1637** | **23.66** | 80.79 | **15.82** | 84.56 |
| `pretrain_domain_code` | +26,847（领域续训） | +1.1 亿 | 3.5095 | 33.43 | **28.72** | **22.93** | **28.02** |

> 新协议 = `--split val`（均匀分块 + 双端对齐文档边界，497 个窗口，与训练集**文档级零重叠**）。
> 结论方向与旧口径一致但幅度更清晰：v1→v2 通用 ppl **33.56 → 15.82**；领域续训把代码领域 ppl
> **84.56 → 28.02（−67%）**，代价是通用 ppl **15.82 → 22.93（+45%）**。

10 小时把同口径 ppl 从 37.8 降到 23.7（新协议 33.6 → 15.8），是本仓库"规模换质量"最直接的一组数据。

**领域增量预训练（`dataset/IndustryCorpus2_computer_programming_code_high`）**：74 分钟、lr=1e-4、1 epoch、
混入 15% 通用语料，把代码领域 ppl 从 **84.56 打到 28.02（−67%）**；但代价是通用 ppl 从 15.82 退到 22.93
（**+45%**）——15% 混料不足以抵消 1e-4 一步到位的偏移，属于典型的部分灾难性遗忘。下游对话 SFT 的 eval_loss
也从 2.7795 升到 2.9087，样例里甚至出现"抱歉，我无法回答这个问题。但我可以告诉你关于计算机程序的语法和编程模型"。
结论：**领域自适应必须双口径验收**（`bash scripts/run_domain_compare.sh`），并按需降低 lr（1e-5~3e-5）、
提高混料比（30%+）或减少步数；本次数据保留下来正是为了说明"领域增益 ≠ 无损"。

### ⚠️ CoT 评测集口径修正（重要）

`scripts/build_cot_sft.py` 旧实现用"换 seed"来避免训练/评测重合，但 **easy profile 的操作数只有 1..9，
全部唯一题目仅 1063 个**，而训练集有数万行——实测旧 `cot_eval_easy_zh.jsonl` 的 156 条题目
**100%** 出现在 60k 训练集里，旧 `cot_eval_zh.jsonl` 也有 **33.5%** 重合。
也就是说下面表中的"easy 90%→100%"测的是**记忆**，不是泛化。

现已修复：生成器改为"先去重建池、再把留出集切出来"，并在落盘前自检零重合；同时提供两份
**与训练集零重合**的留出集：

```bash
# 与 sft_cot_easy_60k.jsonl 零重合（用 --easy-max 20 扩大题目空间）
dataset/sft/cot_eval_easy_disjoint.jsonl
# 与 sft_cot_hard_80k.jsonl 零重合
dataset/sft/cot_eval_hard_disjoint.jsonl
```

`scripts/run_downstream_evals.sh` 已优先使用这两份文件。**下表中的 easy 100% 需要在 GPU 上用新留出集重测**；
`cot_eval_easy_disjoint.jsonl` 对应的训练集是 `sft_cot_easy_disjoint60k.jsonl`（9500+ 唯一题目），
需要用 `scripts/build_cot_sft.py --profile easy --easy-max 20` 重新生成配套训练集后再 SFT。

### 下一步实验：容量 vs 步数（isotoken 消融）

hard CoT（多位数运算）的瓶颈究竟是**容量**还是**训练量**，可以用同 token 预算下的扩容消融直接回答：

```bash
# 512/10/8 = 47.9M（已有基线 pretrain_v2_full，8.6 亿 tokens）
# 768/12/12 = 109.6M，同样 8.6 亿 tokens => 52,490 步 @ bs16×ctx1024
DRY_RUN=1 bash scripts/run_capacity_ablation.sh          # 只打印命令，可在无 GPU 环境检查
setsid nohup env TAG=cap768 TOKENS=8.6e8 BS=16 ACCUM=1 \
  bash scripts/run_capacity_ablation.sh > models/checkpoints/cap768.log 2>&1 &
```

判据：若 cap768 在 `cot_eval_hard_disjoint.jsonl` 上显著超过 512/10/8，则"容量墙"成立，
应继续扩宽/加深而不是加步数；若持平，则瓶颈在数据/监督方式（例如需要 scratchpad 式多步监督）。

**⚠️ 该实验需要单卡 ≥16GB，本仓库开发环境无 GPU，脚本只完成到"命令行拼装 + 语法检查"，
训练尚未执行。**

### 下游 SFT / CoT（v2 基座，`models/checkpoints/sft_v2_*`）

| run | 数据 | 步数 | eval_loss |
|---|---|---|---|
| `sft_v2_chat` | `sft_data_zh.jsonl` 60k × 2 epochs | 14,700 | 2.7795 |
| `sft_v2_cot_easy` | `sft_cot_easy_60k.jsonl`（45k 合成 + 15k 通用）× 2 epochs | 14,700 | 1.7397（best 1.6109@12499） |
| `sft_v2_cot_hard` | `sft_cot_hard_80k.jsonl`（60k 合成 + 20k 通用）× 3 epochs | 29,400 | 1.3565 |

思考模式留出集（60 题，贪心解码，`--repetition-penalty 1.0`）：

| 任务档次 | 模型 | plain | single（单通道 CoT） | two-phase |
|---|---|---|---|---|
| **easy**（1~2 位数单/两步） | 预训练基座（v1） | 0.0% | 1.7% | 1.7% |
| | CoT-SFT v1（20k 条） | 90.0% | 90.0% | 90.0% |
| | **CoT-SFT v2（60k 条 × 2ep）** | **100.0%** | **100.0%** | **98.3%** |
| | CoT-SFT domain（代码领域基座） | **100.0%** | **100.0%** | **98.3%** |
| **hard**（多位数四则 + 应用题） | CoT-SFT v1（30k 条） | 1.7% | 3.3% | 3.3% |
| | CoT-SFT v2（80k 条 × 3ep） | 1.7% | 3.3% | 3.3% |
| | CoT-SFT domain | 1.7% | 1.7% | 1.7% |

结论：**更强基座 + 更多 CoT 数据把 easy 从 90% 推到 100%**；hard 仍是容量墙（47.9M 参数在多位数乘加上算不对），
需要放大模型或走「工具调用范式」（模型只生成算式，由 Python 结算）。领域基座在 easy 上追平 v2（窄任务靠 SFT
重新学会），但 hard 与对话质量双双下滑——**领域增益没有换来推理能力**。详见 `minigpt/README.md` §8。

### 生成示例（SFT 后，温度0.8/top-k50/top-p0.92）
```
用户: 什么是AI？
模型: AI（人工智能）是一种模拟人类智能的技术，它允许计算机执行特定的任务，
      例如语音识别、图像识别、自然语言处理、机器翻译、智能控制等。……
用户: 用一句话解释机器学习。
模型: 机器学习是人工智能的一个分支，它让计算机从数据中学习，并根据规律进行预测和决策。
```
更多见 `models/checkpoints/sft_v1_512/final_samples.txt`、`models/checkpoints/samples_v2_chat.txt`、
`models/checkpoints/samples_v2_cot.txt`。

## 🤖 交给 Agent 自动训练（Skill）

本仓库根目录提供可被 agent 自动发现的 skill：**`skills/minigpt-train/`**（DSH 会扫描 `skills/`、`.dsh/skills/`、`.agents/skills/`）。
把仓库交给 agent 后，它会：

1. 读 `skills/minigpt-train/SKILL.md` 了解训练流程与决策树；
2. 运行 `bash skills/minigpt-train/scripts/preflight.sh` 做环境体检（依赖/GPU/显存/磁盘/语料/分词器/服务/单测，输出 JSON）；
3. 按预算执行 `bash skills/minigpt-train/scripts/pipeline.sh --smoke|--full`：
   - `--smoke`：约 5 分钟小模型端到端冒烟（含显存守卫）；
   - `--full`：全量语料 `.bin`（缺则构建）→ 预训练（自愈守护 + `torch.compile` + 空间守护）→ 自动 SFT/CoT/评测 → 看板 `:8099`；
4. 参考 `references/training-recipes.md`（超参/吞吐/预算换算）与 `references/troubleshooting.md`（OOM/续训/速度/磁盘等排障）自行优化。

**环境自适应**：skill 会按硬件自动选预设并调整参数——
无 GPU → `cpu` 预设（128/2/4、ctx128、fp32、2 万行小语料，仅机制验证，实测 0.12–0.17k tok/s）；
单卡 5.5–8GB → `gpu-small`（512/10/8、ctx512、bs8、fp16 + compile，22–30k tok/s）；
单卡更大显存 → `gpu-mid`/`gpu-large`；**多卡 → `multi-gpu`**（torchrun DDP、每卡 batch 不变、lr×√N、步数按有效 batch 重算）。
可用 `--preset` 手动覆盖、`--hours` 指定预算自动换算目标步数、`--dry-run` 先看计划。

无需人工执行脚本；agent 会后台运行长任务、轮询 `scripts/train_dashboard.py --once`，并按 skill 中的报告模板给出产物路径与指标对比。
