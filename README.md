
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

训练前可用下面两张表粗略评估资源需求（默认架构：GELU 前馈 + 独立 QKV + 不共享输出头，vocab=32000）。

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
├── notebooks/                 # 15 个课程 notebook，01_~15_ 前缀按课程顺序排序
├── minigpt/                   # 可复用代码包，notebook 通过 %run minigpt/... 引用
│   ├── config.py              # 集中管理超参与路径（运行前改这里）
│   ├── model/
│   │   ├── attention.py       # 自注意力/多头注意力/FlashAttention/RoPE
│   │   └── transformer.py     # LayerNorm/FFN/TransformerBlock/GPTConfig/MiniGPT
│   ├── data/
│   │   ├── pretrain_dataset.py # 预训练二进制数据集(np.memmap)
│   │   └── sft_dataset.py     # SFT 指令数据集/损失掩码
│   └── train/
│       ├── trainer.py         # 训练器(单卡/DDP、混合精度、梯度累积)
│       ├── pretrainer.py      # 预训练入口(DDP)
│       └── pretrainer_single.py # 单卡教学版训练器
├── scripts/                   # 下载数据 / 训练分词器 / 验证 / 环境检查 / 启动
│   ├── download_data.sh       # 从 ModelScope 下载 pretrain_t2t_mini.jsonl
│   ├── train_tokenizer.py     # 训练 BPE 分词器（对应 notebook 01，支持 --max-lines 子集）
│   ├── validate_pretrain.py   # 单卡端到端预训练验证（对应 notebook 09/10）
│   ├── validate_ddp.py        # DDP 链路验证（nproc=1 单卡可跑，nproc=2 需真多卡）
│   ├── check_env.py           # 打印环境信息(Python/PyTorch/CUDA/GPU/CPU/包版本)
│   ├── estimate_resources.py  # 估算参数量/显存/起步 GPU/训练时长
│   └── pretrain_start.sh      # torchrun DDP 多卡启动
├── tests/                     # 单元测试(pytest，CPU 即可运行)
│   ├── conftest.py            # 共享 fixture(小模型/迷你分词器)
│   ├── test_attention.py      # 因果掩码/RoPE/padding 掩码/flash-attn 一致性
│   ├── test_transformer.py    # GPTConfig/MiniGPT 前向/pos_cis 扩展/生成
│   ├── test_data.py           # texts_to_bin/二进制数据集/数据集划分
│   ├── test_trainer.py        # 梯度范数/梯度累积
│   └── test_tokenizer.py      # 分词器编解码 round-trip/特殊 token
└── img/                       # 图片素材
```

> 涉及 `minigpt` 包的 notebook 开头都有一个「设置」单元格，会自动切到项目根目录，
> 因此无论从哪个目录启动 Jupyter，`%run minigpt/...` 都能正常找到代码。

## 💥 工程化说明

正式训练脚本的超参与路径已统一收敛到 [`minigpt/config.py`](./minigpt/config.py)，运行前只需改这一处即可，无需再逐个脚本/notebook 找硬编码地址。

模型与训练器都支持一些**前沿开关**（在 `GPTConfig` / `train_args` 中，默认关闭以兼容旧 checkpoint，开启即为前沿配置）：

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


最后，感谢您阅读这个教程。如果觉得对您有所帮助，可以考虑送我一杯奶茶作为鼓励😊

![a cup of tea](./img/cup_of_tea.jpg)

---

## 🔧 生产级训练/推理（非 notebook demo）

仓库除教学 notebook 外，提供一套按生产规范组织的训练/推理/验证链路（配置集中化、CLI 覆盖、
DDP/torchrun、AMP+梯度累积、checkpoint/RNG 恢复、tensorboard、采样推理、pytest）。

### 配置
- `minigpt/config.py`：`ModelConfig / DataConfig / TrainConfig / PathConfig` 默认值全部基于仓库根自动定位；
  任意入口都接受扁平 CLI 覆盖：`--model_emb_dim 384 --train_batch_size 8 --paths_output_dir ...`；
  运行配置会以 `config.json` 快照写入输出目录，可复现。

### 数据管线（jsonl → .bin）
```bash
python scripts/build_pretrain_bin.py build \
    --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
    --tokenizer-dir models/tokenizer_qwen2 \
    --out-bin dataset/bins/pretrain_qwen.bin --max-lines 600000
python scripts/build_pretrain_bin.py info --bin dataset/bins/pretrain_qwen.bin
```
自动按词表选择 uint16/uint32 并生成 `.meta.json`；`TokenBinDataset` 按元数据自动识别。

### 预训练
```bash
# 单卡
python3 -m minigpt.train.pretrainer \
    --model_emb_dim 384 --model_n_layers 8 --model_n_heads 8 --model_context_length 512 \
    --train_batch_size 8 --train_warmup_steps 800 --train_eval_steps 400 --train_save_steps 2000 \
    --paths_output_dir models/checkpoints/pretrain_qwen_v1
# 多卡 DDP
NPROC=2 bash scripts/pretrain_start.sh --paths_output_dir models/checkpoints/pretrain_ddp
```
产物：`checkpoint-{step}.pth`（含 optimizer/RNG/scaler）、`final.pt`（含 config，供推理）、
`tensorboard/`、`metrics.json`、`sample.txt`。续训：`--paths_last_checkpoint_path .../checkpoint-N.pth`。

### 采样推理
```bash
python scripts/generate.py --checkpoint models/checkpoints/pretrain_qwen_v1/final.pt \
    --tokenizer-dir models/tokenizer_qwen2 --prompt "什么是AI？" --chat \
    --do-sample --temperature 0.8 --top-k 50 --top-p 0.9 --repeat-penalty 1.1 \
    --max-new-tokens 200
# 或 --interactive / --prompt-file
```

### 评估与测试
```bash
python scripts/evaluate_pretrain.py --checkpoint models/checkpoints/pretrain_qwen_v1/final.pt \
    --tokenizer-dir models/tokenizer_qwen2 --bin dataset/bins/pretrain_qwen.bin --max-rows 2000
pytest tests/ -q        # 数据管线 / 采样 / config / 模型 / trainer 单元测试
```

### 说明
- `models/tokenizer_qwen2`：从 ModelScope 获取的 Qwen2.5-0.5B tokenizer（151,665 词表，现代中文 BPE，
  自带 `<|im_start|>/<|im_end|>` chat 模板），训练侧由此推导词表大小；默认启用输入/输出嵌入权重共享（`--model_tie_word_embeddings`）。
- RTX 20 系（Turing）不支持 FlashAttention-2，默认 `flash_attn=False`。
