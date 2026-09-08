
<h1 align="center"> ScratchLLMStepByStep</h1>

欢迎来到这套全面的从零开始编写并训练大语言模型的教程！本项目旨在为对语言模型和深度学习感兴趣的开发者提供一套系统的、易于理解的学习资源。通过本系列教程，您将逐步了解并掌握大语言模型的基本概念、核心算法及其实现细节。

本教程将会带你从分词器训练开始，一步一步编写和实现自己的attention、transformer以及gptmodel，并对这个模型进行预训练、监督微调(SFT)，最终训练出一个可以进行对话聊天的大语言模型。

## 💥 目标受众

本教程适合具有以下背景的读者：
- 具备基本的编程知识，尤其是Python
- 对机器学习和深度学习有一定的了解
- 希望深入理解语言模型的工作原理和实现方法

## 💥 章节结构

- [带你从零训练tokenizer](./01_分词器训练.ipynb)
- [词嵌入和位置嵌入](./02_模型结构之词嵌入和位置编码.ipynb)
- [从零认识自注意力](./03_模型结构之自注意力.ipynb)
- [实现因果注意力机制](./04_模型结构之因果注意力.ipynb)
- [实现多头注意力](./05_模型结构之多头注意力.ipynb)
- [构建TransformerBlock](./06_模型结构之TransformerBlock.ipynb)
- [构建MiniGPT](./07_模型结构之MiniGPT.ipynb)
- [预训练之高效数据加载](./08_预训练之高效数据加载.ipynb)
- [预训练之从零起步](./09_预训练之从零起步.ipynb)
- [预训练之运算加速](./10_预训练之运算加速.ipynb)
- [预训练之多卡并行](./11_预训练之多卡并行.ipynb)
- [SFT之分类微调](./12_SFT之分类微调.ipynb)
- [SFT指令微调之数据处理](./13_SFT指令微调之数据处理.ipynb)
- [SFT之指令微调训练](./14_SFT指令微调之训练.ipynb)
- [模型推理之选词算法](./15_模型推理之选词算法.ipynb)

## 💥 数据集
相关训练所需数据集的下载地址。
- [分词器训练数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)
- [预训练数据集](http://share.mobvoi.com:5000/sharing/O91blwPkY)
- [SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl)

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

## 💥 目录结构

可复用的代码已整理为 `minigpt` 包，notebook 通过 `%run minigpt/...` 引用：

```
minigpt/
├── config.py                  # 集中管理超参与路径（运行前改这里）
├── model/
│   ├── attention.py           # 自注意力/多头注意力/FlashAttention/RoPE
│   └── transformer.py         # LayerNorm/FFN/TransformerBlock/GPTConfig/MiniGPT
├── data/
│   ├── pretrain_dataset.py    # 预训练二进制数据集(np.memmap)
│   └── sft_dataset.py         # SFT 指令数据集/损失掩码
└── train/
    ├── trainer.py             # 训练器(单卡/DDP、混合精度、梯度累积)
    ├── pretrainer.py          # 预训练入口(DDP)
    └── pretrainer_single.py   # 单卡教学版训练器
```

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
