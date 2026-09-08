"""集中管理训练超参数与路径。

此前这些数值散落在 pretrainer.py、各 notebook 的硬编码字符串里，读者需要逐个文件修改。
这里统一收敛，运行前只需改本文件即可。

说明：notebook 里的演示路径仍保留了绝对地址（它们是逐步讲解的教学代码），
正式训练脚本(pretrainer.py)已改为从这里读取。
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """预训练超参数。"""
    epochs: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 8
    train_ratio: float = 0.9997
    weight_decay: float = 0.01


@dataclass
class PathConfig:
    """数据集 / 模型 / tokenizer 路径，按需修改成本机路径。"""
    # 预训练数据集（.bin 二进制格式，由 texts_to_bin 生成）
    pretrain_dataset: str = "/data2/minigpt/dataset/pretrain/mobvoi_seq_monkey_general_open_corpus.bin"
    # SFT 指令微调数据集（.jsonl 格式）
    sft_dataset: str = "/data2/minigpt/dataset/sft/sft_data_zh.jsonl"
    # 分词器目录（分词器训练.ipynb 产出）
    tokenizer_dir: str = "/data2/minigpt/models/tokenizer_v3"
    # 模型输出目录
    output_dir: str = "/data2/minigpt/models/20241210"
    # 续训用的预训练 checkpoint（为空表示从头训练）
    last_checkpoint_path: str = "/data2/minigpt/models/20241210/checkpoint-450000.pth"


paths = PathConfig()
