"""pytest 共享 fixture 与路径配置。

运行方式（在项目根目录）：
    PYTHONPATH=. pytest tests/ -q
"""
import json
import sys
from pathlib import Path

# 确保测试能 import minigpt 包，以及 scripts/ 下的工具脚本（如 train_tokenizer.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest
from transformers import AutoTokenizer

from minigpt.model.transformer import GPTConfig


@pytest.fixture
def small_config():
    """一个足够小、可在 CPU 上快速前向/反向的模型配置（确定性：drop_rate=0）。"""
    return GPTConfig(
        vocab_size=1024,
        context_length=32,
        emb_dim=32,
        n_layers=2,
        n_heads=4,
        drop_rate=0.0,
        qkv_bias=False,
        flash_attn=False,
    )


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """用合成中英文文本训练一个迷你 BPE 分词器，供数据/训练类测试复用（只训练一次）。"""
    from train_tokenizer import train_tokenizer  # scripts/train_tokenizer.py

    data_dir = tmp_path_factory.mktemp("tiny_tokenizer")
    data_path = data_dir / "corpus.jsonl"
    samples = [
        "在查处虚开增值税专用发票案件中，常常涉及进项留抵税额的认定和处理。",
        "Transformer 通过自注意力机制建模上下文关系。",
        "The transformer uses self-attention to model context.",
        "秋天来了，树叶黄了。",
        "gradient descent 是常用的优化算法。",
        "hello world, this is a sample sentence.",
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")

    output_dir = data_dir / "tokenizer"
    train_tokenizer(str(data_path), str(output_dir), vocab_size=1000)
    return AutoTokenizer.from_pretrained(str(output_dir), use_fast=False)
