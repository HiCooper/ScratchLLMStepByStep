"""数据模块的单元测试：texts_to_bin 序列化、二进制数据集、训练/评估划分。"""
import json

import torch

from minigpt.data.pretrain_dataset import texts_to_bin, PretrainBinaryDataset, split_dataset


def _write_corpus(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")


def test_texts_to_bin_and_binary_dataset(tmp_path, tiny_tokenizer):
    data_path = tmp_path / "corpus.jsonl"
    _write_corpus(data_path, ["你好世界", "hello world", "在查处虚开增值税专用发票案件中", "the quick brown fox"])

    bin_path = tmp_path / "corpus.bin"
    texts_to_bin(str(data_path), str(bin_path), tiny_tokenizer, content_key="text")

    ds = PretrainBinaryDataset(str(bin_path), max_tokens=8)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == y.shape == (7,)  # 每样本 8 个 token，input=item[:-1]、target=item[1:]
    # 预训练目标 = 输入右移一位：y[i] == x[i+1]
    assert torch.equal(x[1:], y[:-1])


def test_split_dataset_ratio(tmp_path, tiny_tokenizer):
    data_path = tmp_path / "corpus.jsonl"
    _write_corpus(data_path, [f"样本编号 {i} 这是一段用于测试的文本。" for i in range(20)])

    bin_path = tmp_path / "corpus.bin"
    texts_to_bin(str(data_path), str(bin_path), tiny_tokenizer, content_key="text")

    ds = PretrainBinaryDataset(str(bin_path), max_tokens=8)
    train_set, eval_set = split_dataset(ds[:], 0.8)
    assert len(train_set) == int(len(ds) * 0.8)
    assert len(train_set) + len(eval_set) == len(ds)
