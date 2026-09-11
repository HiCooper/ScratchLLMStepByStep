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


# ---------------------------------------------------------------------------
# 结构与兼容性回归（本轮目录扫描的产出）
#
# 1) PretrainBinaryDataset 旧实现硬编码 uint16 且不看 .meta.json：uint32 产物会被读成
#    "token 数翻倍、窗口全错位"，max_tokens 大于总 token 时静默得到空数据集。
# 2) 两个 split_dataset（2 段按窗口 / 3 段按样本）与两个 Trainer（教学 / 生产）同名，
#    容易误用；已改名并保留兼容别名供 notebook 的 %run 使用。
# ---------------------------------------------------------------------------
import pathlib

import numpy as np
import pytest

from minigpt.data import pretrain_dataset as pds
from minigpt.data import sft_dataset as sds


def test_pretrain_binary_dataset_reads_dtype_from_meta(tmp_path):
    """uint32 产物必须按 uint32 读（旧实现按 uint16 会得到双倍 token 数与错位窗口）。"""
    tokens = np.arange(0, 4000, dtype=np.uint32)
    bin_path = tmp_path / "u32.bin"
    tokens.tofile(str(bin_path))
    (tmp_path / "u32.meta.json").write_text(
        json.dumps({"dtype": "uint32", "tokens": int(tokens.size), "vocab_size": 151665}),
        encoding="utf-8")

    ds = PretrainBinaryDataset(str(bin_path), max_tokens=100)
    assert ds.dtype == "uint32"
    assert ds.total_tokens == tokens.size          # 而不是 8000（按 uint16 误读）
    assert len(ds) == tokens.size // 100
    x, y = ds[0]
    assert int(x[1]) == 1 and int(y[0]) == 1        # 目标=输入右移一位


def test_pretrain_binary_dataset_rejects_dtype_mismatch(tmp_path):
    """文件字节数不是 itemsize 整数倍时必须报错，而不是静默切成错位窗口。"""
    bin_path = tmp_path / "odd.bin"
    bin_path.write_bytes(b"\x01\x02\x03")           # 3 字节，uint16 读不了
    (tmp_path / "odd.meta.json").write_text(json.dumps({"dtype": "uint16"}), encoding="utf-8")
    with pytest.raises(ValueError, match="不是.*整数倍"):
        PretrainBinaryDataset(str(bin_path), max_tokens=8)


def test_pretrain_binary_dataset_rejects_tiny_corpus(tmp_path):
    bin_path = tmp_path / "tiny.bin"
    np.arange(4, dtype=np.uint16).tofile(str(bin_path))
    with pytest.raises(ValueError, match="不足一个窗口"):
        PretrainBinaryDataset(str(bin_path), max_tokens=64)


def test_pretrain_binary_dataset_works_without_meta(tmp_path, capsys):
    """无 meta 时保持旧行为（按 uint16）并显式提示，不静默。"""
    bin_path = tmp_path / "nometa.bin"
    np.arange(200, dtype=np.uint16).tofile(str(bin_path))
    ds = PretrainBinaryDataset(str(bin_path), max_tokens=16)
    assert ds.dtype == "uint16" and len(ds) == 200 // 16
    assert "没有 .meta.json" in capsys.readouterr().out


def test_split_dataset_aliases_point_to_single_impl():
    """两个 split_dataset 语义不同：预训练 2 段（按窗口）、SFT 3 段（按样本）。"""
    assert pds.split_dataset is pds.split_train_eval_random
    assert sds.split_dataset is sds.split_train_eval_test
    assert pds.split_dataset is not sds.split_dataset
    # 名字与语义必须对得上（防再次同名混淆）
    import inspect
    assert len(inspect.signature(pds.split_train_eval_random).parameters) == 2
    assert len(inspect.signature(sds.split_train_eval_test).parameters) == 3


def test_teaching_trainer_is_not_production_trainer():
    """教学版与生产版 Trainer 不得同名（旧实现两个类都叫 Trainer）。"""
    import importlib.util
    from minigpt.train.trainer import Trainer as ProdTrainer
    root = pathlib.Path(pds.__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "pretrainer_single", str(root / "minigpt" / "train" / "pretrainer_single.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.Trainer is mod.SingleCardTrainer      # notebook 的 %run 兼容别名
    assert mod.SingleCardTrainer is not ProdTrainer
    assert ProdTrainer.__name__ == "Trainer" and mod.SingleCardTrainer.__name__ == "SingleCardTrainer"


def test_notebook_legacy_symbols_still_importable():
    """notebook 08/09/10/11 通过 %run 使用这些符号，改名/清理时不能弄丢。"""
    for sym in ("read_text_dataset", "PretrainTextDataset", "texts_to_bin",
                "PretrainBinaryDataset", "create_dataloaders", "split_dataset",
                "TokenBinDataset", "tokenize_jsonl_to_bin", "split_train_eval_blocks"):
        assert hasattr(pds, sym), f"pretrain_dataset 缺少 {sym}"
    for sym in ("InstructionDataset", "collate", "create_batch_collator",
                "split_dataset", "calc_label", "resolve_stop_token_ids"):
        assert hasattr(sds, sym), f"sft_dataset 缺少 {sym}"


def test_dead_code_removed():
    """create_dataloaders_from_texts 零引用，已删（防回归）。"""
    assert not hasattr(pds, "create_dataloaders_from_texts")
