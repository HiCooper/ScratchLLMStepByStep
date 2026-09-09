import json
import os

import pytest

from minigpt.data.pretrain_dataset import TokenBinDataset, tokenize_jsonl_to_bin
from transformers import AutoTokenizer

TOKENIZER = os.path.abspath("models/tokenizer_v3")


@pytest.fixture()
def tiny_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    lines = [{"text": f"第{i}句中文测试文本" * 3} for i in range(20)]
    with open(p, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return p


def test_tokenize_bin_roundtrip(tmp_path, tiny_jsonl):
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    out = tmp_path / "data.bin"
    lines, tokens, dtype = tokenize_jsonl_to_bin(str(tiny_jsonl), str(out), tok)
    assert lines == 20 and dtype == "uint16"
    ds = TokenBinDataset(str(out), 64)
    assert len(ds) == tokens // 64
    x, y = ds[0]
    assert x.shape[0] == 63 and y.shape[0] == 63
