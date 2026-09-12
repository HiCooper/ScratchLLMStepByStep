import json
import os

import pytest

from minigpt.data.pretrain_dataset import (TokenBinDataset, split_train_eval_blocks,
                                           tokenize_jsonl_to_bin, validate_bin_tokenizer)


@pytest.fixture()
def tiny_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    lines = [{"text": f"第{i}句中文测试文本" * 3} for i in range(20)]
    with open(p, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return p


@pytest.fixture()
def tiny_bin(tmp_path, tiny_jsonl, tiny_tokenizer):
    out = tmp_path / "data.bin"
    lines, tokens, dtype = tokenize_jsonl_to_bin(str(tiny_jsonl), str(out), tiny_tokenizer)
    return out, lines, tokens, dtype


def test_tokenize_bin_roundtrip(tiny_bin):
    out, lines, tokens, dtype = tiny_bin
    assert lines == 20 and dtype == "uint16"
    ds = TokenBinDataset(str(out), 64)
    assert len(ds) == tokens // 64
    x, y = ds[0]
    assert x.shape[0] == 63 and y.shape[0] == 63


def test_tokenbdataset_uses_memmap(tiny_bin):
    """旧实现用 np.fromfile 把整个 bin 读进内存（每个 worker 一份），必须改回 memmap。"""
    import numpy as np
    out, _, _, _ = tiny_bin
    ds = TokenBinDataset(str(out), 32)
    assert isinstance(ds.flat, np.memmap), f"期望 np.memmap，实际 {type(ds.flat)}"
    assert ds.total_tokens == ds.flat.size


# ---------------------------------------------------------------------------
# 验证集切分：均匀分块 + 双端对齐文档边界，且不得有文档同时出现在 train/eval
# （旧实现 random_split 会让文档前半进训练集、后半进验证集；只取尾部又失去代表性）
# ---------------------------------------------------------------------------
def _make_stream_ds(tmp_path, seq_per_doc, n_docs, vocab_size=1000, eos_id=2, max_len=16):
    """构造一个"文档 + eos"交替的 token 流，便于验证文档边界对齐。"""
    import numpy as np
    tokens = []
    for d in range(n_docs):
        tokens.extend([10 + (d % 50)] * seq_per_doc)
        tokens.append(eos_id)
    arr = np.asarray(tokens, dtype=np.uint16)
    p = tmp_path / "stream.bin"
    arr.tofile(str(p))
    p.with_suffix(".meta.json").write_text(json.dumps(
        {"dtype": "uint16", "lines": n_docs, "tokens": int(arr.size),
         "vocab_size": vocab_size, "eos_id": eos_id, "content_key": "text",
         "tokenizer": "synthetic"}), encoding="utf-8")
    return TokenBinDataset(str(p), max_len)


def _doc_of_window(ds, eos_id=2):
    """返回 {窗口下标: 该窗口覆盖到的文档 id 集合}。"""
    import numpy as np
    flat = np.asarray(ds.flat)
    eos = np.nonzero(flat == eos_id)[0]
    starts = np.concatenate([[0], eos + 1])
    ends = np.concatenate([eos + 1, [flat.size]])
    L = ds.max_len
    out = {}
    for di, (s, e) in enumerate(zip(starts, ends)):
        for w in range(int(s) // L, min(len(ds), -(-int(e) // L))):
            out.setdefault(w, set()).add(di)
    return out


def test_split_blocks_are_disjoint(tmp_path):
    ds = _make_stream_ds(tmp_path, seq_per_doc=40, n_docs=2000, max_len=16)
    train, val, info = split_train_eval_blocks(ds, eval_ratio=0.05, eos_id=2, n_blocks=8)
    assert info["split"] == "blocked-document-aligned"
    assert not (set(train.indices) & set(val.indices))
    assert len(train) + len(val) + (len(ds) - len(train) - len(val)) == len(ds)
    assert len(val) >= 8


def test_split_blocks_have_no_document_overlap(tmp_path):
    """核心保证：没有任何一篇文档同时出现在训练集和验证集。"""
    ds = _make_stream_ds(tmp_path, seq_per_doc=40, n_docs=2000, max_len=16)
    train, val, info = split_train_eval_blocks(ds, eval_ratio=0.05, eos_id=2, n_blocks=8)
    w2doc = _doc_of_window(ds)
    val_docs, train_docs = set(), set()
    for w in val.indices:
        val_docs |= w2doc.get(w, set())
    for w in train.indices:
        train_docs |= w2doc.get(w, set())
    assert not (val_docs & train_docs), f"{len(val_docs & train_docs)} 篇文档同时出现在 train 与 val"


def test_split_blocks_are_spread_across_stream(tmp_path):
    """验证块必须分散在整个 token 流上，而不是集中在尾部（语料按领域排序）。"""
    ds = _make_stream_ds(tmp_path, seq_per_doc=40, n_docs=2000, max_len=16)
    train, val, info = split_train_eval_blocks(ds, eval_ratio=0.05, eos_id=2, n_blocks=8)
    starts = [b["doc_start_token"] for b in info["blocks"]]
    assert len(starts) == 8
    assert starts == sorted(starts)
    # 首块应在前 10%，末块应在后 90% 之后 => 覆盖整条流
    assert starts[0] < ds.total_tokens * 0.1
    assert starts[-1] > ds.total_tokens * 0.8
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert max(gaps) < 3 * (sum(gaps) / len(gaps)), "块分布不均，像集中在某一段"


def test_split_blocks_aligned_to_document_boundaries(tmp_path):
    ds = _make_stream_ds(tmp_path, seq_per_doc=40, n_docs=2000, max_len=16)
    train, val, info = split_train_eval_blocks(ds, eval_ratio=0.05, eos_id=2, n_blocks=8)
    for b in info["blocks"]:
        s = b["doc_start_token"]
        assert s == 0 or int(ds.flat[s - 1]) == 2, "块起点没有对齐到文档边界"
        e = b["doc_end_token"]
        assert e == 0 or int(ds.flat[e - 1]) == 2, "块终点没有对齐到文档结束"


def test_split_blocks_respect_max_eval(tmp_path):
    ds = _make_stream_ds(tmp_path, seq_per_doc=40, n_docs=2000, max_len=16)
    train, val, info = split_train_eval_blocks(ds, eval_ratio=0.5, max_eval=12, eos_id=2,
                                               n_blocks=4)
    assert len(val) <= 12 * 3          # 允许文档边界对齐带来的少量上浮
    assert info["requested_eval_ratio"] == 0.5
    assert info["actual_eval_ratio"] < 0.5


def test_split_blocks_without_eos_still_works(tmp_path):
    ds = _make_stream_ds(tmp_path, seq_per_doc=40, n_docs=2000, max_len=16)
    train, val, info = split_train_eval_blocks(ds, eval_ratio=0.05, eos_id=None, n_blocks=4)
    assert len(train) >= 1 and len(val) >= 1
    assert info["eos_id"] is None


def test_split_blocks_rejects_tiny_dataset(tmp_path):
    ds = _make_stream_ds(tmp_path, seq_per_doc=8, n_docs=2, max_len=16)
    with pytest.raises(ValueError):
        split_train_eval_blocks(ds, eval_ratio=0.1, eos_id=2, n_blocks=32)


# ---------------------------------------------------------------------------
# 词表一致性：训练侧以前只按 len(tokenizer) 建模型，不比对 meta
# ---------------------------------------------------------------------------
def test_validate_bin_tokenizer_ok(tiny_bin, tiny_tokenizer):
    out, _, _, _ = tiny_bin
    info = validate_bin_tokenizer(str(out), tiny_tokenizer)
    assert info["checked"] is True
    assert info["bin_vocab"] == len(tiny_tokenizer)


def test_validate_bin_tokenizer_mismatch_raises(tiny_bin, tiny_tokenizer):
    out, _, _, _ = tiny_bin
    class FakeTok:
        def __len__(self):
            return len(tiny_tokenizer) + 1
    with pytest.raises(ValueError, match="不配套"):
        validate_bin_tokenizer(str(out), FakeTok())


def test_validate_bin_tokenizer_missing_meta(tmp_path, tiny_tokenizer):
    out = tmp_path / "nometa.bin"
    out.write_bytes(b"\x01\x00\x02\x00")
    info = validate_bin_tokenizer(str(out), tiny_tokenizer)
    assert info["checked"] is False


def test_tokenize_skips_malformed_lines(tmp_path, tiny_tokenizer):
    """脏行必须**跳过并计数**，不能让一行坏数据毁掉整轮构建。

    真实事故：上游 `pretrain_t2t.jsonl` 第 8,441,338 行开引号被 0x02 替换，
    旧实现在跑满 65 分钟、写完 3.15GB 之后抛 JSONDecodeError，连 meta 都没落盘。
    """
    import json
    from minigpt.data.pretrain_dataset import load_bin_meta, tokenize_jsonl_to_bin

    src = tmp_path / "corpus.jsonl"
    with open(src, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "正常的一行文本" * 10}, ensure_ascii=False) + "\n")
        f.write('{"text": \x02坏行：开引号被控制符替换' + "x" * 50 + "\n")   # 非法 JSON
        f.write("\n")                                                      # 空行
        f.write(json.dumps({"text": "第三行也正常" * 10}, ensure_ascii=False) + "\n")
    out = tmp_path / "corpus.bin"
    lines, tokens, dtype = tokenize_jsonl_to_bin(str(src), str(out), tiny_tokenizer)

    assert tokens > 0 and out.exists()
    meta, _ = load_bin_meta(str(out))
    assert meta["bad_lines"] == 1, f"坏行数应记为 1，实际 {meta.get('bad_lines')}"
    # 行号仍然连续计数（跳过的行不占行号空洞），且坏行没有产 token
    assert lines == 4
    assert meta["tokens"] == tokens
