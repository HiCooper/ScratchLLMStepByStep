"""checkpoint 结构推断 / 容错加载的单元测试。

回归背景：`best.pt`、周期 `checkpoint-*.pth` 早期不写 `config`，下游脚本按默认 GPTConfig()
（768/12）建模型去加载 512/10 权重会 `size mismatch ... 32000x512 vs 32000x768`。
"""
import pytest
import torch

from minigpt.model.checkpoint import (build_model_from_checkpoint, infer_model_kwargs,
                                      model_kwargs_from_checkpoint)
from minigpt.model.transformer import GPTConfig, MiniGPT


def _make(emb=32, layers=2, heads=4, ctx=64, vocab=128, **kw):
    cfg = GPTConfig(emb_dim=emb, n_layers=layers, n_heads=heads, context_length=ctx,
                    vocab_size=vocab, drop_rate=0.0, **kw)
    return MiniGPT(cfg), cfg


def test_infer_from_state_dict():
    model, cfg = _make()
    kws = infer_model_kwargs(model.state_dict())
    assert kws["emb_dim"] == 32
    assert kws["n_layers"] == 2
    assert kws["vocab_size"] == 128
    assert kws["context_length"] == 64
    assert kws["n_heads"] == max(1, 32 // 64)      # 无法反推，走 emb_dim//64 兜底
    assert kws["tie_word_embeddings"] is False     # 未共享时 out_head.weight 独立存在


def test_infer_tie_word_embeddings():
    model, _ = _make(tie_word_embeddings=True)
    assert infer_model_kwargs(model.state_dict())["tie_word_embeddings"] is True


def test_config_wins_over_inference():
    model, cfg = _make()
    ck = {"model_state": model.state_dict(), "config": cfg.to_dict()}
    kws, inferred = model_kwargs_from_checkpoint(ck, vocab_size=128)
    assert inferred is False
    assert kws["emb_dim"] == 32 and kws["n_layers"] == 2 and kws["n_heads"] == 4


def test_load_without_config_roundtrip(tmp_path):
    """没有 config 的 checkpoint 也能正确重建（这正是 best.pt 踩过的坑）。"""
    model, cfg = _make(emb=32, layers=2, heads=4, ctx=64, vocab=128)
    path = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict(), "step": 7}, path)

    loaded, ck, kws = build_model_from_checkpoint(str(path), tokenizer=None, device="cpu")
    assert kws["emb_dim"] == 32 and kws["n_layers"] == 2
    assert ck["step"] == 7
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        assert loaded(x).shape == (1, 8, 128)


def test_vocab_mismatch_raises_readable_error(tmp_path):
    """tokenizer 词表与 checkpoint embedding 行数不一致时必须给出可读报错（而不是 size mismatch 堆栈）。"""
    model, cfg = _make(vocab=128)
    path = tmp_path / "ck.pt"
    torch.save({"model_state": model.state_dict()}, path)

    class _Tok:
        def __len__(self):
            return 200

    with pytest.raises(ValueError, match="词表"):
        build_model_from_checkpoint(str(path), tokenizer=_Tok(), device="cpu")


def test_vocab_matches_tokenizer(tmp_path):
    model, cfg = _make(vocab=128)
    path = tmp_path / "ck.pt"
    torch.save({"model_state": model.state_dict()}, path)

    class _Tok:
        def __len__(self):
            return 128

    loaded, _, kws = build_model_from_checkpoint(str(path), tokenizer=_Tok(), device="cpu")
    assert kws["vocab_size"] == 128
    with torch.no_grad():
        assert loaded(torch.randint(0, 128, (1, 4))).shape == (1, 4, 128)
