"""checkpoint 结构推断 / 容错加载的单元测试。

回归背景：`best.pt`、周期 `checkpoint-*.pth` 早期不写 `config`，下游脚本按默认 GPTConfig()
（768/12）建模型去加载 512/10 权重会 `size mismatch ... 32000x512 vs 32000x768`。
"""
import warnings

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
    with pytest.warns(UserWarning, match="n_heads"):      # 头数只能猜，见 P0 回归测试
        kws = infer_model_kwargs(model.state_dict())
    assert kws["emb_dim"] == 32
    assert kws["n_layers"] == 2
    assert kws["vocab_size"] == 128
    assert kws["context_length"] == 64
    assert kws["n_heads"] == max(1, 32 // 64)      # 无法反推，走 emb_dim//64 兜底
    assert kws["tie_word_embeddings"] is False     # 未共享时 out_head.weight 独立存在


def test_infer_tie_word_embeddings():
    model, _ = _make(tie_word_embeddings=True)
    kws = infer_model_kwargs(model.state_dict(), n_heads=4)
    assert kws["tie_word_embeddings"] is True


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

    loaded, ck, kws = build_model_from_checkpoint(str(path), tokenizer=None, device="cpu",
                                                  n_heads=4)
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
        build_model_from_checkpoint(str(path), tokenizer=_Tok(), device="cpu", n_heads=4)


def test_vocab_matches_tokenizer(tmp_path):
    model, cfg = _make(vocab=128)
    path = tmp_path / "ck.pt"
    torch.save({"model_state": model.state_dict()}, path)

    class _Tok:
        def __len__(self):
            return 128

    loaded, _, kws = build_model_from_checkpoint(str(path), tokenizer=_Tok(), device="cpu",
                                                 n_heads=4)
    assert kws["vocab_size"] == 128
    with torch.no_grad():
        assert loaded(torch.randint(0, 128, (1, 4))).shape == (1, 4, 128)


# ---------------------------------------------------------------------------
# P2：新架构字段的反推与宽容加载
# ---------------------------------------------------------------------------
def _tiny_state(**cfg_kw):
    from minigpt.model.transformer import GPTConfig, MiniGPT
    m = MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                          context_length=16, drop_rate=0.0, **cfg_kw))
    return m.state_dict()


def test_infer_new_architecture_fields():
    from minigpt.model.checkpoint import infer_model_kwargs
    sd = _tiny_state()                       # 默认：layernorm + GELU + 无 head bias
    kws = infer_model_kwargs(sd, n_heads=2)
    assert kws["norm_type"] == "layernorm"
    assert kws["lm_head_bias"] is False
    assert kws["ffn_hidden_dim"] == 4 * 32    # GELU: 4d
    assert "out_head.bias" not in sd

    sd_rn = _tiny_state(norm_type="rmsnorm")
    assert infer_model_kwargs(sd_rn, n_heads=2)["norm_type"] == "rmsnorm"

    sd_sw = _tiny_state(use_swiglu=True)
    kws_sw = infer_model_kwargs(sd_sw, n_heads=2)
    assert kws_sw["use_swiglu"] is True
    assert kws_sw["ffn_hidden_dim"] == 64      # 8/3*32=85.3 就近对齐到 64 的倍数

    sd_b = _tiny_state(lm_head_bias=True)
    assert infer_model_kwargs(sd_b, n_heads=2)["lm_head_bias"] is True


def test_tolerant_load_ignores_legacy_head_bias():
    """旧产物带 out_head.bias、新结构不带：良性差异应被忽略而不是报错。"""
    from minigpt.model.checkpoint import load_state_into_model
    from minigpt.model.transformer import GPTConfig, MiniGPT
    sd = _tiny_state()
    sd["out_head.bias"] = torch.zeros(64)      # 模拟旧 checkpoint
    model = MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                              context_length=16, drop_rate=0.0))
    load_state_into_model(model, sd)           # 不应抛异常


def test_tolerant_load_still_rejects_real_mismatch():
    """真实的架构不匹配必须显式失败，绝不能静默续训到错误权重。"""
    from minigpt.model.checkpoint import load_state_into_model
    from minigpt.model.transformer import GPTConfig, MiniGPT
    sd = _tiny_state()
    sd["decode_layers.0.ffn.layers.0.weight"] = torch.zeros(999, 32)   # 形状/结构不符
    model = MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                              context_length=16, drop_rate=0.0))
    try:
        load_state_into_model(model, sd)
    except RuntimeError as exc:
        msg = str(exc)
        assert "缺少关键权重" in msg or "未知权重" in msg or "形状不匹配" in msg
    else:
        raise AssertionError("结构不匹配时必须报错")


# ---------------------------------------------------------------------------
# P0：n_heads 是唯一无法从权重反推的字段，猜错**不会**触发形状报错
# （Q/K/V/O 都是 (emb, emb)，pos_cis 是 non-persistent buffer），只会静默改变 head_dim。
# 实测 `_make(emb=384, heads=8)` 被推成 n_heads=6：missing=0/unexpected=0，logits 已不同。
# ---------------------------------------------------------------------------
def test_infer_n_heads_warns_and_can_be_overridden():
    model, _ = _make(emb=384, layers=2, heads=8, ctx=64, vocab=128)

    with pytest.warns(UserWarning, match="n_heads"):
        guessed = infer_model_kwargs(model.state_dict())
    assert guessed["n_heads"] == 384 // 64 == 6      # 真值是 8 —— 猜错且不报错，正是危险所在

    with warnings.catch_warnings():
        warnings.simplefilter("error")               # 显式给出时不应再有任何告警
        exact = infer_model_kwargs(model.state_dict(), n_heads=8)
    assert exact["n_heads"] == 8


def test_config_less_checkpoint_needs_explicit_n_heads(tmp_path):
    """config-less 老产物：默认告警，显式 --n-heads 后可精确重建原模型。"""
    model, _ = _make(emb=384, layers=2, heads=8, ctx=64, vocab=128)
    path = tmp_path / "nocfg.pt"
    torch.save({"model_state": model.state_dict()}, path)

    with pytest.warns(UserWarning, match="n_heads"):
        _, _, kws = build_model_from_checkpoint(str(path), device="cpu")
    assert kws["n_heads"] == 6

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded, _, kws2 = build_model_from_checkpoint(str(path), device="cpu", n_heads=8)
    assert kws2["n_heads"] == 8

    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        ref = model.eval()(x)
        got = loaded.eval()(x)
    assert torch.allclose(ref, got, atol=1e-6), "显式的 n_heads 必须能精确复现原模型输出"


def test_explicit_n_heads_ignored_when_config_present():
    """checkpoint 自带 config 时以 config 为准（并出声提示），不能按 flag 静默改结构。"""
    model, cfg = _make(emb=384, layers=2, heads=8, ctx=64, vocab=128)
    ck = {"model_state": model.state_dict(), "config": cfg.to_dict()}

    with pytest.warns(UserWarning, match="以 config"):
        kws, inferred = model_kwargs_from_checkpoint(ck, vocab_size=128, n_heads=4)
    assert inferred is False
    assert kws["n_heads"] == 8
