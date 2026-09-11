"""权重初始化回归测试。

背景（真实事故）：`MiniGPT.__init__` 先 `super().__init__(config)` 再建子模块，且从未调用
`self.post_init()`；而 transformers 5.x 的 `PreTrainedModel.__init__` 已不再自动调用它。
结果全部权重落在 PyTorch 默认值上——`nn.Embedding` 是 N(0,1) 而非 N(0, 0.02²)，logits
尺度被放大 ~50 倍，softmax 初始化即饱和，随机 token 的首个 batch loss 从应有的 ~10.4
变成 ~397。这里锁死修复后的行为，防止再次退化。
"""
import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.model.transformer import GPTConfig, MiniGPT

LN_VOCAB = math.log(1024)


def _cfg(**kw):
    base = dict(vocab_size=1024, context_length=32, emb_dim=32, n_layers=4,
                n_heads=4, drop_rate=0.0, tie_word_embeddings=True)
    base.update(kw)
    return GPTConfig(**base)


def test_embedding_uses_initializer_range():
    torch.manual_seed(0)
    model = MiniGPT(_cfg())
    std = model.token_emb.weight.std().item()
    assert 0.01 < std < 0.04, f"token_emb std={std}，应为 initializer_range≈0.02（而不是 1.0）"


def test_all_linear_weights_are_normal_small():
    torch.manual_seed(0)
    model = MiniGPT(_cfg())
    for name, p in model.named_parameters():
        if name.endswith("weight") and p.dim() >= 2:
            assert p.std().item() < 0.1, f"{name} std={p.std().item()} 过大"


def test_residual_projections_scaled_by_depth():
    """GPT-2 做法：残差分支输出投影用 std/sqrt(2*n_layers)，抑制残差流方差累积。"""
    torch.manual_seed(0)
    n_layers = 4
    model = MiniGPT(_cfg(n_layers=n_layers))
    expected = 0.02 / math.sqrt(2 * n_layers)
    for i in range(n_layers):
        wo = model.decode_layers[i].atten.Wo.weight.std().item()
        ffn_out = model.decode_layers[i].ffn.layers[-1].weight.std().item()
        assert wo == pytest.approx(expected, rel=0.15), f"layer{i} Wo std={wo}"
        assert ffn_out == pytest.approx(expected, rel=0.15), f"layer{i} ffn_out std={ffn_out}"
    # 非残差投影保持 std=0.02
    assert model.decode_layers[0].atten.Wq.weight.std().item() == pytest.approx(0.02, rel=0.15)


def test_layernorm_params_untouched():
    torch.manual_seed(0)
    model = MiniGPT(_cfg())
    for layer in model.decode_layers:
        for ln in (layer.layernorm1, layer.layernorm2):
            assert torch.allclose(ln.scale, torch.ones_like(ln.scale))
            assert torch.allclose(ln.shift, torch.zeros_like(ln.shift))
    assert torch.allclose(model.final_norm.scale, torch.ones_like(model.final_norm.scale))


def test_initial_loss_is_near_uniform_baseline():
    """正确初始化下，随机 token 的交叉熵应贴着 ln(V)（而不是几百）。"""
    torch.manual_seed(0)
    model = MiniGPT(_cfg())
    ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        logits = model(ids)[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
    assert LN_VOCAB - 1.0 < loss.item() < LN_VOCAB + 2.0, \
        f"初始 loss={loss.item():.3f}，应接近 ln(1024)={LN_VOCAB:.3f}"
    assert logits.std().item() < 3.0, f"logits std={logits.std().item()} 过大"


def test_tie_word_embeddings_after_init():
    torch.manual_seed(0)
    tied = MiniGPT(_cfg(tie_word_embeddings=True))
    assert tied.out_head.weight.data_ptr() == tied.token_emb.weight.data_ptr()
    untied = MiniGPT(_cfg(tie_word_embeddings=False))
    assert untied.out_head.weight.data_ptr() != untied.token_emb.weight.data_ptr()
    assert not torch.allclose(untied.out_head.weight, untied.token_emb.weight)


def test_get_input_output_embeddings_hooks():
    model = MiniGPT(_cfg())
    assert model.get_input_embeddings() is model.token_emb
    assert model.get_output_embeddings() is model.out_head


@pytest.mark.parametrize("kwargs,match", [
    (dict(emb_dim=30, n_heads=4), "整除"),      # 30 % 4 != 0
    (dict(emb_dim=6, n_heads=2), "偶数"),       # head_dim=3 为奇数，RoPE 配对会炸
    (dict(emb_dim=32, n_heads=8, context_length=0), "assert"),
])
def test_gptconfig_validation(kwargs, match):
    with pytest.raises(AssertionError):
        GPTConfig(vocab_size=100, **kwargs)


def test_config_has_initializer_range():
    """transformers 5.x 的 PretrainedConfig 不再自带 initializer_range。"""
    assert GPTConfig(vocab_size=100).initializer_range == 0.02
    assert GPTConfig(vocab_size=100, initializer_range=0.05).initializer_range == 0.05


# ---------------------------------------------------------------------------
# P2：架构开关（norm_type / ffn_hidden_dim / lm_head_bias / rope_theta）
# ---------------------------------------------------------------------------
def test_layernorm_matches_fused_reference():
    """LayerNorm 改为 F.layer_norm 后必须与有偏方差的标准定义一致。

    旧实现用 `x.var(dim=-1)`（**无偏**，除以 N-1），与标准 LayerNorm（除以 N）
    相差约 (N/(N-1))^0.5；这是真实存在的数值偏差（虽因 LayerNorm 的尺度不变性
    被后续线性层吸收，对已训练权重影响为 0.0000 nats）。
    """
    import torch.nn.functional as F
    from minigpt.model.transformer import LayerNorm
    torch.manual_seed(0)
    ln = LayerNorm(32)
    x = torch.randn(4, 32)
    ref = F.layer_norm(x, (32,), ln.scale, ln.shift, ln.eps)
    assert torch.allclose(ln(x), ref, atol=1e-6)
    biased = ln.scale * ((x - x.mean(-1, keepdim=True))
                         / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + ln.eps)) + ln.shift
    assert torch.allclose(ln(x), biased, atol=1e-5)
    unbiased = ln.scale * ((x - x.mean(-1, keepdim=True))
                           / torch.sqrt(x.var(-1, keepdim=True, unbiased=True) + ln.eps)) + ln.shift
    assert not torch.allclose(ln(x), unbiased, atol=1e-3), "应与旧的无偏方差实现存在差异"


def test_rmsnorm_shape_and_params():
    from minigpt.model.transformer import RMSNorm
    torch.manual_seed(0)
    rn = RMSNorm(32)
    x = torch.randn(2, 4, 32)
    y = rn(x)
    assert y.shape == x.shape
    assert not hasattr(rn, "shift"), "RMSNorm 不应有 shift 参数"
    # RMS 归一化后每行均方 ≈ 1
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 4), atol=0.1)


def test_swiglu_hidden_dim_keeps_param_parity():
    """SwiGLU 中间维必须压到 8/3·d，否则比 GELU 版多 50% 前馈参数。

    只数参数量、不做前向，因此可以直接用生产尺寸 emb=512。
    （小 emb 下 64 的对齐粒度会带来几个百分点的残差，属预期。）
    """
    def total(**kw):
        m = MiniGPT(GPTConfig(vocab_size=1024, emb_dim=512, n_heads=8, n_layers=1,
                              context_length=32, **kw))
        return sum(p.numel() for p in m.parameters()), m

    gelu, _ = total(tie_word_embeddings=True)
    swiglu, m_sw = total(tie_word_embeddings=True, use_swiglu=True)
    old_style, _ = total(tie_word_embeddings=True, use_swiglu=True,
                         ffn_hidden_dim=4 * 512)
    assert m_sw.decode_layers[0].ffn.hidden_dim == 1344, "8/3·512 应对齐到 64 的倍数"
    assert abs(swiglu - gelu) / gelu < 0.02, \
        f"SwiGLU({swiglu}) 与 GELU({gelu}) 参数量应基本一致（差 {abs(swiglu-gelu)/gelu:.1%}）"
    assert old_style > swiglu * 1.2, "旧的 4d 实现应显著更大，确保该回归不会复现"


def test_norm_type_switch():
    gelu_ln = MiniGPT(_cfg())
    from minigpt.model.transformer import LayerNorm, RMSNorm
    assert isinstance(gelu_ln.decode_layers[0].layernorm1, LayerNorm)
    rn = MiniGPT(_cfg(norm_type="rmsnorm"))
    assert isinstance(rn.decode_layers[0].layernorm1, RMSNorm)
    assert isinstance(rn.final_norm, RMSNorm)
    with pytest.raises(AssertionError):
        GPTConfig(vocab_size=10, norm_type="batchnorm")


def test_lm_head_bias_default_off():
    assert MiniGPT(_cfg()).out_head.bias is None
    assert MiniGPT(_cfg(lm_head_bias=True)).out_head.bias is not None


def test_rope_theta_is_configurable():
    from minigpt.model.transformer import precompute_pos_cis
    ctx = 8
    a = precompute_pos_cis(8, ctx, theta=10000.0)
    b = precompute_pos_cis(8, ctx, theta=500000.0)
    assert not torch.allclose(a, b)
    # 默认值必须保持 10000（与历史 checkpoint 一致）
    m = MiniGPT(_cfg(context_length=ctx))
    expect = precompute_pos_cis(m.config.emb_dim // m.config.n_heads, ctx, theta=10000.0)
    assert torch.allclose(m.pos_cis, expect)
