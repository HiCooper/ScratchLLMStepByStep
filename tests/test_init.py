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
