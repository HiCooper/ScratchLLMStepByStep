"""注意力模块的单元测试：因果掩码、前向形状、RoPE、padding 掩码、flash-attn 一致性。"""
import torch
import pytest

from minigpt.model.attention import (
    MultiHeadAttention,
    FlashMultiHeadAttention,
    apply_rotary_emb,
)
from minigpt.model.transformer import precompute_pos_cis


def test_causal_mask_is_additive():
    ctx = 8
    attn = MultiHeadAttention(dim_in=32, dim_out=32, context_length=ctx, dropout_rate=0.0, num_heads=4)
    mask = attn.causal_mask
    assert mask.shape == (ctx, ctx)
    # 下三角（含对角线）为 0，上三角（未来位置）为 -inf
    tril = torch.tril(torch.ones(ctx, ctx, dtype=torch.bool))
    triu = torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), diagonal=1)
    assert torch.all(mask[tril] == 0.0)
    assert torch.all(torch.isinf(mask[triu]))


def test_attention_forward_shape_and_kv_cache():
    torch.manual_seed(0)
    attn = MultiHeadAttention(dim_in=32, dim_out=32, context_length=16, dropout_rate=0.0, num_heads=4)
    x = torch.randn(2, 8, 32)
    pos_cis = precompute_pos_cis(8, 8)
    out, past_kv = attn(x, pos_cis)
    assert out.shape == (2, 8, 32)
    # 缓存 k/v 形状：b, seq, heads, head_dim
    assert past_kv[0].shape == (2, 8, 4, 8)
    assert past_kv[1].shape == (2, 8, 4, 8)


def test_attention_padding_mask_finite():
    torch.manual_seed(0)
    attn = MultiHeadAttention(dim_in=32, dim_out=32, context_length=16, dropout_rate=0.0, num_heads=4)
    x = torch.randn(2, 8, 32)
    pos_cis = precompute_pos_cis(8, 8)
    # attention_mask 期望形状为 (b, 1, 1, seq)，1=有效、0=填充
    am = torch.ones(2, 8)
    am[:, -2:] = 0
    out, _ = attn(x, pos_cis, attention_mask=am.unsqueeze(1).unsqueeze(2))
    assert torch.isfinite(out).all()


def test_rotary_emb_preserves_shape_and_dtype():
    torch.manual_seed(0)
    xq = torch.randn(2, 8, 4, 8)  # b, seq, heads, head_dim
    xk = torch.randn(2, 8, 4, 8)
    pos_cis = precompute_pos_cis(8, 8)
    q, k = apply_rotary_emb(xq, xk, pos_cis)
    assert q.shape == xq.shape and k.shape == xk.shape
    assert q.dtype == xq.dtype and k.dtype == xk.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash-attn 需要 CUDA")
def test_flash_attention_matches_standard():
    pytest.importorskip("flash_attn")
    torch.manual_seed(0)
    dim, ctx, heads = 32, 16, 4
    flash = FlashMultiHeadAttention(dim_in=dim, dim_out=dim, context_length=ctx, dropout_rate=0.0, num_heads=heads)
    std = MultiHeadAttention(dim_in=dim, dim_out=dim, context_length=ctx, dropout_rate=0.0, num_heads=heads)
    std.load_state_dict(flash.state_dict())  # 两实现共享同一份权重

    x = torch.randn(2, 8, dim, device="cuda")
    pos_cis = precompute_pos_cis(dim // heads, 8).to("cuda")
    flash.eval()
    std.eval()
    with torch.no_grad():
        out_flash, _ = flash(x, pos_cis)
        out_std, _ = std(x, pos_cis)
    # flash-attn 与标准注意力输出应一致（此前这里有个输入形状转置的 bug）
    assert torch.allclose(out_flash, out_std, atol=1e-3, rtol=1e-3)
