import torch

from minigpt.model.transformer import (GPTConfig, MiniGPT, apply_repetition_penalty,
                                       filter_logits_top_k_top_p)


def _model():
    return MiniGPT(GPTConfig(emb_dim=64, n_layers=2, n_heads=4, context_length=64,
                             vocab_size=1000, tie_word_embeddings=True))


def test_topk_top_p_filter_shape():
    logits = torch.randn(2, 1000)
    out = filter_logits_top_k_top_p(logits, top_k=10, top_p=0.9)
    assert out.shape == logits.shape
    finite_after = (out > float("-1e30")).sum(dim=-1)
    assert (finite_after <= 10).all()


def test_repetition_penalty_changes_seen_token():
    logits = torch.randn(1, 1000)
    seq = torch.tensor([[10, 20]])
    out = apply_repetition_penalty(logits, seq, 1.2)
    assert not torch.allclose(logits[0, 10], out[0, 10])
    # 惩罚后分数更小（正 logit 被除以 >1）
    assert out[0, 10] <= logits[0, 10] + 1e-6 or torch.isclose(out[0, 10], logits[0, 10])


def test_generate_sampling_deterministic_with_seed():
    m = _model()
    ids = torch.tensor([[10, 20, 30]])
    torch.manual_seed(7)
    a = m.generate(ids, 16, eos_token_id=999, do_sample=True, temperature=0.9,
                   top_k=50, use_kv_cache=False)
    torch.manual_seed(7)
    b = m.generate(ids, 16, eos_token_id=999, do_sample=True, temperature=0.9,
                   top_k=50, use_kv_cache=False)
    assert torch.equal(a, b)
