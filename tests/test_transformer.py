"""模型结构的单元测试：GPTConfig 默认值、前向形状、pos_cis 动态扩展、生成。"""
import torch

from minigpt.model.transformer import GPTConfig, MiniGPT


def test_gptconfig_defaults():
    cfg = GPTConfig()
    assert cfg.vocab_size == 32000
    assert cfg.context_length == 1024
    assert cfg.emb_dim == 768
    assert cfg.n_layers == 12
    assert cfg.n_heads == 12
    # 前沿架构开关默认关闭（以兼容旧 checkpoint）
    assert cfg.use_swiglu is False
    assert cfg.qkv_merged is False
    assert cfg.tie_word_embeddings is False
    assert cfg.flash_attn is False


def test_minigpt_forward_shape(small_config):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    x = torch.randint(0, small_config.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, small_config.vocab_size)


def test_minigpt_pos_cis_regrowth(small_config):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    # 输入长度超过预计算的 context_length，触发 pos_cis 动态扩展
    seq = small_config.context_length + 8
    x = torch.randint(0, small_config.vocab_size, (1, seq))
    logits = model(x)
    assert logits.shape == (1, seq, small_config.vocab_size)


def test_minigpt_generate_kv_cache_and_no_cache(small_config):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    model.eval()
    prompt = torch.randint(0, small_config.vocab_size, (1, 8))
    with torch.no_grad():
        out_kv = model.generate(prompt, max_length=16, use_kv_cache=True)
        out_nokv = model.generate(prompt, max_length=16, use_kv_cache=False)
    # 输出长度 = prompt 长度 + max_length
    assert out_kv.shape == out_nokv.shape == (1, 8 + 16)
    # 生成的 token id 都应在词表范围内
    assert (out_kv < small_config.vocab_size).all()
    assert (out_nokv < small_config.vocab_size).all()
