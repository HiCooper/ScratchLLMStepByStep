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


# ---------------------------------------------------------------------------
# 回归：KV-cache 必须与全量前向数值等价
#
# 真实事故：注意力里用 `if not use_kv_cache` 决定是否加因果掩码，但 generate() 走 KV cache
# 时首步喂的是**完整 prompt**（prefill），于是整段 prompt 被双向注意力处理、被污染的 K/V
# 进入缓存，最后位置 logits 与正确因果前向相差 1.26（相对 3%）。
# 旧测试只断言 past_kv 的 **形状**，所以完全没发现。
# ---------------------------------------------------------------------------
def test_kv_cache_prefill_is_causal(small_config):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    model.eval()
    x = torch.randint(0, small_config.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(x, use_kv_cache=False)[0, -1]
        prefill = model(x, use_kv_cache=True, past_kvs=None, return_dict=True)["logits"][0, -1]
    assert torch.allclose(full, prefill, atol=1e-5), \
        f"KV-cache prefill 与全量前向不一致，最大差 {(full - prefill).abs().max().item()}"


def test_kv_cache_incremental_matches_full_forward(small_config):
    """逐 token 增量前向的每个位置都应等于全量前向（这是 KV cache 的定义性质）。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    model.eval()
    x = torch.randint(0, small_config.vocab_size, (2, 10))
    with torch.no_grad():
        full = model(x, use_kv_cache=False)
        past, per_step = None, []
        for t in range(x.shape[1]):
            out = model(x[:, t:t + 1], use_kv_cache=True, past_kvs=past, return_dict=True)
            past = out["past_key_values"]
            per_step.append(out["logits"][:, 0])
        step_logits = torch.stack(per_step, dim=1)
    assert torch.allclose(full, step_logits, atol=1e-5), \
        f"增量前向与全量前向不一致，最大差 {(full - step_logits).abs().max().item()}"


def test_generate_kv_cache_matches_no_cache_greedy(small_config):
    """贪心解码下，走 KV cache 与不走必须产出完全相同的序列。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    model.eval()
    prompt = torch.randint(0, small_config.vocab_size, (2, 6))
    with torch.no_grad():
        a = model.generate(prompt, max_length=12, do_sample=False, use_kv_cache=True)
        b = model.generate(prompt, max_length=12, do_sample=False, use_kv_cache=False)
    assert torch.equal(a, b)


def test_generate_accepts_attention_mask(small_config):
    """attention_mask 曾是 **kwargs 里的隐式参数，会与内部同名实参冲突直接 TypeError。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    model.eval()
    prompt = torch.randint(1, small_config.vocab_size, (2, 8))
    mask = torch.ones_like(prompt)
    mask[1, :3] = 0
    with torch.no_grad():
        out = model.generate(prompt, max_length=5, use_kv_cache=True, attention_mask=mask)
    assert out.shape == (2, 13)


def test_generate_attention_mask_masks_padding(small_config):
    """被 mask 掉的位置，其 token 内容不应影响生成结果（增量与全量两条路径都成立）。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    model.eval()
    prompt = torch.randint(1, small_config.vocab_size, (2, 9))
    mask = torch.ones_like(prompt)
    mask[1, :4] = 0
    other = prompt.clone()
    other[1, :4] = 7
    for use_kv in (True, False):
        with torch.no_grad():
            a = model.generate(prompt, max_length=6, do_sample=False,
                               use_kv_cache=use_kv, attention_mask=mask)
            b = model.generate(other, max_length=6, do_sample=False,
                               use_kv_cache=use_kv, attention_mask=mask)
        assert torch.equal(a[:, prompt.shape[1]:], b[:, prompt.shape[1]:]), \
            f"use_kv_cache={use_kv} 时 padding 内容泄漏进了生成结果"

