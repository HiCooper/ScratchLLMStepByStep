"""`generate()` 的 padding 掩码窗口对齐（真实事故回归）。

旧实现（`MiniGPT.generate`，use_kv_cache=False 分支）：

    step_input = input_ids[:, -context_length:]              # 取"最后" ctx 个 token
    step_mask  = attention_mask[:, :past_len + len]          # 却取"最前" ctx 个 mask

prompt 长于 `context_length` 时，喂进去的是一整段 padding（通常全 0），
`pad_bias = (mask == 0) * finfo.min` 把所有位置压平，softmax 退化成**均匀分布**——
不报错、不 NaN，只是静默掉点。这里用"记录 generate 实际传给 forward 的掩码"的方式锁住它。
"""
import torch

from minigpt.model.transformer import GPTConfig, MiniGPT


def _cfg(ctx=8):
    torch.manual_seed(0)
    return GPTConfig(emb_dim=64, n_layers=2, n_heads=4, context_length=ctx,
                     vocab_size=64, tie_word_embeddings=True, drop_rate=0.0)


class _MaskRecorder(MiniGPT):
    """记录每次 forward 收到的 attention_mask，用于验证 generate 的切片口径。"""

    def __init__(self, config):
        super().__init__(config)
        self.seen_masks = []

    def forward(self, *args, **kwargs):
        mask = kwargs.get("attention_mask")
        self.seen_masks.append(None if mask is None else mask.clone())
        return super().forward(*args, **kwargs)


def test_no_cache_generation_masks_the_tail_window():
    ctx = 8
    m = _MaskRecorder(_cfg(ctx)).eval()
    ids = torch.randint(0, 64, (1, 20))          # prompt 远长于 ctx
    mask = torch.zeros(1, 20)
    mask[:, 12:] = 1                             # 只有最后 8 个位置有效

    m.generate(ids, max_length=3, use_kv_cache=False, attention_mask=mask)

    assert m.seen_masks, "generate 没有调用 forward？"
    for i, seen in enumerate(m.seen_masks):
        assert seen is not None
        # 每一步都必须对齐到"最后 ctx 个位置"，而不是前缀 [0, ctx)
        assert torch.equal(seen, mask[:, -ctx:]), (
            f"第 {i} 步的掩码窗口错了：step_input 取尾部 {ctx} 个 token，"
            f"掩码却喂了 {seen.sum().item()} 个有效位（应为 {int(mask[:, -ctx:].sum())}）")


def test_cache_generation_masks_grow_with_prefix():
    """带 KV cache 的分支走相反方向：序列从位置 0 连续增长，掩码应取前缀且逐步变长。"""
    m = _MaskRecorder(_cfg(ctx=8)).eval()
    ids = torch.randint(0, 64, (1, 6))
    mask = torch.ones(1, 6)

    m.generate(ids, max_length=3, use_kv_cache=True, attention_mask=mask)

    lengths = [int(s.shape[1]) for s in m.seen_masks if s is not None]
    assert lengths == [6, 7, 8], f"prefill + 增量三步的掩码长度应为 6/7/8，实际 {lengths}"


def test_all_padding_mask_degenerates_to_uniform_attention():
    """把失效模式本身也钉住：全 padding 掩码会让注意力变成均匀分布（旧 bug 的症状）。"""
    m = MiniGPT(_cfg(ctx=8)).eval()
    att = m.decode_layers[0].atten
    att.capture_attention = True
    step = torch.randint(0, 64, (1, 8))

    with torch.no_grad():
        m(step, attention_mask=torch.zeros(1, 8))
        uniform = att.last_attention[0, 0, -1]
        m(step, attention_mask=torch.ones(1, 8))
        causal = att.last_attention[0, 0, -1]

    assert torch.allclose(uniform, torch.full_like(uniform, 1.0 / 8)), \
        "全 padding 行应退化成均匀注意力（这正是旧实现在长 prompt 下实际喂进去的东西）"
    assert not torch.allclose(causal, uniform), "正常因果掩码不应等于均匀分布"
