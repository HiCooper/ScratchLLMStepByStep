import torch
from torch import nn
from typing import Tuple, List

# flash-attn 是可选依赖：只有使用 FlashMultiHeadAttention 时才需要，未安装时模块仍可正常导入。
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)   
)
batch = torch.stack((inputs, inputs), dim=0)

class SelfAttentionV1(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.Wq = nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=True)
        self.Wk = nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=True)
        self.Wv = nn.Parameter(torch.rand(dim_in, dim_out), requires_grad=True)

    def forward(self, x):
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        atten_scores = q @ k.T
        atten_weights = torch.softmax(atten_scores/self.dim_out ** 0.5, dim=-1)
        context_vecs = atten_weights @ v
        return context_vecs

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout_rate, qkv_bias=False):
        super().__init__()
        self.dim_out = dim_out
        self.Wq = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wk = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wv = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, seq_len, dim = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        atten_scores = q @ k.transpose(1, 2)  # 将第1维和第2维转置，从第0维开始
        atten_scores = atten_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        atten_weights = torch.softmax(atten_scores/self.dim_out ** 0.5, dim=-1)
        context_vecs = self.dropout(atten_weights) @ v
        return context_vecs
    

def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert ndim >= 2, f"expect x to have at least 2 dims, got {ndim}"
        assert pos_cis.shape == (x.shape[1], x.shape[-1]), f"{pos_cis.shape} == {(x.shape[1], x.shape[-1])} ?"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout_rate, num_heads, qkv_bias=False, qkv_merged=False):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads   # 每个头的维度
        self.qkv_merged = qkv_merged
        if qkv_merged:
            # 合并 QKV 投影：单次矩阵乘法产出 3*dim_out，比三个独立 Linear 更高效
            self.Wqkv = nn.Linear(dim_in, 3 * dim_out, bias=qkv_bias)
        else:
            self.Wq = nn.Linear(dim_in, dim_out, bias=qkv_bias)
            self.Wk = nn.Linear(dim_in, dim_out, bias=qkv_bias)
            self.Wv = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.Wo = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout_rate)

        # 加法掩码形式的因果掩码：上三角(未来位置)为 -inf，下三角(含对角线)为 0。
        # 直接与注意力得分相加即可完成因果遮蔽，避免在 (b, heads, seq, seq) 上做 masked_fill。
        self.register_buffer("causal_mask", torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1))

    def _project_qkv(self, x):
        if self.qkv_merged:
            q, k, v = self.Wqkv(x).chunk(3, dim=-1)
            return q, k, v
        return self.Wq(x), self.Wk(x), self.Wv(x)

    def forward(self, x, pos_cis, attention_mask=None, use_kv_cache=False, past_kv:Tuple[torch.Tensor]=None):
        # 输入形状：训练/首次前向为 (b, seq_len, dim)，增量推理时为 (b, 1, dim)
        b, num_tokens, dim_in = x.shape

        # 求 Q/K/V。增量模式下只对最新 1 个 token 计算 q/k/v，并与已缓存的 k/v 拼接，
        # 从而避免每步重算整个序列的投影与注意力（真正的 KV cache）。
        if use_kv_cache and past_kv is not None:
            past_k, past_v = past_kv
        else:
            past_k, past_v = None, None

        q, k_new, v_new = self._project_qkv(x)   # 每个 (b, num_tokens, dim_out)

        # 变换形状，将最后一维拆成多头，每个头有 head_dim 维，矩阵形状由三维变为四维。
        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k_new = k_new.view(b, num_tokens, self.num_heads, self.head_dim)
        v_new = v_new.view(b, num_tokens, self.num_heads, self.head_dim)

        # pos_cis 已按绝对位置对齐（增量时只有当前 token 的 1 个位置）
        q, k_new = apply_rotary_emb(q, k_new, pos_cis) if pos_cis is not None else (q, k_new)

        if past_k is not None:
            k = torch.cat((past_k, k_new), dim=1)
            v = torch.cat((past_v, v_new), dim=1)
        else:
            k, v = k_new, v_new
        past_kv = (k, v)

        # 交换第2维和第3维，这一步过后，形状变为：b, num_heads, num_tokens, head_dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        num_queries = q.shape[2]
        num_keys = k.shape[2]

        # 计算注意力分数，形状变为: b, num_heads, num_queries, num_keys
        atten_scores = q @ k.transpose(2, 3)
        scaled_atten_scores = atten_scores / self.head_dim ** 0.5

        # 加法掩码：因果掩码（仅完整前向需要，增量时 q 只有最新 token，天然看不到未来）
        # 与 padding 掩码合并成一份偏置，一次性加到得分上，避免 masked_fill。
        additive_mask = None
        if not use_kv_cache:
            additive_mask = self.causal_mask[:num_queries, :num_keys].to(scaled_atten_scores.dtype)
        if attention_mask is not None:
            # attention_mask: (batch, 1, 1, num_keys)，1=有效、0=填充
            pad_bias = (attention_mask == 0).to(scaled_atten_scores.dtype) * torch.finfo(scaled_atten_scores.dtype).min
            additive_mask = pad_bias if additive_mask is None else additive_mask + pad_bias
        if additive_mask is not None:
            scaled_atten_scores = scaled_atten_scores + additive_mask

        atten_weights = torch.softmax(scaled_atten_scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vecs = atten_weights @ v   # shape: b, num_heads, num_queries, head_dim
        context_vecs = context_vecs.transpose(1, 2)  # shape: b, num_queries, num_heads, head_dim
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.dim_out)
        output = self.Wo(context_vecs)

        return output, past_kv


class FlashMultiHeadAttention(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x, pos_cis=None, attention_mask=None, use_kv_cache=False, past_kv:Tuple[torch.Tensor]=None):
        # 输入形状：训练/首次前向为 (b, seq_len, dim)，增量推理时为 (b, 1, dim)
        b, num_tokens, dim_in = x.shape

        # 增量模式下只对最新 token 计算 q/k/v，并与缓存拼接（真·KV cache）。
        if use_kv_cache and past_kv is not None:
            past_k, past_v = past_kv
        else:
            past_k, past_v = None, None

        q, k_new, v_new = self._project_qkv(x)

        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k_new = k_new.view(b, num_tokens, self.num_heads, self.head_dim)
        v_new = v_new.view(b, num_tokens, self.num_heads, self.head_dim)

        q, k_new = apply_rotary_emb(q, k_new, pos_cis) if pos_cis is not None else (q, k_new)

        if past_k is not None:
            k = torch.cat((past_k, k_new), dim=1)
            v = torch.cat((past_v, v_new), dim=1)
        else:
            k, v = k_new, v_new
        past_kv = (k, v)

        # flash_attn_func 的输入/输出均为 (batch, seqlen, nheads, headdim)，无需转置到 head-first，
        # 这与非 flash 的 MultiHeadAttention（需要 transpose 后做批量 matmul）不同。
        input_dtype = q.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = torch.float16
            q = q.to(dtype=target_dtype)
            k = k.to(dtype=target_dtype)
            v = v.to(dtype=target_dtype)

        dropout_rate = self.dropout.p if self.training else 0.0
        if flash_attn_func is None:
            raise ImportError("使用 FlashMultiHeadAttention 需要安装 flash-attn 包。")
        # 增量时 q 只有最新 1 个 token，天然看不到未来，无需 causal；完整前向才需要 causal。
        causal = not use_kv_cache
        # 注：flash_attn 不支持任意 padding 掩码，SFT(需要 padding)应使用非 flash 注意力。
        context_vecs = flash_attn_func(q, k, v, dropout_p=dropout_rate, softmax_scale=self.head_dim ** -0.5, causal=causal)
        context_vecs = context_vecs.to(dtype=input_dtype)

        output = self.Wo(context_vecs.contiguous().view(b, num_tokens, self.dim_out))

        return output, past_kv
