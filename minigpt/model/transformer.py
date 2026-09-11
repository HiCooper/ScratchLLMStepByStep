import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from minigpt.model.attention import MultiHeadAttention, FlashMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutputWithPast

class LayerNorm(nn.Module):
    """LayerNorm（保留 scale/shift 参数名，与历史 checkpoint 完全兼容）。

    实现改为调用 `F.layer_norm`：融合内核比手写 mean/var 少物化 4~5 个中间张量，
    且 CUDA 实现内部按 fp32 累加（手写版在 fp16 下会以 fp16 统计均值/方差）。
    """

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.scale, self.shift, self.eps)


class RMSNorm(nn.Module):
    """RMSNorm（LLaMA/Qwen 主流）：省掉均值项与 shift，参数更少、少一个减均值的规约。

    通过 `GPTConfig(norm_type="rmsnorm")` 启用。它与 LayerNorm 的 checkpoint 不兼容
    （没有 `shift`），`checkpoint.py` 会按 `layernorm*/shift` 是否存在自动反推。
    """

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.scale * xf).to(dtype)


def build_norm(norm_type: str, emb_dim: int):
    if norm_type == "rmsnorm":
        return RMSNorm(emb_dim)
    if norm_type == "layernorm":
        return LayerNorm(emb_dim)
    raise ValueError(f"未知的 norm_type: {norm_type!r}（可选 layernorm | rmsnorm）")


def _round_to_multiple(value: float, multiple: int = 64) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


class FeedForward(nn.Module):
    def __init__(self, emb_dim:int, use_swiglu:bool=False, hidden_dim:int=0):
        """前馈层。

        `hidden_dim=0` 表示按经验公式自动取：
          - GELU 版：4×emb_dim（GPT-2 传统）
          - SwiGLU 版：8/3×emb_dim 对齐到 64 的倍数

        真实问题：旧实现 SwiGLU 直接沿用 4×emb_dim，即 3 个 4d 矩阵，比 GELU 版
        （2 个 4d 矩阵）多 50% 前馈参数——实测 512/10/8 总参数从 47.9M 涨到 58.4M
        （+22%），与"门控换质量"的初衷相悖。LLaMA 正是把中间维压到 8/3·d 对齐参数量。
        """
        super().__init__()
        self.use_swiglu = use_swiglu
        self.hidden_dim = int(hidden_dim) or (
            _round_to_multiple(8 * emb_dim / 3) if use_swiglu else 4 * emb_dim)
        if use_swiglu:
            # SwiGLU：gate 分支与 up 分支逐元素相乘，再 down 投影。
            # 现代 LLM(LLaMA/Mistral/Qwen)普遍采用；无 bias 与 qkv_bias=False 保持一致。
            self.gate_proj = nn.Linear(emb_dim, self.hidden_dim, bias=False)
            self.up_proj = nn.Linear(emb_dim, self.hidden_dim, bias=False)
            self.down_proj = nn.Linear(self.hidden_dim, emb_dim, bias=False)
        else:
            self.layers = nn.Sequential(
                nn.Linear(emb_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, emb_dim),
            )

    def forward(self, x):
        if self.use_swiglu:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        attn_kwargs = {
            'dim_in': kwargs['emb_dim'],
            'dim_out': kwargs['emb_dim'],
            'context_length': kwargs['context_length'],
            'num_heads': kwargs['n_heads'],
            'dropout_rate': kwargs['drop_rate'],
            'qkv_bias': kwargs['qkv_bias'],
            'qkv_merged': kwargs.get('qkv_merged', False),
        }
        if kwargs.get('flash_attn'):
            self.atten = FlashMultiHeadAttention(**attn_kwargs)
        else:
            self.atten = MultiHeadAttention(**attn_kwargs)
        self.ffn = FeedForward(kwargs['emb_dim'], use_swiglu=kwargs.get('use_swiglu', False),
                               hidden_dim=kwargs.get('ffn_hidden_dim', 0))
        self.use_checkpoint = kwargs.get('use_checkpoint', False)
        self.drop = nn.Dropout(kwargs['drop_rate'])
        norm_type = kwargs.get('norm_type', 'layernorm')
        self.layernorm1 = build_norm(norm_type, kwargs['emb_dim'])
        self.layernorm2 = build_norm(norm_type, kwargs['emb_dim'])

    def forward(self, x, pos_cis, attention_mask=None, use_kv_cache=False, past_kv=None):
        shortcut = x
        x = self.layernorm1(x)
        x, past_kv = self.atten(x, pos_cis, attention_mask, use_kv_cache, past_kv)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.layernorm2(x)
        # 激活重计算：训练时对 FFN 用 checkpoint 以省显存，推理/无梯度时直通
        if self.use_checkpoint and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False)
        else:
            x = self.ffn(x)
        x = self.drop(x)
        x = x + shortcut

        return x, past_kv

def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    """RoPE 的复数旋转因子（与历史实现一致，theta 可配置以便长上下文外推）。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis

def attention_mask_to_additive(attention_mask):
    """将 (batch, seq_len) 的注意力掩码(1=有效, 0=填充)扩展成 (batch, 1, 1, seq_len)。

    这里只做维度扩展，真正的加法偏置在注意力内部按得分 dtype 生成，避免 dtype 不匹配。
    """
    return attention_mask.unsqueeze(1).unsqueeze(2)
        

# ---------------------------------------------------------------------------
# 采样解码工具（temperature / top-k / top-p / repetition penalty）
# ---------------------------------------------------------------------------
def apply_repetition_penalty(logits, seq_ids, penalty: float):
    """对当前序列中出现过的 token 施加惩罚：logit>0 除以 penalty，logit<0 乘 penalty。"""
    if penalty is None or penalty <= 1.0:
        return logits
    logits = logits.clone()
    for b in range(logits.size(0)):
        seen = torch.unique(seq_ids[b])
        g = logits[b, seen]
        logits[b, seen] = torch.where(g > 0, g / penalty, g * penalty)
    return logits


def filter_logits_top_k_top_p(logits, top_k=None, top_p=None):
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth,
                             torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, sorted_idx, sorted_logits)
        logits = out
    return logits


def sample_next_token(logits, do_sample: bool, temperature: float = 1.0,
                      top_k=None, top_p=None):
    if temperature != 1.0:
        logits = logits / max(float(temperature), 1e-6)
    logits = filter_logits_top_k_top_p(logits, top_k, top_p)
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class GPTConfig(PretrainedConfig):
    # 每个模型都必须有一个独特的model_type，否则会报"Should have a `model_type` key in its config.json"
    model_type = "minigpt"

    def __init__(self, **kwargs):
        self.context_length = kwargs.get('context_length', 1024)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.emb_dim = kwargs.get('emb_dim', 768)
        self.drop_rate = kwargs.get('drop_rate', 0.1)
        self.n_layers = kwargs.get('n_layers', 12)
        self.n_heads = kwargs.get('n_heads', 12)
        self.qkv_bias = kwargs.get('qkv_bias', False)
        self.flash_attn = kwargs.get('flash_attn', False)
        # 初始化标准差。transformers 5.x 的 PretrainedConfig 不再自带 initializer_range，
        # 必须显式声明，否则 _init_weights 拿不到统一来源（默认 0.02，与 GPT-2/LLaMA 一致）。
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        # 前沿架构开关（默认关闭以兼容旧 checkpoint，开启即前沿配置）
        self.use_swiglu = kwargs.get('use_swiglu', False)
        self.qkv_merged = kwargs.get('qkv_merged', False)
        self.tie_word_embeddings = kwargs.get('tie_word_embeddings', False)
        self.use_checkpoint = kwargs.get('use_checkpoint', False)
        # 归一化类型：layernorm（默认，兼容历史 checkpoint）| rmsnorm
        self.norm_type = kwargs.get('norm_type', 'layernorm')
        # 前馈中间维：0 = 自动（GELU 4d / SwiGLU 8/3 d 对齐 64）
        self.ffn_hidden_dim = kwargs.get('ffn_hidden_dim', 0)
        # 输出头是否带 bias。共享权重时标准做法是不带（旧默认 True，多 32k 参数且非标准）
        self.lm_head_bias = kwargs.get('lm_head_bias', False)
        # RoPE 基频；默认 10000 与历史 checkpoint 一致
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        super().__init__(**kwargs)
        self._validate()

    def _validate(self):
        """结构合法性校验：尽早失败，避免在 attention 里才炸出难懂的 reshape 错误。

        注意：PretrainedConfig 不会调用 `__post_init__`（已核对 transformers 5.1 源码），
        所以这里显式从 `__init__` 调用，否则校验就是死代码。
        """
        assert self.emb_dim % self.n_heads == 0, \
            f"emb_dim({self.emb_dim}) 必须能被 n_heads({self.n_heads}) 整除"
        head_dim = self.emb_dim // self.n_heads
        assert head_dim % 2 == 0, f"head_dim({head_dim}) 必须为偶数（RoPE 按相邻两维配对）"
        assert self.n_layers >= 1 and self.context_length >= 1 and self.vocab_size >= 1
        assert self.initializer_range > 0
        assert self.norm_type in ('layernorm', 'rmsnorm'), f"未知 norm_type: {self.norm_type}"
        assert self.rope_theta > 0


class MiniGPT(PreTrainedModel):
    config_class = GPTConfig

    @classmethod  
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  
        # 自定义加载逻辑，通常包括加载权重和配置  
        model = super(MiniGPT, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)  
        return model  

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.context_length = config.context_length
        self.num_heads = config.n_heads
        self.n_layers = config.n_layers
        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.decode_layers = nn.ModuleList([
            TransformerBlock(**(config.to_dict())) for _ in range(config.n_layers)
        ])

        pos_cis = precompute_pos_cis(config.emb_dim // config.n_heads, config.context_length,
                                     theta=config.rope_theta)
        self.register_buffer("pos_cis", pos_cis, persistent=False)
        self.final_norm = build_norm(config.norm_type, config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=config.lm_head_bias)

        # ---- 初始化 ----
        # 真实事故：这里以前没有任何初始化调用。transformers 5.x 的 PreTrainedModel.__init__
        # 已经不再自动调用 post_init()，而 MiniGPT 也没有 _init_weights，于是全部权重落在
        # PyTorch 默认值上——nn.Embedding 是 N(0,1) 而不是 N(0, 0.02²)，logits 尺度被放大约
        # 50 倍（实测 std 22.7 vs 0.5），softmax 在初始化即饱和：随机 token 的首个 batch
        # loss 从应有的 ~10.4 变成 ~397，真实语料 A/B 里 80 步后仍落后正确初始化 11.8 nats。
        std = float(config.initializer_range or 0.02)
        # GPT-2 的做法：残差分支输出投影按 1/sqrt(2*n_layers) 缩放，抑制残差流方差累积
        self._residual_std = std / math.sqrt(2 * max(1, config.n_layers))
        for layer in self.decode_layers:
            layer.atten.Wo._is_residual_out = True
            ffn_out = layer.ffn.down_proj if config.use_swiglu else layer.ffn.layers[-1]
            ffn_out._is_residual_out = True
        # post_init 会遍历子模块执行 _init_weights（并完成 HF 侧其余登记）
        self.post_init()

        if config.tie_word_embeddings:
            # 权重共享：输入嵌入与输出头共用同一份权重，显著减少参数量（输出头偏置仍独立）
            self.out_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        """HF 初始化钩子（由 post_init -> initialize_weights 逐模块调用）。"""
        std = float(getattr(self.config, "initializer_range", None) or 0.02)
        if isinstance(module, nn.Linear):
            target = self._residual_std if getattr(module, "_is_residual_out", False) else std
            nn.init.normal_(module.weight, mean=0.0, std=target)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def get_output_embeddings(self):
        return self.out_head

    def set_output_embeddings(self, new_embeddings):
        self.out_head = new_embeddings

    def forward(self,
                inputs:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                position_ids:Optional[torch.Tensor]=None,
                use_kv_cache=False,
                past_kvs=None,
                return_dict=False,
                **kwargs):

        if not past_kvs:
            past_kvs = [None for _ in range(self.n_layers)]
        if 'input_ids' in kwargs:
            inputs = kwargs['input_ids']
        assert isinstance(inputs, torch.Tensor), f"expect torch.Tensor, but got{type(inputs)}"
        b, seq_len = inputs.shape

        # 位置编码：增量推理时只处理最新 token，需按绝对位置取旋转编码，而非窗口内相对位置
        if position_ids is None:
            if use_kv_cache and past_kvs[0] is not None:
                start_pos = past_kvs[0][0].shape[1]  # 已缓存的 kv 长度即当前 token 的绝对位置
            else:
                start_pos = 0
            position_ids = torch.arange(start_pos, start_pos + seq_len, device=inputs.device)

        # pos_cis 只预计算了 context_length 长度，超出时按需动态扩展
        # 注意：pos_cis 的最后一维是 head_dim/2，而 precompute_pos_cis 需要传完整 head_dim
        max_pos = position_ids.max().item()
        if max_pos >= self.pos_cis.shape[0]:
            self.pos_cis = precompute_pos_cis(self.pos_cis.shape[1] * 2, max_pos + 1,
                                              theta=self.config.rope_theta).to(self.pos_cis.device)
        pos_cis = self.pos_cis[position_ids]

        x = self.token_emb(inputs)
        x = self.drop_emb(x)

        # 支持注意力掩码计算（1=有效，0=填充）
        # 形状约定：attention_mask 必须覆盖**完整 key 长度**（已缓存长度 + 本步输入长度），
        # 因为注意力偏置是加在 (q, k) 得分上的。增量推理时若只传当前 token 的掩码，
        # 早期位置的 padding 就屏蔽不掉了。
        if attention_mask is not None:
            assert isinstance(attention_mask, torch.Tensor), f"expect torch.Tensor, but got{type(attention_mask)}"
            past_len = past_kvs[0][0].shape[1] if (use_kv_cache and past_kvs[0] is not None) else 0
            expect = past_len + seq_len
            assert attention_mask.dim() == 2 and attention_mask.size(0) == b, \
                f"attention_mask 应为 (batch, seq_len)，实际 {tuple(attention_mask.size())}"
            assert attention_mask.size(1) == expect, \
                f"attention_mask 长度 {attention_mask.size(1)} 必须等于 已缓存 {past_len} + 本步 {seq_len} = {expect}"
            attention_mask = attention_mask_to_additive(attention_mask)

        for i, block in enumerate(self.decode_layers):
            x, past_kvs[i] = block(x, pos_cis, attention_mask, use_kv_cache, past_kvs[i])

        x = self.final_norm(x)
        logits = self.out_head(x)
        if not return_dict:
            return logits

        # 每次前向都新建输出对象，避免复用可变单例导致的隐式状态共享
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_kvs)
 
    @torch.inference_mode()
    def generate(self, input_ids, max_length=512, eos_token_id=-1,
                 do_sample=False, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, attention_mask=None, **kwargs):
        """自回归生成。

        `eos_token_id` 可以是单个 id，也可以是 id 列表（chat 模型需要同时接受
        `<|im_end|>` 与 `<|endoftext|>` 两个停止符——qwen2 词表下二者并不相同：
        im_end=151645、eos=151643，只传 eos 会导致生成永远停不下来）。
        `max_length` 是**新增** token 数，不是总长度。
        """
        assert isinstance(max_length, int) and max_length > 0
        eos_reached = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = sorted({int(x) for x in eos_token_id if x is not None and int(x) >= 0})
        else:
            eos_ids = [int(eos_token_id)] if int(eos_token_id) >= 0 else []
        eos_tensor = torch.tensor(eos_ids, dtype=input_ids.dtype, device=input_ids.device) if eos_ids else None
        past_kvs = None
        use_kv_cache = kwargs.pop('use_kv_cache', True)

        for _ in range(max_length):
            if use_kv_cache:
                # 增量推理：第一步用完整 prompt，之后每步只送入最新 1 个 token
                step_input = input_ids if past_kvs is None else input_ids[:, -1:]
            else:
                # 不使用缓存时，每次都对最后 context_length 个 token 完整前向
                step_input = input_ids[:, -self.context_length:]
            # attention_mask 需与「已缓存长度 + 本步长度」对齐；随生成同步增长（新 token 视为有效）
            step_mask = attention_mask
            if attention_mask is not None:
                past_len = 0 if past_kvs is None else past_kvs[0][0].shape[1]
                step_mask = attention_mask[:, :past_len + step_input.shape[1]]
            output = self(step_input, attention_mask=step_mask, use_kv_cache=use_kv_cache,
                          past_kvs=past_kvs, return_dict=True, **kwargs)  # shape: batch, n_tokens, vocab_size
            past_kvs = output["past_key_values"] if use_kv_cache else None
            # 只取每个序列最后一个token的输出向量作为logits, shape变为: batch, vocab_size
            logits = output["logits"][:, -1, :]
            # 采样策略：repetition penalty -> temperature/top-k/top-p -> 采样/贪心
            logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
            next_token_ids = sample_next_token(
                logits, do_sample=do_sample, temperature=temperature,
                top_k=top_k, top_p=top_p)
            input_ids = torch.cat((input_ids, next_token_ids), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))), dim=1)
            # 更新 eos_reached
            if eos_tensor is not None:
                eos_reached |= torch.isin(next_token_ids.squeeze(-1), eos_tensor)
            if eos_reached.all():
                break

        return input_ids
