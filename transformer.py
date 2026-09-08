import torch
import torch.nn as nn
import importlib  
import attention_v1
# 使用 importlib.reload 来重新加载该模块  
importlib.reload(attention_v1) 

from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from attention_v1 import MultiHeadAttention, FlashMultiHeadAttention
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_CONFIG = {
    "vocab_size": 32000, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean)/(torch.sqrt(var + self.eps))
        return self.scale * x_norm + self.shift


class FeedForward(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )
    def forward(self, x):
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
            'qkv_bias': kwargs['qkv_bias']  
        }  
        if kwargs.get('flash_attn'):
            self.atten = FlashMultiHeadAttention(**attn_kwargs)
        else:
            self.atten = MultiHeadAttention(**attn_kwargs)
        self.ffn = FeedForward(kwargs['emb_dim'])
        self.drop = nn.Dropout(kwargs['drop_rate'])
        self.layernorm1 = LayerNorm(kwargs['emb_dim'])
        self.layernorm2 = LayerNorm(kwargs['emb_dim'])

    def forward(self, x, pos_cis, attention_mask=None, use_kv_cache=False, past_kv=None):
        shortcut = x
        x = self.layernorm1(x)
        x, past_kv = self.atten(x, pos_cis, attention_mask, use_kv_cache, past_kv)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.drop(x)
        x = x + shortcut

        return x, past_kv

def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
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
        super().__init__(**kwargs)

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

        pos_cis = precompute_pos_cis(config.emb_dim // config.n_heads, config.context_length)
        self.register_buffer("pos_cis", pos_cis, persistent=False)
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size)

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
            self.pos_cis = precompute_pos_cis(self.pos_cis.shape[1] * 2, max_pos + 1).to(self.pos_cis.device)
        pos_cis = self.pos_cis[position_ids]

        x = self.token_emb(inputs)
        x = self.drop_emb(x)

        # 支持注意力掩码计算（1=有效，0=填充）
        if attention_mask is not None:
            assert isinstance(attention_mask, torch.Tensor), f"expect torch.Tensor, but got{type(attention_mask)}"
            assert attention_mask.size() == inputs.size(), f"size of inputs {inputs.size()} and attention_mask {attention_mask.size()} must be the same."
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
    def generate(self, input_ids, max_length=512, eos_token_id=-1, **kwargs):
        assert isinstance(max_length, int) and max_length > 0
        eos_reached = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        past_kvs = None
        attention_mask = None
        use_kv_cache = kwargs.pop('use_kv_cache', True)

        for _ in range(max_length):
            if use_kv_cache:
                # 增量推理：第一步用完整 prompt，之后每步只送入最新 1 个 token
                step_input = input_ids if past_kvs is None else input_ids[:, -1:]
            else:
                # 不使用缓存时，每次都对最后 context_length 个 token 完整前向
                step_input = input_ids[:, -self.context_length:]
            output = self(step_input, attention_mask=attention_mask, use_kv_cache=use_kv_cache,
                          past_kvs=past_kvs, return_dict=True, **kwargs)  # shape: batch, n_tokens, vocab_size
            past_kvs = output["past_key_values"] if use_kv_cache else None
            # 只取每个序列最后一个token的输出向量作为logits, shape变为: batch, vocab_size
            logits = output["logits"][:, -1, :]
            # 使用softmax函数将logits转换为下一个token的概率分布，shape仍是: batch, vocab_size
            probs = torch.softmax(logits, dim=-1)
            # 取概率最大的作为next_input_ids，形状变为：batch, 1
            next_token_ids = torch.argmax(probs, dim=-1, keepdim=True)
            # 将next_token_id连接到下一个token的结尾， 形状变为：batch, n_tokens+1
            input_ids = torch.cat((input_ids, next_token_ids), dim=1)
            # 更新 eos_reached
            eos_reached |= (next_token_ids.squeeze(-1) == eos_token_id)
            if eos_reached.all():
                break

        return input_ids
