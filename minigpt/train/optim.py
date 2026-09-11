"""优化器构造：参数分组（weight decay 规范）。

单独成模块的原因：`AdamW(model.parameters(), weight_decay=w)` 会把 weight decay 施加到
**所有**参数上，包括 LayerNorm/RMSNorm 的 scale/shift 与所有 bias。对这些 1 维参数做衰减会把
归一化尺度往 0 拉，是明确的坏做法——nanoGPT / LLaMA / OLMo 等主流实现都把它们排除在衰减之外。
这是最容易被照抄进生产的超参错误，因此显式建模并加单测。
"""
from __future__ import annotations

import torch

# 与主流实现一致的默认值（显式写出，避免依赖 torch 默认值随版本变化）
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8


def is_decay_param(name: str, param: torch.nn.Parameter) -> bool:
    """判断某参数是否应施加 weight decay。

    标准规则（nanoGPT 等）：**只衰减维度 >= 2 且不是 bias 的参数**。
    LayerNorm/RMSNorm 的 scale/shift 都是 1 维，天然落入"不衰减"一侧。
    """
    return param.dim() >= 2 and not name.endswith(".bias")


def build_param_groups(model, weight_decay: float):
    """按是否衰减分成两组，返回可直接传给优化器的 param_groups。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if is_decay_param(name, param) else no_decay).append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        # 归一化尺度/bias：显式 0 衰减
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model, lr: float, weight_decay: float = 0.01,
                    betas=DEFAULT_BETAS, eps: float = DEFAULT_EPS):
    """构造 AdamW（参数已按 weight decay 规范分组）。"""
    return torch.optim.AdamW(build_param_groups(model, weight_decay),
                             lr=float(lr), betas=tuple(betas), eps=float(eps))


def param_group_summary(optimizer):
    """给日志/自检用：每组参数个数、元素数与 weight_decay。"""
    out = []
    for g in optimizer.param_groups:
        out.append({
            "weight_decay": g.get("weight_decay"),
            "tensors": len(g["params"]),
            "numel": int(sum(p.numel() for p in g["params"])),
        })
    return out
