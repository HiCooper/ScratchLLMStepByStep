"""分布式/编译包装相关的纯工具（从 trainer.py 抽出，可在单进程 CPU 上单测）。"""
from __future__ import annotations

import inspect

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def unwrap_model(model):
    """取出真实模型：兼容 `DistributedDataParallel` 与 `torch.compile(OptimizedModule)`。

    真实事故：train() 里先 DDP 再 compile，嵌套顺序是
    `OptimizedModule(DistributedDataParallel(MiniGPT))`。原来先判 `isinstance(DDP)`
    对最外层 OptimizedModule 为 False，再取 `_orig_mod` 就拿到 DDP 本身，保存出来的
    state_dict 键全部带 `module.` 前缀（实测 `module.token_emb.weight`）——multi-GPU
    （预设里 torch_compile=True）产出的 checkpoint 会被 checkpoint.py 判成"缺少关键
    权重"而无法加载、也无法续训。因此必须先剥 compile 外壳，再剥 DDP。
    """
    inner = getattr(model, "_orig_mod", model)      # 先剥 torch.compile 外壳
    return inner.module if isinstance(inner, DistributedDataParallel) else inner


def ddp_kwargs(local_rank=None):
    """DDP 构造参数：关闭每 forward 的 buffer 广播。

    模型 buffer 全是常量（每层 causal_mask 512×512×4B，10 层合计 10.5MB），DDP 默认
    每个 forward 都广播一遍，纯属浪费带宽外加一次集合通信同步。
    torch 2.13 起 `broadcast_buffers` 已废弃，优先用 `forward_sync_buffers`。
    """
    kwargs = {}
    if local_rank is not None:
        kwargs["device_ids"] = [local_rank]
    if "forward_sync_buffers" in inspect.signature(DistributedDataParallel.__init__).parameters:
        kwargs["forward_sync_buffers"] = False
    else:
        kwargs["broadcast_buffers"] = False
    return kwargs


def wrap_ddp(model, local_rank):
    """包一层 DDP，并屏蔽复数 buffer `pos_cis`（NCCL 不支持复数）。"""
    model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
    kwargs = ddp_kwargs(local_rank)
    wrapped = DistributedDataParallel(model, **kwargs)
    print(f"packaged model with DDP in cuda:{local_rank} ({kwargs})")
    return wrapped


def distributed_scalar_mean(total: float, count: int, device, enabled: bool) -> float:
    """跨 rank 的加权平均（**所有 rank 都必须调用**，内部含集合通信）。

    只做一次 all_reduce，替代"每个 micro-batch 一次 dist.reduce"。
    """
    if not enabled:
        return float(total) / max(1, int(count))
    t = torch.tensor([float(total), float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0] / max(1.0, float(t[1])))
