"""学习率调度与 loss 记账（从 trainer.py 抽出，便于单独测试与复用）。

这里只放**纯函数/纯状态**，不依赖模型、优化器或分布式环境。
"""
from __future__ import annotations

import math


def get_dynamic_lr(target_lr: float, cur_step: int, warmup_steps: int, decay_steps: int) -> float:
    """线性 warmup + 余弦退火到 `target_lr/10`。

    `cur_step` 是**即将执行的更新序号（1-based）**。

    真实事故：调用方以前传"已完成的更新数"，第一次更新时 cur_step=0 → warmup 分支算出
    lr=0 → 上层走 `if lr <= 0: return target_lr` 直接返回、**没有写回 param_group**，
    优化器保持初始 lr（=峰值）：warmup 首步被整段跳过，fp16 下很容易打出 loss 尖峰。
    另外 `warmup_steps == decay_steps` 时 `(cur-warmup)/(decay-warmup)` 会 ZeroDivisionError。
    """
    min_lr = target_lr / 10
    warmup_steps = max(0, int(warmup_steps))
    decay_steps = max(1, int(decay_steps))
    if warmup_steps > 0 and cur_step <= warmup_steps:
        return target_lr * (cur_step / warmup_steps)
    if cur_step >= decay_steps:
        return min_lr
    # progress 以 cur_step-1 为基准：warmup_steps=0 时第一次更新恰好等于 target_lr
    progress = (cur_step - 1 - warmup_steps) / max(1, decay_steps - 1 - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + (target_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class LossAccumulator:
    """按 (sum, count) 记账的 train loss 累加器。

    旧实现用"loss 之和 ÷ (eval_steps × grad_accum)"：在最后一个不完整的 eval 窗口、
    以及 batch 数不能被 accumulation 整除时都会系统性偏小（实测残窗 1.418，真值约 2.85）。
    改成同时记 micro-batch 数后，取平均与 eval 间隔是否整除无关。

    分布式下各 rank 只做本地累加，需要全局均值时调用 `distributed_mean()` 一次性
    集合通信（旧实现每个 micro-batch 都做一次 `dist.reduce`）。
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0
        self.last_mean = None
        self.last_count = 0

    def add(self, loss) -> None:
        self.total += float(loss.detach() if hasattr(loss, "detach") else loss)
        self.count += 1

    def mean(self) -> float:
        return self.total / max(1, self.count)

    def snapshot_and_reset(self) -> tuple[float, int]:
        """记录最近一个窗口的均值与样本数，然后清零。"""
        mean = self.mean()
        self.last_mean, self.last_count = mean, self.count
        self.total, self.count = 0.0, 0
        return mean, self.last_count
