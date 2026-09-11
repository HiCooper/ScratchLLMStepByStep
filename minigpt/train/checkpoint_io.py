"""训练 checkpoint 的读写（payload 组装 + RNG/缩放器恢复），从 trainer.py 抽出。

抽出来的价值：RNG 往返、best 记录恢复、旧产物兼容这些逻辑可以脱离 Trainer
单独测试（以前只能靠"跑一遍训练"间接覆盖）。
"""
from __future__ import annotations

import os
import random as _random

import numpy as np
import torch

from minigpt.model.checkpoint import load_state_into_model
from minigpt.train.ddp_utils import unwrap_model


def build_payload(model, optimizer, *, epoch: int, step: int, best_eval_loss: float,
                  best_step, scaler=None, config=None) -> dict:
    """组装一个自包含的训练 checkpoint（模型/优化器/缩放器/步数/最优记录/RNG）。"""
    payload = {
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_eval_loss": best_eval_loss,
        "best_step": best_step,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": _random.getstate(),
        },
    }
    if config is not None:
        payload["config"] = config
    return payload


def save_training_checkpoint(path, model, optimizer, **kw) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(build_payload(model, optimizer, **kw), path)


def restore_rng_state(rng: dict) -> None:
    """恢复四路 RNG。

    注意：`torch.load(map_location=device)` 会把 RNG 的 ByteTensor 也搬到 GPU，而
    `set_rng_state` 要求 CPU ByteTensor，这里统一 `.cpu()` 修正（真实踩过的坑）。
    """
    if not rng:
        return
    torch_state = rng.get("torch")
    if torch_state is not None:
        if torch.is_tensor(torch_state):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if rng.get("cuda") and torch.cuda.is_available():
        cuda_states = [st.cpu() if torch.is_tensor(st) else st for st in rng["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("random") is not None:
        _random.setstate(tuple(rng["random"]))


def load_training_checkpoint(path, model, optimizer=None, scaler=None, device="cpu",
                             verbose=False) -> dict:
    """载入 checkpoint 并就地恢复模型/优化器/缩放器/RNG/最优记录。

    返回 {'epoch', 'step', 'best_eval_loss', 'best_step'}。
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    # 宽容加载：跨版本续训时良性键差异（如旧产物的 out_head.bias）不应中断训练，
    # 但真实的架构不匹配必须显式失败，避免"静默续训到错误权重"
    load_state_into_model(unwrap_model(model), ck["model_state"], verbose=verbose)
    if optimizer is not None and ck.get("optimizer_state") is not None:
        optimizer.load_state_dict(ck["optimizer_state"])
    if scaler is not None and ck.get("scaler_state") is not None:
        scaler.load_state_dict(ck["scaler_state"])
    restore_rng_state(ck.get("rng_state"))

    best = ck.get("best_eval_loss")
    print(f"load from checkpoint: {path}, last_epoch:{ck.get('epoch', 0)}, "
          f"last_step: {ck.get('step', 0)}")
    return {
        "epoch": ck.get("epoch", 0),
        "step": ck.get("step", 0),
        # 续训时恢复"历史最优"记录，避免 best.pt 覆盖后指标从 inf 重新起算
        "best_eval_loss": float(best) if best is not None else None,
        "best_step": ck.get("best_step"),
    }
