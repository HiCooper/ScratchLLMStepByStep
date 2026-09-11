"""新抽出模块的单测：schedule / ddp_utils / checkpoint_io。

这些逻辑以前只能靠"跑一遍完整训练"间接覆盖（RNG 往返、DDP 解包、LR 调度都属于
出过真实事故的地方），抽成纯模块后可以精确断言。
"""
import math
import random

import numpy as np
import pytest
import torch

from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train import checkpoint_io
from minigpt.train.ddp_utils import ddp_kwargs, distributed_scalar_mean, unwrap_model
from minigpt.train.schedule import LossAccumulator, get_dynamic_lr


# ------------------------------------------------------------------ schedule
def test_get_dynamic_lr_warmup_and_decay():
    f = get_dynamic_lr
    assert f(1e-3, 1, 100, 1000) == 1e-3 / 100     # 第一次更新 = target/warmup
    assert f(1e-3, 100, 100, 1000) == 1e-3         # warmup 结束 = target
    assert f(1e-3, 1000, 100, 1000) == 1e-4        # 末端 = min_lr
    assert f(1e-3, 99999, 100, 1000) == 1e-4       # 超出后保持 min_lr
    assert f(1e-3, 1, 0, 1000) == 1e-3             # warmup=0 => 直接 target
    mid = f(1e-3, 550, 100, 1000)
    assert 1e-4 < mid < 1e-3, "warmup 与末端之间必须单调过渡"


def test_get_dynamic_lr_is_monotonic_after_warmup():
    vals = [get_dynamic_lr(1e-3, s, 10, 100) for s in range(10, 101)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), "退火阶段必须单调不增"


def test_get_dynamic_lr_no_div_by_zero():
    assert get_dynamic_lr(1e-3, 100, 100, 100) == 1e-3   # 旧实现 ZeroDivisionError
    assert get_dynamic_lr(1e-3, 101, 100, 100) == 1e-4


def test_loss_accumulator_counts_micro_batches():
    acc = LossAccumulator()
    for v in (1.0, 2.0, 3.0):
        acc.add(torch.tensor(v))
    assert acc.count == 3 and acc.mean() == pytest.approx(2.0)
    mean, n = acc.snapshot_and_reset()
    assert mean == pytest.approx(2.0) and n == 3
    assert acc.count == 0 and acc.last_mean == pytest.approx(2.0)


def test_distributed_scalar_mean_local_path():
    assert distributed_scalar_mean(6.0, 3, device="cpu", enabled=False) == pytest.approx(2.0)
    assert distributed_scalar_mean(0.0, 0, device="cpu", enabled=False) == 0.0


# ------------------------------------------------------------------ ddp utils
def test_unwrap_model_handles_plain_and_compiled():
    model = MiniGPT(GPTConfig(vocab_size=16, emb_dim=16, n_layers=1, n_heads=2,
                              context_length=8, drop_rate=0.0))
    assert unwrap_model(model) is model
    compiled = torch.compile(model)
    assert unwrap_model(compiled) is model


def test_ddp_kwargs_disables_buffer_broadcast():
    kw = ddp_kwargs(local_rank=1)
    assert kw["device_ids"] == [1]
    assert kw.get("forward_sync_buffers") is False or kw.get("broadcast_buffers") is False


# ------------------------------------------------------------- checkpoint_io
def _tiny_model():
    torch.manual_seed(0)
    return MiniGPT(GPTConfig(vocab_size=32, emb_dim=16, n_layers=1, n_heads=2,
                             context_length=8, drop_rate=0.0))


def test_checkpoint_payload_roundtrip(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ck.pth"
    checkpoint_io.save_training_checkpoint(
        str(path), model, opt, epoch=2, step=17, best_eval_loss=1.25, best_step=11,
        scaler=None, config={"model_type": "minigpt"})

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ck["step"] == 17 and ck["epoch"] == 2
    assert ck["best_eval_loss"] == 1.25 and ck["best_step"] == 11
    assert ck["config"]["model_type"] == "minigpt"
    assert set(ck["rng_state"]) >= {"torch", "numpy", "random"}
    # 键名不得带 DDP 的 module. 前缀
    assert not any(k.startswith("module.") for k in ck["model_state"])

    model2 = _tiny_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=5e-4)
    info = checkpoint_io.load_training_checkpoint(str(path), model2, opt2, device="cpu")
    assert info["step"] == 17 and info["best_eval_loss"] == 1.25
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b), "权重未正确恢复"


def test_restore_rng_state_reproduces_sequence(tmp_path):
    """RNG 往返必须能复现同一串随机数（续训可复现性的基础）。"""
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    for _ in range(3):
        random.random(); np.random.rand(); torch.rand(1)

    state = {
        "torch": torch.get_rng_state(),
        "cuda": None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    expect = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    for _ in range(10):     # 把状态打乱
        random.random(); np.random.rand(); torch.rand(1)
    checkpoint_io.restore_rng_state(state)
    got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert got == pytest.approx(expect), f"RNG 未复现: {got} vs {expect}"


def test_restore_rng_state_tolerates_empty():
    checkpoint_io.restore_rng_state({})
    checkpoint_io.restore_rng_state(None)


def test_checkpoint_io_ignores_legacy_head_bias(tmp_path):
    """旧产物带 out_head.bias、新结构不带：续训不能因此中断。"""
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pth"
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["out_head.bias"] = torch.zeros(32)          # 模拟旧结构
    torch.save({"model_state": sd, "optimizer_state": opt.state_dict(),
                "epoch": 0, "step": 5}, str(path))

    model2 = _tiny_model()
    info = checkpoint_io.load_training_checkpoint(
        str(path), model2, torch.optim.AdamW(model2.parameters(), lr=1e-3), device="cpu")
    assert info["step"] == 5
