"""训练器的单元测试：梯度范数、梯度累积的步数计数。"""
import torch

from minigpt.model.transformer import MiniGPT
from minigpt.train.trainer import Trainer


def test_grad_norm_nonzero(small_config):
    """回归测试：修复前 grad_norm 因在 zero_grad 之后计算而恒为 0。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, {"train_batch_size": 2}, device="cpu")

    X = torch.randint(0, small_config.vocab_size, (2, 16))
    Y = torch.randint(0, small_config.vocab_size, (2, 16))
    _, did_update = trainer._train_step(X, Y, None)

    assert did_update is True
    assert trainer.last_grad_norm > 0


def test_gradient_accumulation_step_count(small_config):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer,
        {"train_batch_size": 2, "gradient_accumulation_steps": 2},
        device="cpu",
    )

    X = torch.randint(0, small_config.vocab_size, (2, 16))
    Y = torch.randint(0, small_config.vocab_size, (2, 16))
    _, upd1 = trainer._train_step(X, Y, None)
    _, upd2 = trainer._train_step(X, Y, None)

    assert upd1 is False  # 第 1 个 micro-batch 只累积梯度，不更新参数
    assert upd2 is True   # 第 2 个 micro-batch 累积满，触发一次参数更新
