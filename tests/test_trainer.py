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


def test_save_best_checkpoint_on_eval_improvement(small_config, tmp_path):
    """eval_loss 创新低时写出 best.pt 并记录 best_step；变差时不再覆盖。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer,
        {"train_batch_size": 2, "output_dir": str(tmp_path), "save_best": True},
        device="cpu",
    )

    trainer.step, trainer.cur_epoch = 100, 0
    trainer._record_metrics(1.0, 2.0, 0.5, 1e-3)
    assert (tmp_path / "best.pt").is_file()
    assert trainer.best_eval_loss == 2.0 and trainer.best_step == 100

    first = (tmp_path / "best.pt").read_bytes()
    trainer.step = 200
    trainer._record_metrics(1.0, 2.5, 0.5, 1e-3)          # 变差：不覆盖
    assert (tmp_path / "best.pt").read_bytes() == first
    assert trainer.best_eval_loss == 2.0 and trainer.best_step == 100

    trainer.step = 300
    trainer._record_metrics(1.0, 1.5, 0.5, 1e-3)          # 变好：覆盖
    assert trainer.best_eval_loss == 1.5 and trainer.best_step == 300


def test_save_best_can_be_disabled(small_config, tmp_path):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer,
        {"train_batch_size": 2, "output_dir": str(tmp_path), "save_best": False},
        device="cpu",
    )
    trainer.step = 10
    trainer._record_metrics(1.0, 1.0, 0.5, 1e-3)
    assert not (tmp_path / "best.pt").exists()
    assert trainer.best_eval_loss == 1.0                   # 仍记录指标，只是不落盘
