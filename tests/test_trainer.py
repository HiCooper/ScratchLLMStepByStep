"""训练器的单元测试：梯度范数、梯度累积的步数计数。"""
import torch

from minigpt.model.transformer import GPTConfig, MiniGPT
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


class _ListDS(torch.utils.data.Dataset):
    """最小可训练数据集：返回 (X, Y) 使 Trainer 能跑完 1 个 epoch。"""

    def __init__(self, n=8, seq=8, vocab=16, base=0):
        self.n, self.seq, self.vocab, self.base = n, seq, vocab, base
        self.X = torch.randint(base, base + vocab, (n, seq))
        self.Y = torch.randint(base, base + vocab, (n, seq))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def _collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def _build(tmp_path, train_args_extra, ckpt_step=100000, ckpt_epoch=5):
    """构造一个"从高步数 checkpoint 续训"的 trainer。"""
    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(emb_dim=16, n_layers=1, n_heads=2, context_length=8,
                              vocab_size=16, drop_rate=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ck = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
          "epoch": ckpt_epoch, "step": ckpt_step}
    ckpt_file = tmp_path / "checkpoint.pth"
    torch.save(ck, ckpt_file)

    args = {"train_batch_size": 4, "eval_steps": 2, "save_steps": 10000,
            "warmup_steps": 1, "gradient_accumulation_steps": 1, "grad_clip": 1.0,
            "output_dir": str(tmp_path / "out"), "save_best": False,
            "last_checkpoint_path": str(ckpt_file), "num_train_epochs": 5,
            "use_mixed_precision": False}
    args.update(train_args_extra)
    trainer = Trainer(model, optimizer, args, device="cpu")
    ds = _ListDS(8)
    trainer.set_dataset(ds, ds, _collate)
    return trainer


def test_incremental_pretrain_without_reset_step_does_nothing(tmp_path):
    """回归（真实事故）：从 211000 步基座 + max_steps=100 会被判定"已训完"，一步都不训。"""
    tr = _build(tmp_path, {"max_steps": 100}, ckpt_step=100000)
    tr.train()
    assert tr.step == 100          # 被 min(step, effective_max) 截断
    assert not (tmp_path / "best.pt").exists()


def test_incremental_pretrain_with_reset_step_actually_trains(tmp_path):
    """reset_step=True 时步数归零，真正从基座继续训练额外的 N 步。"""
    tr = _build(tmp_path, {"max_steps": 4, "reset_step": True}, ckpt_step=100000)
    tr.train()
    assert tr.step == 4            # 真正训满 4 步，而不是立刻结束


def test_extra_steps_means_train_n_more_steps(tmp_path):
    """extra_steps=N 的语义是"从现在起再训 N 步"（隐含步数归零）。"""
    tr = _build(tmp_path, {"extra_steps": 3}, ckpt_step=50, ckpt_epoch=0)
    tr.train()
    assert tr.effective_max == 3
    assert tr.step == 3
