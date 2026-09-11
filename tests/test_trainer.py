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


def _build(tmp_path, train_args_extra, ckpt_step=100000, ckpt_epoch=5, n=8):
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
    ds = _ListDS(n)
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


# ---------------------------------------------------------------------------
# 回归：LR 调度
#
# 真实事故：`_adjust_lr` 传的是"已完成更新数"，第一次更新时算出 lr=0 → 走
# `if lr <= 0: return target_lr` 提前返回且**不写回 param_group**，优化器保持初始
# lr（=峰值），warmup 首步被跳过；且 horizon 用 total_steps 而非 effective_max，
# 增量续训时余弦永远走不完。另有 warmup_steps==decay_steps 的 ZeroDivisionError。
# ---------------------------------------------------------------------------
def test_lr_warmup_first_update_is_not_peak():
    f = Trainer._get_dynamic_lr
    assert f(1e-3, 1, 100, 1000) == 1e-3 / 100      # 第 1 次更新 = target/warmup
    assert f(1e-3, 100, 100, 1000) == 1e-3          # warmup 结束 = target
    assert f(1e-3, 1000, 100, 1000) == 1e-4         # horizon 末端 = min_lr
    assert f(1e-3, 5000, 100, 1000) == 1e-4         # 超出后保持 min_lr
    assert f(1e-3, 1, 0, 1000) == 1e-3              # warmup=0 => 直接 target


def test_lr_schedule_no_div_by_zero():
    f = Trainer._get_dynamic_lr
    assert f(1e-3, 100, 100, 100) == 1e-3           # 旧实现在这里 ZeroDivisionError
    assert f(1e-3, 101, 100, 100) == 1e-4
    assert 0 < f(1e-3, 50, 100, 100) < 1e-3


def test_adjust_lr_uses_next_update_index(small_config):
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = Trainer(model, opt, {"warmup_steps": 10}, device="cpu")
    tr.total_steps, tr.effective_max = 100, 100
    lr = tr._adjust_lr()
    assert lr == 1e-3 / 10, "第 1 次更新必须处于 warmup 起点"
    assert opt.param_groups[0]["lr"] == lr, "必须写回 param_group，否则优化器仍用峰值 lr"


def test_adjust_lr_horizon_follows_effective_max(small_config):
    """horizon 必须用 effective_max：否则 max_steps/增量续训时余弦走不完。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = Trainer(model, opt, {"warmup_steps": 1}, device="cpu")
    tr.total_steps = 1_000_000      # epochs×每epoch步数（远大于本次目标）
    tr.effective_max = 100
    tr.step = 99                    # 最后一次更新
    lr = tr._adjust_lr()
    assert lr == 1e-4, f"末端 lr={lr}，应已衰减到 min_lr（说明 horizon 用错成 total_steps）"


def test_eval_triggers_on_exact_step_boundary(tmp_path):
    """eval 必须恰好每 eval_steps 次更新触发一次（旧实现 (step+1)%eval_steps 差 1）。"""
    tr = _build(tmp_path, {"eval_steps": 3, "max_steps": 6, "reset_step": True})
    seen = []
    tr._record_metrics = lambda tl, el, gn, lr: seen.append((tr.step, tl))
    tr.train()
    assert [s for s, _ in seen] == [3, 6], f"eval 触发在 {[s for s, _ in seen]}，应为 [3, 6]"


def test_train_loss_accounting_averages_micro_batches(tmp_path):
    """train_loss 必须按实际 micro-batch 数取平均，与 eval 间隔是否整除无关。

    数据集取 32 条 / batch 4 = 8 个 batch/epoch，累积 2 => 4 次更新/epoch；
    max_steps=eval_steps=4 => 恰好 8 个 micro-batch 后触发一次 eval。
    旧实现用 `/(eval_steps × accum)` 作分母，在不整除/残窗时会系统性偏小。
    """
    tr = _build(tmp_path, {"eval_steps": 4, "max_steps": 4, "reset_step": True,
                           "gradient_accumulation_steps": 2, "num_train_epochs": 1},
                n=32)
    seen = []
    tr._record_metrics = lambda tl, el, gn, lr: seen.append((tr.step, tl))
    tr.train()
    assert len(seen) == 1 and seen[0][0] == 4
    assert tr.last_train_loss_count == 8, \
        f"参与平均的 micro-batch 数={tr.last_train_loss_count}，应为 8"
    assert tr.micro_count == 0, "eval 触发后计数器应清零"
    assert seen[0][1] == tr.last_train_loss
    # 8 个 micro-batch 的平均，量级应在 ln(16)=2.77 附近
    assert 1.0 < seen[0][1] < 5.0, f"train_loss={seen[0][1]} 不合理（疑似分母错）"


# ---------------------------------------------------------------------------
# DDP 通信优化
# ---------------------------------------------------------------------------
def test_ddp_disables_buffer_broadcast():
    """每 forward 广播 10.5MB 常量 buffer 纯属浪费（causal_mask 每层 1.05MB × 10 层）。"""
    kw = Trainer._ddp_kwargs(local_rank=0)
    assert kw["device_ids"] == [0]
    assert kw.get("forward_sync_buffers") is False or kw.get("broadcast_buffers") is False


class _StubDDP:
    """模拟 DDP 外壳：只提供 no_sync / 前向 / parameters。"""

    def __init__(self, model):
        self.module = model
        self.no_sync_calls = 0

    def no_sync(self):
        self.no_sync_calls += 1
        from contextlib import nullcontext
        return nullcontext()

    def __call__(self, *a, **kw):
        return self.module(*a, **kw)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()


def test_no_sync_only_on_non_last_micro_batch(small_config):
    """累积期间必须跳过 all-reduce；旧实现每个 micro-batch 都同步一次（通信量 ×accum）。"""
    torch.manual_seed(0)
    stub = _StubDDP(MiniGPT(small_config))
    opt = torch.optim.AdamW(stub.parameters(), lr=1e-3)
    tr = Trainer(stub, opt, {"gradient_accumulation_steps": 3}, device="cpu")
    tr.ddp = True

    X = torch.randint(0, small_config.vocab_size, (2, 8))
    Y = torch.randint(0, small_config.vocab_size, (2, 8))
    _, u1 = tr._train_step(X, Y, None)     # micro 1：不同步
    _, u2 = tr._train_step(X, Y, None)     # micro 2：不同步
    assert stub.no_sync_calls == 2
    _, u3 = tr._train_step(X, Y, None)     # micro 3（最后一个）：必须同步
    assert u1 is False and u2 is False and u3 is True
    assert stub.no_sync_calls == 2, "最后一个 micro-batch 不应再跳过同步"


def test_grad_norm_matches_manual_computation(small_config):
    """_foreach_norm 批量实现必须与逐参数平方和开方等价（只做一次 host 同步）。"""
    torch.manual_seed(0)
    model = MiniGPT(small_config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tr = Trainer(model, opt, {"train_batch_size": 2}, device="cpu")
    X = torch.randint(0, small_config.vocab_size, (2, 16))
    Y = torch.randint(0, small_config.vocab_size, (2, 16))
    tr._train_step(X, Y, None)
    manual = sum(float(p.grad.detach().norm(2)) ** 2 for p in model.parameters()
                 if p.grad is not None) ** 0.5
    # _train_step 之后梯度已被 zero_grad 清空，改用未清零的路径比较
    model.zero_grad(set_to_none=True)
    logits = model(X)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), Y.flatten()).backward()
    fast = tr._calc_grad_norm()
    manual = sum(float(p.grad.detach().norm(2)) ** 2 for p in model.parameters()
                 if p.grad is not None) ** 0.5
    assert abs(fast - manual) < 1e-4, f"fast={fast} manual={manual}"


def test_final_train_loss_not_zero_when_last_eval_at_end(tmp_path):
    """回归：最后一次 eval 恰好落在终点时，metrics 里的 train_loss 不能记成 0。

    真实问题：eval 触发会清零 loss 累加器，随后收尾再取一次全局均值就得到 0.0——
    冒烟跑出来的 metrics.json 里 train_loss=0.0 正是这个原因。
    """
    tr = _build(tmp_path, {"eval_steps": 4, "max_steps": 4, "reset_step": True,
                           "gradient_accumulation_steps": 2, "num_train_epochs": 1},
                n=32)
    tr.train()
    tl = tr.final_metrics.get("train_loss")
    assert tl is not None and tl > 0.5, f"train_loss={tl}（0 或过小说明取了空累加器）"
