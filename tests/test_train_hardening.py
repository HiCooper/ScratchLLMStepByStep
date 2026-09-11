"""训练逻辑生产级加固的回归测试（对应评审里 5 个"未达标"项）。

1. eval loader 必须 shuffle=False（且不消耗全局 RNG）
2. 训练数据顺序必须确定、可复现，且与全局 RNG 无关（否则续训跳过的是另一批样本）
3. checkpoint 必须原子写（失败不能破坏上一个可用存档）
4. NCCL 超时必须可配且足够长（要覆盖 rank0 独占的 eval + 落盘）
5. weight decay 必须分组（LayerNorm/RMSNorm 尺度与 bias 不衰减）
"""
import os

import pytest
import torch

from minigpt.config import TrainConfig
from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train import checkpoint_io
from minigpt.train.optim import build_optimizer, build_param_groups, is_decay_param
from minigpt.train.trainer import Trainer


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                             context_length=8, drop_rate=0.0))


class _IdsDS(torch.utils.data.Dataset):
    """每条样本的 X 是它自己的编号，便于断言"取到了哪些样本"。"""

    def __init__(self, n=64, seq=8):
        self.n, self.seq = n, seq

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (torch.full((self.seq,), i, dtype=torch.long),
                torch.zeros(self.seq, dtype=torch.long))


def _collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def _ids(loader):
    """把 loader 走一遍，返回每个 batch 的样本编号（取 X 的第一个 token）。"""
    return [[int(x) for x in batch[0][:, 0]] for batch in loader]


def _make(n=64, batch=4, accum=1, seed=123, **extra):
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-3)
    args = {"train_batch_size": batch, "gradient_accumulation_steps": accum,
            "num_train_epochs": 2, "seed": seed}
    args.update(extra)
    tr = Trainer(model, opt, args, device="cpu")
    tr.set_dataset(_IdsDS(n), _IdsDS(16), _collate)
    return tr


# --------------------------------------------------------------- 1) eval loader
def test_eval_loader_is_not_shuffled():
    tr = _make()
    tr._init_dataloader()
    first = _ids(tr.eval_loader)
    second = _ids(tr.eval_loader)
    assert first == second, "eval 顺序必须稳定"
    # 顺序就是原始下标顺序（0,1,2,...）
    flat = [i for b in first for i in b]
    assert flat == list(range(16))


def test_eval_does_not_consume_global_rng():
    """旧实现 eval_loader shuffle=True，会消耗训练 RNG，使 eval 疏密改变训练数据顺序。"""
    tr = _make()
    tr._init_dataloader()
    before = torch.get_rng_state()
    for _ in tr.eval_loader:
        pass
    after = torch.get_rng_state()
    assert torch.equal(before, after), "eval 不应触碰全局 RNG"


# ------------------------------------------------------- 2) train order determinism
def test_train_order_deterministic_across_instances():
    a = _make(seed=7)
    a._init_dataloader()
    a.set_train_epoch(0)
    order_a = _ids(a.train_loader)

    b = _make(seed=7)
    b._init_dataloader()
    b.set_train_epoch(0)
    assert _ids(b.train_loader) == order_a, "同 seed 的数据顺序必须一致"


def test_train_order_independent_of_global_rng():
    """核心回归：训练顺序不得依赖全局 RNG，否则续训恢复 RNG 后会拿到另一个排列。"""
    tr = _make(seed=7)
    tr._init_dataloader()
    tr.set_train_epoch(0)
    base = _ids(tr.train_loader)

    torch.manual_seed(999999)          # 任意扰动全局 RNG
    tr.set_train_epoch(0)
    assert _ids(tr.train_loader) == base, "数据顺序被全局 RNG 影响了"


def test_train_order_differs_between_epochs():
    tr = _make(seed=7)
    tr._init_dataloader()
    tr.set_train_epoch(0)
    e0 = _ids(tr.train_loader)
    tr.set_train_epoch(1)
    e1 = _ids(tr.train_loader)
    assert e0 != e1, "不同 epoch 必须重新打乱"
    # 且每个 epoch 覆盖全部样本一次（drop_last=True 时是前 N//batch*batch 条）
    assert sorted(i for b in e0 for i in b) == sorted(i for b in e1 for i in b)


def test_resume_skips_the_same_samples():
    """续训语义：跳过前 k 个 micro-batch 后，剩下的必须与未中断时完全一致。"""
    tr = _make(n=64, batch=4, seed=5)
    tr._init_dataloader()
    tr.set_train_epoch(0)
    full = _ids(tr.train_loader)

    resumed = _make(n=64, batch=4, seed=5)
    resumed._init_dataloader()
    resumed.step = 3                       # 已完成 3 次更新（accum=1 => 跳过 3 个 batch）
    torch.manual_seed(4242)                # 模拟"恢复了一份中段 RNG 状态"
    resumed.set_train_epoch(0)
    skip_micro = resumed.step - 0 * resumed.updates_per_epoch

    seen = []
    for i, batch in enumerate(resumed.train_loader):
        if i < skip_micro:      # 与 _train_epoch 的 skip 逻辑一致
            continue
        seen.append([int(x) for x in batch[0][:, 0]])
    assert seen == full[skip_micro:], "续训跳过的不是同一批样本（数据会重复/漏训）"


# ------------------------------------------------------------ 3) atomic checkpoint
def _save(path):
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-3)
    checkpoint_io.save_training_checkpoint(str(path), model, opt, epoch=0, step=1,
                                          best_eval_loss=1.0, best_step=1,
                                          config=model.config.to_dict())


def test_atomic_save_leaves_no_temp_file(tmp_path):
    path = tmp_path / "ck.pth"
    _save(path)
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp*")), "临时文件未清理"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ck["step"] == 1


def test_atomic_save_keeps_previous_on_failure(tmp_path, monkeypatch):
    """真实风险：写坏文件后"续训读最近 checkpoint"会把一次可恢复故障放大成整轮报废。"""
    path = tmp_path / "ck.pth"
    _save(path)
    good = path.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(checkpoint_io.torch, "save", boom)
    with pytest.raises(RuntimeError):
        _save(path)

    assert path.read_bytes() == good, "失败时不得破坏上一个可用 checkpoint"
    assert not list(tmp_path.glob("*.tmp*")), "失败后临时文件必须被清理"


def test_atomic_save_overwrites_atomically(tmp_path):
    path = tmp_path / "ck.pth"
    _save(path)
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-3)
    checkpoint_io.save_training_checkpoint(str(path), model, opt, epoch=1, step=99,
                                          best_eval_loss=0.5, best_step=99)
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ck["step"] == 99 and ck["epoch"] == 1


# ---------------------------------------------------------------- 4) DDP timeout
def test_ddp_timeout_default_and_override():
    assert _make().ddp_timeout_seconds == 1800
    assert _make(ddp_timeout_seconds=60).ddp_timeout_seconds == 60
    # 必须显著大于旧的 120s：rank0 的完整 eval + 数百 MB 落盘都在 barrier 内
    assert TrainConfig().ddp_timeout_seconds >= 600


def test_deterministic_cudnn_flag_is_configurable():
    assert _make().deterministic_cudnn is False
    assert _make(deterministic_cudnn=True).deterministic_cudnn is True
    tr = _make(deterministic_cudnn=True)
    tr.set_seed(123)
    assert torch.backends.cudnn.deterministic is True
    tr2 = _make(deterministic_cudnn=False)
    tr2.set_seed(123)
    assert torch.backends.cudnn.deterministic is False


# --------------------------------------------------------- 5) weight decay groups
def test_is_decay_param_rule():
    m = _tiny_model()
    by_name = dict(m.named_parameters())
    assert is_decay_param("decode_layers.0.ffn.layers.0.weight", by_name["decode_layers.0.ffn.layers.0.weight"])
    assert not is_decay_param("decode_layers.0.ffn.layers.0.bias", by_name["decode_layers.0.ffn.layers.0.bias"])
    assert not is_decay_param("final_norm.scale", by_name["final_norm.scale"])
    assert not is_decay_param("final_norm.shift", by_name["final_norm.shift"])
    assert is_decay_param("token_emb.weight", by_name["token_emb.weight"])


def test_weight_decay_groups_partition_all_params():
    m = _tiny_model()
    groups = build_param_groups(m, 0.1)
    assert len(groups) == 2
    decay_ids = {id(p) for p in groups[0]["params"]}
    nodecay_ids = {id(p) for p in groups[1]["params"]}
    assert not (decay_ids & nodecay_ids)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    # 覆盖全部可训练参数，一个不漏
    all_ids = {id(p) for p in m.parameters() if p.requires_grad}
    assert decay_ids | nodecay_ids == all_ids
    # 归一化参数必须落在 0 衰减组
    for name, p in m.named_parameters():
        if name.endswith((".scale", ".shift", ".bias")):
            assert id(p) in nodecay_ids, f"{name} 不应被 weight decay"


def test_norm_params_are_not_decayed_by_optimizer_step():
    """梯度为 0 时，被衰减的参数会因 decoupled decay 变小；不衰减的必须纹丝不动。"""
    m = _tiny_model()
    opt = build_optimizer(m, lr=0.1, weight_decay=0.5)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    scale_before = m.final_norm.scale.detach().clone()
    w_before = m.decode_layers[0].ffn.layers[0].weight.detach().clone()
    opt.step()
    assert torch.allclose(m.final_norm.scale, scale_before), "LayerNorm scale 被 decay 了"
    assert not torch.allclose(m.decode_layers[0].ffn.layers[0].weight, w_before), \
        "2 维权重应当被 decay"


def test_build_optimizer_pins_betas_and_eps():
    m = _tiny_model()
    opt = build_optimizer(m, lr=1e-3, weight_decay=0.01)
    assert opt.defaults["betas"] == (0.9, 0.999)
    assert opt.defaults["eps"] == 1e-8
