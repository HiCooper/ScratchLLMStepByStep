"""DDP + torch.compile 解包顺序的回归测试。

真实事故：`train()` 里先包 DDP 再 `torch.compile`，嵌套顺序是
`OptimizedModule(DistributedDataParallel(MiniGPT))`。旧的 `_unwrap()` 先判
`isinstance(model, DistributedDataParallel)`——对最外层 OptimizedModule 为 False——
再去取 `_orig_mod`，于是返回的是 **DDP 本身**，`state_dict()` 的键全部带 `module.`
前缀（实测 `module.token_emb.weight`）。后果是 multi-GPU 预设（`torch_compile=True`）
产出的 checkpoint 被 `checkpoint.py` 判成"缺少关键权重"，既无法评测也无法续训。

本测试用 gloo 单进程组在 CPU 上复现这个嵌套，无需真实多卡。
"""
import os
import socket

import pytest
import torch

from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.trainer import Trainer


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tiny_model():
    torch.manual_seed(0)
    return MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                             context_length=16, drop_rate=0.0))


def _make_trainer(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return Trainer(model, opt, {"output_dir": None}, device="cpu", verbose=False)


@pytest.fixture(scope="module")
def pg():
    """单进程 gloo 组；不可用时跳过而不是让整个测试套件失败。"""
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    try:
        dist.init_process_group("gloo", rank=0, world_size=1)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"无法初始化 gloo 进程组: {exc}")
    yield dist
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_ref(trainer, model):
    """复刻修复后的解包逻辑，返回真实 MiniGPT。"""
    trainer.model = model
    return trainer._unwrap()


def test_unwrap_plain_model():
    model = _tiny_model()
    t = _make_trainer(model)
    assert t._unwrap() is model


def test_unwrap_ddp_only(pg):
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = _tiny_model()
    model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
    wrapped = DDP(model)
    t = _make_trainer(wrapped)
    unwrapped = t._unwrap()
    assert isinstance(unwrapped, MiniGPT)
    assert not any(k.startswith("module.") for k in unwrapped.state_dict())


def test_unwrap_ddp_then_compile_has_no_module_prefix(pg):
    """核心回归：DDP 内层 + torch.compile 外层的嵌套必须解到 MiniGPT。"""
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = _tiny_model()
    model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
    nested = torch.compile(DDP(model))          # 与 trainer.train() 的嵌套顺序一致

    t = _make_trainer(nested)
    unwrapped = t._unwrap()

    assert isinstance(unwrapped, MiniGPT), \
        f"_unwrap 返回了 {type(unwrapped).__name__}，应为 MiniGPT"
    keys = list(unwrapped.state_dict())
    assert not any(k.startswith("module.") for k in keys), \
        f"state_dict 仍带 module. 前缀: {keys[:3]}"
    assert "token_emb.weight" in keys


def test_saved_checkpoint_loads_back_after_ddp_compile(pg, tmp_path):
    """端到端：DDP+compile 下保存的 checkpoint 必须能被 checkpoint.py 正常重建。"""
    from torch.nn.parallel import DistributedDataParallel as DDP
    from minigpt.model.checkpoint import model_kwargs_from_checkpoint

    model = _tiny_model()
    model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
    nested = torch.compile(DDP(model))
    t = _make_trainer(nested)
    t.step = 3
    t.extra_ckpt = model.config.to_dict()
    path = tmp_path / "checkpoint-3.pth"
    t._save_model(str(path), epoch=0)

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    keys = list(ck["model_state"])
    assert not any(k.startswith("module.") for k in keys)
    # checkpoint.py 的加载路径必须能认出结构（否则就是"缺少关键权重"错误）
    kws, inferred = model_kwargs_from_checkpoint(ck, vocab_size=64)
    assert not inferred
    reloaded = MiniGPT(GPTConfig(**{k: v for k, v in kws.items()}))
    reloaded.load_state_dict(ck["model_state"])
