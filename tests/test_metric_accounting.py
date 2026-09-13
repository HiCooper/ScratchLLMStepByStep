"""训练指标的记账口径：吞吐（tok/s）与 eval loss 必须按**正确的分母**统计。

两处真实偏差（都会静默给出错数字，不报错）：

1. 吞吐：`metrics.before_zero_grad(tokens=X.numel())` 里 X 只是**最后一个** micro-batch，
   DDP 下还只算了 rank0 自己那份 ⇒ accum=N / N 卡时 tok/s 低报约 N 倍、ETA 高报约 N 倍。
2. eval loss：`total_loss / num_batches` 是"各 batch 均值的均值"，而 eval loader 是
   `drop_last=False` 且 SFT 的 label 带 -100 掩码 ⇒ 样本少的那个 batch 与满 batch 等权，
   `best.pt` 的挑选取决于 batch 组成而不是模型质量。
"""
import torch
import torch.nn.functional as f

from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.optim import build_optimizer
from minigpt.train.trainer import Trainer


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                             context_length=8, drop_rate=0.0))


def _make(batch=4, accum=1, seq=8):
    model = _tiny_model()
    opt = build_optimizer(model, lr=1e-3)
    tr = Trainer(model, opt, {"train_batch_size": batch,
                              "gradient_accumulation_steps": accum,
                              "num_train_epochs": 1, "seed": 123}, device="cpu")
    return tr


class _Capture:
    """只记 tokens 的 metrics 替身（真实 MetricsLogger 会把它们写进 tensorboard）。"""

    def __init__(self):
        self.tokens = []

    def before_zero_grad(self, step, tokens, batch=None):
        self.tokens.append(tokens)

    def on_train_step(self, *a, **k):
        pass


# --------------------------------------------------------------- 1) 吞吐记账
def test_throughput_counts_all_micro_batches_of_one_update():
    """一次参数更新吃 accum 个 micro-batch 的 token，不能只报最后一个。"""
    tr = _make(batch=4, accum=4, seq=8)          # 每 micro-batch 4×8 = 32 token
    tr.metrics = cap = _Capture()
    X = torch.zeros(4, 8, dtype=torch.long)
    Y = torch.zeros(4, 8, dtype=torch.long)
    for _ in range(4):                           # 累积满 4 次才构成一次更新
        tr._train_step(X, Y, None)
    assert cap.tokens == [128], f"应记 4×32=128，实际 {cap.tokens}"
    # 下一次更新必须从 0 重新累积（否则吞吐会随 step 单调虚增）
    for _ in range(4):
        tr._train_step(X, Y, None)
    assert cap.tokens == [128, 128], f"累积计数未复位：{cap.tokens}"


def test_throughput_covers_all_ranks_for_global_rate():
    """DDP 下 rank0 记的应是**全局**吞吐：乘上 world_size，否则低报卡数倍。"""
    tr = _make(batch=4, accum=2, seq=8)
    tr.world_size = 3
    tr.metrics = cap = _Capture()
    X = torch.zeros(4, 8, dtype=torch.long)
    Y = torch.zeros(4, 8, dtype=torch.long)
    for _ in range(2):
        tr._train_step(X, Y, None)
    assert cap.tokens == [2 * 32 * 3], f"未乘世界大小：{cap.tokens}"


def test_world_size_defaults_to_one():
    assert _make().world_size == 1


# ------------------------------------------------------------- 2) eval 分母
def _masked_batches(lens):
    """构造 (X, Y)：Y 里只有前 L 个位置是监督 token，其余 -100（模拟 SFT 掩码）。"""
    out = []
    for L in lens:
        X = torch.ones(1, 8, dtype=torch.long)
        Y = torch.full((1, 8), -100, dtype=torch.long)
        Y[0, :L] = 1
        out.append((X, Y))
    return out


def test_eval_loss_is_weighted_by_supervised_tokens(monkeypatch):
    """6 个 batch（5 个满 batch + 1 个只监督 1 token）时，口径差异必须体现出来。

    让第 i 个 batch 的 per-token loss 恒为 i（用 patch 掉的 cross_entropy 制造），于是
      - 按监督 token 加权 → Σ i·n_i / Σ n_i
      - 旧的"各 batch 均值的均值" → Σ i / num_batches
    两者在有短 batch 时必然不同；旧的会把那条 1-token 样本算成 1/6 的权重。
    """
    tr = _make()
    tr._unwrap = lambda: tr.model
    lens = [8, 8, 8, 8, 8, 1]                 # 最后一条只有 1 个监督 token
    calls = {"i": 0}

    def fake_cross_entropy(logits, targets, reduction="mean", **kw):
        i = calls["i"]                        # batch 序号 = "该 batch 的 per-token loss"
        calls["i"] += 1
        n = int((targets != -100).sum().item())
        val = float(i) * n if reduction == "sum" else float(i)
        return torch.tensor(val)

    monkeypatch.setattr("minigpt.train.trainer.f.cross_entropy", fake_cross_entropy)

    got = tr._eval_loss(_masked_batches(lens))

    expected = sum(i * n for i, n in enumerate(lens)) / sum(lens)    # 按 token 加权
    old = sum(range(len(lens))) / len(lens)                          # 各 batch 均值的均值
    assert abs(got - expected) < 1e-6, f"分母不是监督 token 数：got={got}, want={expected}"
    assert abs(old - expected) > 1e-3, "该用例应当能区分两种口径（否则测试无意义）"


def test_eval_loss_without_masking_matches_plain_mean():
    """预训练没有 -100：新口径必须与旧的"各 batch 均值的均值"数值一致（向后兼容）。

    等长 batch 下 Σloss_i·n / Σn == Σloss_i / B，所以这里两种算法必须完全相等——
    这条守的是"改了 SFT 的口径、别顺手改了基座那个已经进报告的数字"。
    """
    tr = _make()
    model = tr._unwrap()
    torch.manual_seed(0)
    loader = [(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8))) for _ in range(3)]

    per_batch = []
    with torch.no_grad():
        for X, Y in loader:
            logits = model(X)
            per_batch.append(float(f.cross_entropy(logits.flatten(0, 1), Y.flatten())))

    old_style = sum(per_batch) / len(per_batch)       # 旧实现：total_loss / num_batches
    got = tr._eval_loss(loader)
    assert abs(got - old_style) < 1e-6, f"无掩码时口径变了：got={got}, old={old_style}"
