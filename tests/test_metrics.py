"""训练指标记录（TensorBoard）单测：标量 / 直方图 / 嵌入投影 / 文本。"""
import glob
import struct

import torch
from torch.utils.tensorboard import SummaryWriter

from minigpt.model.transformer import GPTConfig, MiniGPT
from minigpt.train.metrics import MetricsLogger


def _read_tags(path):
    from tensorboard.compat.proto import event_pb2

    def records(data):
        i = 0
        while i + 12 <= len(data):
            n = struct.unpack("<Q", data[i:i + 8])[0]
            i += 12
            yield data[i:i + n]
            i += n + 4

    tags, kinds = set(), {}
    for raw in records(open(path, "rb").read()):
        ev = event_pb2.Event()
        ev.ParseFromString(raw)
        kinds[ev.WhichOneof("what")] = kinds.get(ev.WhichOneof("what"), 0) + 1
        if ev.WhichOneof("what") == "summary":
            tags |= {v.tag for v in ev.summary.value}
    return tags, kinds


def test_metrics_logger_writes_panels(tmp_path):
    writer = SummaryWriter(str(tmp_path))
    model = MiniGPT(GPTConfig(emb_dim=32, n_layers=1, n_heads=2, context_length=32,
                              vocab_size=100, tie_word_embeddings=True))
    logger = MetricsLogger(writer, model, tokenizer=None, device="cpu",
                           log_hist_every=1, log_embedding_every=1,
                           log_attention_every=0, log_graph=False,
                           sample_prompts=[])
    x = torch.randint(0, 100, (2, 8))
    logger.on_train_step(step=1, loss=3.0, lr=1e-3, grad_norm=0.5, batch=(x, x))
    logger.on_eval(step=2, train_loss=3.0, eval_loss=3.2, lr=1e-3, grad_norm=0.5)
    writer.flush()
    writer.close()

    path = sorted(glob.glob(str(tmp_path / "events*")))[-1]
    tags, kinds = _read_tags(path)
    assert "train/loss" in tags and "eval/loss" in tags and "eval/perplexity" in tags
    assert any(t.startswith("weights/") for t in tags)
    # 嵌入投影：<tag> 与 <tag>/metadata 均需写入（torch add_embedding 在部分版本会静默失效）
    assert "token_embedding" in tags and "token_embedding/metadata" in tags
