"""训练过程指标记录（TensorBoard）：标量 / 直方图 / 图像 / 模型图 / 嵌入投影 / 文本。

设计要点
- 全部为可选、按间隔触发，默认开销很小；仅在主进程（rank0）实例化。
- 超大张量（如 151k 词表 embedding）默认跳过直方图（log_hist_max_numel），避免 event 文件膨胀。
- 图像面板复用真实训练/评估批次：注意力热力图来自第一个 TransformerBlock 捕获的权重
  （需 attention.py 中 capture_attention=True，由本模块按需开启/关闭）。
- 嵌入投影（projector）只写 log(max_tokens) 个子集向量，并带 token 文本 metadata。

对应面板与配置项见 minigpt/README.md。
"""
from __future__ import annotations

import io
import time
from collections import deque

import torch


class MetricsLogger:
    def __init__(self, writer, model, tokenizer=None, device="cuda:0",
                 log_hist_every=500, log_hist_max_numel=2_000_000,
                 log_embedding_every=2000, projector_max_tokens=2000,
                 log_attention_every=1000, log_graph=False,
                 log_samples_every=0, sample_max_new_tokens=60,
                 sample_prompts=None, tag_prefix=""):
        self.writer = writer
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tag_prefix = tag_prefix
        self.log_hist_every = int(log_hist_every or 0)
        self.log_hist_max_numel = int(log_hist_max_numel)
        self.log_embedding_every = int(log_embedding_every or 0)
        self.projector_max_tokens = int(projector_max_tokens)
        self.log_attention_every = int(log_attention_every or 0)
        self.log_graph = bool(log_graph)
        self.log_samples_every = int(log_samples_every or 0)
        self.sample_max_new_tokens = int(sample_max_new_tokens)
        self.sample_prompts = sample_prompts or []

        self._history = {"train_loss": [], "eval_loss": [], "lr": []}
        self._steps = deque(maxlen=50)
        self._times = deque(maxlen=50)
        self._graph_logged = False
        self._probe_ids = None
        self._last_hist_step = -10 ** 9
        self._last_embed_step = -10 ** 9
        self._last_attn_step = -10 ** 9

    # ------------------------------------------------------------------ utils
    def _t(self, tag: str) -> str:
        return f"{self.tag_prefix}{tag}" if self.tag_prefix else tag

    def _unwrap(self):
        model = self.model
        module = getattr(model, "module", None)  # DDP
        return module if module is not None else model

    def _capture_attention(self, enable: bool):
        try:
            block0 = self._unwrap().decode_layers[0]
            block0.atten.capture_attention = enable
        except Exception:  # noqa: BLE001
            pass

    def _tokens_per_sec(self, step, tokens):
        now = time.time()
        self._steps.append(step)
        self._times.append((now, tokens))
        if len(self._times) >= 2:
            (t0, _), (t1, n) = self._times[0], self._times[-1]
            span = max(t1 - t0, 1e-6)
            total = sum(n for _, n in list(self._times)[1:])
            if span > 0:
                self.writer.add_scalar(self._t("train/tokens_per_sec"), total / span, step)

    # ------------------------------------------------------------- train step
    def on_train_step(self, step, loss, lr, grad_norm, batch=None):
        """每个参数更新步调用。batch 为 (X, Y) 或 X，用于图像/模型图探针。"""
        self.writer.add_scalar(self._t("train/loss"), float(loss), step)
        self.writer.add_scalar(self._t("train/lr"), float(lr), step)
        if grad_norm is not None:
            self.writer.add_scalar(self._t("train/grad_norm"), float(grad_norm), step)

        if batch is not None:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            if torch.is_tensor(x) and self._probe_ids is None:
                self._probe_ids = x[:1, : min(32, x.shape[-1])].detach().to(self.device)
            if self.log_graph and not self._graph_logged and torch.is_tensor(x):
                self.log_graph_once(x[:1, : min(32, x.shape[-1])])

        if self.log_hist_every and step % self.log_hist_every == 0:
            self.log_histograms(step)

    # --------------------------------------------------------------- histograms
    def log_histograms(self, step):
        model = self._unwrap()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.numel() > self.log_hist_max_numel:
                    continue
                if p.is_floating_point():
                    self.writer.add_histogram(self._t(f"weights/{name}"), p.detach().float().cpu(), step)
                if p.grad is not None:
                    self.writer.add_histogram(self._t(f"grads/{name}"), p.grad.detach().float().cpu(), step)
            for name, buf in model.named_buffers():
                if buf.numel() <= 4096 and buf.is_floating_point():
                    self.writer.add_histogram(self._t(f"buffers/{name}"), buf.detach().float().cpu(), step)

    # ------------------------------------------------------------------ graph
    def log_graph_once(self, dummy_ids):
        if self._graph_logged:
            return
        try:
            model = self._unwrap()
            was_training = model.training
            model.eval()
            self.writer.add_graph(model, dummy_ids.detach().to(self.device))
            model.train(was_training)
            self._graph_logged = True
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] add_graph skipped: {exc}")

    # -------------------------------------------------------------- embeddings
    def log_embedding(self, step):
        try:
            model = self._unwrap()
            weight = model.token_emb.weight.detach().float().cpu()
            total = weight.shape[0]
            n = min(self.projector_max_tokens, total)
            if n < total:  # 均匀采样，覆盖整个词表
                idx = torch.linspace(0, total - 1, steps=n).long()
                mat = weight[idx]
                ids = idx.tolist()
            else:
                mat = weight
                ids = list(range(total))
            if self.tokenizer is not None:
                meta = [str(self.tokenizer.convert_ids_to_tokens(int(i))) for i in ids]
            else:
                meta = [str(i) for i in ids]
            self._write_embedding_tensor(self._t("token_embedding"), mat, meta, step)
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] add_embedding skipped: {exc}")


    # ------------------------------------------------------- embedding (manual)
    def _write_embedding_tensor(self, tag, mat, metadata, step):
        """按 TensorBoard projector 插件规范写 tensor 事件（<tag> 与 <tag>/metadata）。

        背景：torch 2.13 + tensorboard 2.21 下 SummaryWriter.add_embedding 会静默不写事件，
        这里直接构造 TensorProto/Summary，兼容 projector 面板的自动发现。
        """
        import numpy as _np
        try:
            from tensorboard.compat.proto import (summary_pb2, tensor_pb2,
                                                  tensor_shape_pb2, types_pb2)

            mat = _np.asarray(mat, dtype=_np.float32)
            dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=int(sz)) for sz in mat.shape]
            t_mat = tensor_pb2.TensorProto(
                dtype=types_pb2.DT_FLOAT,
                tensor_shape=tensor_shape_pb2.TensorShapeProto(dim=dims),
                tensor_content=mat.tobytes(),
            )
            meta_bytes = [[str(m).encode("utf-8")] for m in metadata]
            t_meta = tensor_pb2.TensorProto(
                dtype=types_pb2.DT_STRING,
                tensor_shape=tensor_shape_pb2.TensorShapeProto(
                    dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=len(meta_bytes)),
                         tensor_shape_pb2.TensorShapeProto.Dim(size=1)]),
                string_val=[b for pair in meta_bytes for b in pair],
            )
            summary = summary_pb2.Summary(value=[
                summary_pb2.Summary.Value(tag=tag, tensor=t_mat),
                summary_pb2.Summary.Value(tag=f"{tag}/metadata", tensor=t_meta),
            ])
            self.writer._get_file_writer().add_summary(summary, step)
        except Exception:  # 退化到官方 API（部分版本可用）
            try:
                self.writer.add_embedding(mat, metadata=metadata, tag=tag, global_step=step)
            except Exception as exc:  # noqa: BLE001
                print(f"[metrics] add_embedding skipped: {exc}")

    # ------------------------------------------------------------------ images
    def log_attention_image(self, step):
        if self._probe_ids is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            model = self._unwrap()
            was_training = model.training
            model.eval()
            self._capture_attention(True)
            with torch.no_grad():
                model(self._probe_ids.to(self.device))
            attn = getattr(model.decode_layers[0].atten, "last_attention", None)
            self._capture_attention(False)
            model.train(was_training)
            if attn is None:
                return
            head0 = attn[0, 0].float().cpu()  # (q, k)
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(head0, aspect="auto", cmap="viridis")
            ax.set_title("layer0 head0 attention")
            ax.set_xlabel("key")
            ax.set_ylabel("query")
            fig.colorbar(im, ax=ax, fraction=0.046)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            from PIL import Image  # tensorboard 依赖中自带 Pillow
            import numpy as np
            img = np.asarray(Image.open(buf).convert("RGB"))
            self.writer.add_image(self._t("attention/layer0_head0"),
                                  torch.from_numpy(np.array(img)).permute(2, 0, 1), step)
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] attention image skipped: {exc}")

    def log_loss_figure(self, step):
        if not self._history["eval_loss"]:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(self._history["train_loss"], label="train_loss")
            ax.plot(self._history["eval_loss"], label="eval_loss")
            ax.set_xlabel("eval point")
            ax.set_ylabel("loss")
            ax.legend()
            ax.grid(True)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            from PIL import Image
            import numpy as np
            img = np.asarray(Image.open(buf).convert("RGB"))
            self.writer.add_image(self._t("curves/loss"), torch.from_numpy(np.array(img)).permute(2, 0, 1), step)
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] loss figure skipped: {exc}")

    # ------------------------------------------------------------------- text
    def log_samples(self, step, max_new_tokens=None):
        if self.tokenizer is None or not self.sample_prompts:
            return
        model = self._unwrap()
        was_training = model.training
        model.eval()
        n = max_new_tokens or self.sample_max_new_tokens
        rows = []
        for prompt in self.sample_prompts:
            try:
                ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
                out = model.generate(ids, n, self.tokenizer.eos_token_id,
                                     do_sample=True, temperature=0.8, top_k=50,
                                     top_p=0.92, repetition_penalty=1.15,
                                     use_kv_cache=False)
                text = self.tokenizer.decode(out[0][ids.shape[1]:].tolist(),
                                             skip_special_tokens=True).strip()
            except Exception as exc:  # noqa: BLE001
                text = f"<生成失败: {exc}>"
            rows.append(f"用户: {prompt}\n模型: {text}\n")
        model.train(was_training)
        if rows:
            self.writer.add_text(self._t("samples/generation"), "\n".join(rows), step)

    # ------------------------------------------------------------------- eval
    def on_eval(self, step, train_loss, eval_loss, lr, grad_norm):
        self.writer.add_scalar(self._t("eval/loss"), float(eval_loss), step)
        self.writer.add_scalar(self._t("eval/perplexity"), float(torch.exp(torch.tensor(
            min(float(eval_loss), 80.0))).item()), step)
        self.writer.add_scalar(self._t("train/loss_eval_window"), float(train_loss), step)
        if lr is not None:
            self._history["lr"].append(float(lr))
            self.writer.add_scalar(self._t("train/lr_eval"), float(lr), step)
        self._history["train_loss"].append(float(train_loss))
        self._history["eval_loss"].append(float(eval_loss))

        if self.log_attention_every and step - self._last_attn_step >= self.log_attention_every:
            self._last_attn_step = step
            self.log_attention_image(step)
            self.log_loss_figure(step)
        if self.log_embedding_every and step - self._last_embed_step >= self.log_embedding_every:
            self._last_embed_step = step
            self.log_embedding(step)
        if self.log_samples_every and step % self.log_samples_every == 0:
            self.log_samples(step)

    def on_final(self, step, eval_loss=None, max_new_tokens=None):
        """训练结束：补写嵌入投影、注意力图、损失曲线与最终样本文本。"""
        if self.log_embedding_every:
            self.log_embedding(step)
        if self.log_attention_every:
            self.log_attention_image(step)
        self.log_loss_figure(step)
        self.log_samples(step, max_new_tokens=max_new_tokens)
