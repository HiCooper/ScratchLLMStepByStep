"""checkpoint 加载工具：无论 checkpoint 里有没有 config，都能正确重建模型结构。

背景（真实踩坑）：`best.pt` / 周期 `checkpoint-*.pth` 早期版本没有写入 `config` 字段，
下游脚本若直接用 `GPTConfig()` 默认值建模型（768/12）去加载 512/10 的权重，就会
`size mismatch ... 32000x512 vs 32000x768`。这里统一做两件事：

1. 有 `config` 就用 `config`（最可靠）；
2. 没有 `config` 就从 `model_state` 的权重形状反推结构（emb_dim / n_layers / vocab_size /
   context_length / qkv_bias / use_swiglu / tie_word_embeddings），保证老产物也能用。

`n_heads` 无法从形状反推（Q/K/V/O 都是 (emb, emb)，pos_cis 不入 state_dict），只能按
`emb_dim // 64` 猜测；猜错**不会**触发形状报错，只会静默改变 head_dim 与输出。因此该路径
会发出 UserWarning，并提供 `n_heads=` 显式覆盖口（各下游脚本的 `--n-heads`），
返回值里的 `inferred` 也标注了"结构来自推断"。
"""
from __future__ import annotations

import re
import warnings
from typing import Any

import torch

from minigpt.model.transformer import GPTConfig, MiniGPT

# 反向工程用的键名（与 transformer.py 中的模块命名保持一致）
TOKEN_EMB = "token_emb.weight"
OUT_HEAD_W = "out_head.weight"
OUT_HEAD_B = "out_head.bias"
MASK_SUFFIX = ".atten.causal_mask"
WQ_BIAS_SUFFIX = ".atten.Wq.bias"
SWIGLU_SUFFIX = ".ffn.gate_proj.weight"
GELU_FFN_SUFFIX = ".ffn.layers.0.weight"
LN_SHIFT_SUFFIX = ".layernorm1.shift"
_L0_RE = re.compile(r"^decode_layers\.(\d+)\.")

# GPTConfig 中可以被 checkpoint config 直接覆盖的字段
CONFIG_KEYS = ("emb_dim", "n_layers", "n_heads", "context_length", "drop_rate", "qkv_bias",
               "flash_attn", "tie_word_embeddings", "use_swiglu", "qkv_merged", "use_checkpoint",
               "norm_type", "ffn_hidden_dim", "lm_head_bias", "rope_theta")

# 旧产物存在、新结构不再产生的键：属良性差异，加载时忽略而不是报错
LEGACY_UNEXPECTED_KEYS = ("out_head.bias",)

# n_heads 无法从权重形状反推时的兜底除数（本仓库 512→8、768→12 满足，但 cpu 预设
# 128/2/4、gpu-tiny 预设 384/8/8 **不满足**，所以猜错是常态而不是例外）
N_HEADS_HEURISTIC_DIVISOR = 64


def _is_tied(state_dict: dict) -> bool:
    """判断是否权重共享：`out_head.weight` 与 `token_emb.weight` 指向同一份存储。

    注意：`nn.Module.state_dict()` **总是**包含 `out_head.weight`（哪怕它只是别名），
    所以不能用"键是否存在"来判断，必须比较底层存储指针（torch.save/load 会保留别名关系）。
    """
    a, b = state_dict.get(OUT_HEAD_W), state_dict.get(TOKEN_EMB)
    if a is None or b is None or not (torch.is_tensor(a) and torch.is_tensor(b)):
        return False
    try:
        return a.data_ptr() == b.data_ptr()
    except Exception:  # noqa: BLE001
        return False


def infer_model_kwargs(state_dict: dict, n_heads: int | None = None) -> dict:
    """从 state_dict 的权重形状反推模型结构参数。

    `n_heads` 是**唯一无法反推**的字段（Q/K/V/O 都是 (emb, emb)，pos_cis 是 non-persistent
    buffer），不传就只能按 `emb_dim // N_HEADS_HEURISTIC_DIVISOR` 猜，并发出 UserWarning。

    为什么必须告警（真实隐患）：猜错不会触发任何 missing/unexpected —— 权重形状全对，
    只是 head_dim 变了，模型**静默**算出不同结果。实测 384/8/8 的 state_dict 被推成
    n_heads=6（head_dim 64 而非 48）：`load_state_into_model` 报 missing=0/unexpected=0，
    而 logits max|diff|=0.12、argmax 已经不一致。
    """
    if TOKEN_EMB not in state_dict:
        raise ValueError(f"state_dict 缺少 {TOKEN_EMB}，无法推断模型结构")
    emb = state_dict[TOKEN_EMB]
    vocab_size, emb_dim = int(emb.shape[0]), int(emb.shape[1])

    layers = [int(m.group(1)) for k in state_dict if (m := _L0_RE.match(k))]
    n_layers = (max(layers) + 1) if layers else 1

    ctx = None
    for k, v in state_dict.items():
        if k.endswith(MASK_SUFFIX) and getattr(v, "ndim", 0) == 2:
            ctx = int(v.shape[-1])
            break

    if n_heads is None:
        n_heads = max(1, emb_dim // N_HEADS_HEURISTIC_DIVISOR)
        warnings.warn(
            f"checkpoint 未携带 config：n_heads 无法从权重形状反推，已按 "
            f"emb_dim({emb_dim})//{N_HEADS_HEURISTIC_DIVISOR}={n_heads} 猜测。"
            f"猜错不会报错，只会静默改变 head_dim 与模型输出"
            f"（本仓库 cpu 预设 128/2/4 与 gpu-tiny 预设 384/8/8 都不满足该经验公式）。"
            f"请显式指定头数（CLI 一般提供 --n-heads）以确认。",
            UserWarning, stacklevel=2)

    kws: dict[str, Any] = {
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "n_layers": n_layers,
        "n_heads": int(n_heads),
        "qkv_bias": any(k.endswith(WQ_BIAS_SUFFIX) for k in state_dict),
        "use_swiglu": any(k.endswith(SWIGLU_SUFFIX) for k in state_dict),
        "tie_word_embeddings": _is_tied(state_dict),
        # LayerNorm 有 shift 参数、RMSNorm 没有，据此反推归一化类型
        "norm_type": "layernorm" if any(k.endswith(LN_SHIFT_SUFFIX) for k in state_dict)
                     else "rmsnorm",
        # 输出头 bias 是否存在于权重里（新结构默认不带）
        "lm_head_bias": OUT_HEAD_B in state_dict,
    }
    # 前馈中间维可直接从权重形状读出（SwiGLU 看 gate_proj，GELU 看 layers.0）
    for suffix in (SWIGLU_SUFFIX, GELU_FFN_SUFFIX):
        hit = next((v for k, v in state_dict.items() if k.endswith(suffix)), None)
        if hit is not None and getattr(hit, "ndim", 0) == 2:
            kws["ffn_hidden_dim"] = int(hit.shape[0])
            break
    if ctx:
        kws["context_length"] = ctx
    return kws


def load_state_into_model(model, state_dict: dict, allow_missing=(), allow_unexpected=(),
                          strict_critical: bool = True, verbose: bool = False):
    """把 state_dict 载入模型，对**已知的良性差异**宽容，对其余差异显式报错。

    使用场景：
    - 旧 checkpoint（`out_head.bias` 还在）载入新结构（输出头无 bias）；
    - 续训时模型新增/删除了个别 buffer。
    但绝不放任真实的架构不匹配静默通过：缺失的关键权重一律报错。

    返回 (missing, unexpected)。
    """
    allow_missing = set(allow_missing) | set(LEGACY_UNEXPECTED_KEYS)
    allow_unexpected = set(allow_unexpected) | set(LEGACY_UNEXPECTED_KEYS)
    filtered = {k: v for k, v in state_dict.items()
                if k not in allow_unexpected or k in model.state_dict()}
    try:
        missing, unexpected = model.load_state_dict(filtered, strict=False)
    except RuntimeError as exc:
        # 形状不匹配（例如词表/层数/中间维不一致）——必须带上可读上下文而不是裸堆栈
        raise RuntimeError(
            f"权重形状不匹配，checkpoint 与当前结构不一致：{exc}") from exc
    real_missing = [k for k in missing if k not in allow_missing and ".pos_cis" not in k]
    real_unexpected = [k for k in unexpected if k not in allow_unexpected]
    if verbose and (real_missing or real_unexpected):
        print(f"[ckpt] 权重差异 missing={real_missing[:4]} unexpected={real_unexpected[:4]}")
    if strict_critical and real_missing:
        raise RuntimeError(
            f"加载权重失败：缺少关键权重 {real_missing[:6]}"
            f"{' ...' if len(real_missing) > 6 else ''}；checkpoint 与当前结构不匹配")
    if real_unexpected:
        raise RuntimeError(
            f"加载权重失败：出现未知权重 {real_unexpected[:6]}"
            f"{' ...' if len(real_unexpected) > 6 else ''}；checkpoint 与当前结构不匹配")
    return missing, unexpected


def model_kwargs_from_checkpoint(ck: dict, vocab_size: int | None = None,
                                 n_heads: int | None = None) -> tuple[dict, bool]:
    """返回 (GPTConfig 关键字, 是否由权重形状推断而来)。

    `vocab_size`（一般为 tokenizer 词表大小）必须与 checkpoint 的 embedding 行数一致，
    不一致说明 checkpoint 与 tokenizer 不配套，直接给出可读报错而不是 `size mismatch` 堆栈。

    `n_heads` 只在"checkpoint 没带 config"时生效（有 config 时以 config 为准，它才可靠）：
    它是唯一无法从权重反推的字段，显式传入可消除 `infer_model_kwargs` 的猜测告警；
    不传则沿用 `emb_dim // 64` 兜底并发出 UserWarning。
    """
    cfg = ck.get("config") if isinstance(ck, dict) else None
    state = ck.get("model_state", {}) if isinstance(ck, dict) else {}
    emb_vocab = int(state[TOKEN_EMB].shape[0]) if TOKEN_EMB in state else None

    if isinstance(cfg, dict) and any(k in cfg for k in CONFIG_KEYS):
        kws = {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
        inferred = False
        declared = int(kws.get("vocab_size") or emb_vocab or 0)
        if n_heads is not None and int(n_heads) != int(kws.get("n_heads", n_heads)):
            warnings.warn(
                f"该 checkpoint 自带 config，n_heads 以 config 的 {kws.get('n_heads')} 为准，"
                f"忽略显式传入的 {int(n_heads)}。", UserWarning, stacklevel=2)
    else:
        kws = infer_model_kwargs(state, n_heads=n_heads)
        inferred = True
        declared = emb_vocab

    if vocab_size and declared and int(vocab_size) != declared:
        raise ValueError(
            f"checkpoint 词表({declared}) 与 tokenizer 词表({int(vocab_size)}) 不一致："
            f"请确认 --tokenizer-dir 与训练时使用的分词器相同（这会导致 embedding 形状不匹配）。"
        )
    if vocab_size:
        kws["vocab_size"] = int(vocab_size)
    return kws, inferred


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(f"checkpoint 缺少 model_state 字段: {path}")
    return ck


def build_model_from_checkpoint(path: str, tokenizer=None, device: str = "cpu",
                                verbose: bool = False,
                                n_heads: int | None = None) -> tuple[MiniGPT, dict, dict]:
    """按 checkpoint 自带的 config（缺失则反推）重建模型并加载权重。

    `n_heads` 同 `model_kwargs_from_checkpoint`：仅用于 config-less 老产物，见其文档。

    返回 (model, checkpoint, model_kwargs)。
    """
    ck = load_checkpoint(path, map_location="cpu")
    vocab_size = len(tokenizer) if tokenizer is not None else None
    kws, inferred = model_kwargs_from_checkpoint(ck, vocab_size=vocab_size, n_heads=n_heads)
    if inferred and verbose:
        print(f"[ckpt] {path} 未携带 config，按权重形状推断结构：{kws}")
    gpt = GPTConfig(**kws)
    model = MiniGPT(gpt).to(device)
    # 旧产物带 out_head.bias、新结构不带；良性键差异忽略，关键权重缺失仍报错
    try:
        load_state_into_model(model, ck["model_state"], allow_missing=("out_head.weight",),
                              verbose=verbose)
    except RuntimeError as exc:
        raise RuntimeError(
            f"加载 {path} 失败：{exc}\n推断结构为 {kws}。"
            f"请检查 checkpoint 与 tokenizer 是否匹配。"
        ) from exc
    model.eval()
    return model, ck, kws
