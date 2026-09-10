"""集中管理 MiniGPT 训练/推理的超参数与路径（生产入口统一从本文件取值）。

用法约定：
- 训练入口：`python minigpt/train/pretrainer.py [--flag value ...]`（torchrun 多卡亦同）
- 推理入口：`python scripts/generate.py --checkpoint ...`
- CLI 参数会覆盖下面 dataclass 的默认值；默认路径基于本仓库根目录自动推导，
  不依赖进程启动目录，也不依赖任何 /data2 之类的历史绝对路径。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# 仓库根目录 = minigpt/config.py 的上两级
ROOT = Path(__file__).resolve().parent.parent


def _root_join(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


@dataclass
class ModelConfig:
    """模型架构参数（vocab_size<=0 时由 tokenizer 实际长度推导）。"""
    emb_dim: int = 384
    n_layers: int = 8
    n_heads: int = 8
    context_length: int = 512
    vocab_size: int = 0            # 0 => 从 tokenizer 推导
    drop_rate: float = 0.0
    qkv_bias: bool = False
    flash_attn: bool = False       # RTX20 系不支持 FA2
    tie_word_embeddings: bool = True   # 大词表下输入/输出嵌入共享
    use_swiglu: bool = False
    use_checkpoint: bool = False


@dataclass
class DataConfig:
    """语料与分词产物。"""
    tokenizer_dir: str = _root_join("models", "tokenizer_qwen2")  # Qwen2.5 中文 BPE
    corpus_jsonl: str = _root_join("dataset", "pretrain_t2t_mini.jsonl")
    content_key: str = "text"
    max_lines: int = 300000         # 语料取前 N 行（0=全量）
    tokenized_bin: str = _root_join("dataset", "bins", "pretrain_qwen.bin")
    bin_meta: str = ""              # 空 => tokenized_bin 同目录同名前缀 .meta.json
    eval_ratio: float = 0.002       # 从 .bin 中切出的验证比例（窗口级随机切分）


@dataclass
class TrainConfig:
    """训练超参。"""
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 8
    train_ratio: float = 0.998      # 兼容旧字段（新入口用 eval_ratio）
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 1
    max_steps: int = 0              # 0 => epochs*每epoch步数
    warmup_steps: int = 300
    eval_steps: int = 500
    save_steps: int = 2000
    save_best: bool = True          # eval_loss 创新低时额外保存 best.pt（用于下游/评测选最优权重）
    log_every: int = 20
    grad_clip: float = 1.0
    mixed_precision_dtype: str = "float16"   # float16 | bfloat16 | none
    seed: int = 123
    num_workers: int = 0
    torch_compile: bool = False        # 预训练固定形状下启用可显著提速（SFT 变长序列不建议）
    compile_mode: str = "default"      # torch.compile mode: default/reduce-overhead/max-autotune
    # ---- tensorboard 训练过程指标（见 minigpt/README.md）----
    log_hist_every: int = 500          # 权重/梯度直方图间隔（0=关闭）
    log_hist_max_numel: int = 2_000_000  # 超大张量（如大词表 embedding）跳过直方图
    log_embedding_every: int = 2000     # 嵌入投影（projector）间隔（0=关闭）
    projector_max_tokens: int = 2000    # 投影面板最多写入多少个 token 向量
    log_attention_every: int = 1000     # 注意力热力图间隔（0=关闭）
    log_graph: bool = False             # 是否记录模型计算图（add_graph）
    log_samples_every: int = 0          # 周期性打印/记录生成样本（0=仅结束时）
    sample_max_new_tokens: int = 60
    sample_prompts: str = "什么是AI？|如何保持身体健康？|从前有座山，山上有座庙"


@dataclass
class PathConfig:
    """产物输出路径。"""
    output_dir: str = _root_join("models", "checkpoints", "minigpt_pretrain")
    last_checkpoint_path: str = ""  # 续训用 checkpoint（空=从头开始）
    # ---- 兼容旧引用 ----
    pretrain_dataset: str = _root_join("dataset", "bins", "pretrain_qwen.bin")
    sft_dataset: str = _root_join("dataset", "sft", "sft_data_zh.jsonl")
    tokenizer_dir: str = _root_join("models", "tokenizer_qwen2")


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# 模块级默认配置（旧代码可继续 import）
paths = PathConfig()
train_cfg = TrainConfig()


def build_run_config(cli_args=None, yaml_path: str | None = None) -> RunConfig:
    """由 CLI argparse.Namespace 覆盖默认配置（扁平前缀：model_/data_/train_/paths_）。"""
    cfg = RunConfig()
    if yaml_path:
        with open(yaml_path, encoding="utf-8") as f:
            data = json.load(f)
        apply_mapping(cfg, data)
    if cli_args is not None:
        overrides = {k: v for k, v in vars(cli_args).items()
                     if v is not None and not (isinstance(v, str) and v == "")}
        apply_mapping(cfg, overrides)
    return cfg


def apply_mapping(cfg: RunConfig, mapping: dict) -> None:
    """把 {'model_emb_dim': 384, 'train_batch_size': 8} 这类扁平键写入对应 dataclass。"""
    for key, value in mapping.items():
        if "_" not in key:
            continue
        prefix, attr = key.split("_", 1)
        holder = getattr(cfg, prefix, None)
        if holder is None or not is_dataclass(holder):
            continue
        target = {f.name for f in fields(holder)}
        if attr in target:
            setattr(holder, attr, value)


def add_cli_overrides(parser, section: str, cls) -> None:
    """给 argparse 增加某 dataclass 所有字段的 --<section>_<field> 选项。"""
    for f in fields(cls):
        typ = f.type
        if typ in ("bool", bool):
            kind = bool
        elif typ in ("int", int):
            kind = int
        elif typ in ("float", float):
            kind = float
        else:
            kind = str
        parser.add_argument(f"--{section}_{f.name}", type=kind, default=None,
                            help=f"{section}.{f.name} (默认 {f.default})")


def as_nested_dict(cfg: RunConfig) -> dict:
    return {name: asdict(getattr(cfg, name)) for name in ("model", "data", "train", "paths")}


def dump_run_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(as_nested_dict(cfg), ensure_ascii=False, indent=2),
                          encoding="utf-8")
