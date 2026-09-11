"""集中管理 MiniGPT 训练/推理的超参数与路径（生产入口统一从本文件取值）。

用法约定：
- 训练入口：`python minigpt/train/pretrainer.py [--flag value ...]`（torchrun 多卡亦同）
- 推理入口：`python scripts/generate.py --checkpoint ...`
- CLI 参数会覆盖下面 dataclass 的默认值；默认路径基于本仓库根目录自动推导，
  不依赖进程启动目录，也不依赖任何 /data2 之类的历史绝对路径。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# 仓库根目录 = minigpt/config.py 的上两级
ROOT = Path(__file__).resolve().parent.parent


def _root_join(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


@dataclass
class ModelConfig:
    """模型架构参数（vocab_size<=0 时由 tokenizer 实际长度推导）。

    默认值 = 仓库文档中的生产基线（512/10/8 + ctx512 + 权重共享）。
    注意：默认值必须与 AGENTS.md / README 的"默认配置"一致，否则
    `python -m minigpt.train.pretrainer` 不带参数跑出来的模型和文档描述的不是同一个东西
    （历史上这里是 384/8/8 + qwen2 词表，与文档的 512/10/8 + v3 词表对不上）。
    """
    emb_dim: int = 512
    n_layers: int = 10
    n_heads: int = 8
    context_length: int = 512
    vocab_size: int = 0            # 0 => 从 tokenizer 推导
    drop_rate: float = 0.0
    qkv_bias: bool = False
    qkv_merged: bool = False       # 合并 QKV 投影（此前 GPTConfig 支持但入口没暴露）
    flash_attn: bool = False       # RTX20 系不支持 FA2
    tie_word_embeddings: bool = True   # 大词表下输入/输出嵌入共享
    use_swiglu: bool = False
    use_checkpoint: bool = False
    norm_type: str = "layernorm"       # layernorm（兼容历史 ckpt）| rmsnorm
    ffn_hidden_dim: int = 0            # 0=自动（GELU 4d / SwiGLU 8/3·d 对齐 64）
    lm_head_bias: bool = False         # 共享嵌入时标准做法是不带 bias
    rope_theta: float = 10000.0        # RoPE 基频


@dataclass
class DataConfig:
    """语料与分词产物。"""
    tokenizer_dir: str = _root_join("models", "tokenizer_v3")  # 32k 中文 BPE（生产基线）
    corpus_jsonl: str = _root_join("dataset", "pretrain_t2t_mini.jsonl")
    content_key: str = "text"
    max_lines: int = 0              # 语料取前 N 行（0=全量；CPU/快速验证请显式传小值）
    tokenized_bin: str = _root_join("dataset", "bins", "pretrain_v3_full.bin")
    bin_meta: str = ""              # 空 => tokenized_bin 同目录同名前缀 .meta.json
    eval_ratio: float = 0.002       # 验证集比例（均匀分块 + 双端对齐文档边界）
    eval_blocks: int = 32           # 验证集分成多少块（块越多越能代表整体语料分布；1=只取一段）


@dataclass
class TrainConfig:
    """训练超参。"""
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 1
    max_steps: int = 0              # 优化步上限（绝对值；0 => epochs*每epoch步数）
    reset_step: bool = False        # 续训时把优化步计数归零：领域增量续训必须开（见 SKILL.md §7.5）
    extra_steps: int = 0            # 从恢复点起再训 N 步（隐含步数归零；与 max_steps 二选一）
    warmup_steps: int = 300
    eval_steps: int = 500
    save_steps: int = 2000
    save_best: bool = True          # eval_loss 创新低时额外保存 best.pt（用于下游/评测选最优权重）
    grad_clip: float = 1.0
    mixed_precision_dtype: str = "float16"   # float16 | bfloat16 | none
    seed: int = 123
    num_workers: int = 0
    # NCCL 超时（秒）：必须覆盖 rank0 独占的完整 eval + checkpoint 落盘（都在 barrier 内），
    # 120s 会在慢盘/大验证集时以 NCCL 超时打挂整轮训练
    ddp_timeout_seconds: int = 1800
    # 打开后强制 cudnn.deterministic=True/benchmark=False（严格复现优先于吞吐）
    deterministic_cudnn: bool = False
    # 预训练固定形状下启用可显著提速（+66% 实测）；SFT 变长序列不建议。
    # 生产预设（pipeline.sh / train_pretrain_resilient.sh）默认开启，这里保守取 False。
    torch_compile: bool = False
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
    sft_dataset: str = _root_join("dataset", "sft", "sft_data_zh.jsonl")


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
    """把配置写进 RunConfig，同时支持两种键形式：

    - 扁平：`{'model_emb_dim': 384, 'train_batch_size': 8}`（CLI 覆盖）
    - 嵌套：`{'model': {'emb_dim': 384}}`（dump_run_config 写出的 config.json）

    真实事故：`dump_run_config` 写的是嵌套结构，而这里以前只认扁平键，于是
    `--config-file <dump 出来的 config.json>` 完全 no-op（emb_dim 从 512 静默
    回退到 384），与 pretrainer 的文档承诺矛盾。
    """
    for key, value in mapping.items():
        # 嵌套形式：{section: {field: value}}
        holder = getattr(cfg, key, None)
        if holder is not None and is_dataclass(holder):
            if not isinstance(value, dict):
                continue
            target = {f.name for f in fields(holder)}
            for attr, v in value.items():
                if attr in target:
                    setattr(holder, attr, v)
            continue
        # 扁平形式：<section>_<field>
        if "_" not in key:
            continue
        prefix, attr = key.split("_", 1)
        holder = getattr(cfg, prefix, None)
        if holder is None or not is_dataclass(holder):
            continue
        target = {f.name for f in fields(holder)}
        if attr in target:
            setattr(holder, attr, value)


def str2bool(value):
    """把 CLI 字符串解析成 bool。

    不能用 argparse 的 `type=bool`：`bool("False") == True`，会让
    `--train_torch_compile False` / `--train_save_best False` 这类"关闭开关"全部失效
    （真实事故：冒烟脚本传 False 实际开了 compile，reset_step 也被误开）。
    也不能改成 BooleanOptionalAction：本仓库所有 shell 脚本与 SKILL 文档都写成
    `--flag True/False` 的"带值"形式，BooleanOptionalAction 不接受值参数会直接报错。
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"无法解析为布尔值: {value!r}（可用 true/false/1/0/yes/no/on/off）")


def add_cli_overrides(parser, section: str, cls) -> None:
    """给 argparse 增加某 dataclass 所有字段的 --<section>_<field> 选项。"""
    for f in fields(cls):
        typ = f.type
        if typ in ("bool", bool):
            kind = str2bool
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
