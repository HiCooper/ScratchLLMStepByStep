"""`RunConfig` -> `Trainer.train_args` 的**唯一**翻译层。

真实问题：`pretrainer.py` 与 `sft_trainer.py` 以前各自手写了一份 20 多行的映射（字段还要
改名：`epochs -> num_train_epochs`、`batch_size -> train_batch_size`），两份已经漂移——
`sft_trainer` 漏掉了 `num_workers / torch_compile / compile_mode`，而这三个 flag 由
`add_cli_overrides` 注册进了它的 `--help`，于是 `--train_torch_compile True` 在 SFT 下
是**静默 no-op**（用户以为开了，实际没开）。

现在只有这一份实现，并且：

- `tests/test_train_args.py` 会把本函数的输出键集合与「`trainer.py` 真正
  `train_args.get(...)` 读取的键集合」做双向穷尽校验——新增读取点、或漏传键，测试都会红；
- 两个入口都不再允许出现手写 dict（测试用 AST 校验它们调用了 `as_train_args`）。

`eval_strategy` 已移除：`Trainer` 从来没读过它（只读 `eval_steps`），属于死配置。
"""
from __future__ import annotations

from minigpt.config import PathConfig, TrainConfig

# 混合精度只接受这两种 dtype；其余值（含 "none"）一律按 fp16 处理并由
# use_mixed_precision 开关决定是否启用（见 Trainer.__init__）
_AMP_DTYPES = ("float16", "bfloat16")


def as_train_args(tc: TrainConfig, pc: PathConfig) -> dict:
    """把 `TrainConfig` / `PathConfig` 翻译成 Trainer 消费的扁平 dict（键名保持历史口径）。"""
    return {
        # ---- 步数 / 批大小 / 调度 ----
        "num_train_epochs": tc.epochs,
        "train_batch_size": tc.batch_size,
        "gradient_accumulation_steps": tc.grad_accumulation_steps,
        "max_steps": tc.max_steps,
        "reset_step": tc.reset_step,
        "extra_steps": tc.extra_steps,
        "warmup_steps": tc.warmup_steps,
        "grad_clip": tc.grad_clip,
        # ---- eval / 存档节奏 ----
        "eval_steps": tc.eval_steps,
        "save_strategy": "step",
        "save_steps": tc.save_steps,
        "save_best": tc.save_best,
        # ---- 精度 / 性能 / 复现 ----
        "use_mixed_precision": tc.mixed_precision_dtype != "none",
        "mixed_precision_dtype": tc.mixed_precision_dtype
        if tc.mixed_precision_dtype in _AMP_DTYPES else "float16",
        "torch_compile": tc.torch_compile,
        "compile_mode": tc.compile_mode,
        "num_workers": tc.num_workers,
        "seed": tc.seed,
        "deterministic_cudnn": tc.deterministic_cudnn,
        "ddp_timeout_seconds": tc.ddp_timeout_seconds,
        # ---- 路径 ----
        "output_dir": pc.output_dir,
        "last_checkpoint_path": pc.last_checkpoint_path,
    }
