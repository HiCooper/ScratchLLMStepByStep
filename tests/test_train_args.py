"""`train_args` 翻译层的穷尽性校验（防漂移）。

回归背景：`pretrainer.py` 与 `sft_trainer.py` 以前各自手写一份 `train_args` dict，sft 侧
漏传了 `num_workers / torch_compile / compile_mode`，而这三个 flag 由 `add_cli_overrides`
注册进了 sft_trainer 的 `--help`——于是 `--train_torch_compile True` 在 SFT 下**静默无效**。
本测试从 `trainer.py` 源码里提取"真正被读取的键"，与 `as_train_args()` 的输出做双向校验，
并用 AST 禁止两个入口再写回手写 dict。
"""
import ast
import pathlib
import re

from minigpt.config import PathConfig, TrainConfig
from minigpt.train.train_args import as_train_args

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _keys_read_by_trainer() -> set:
    """`trainer.py` 中 `train_args.get("...")` 实际读取的键（含跨行写法）。"""
    src = (ROOT / "minigpt" / "train" / "trainer.py").read_text(encoding="utf-8")
    return set(re.findall(r'train_args\.get\(\s*[\'"]([^\'"]+)[\'"]', src))


def test_helper_covers_every_key_the_trainer_reads():
    keys = _keys_read_by_trainer()
    assert keys, "未能从 trainer.py 提取到任何 train_args 读取点（正则失效？）"
    produced = set(as_train_args(TrainConfig(), PathConfig()))

    missing = sorted(keys - produced)
    assert not missing, (
        f"Trainer 会读取这些键，但 as_train_args 没有产出：{missing}；"
        f"它们会静默退化成 Trainer 里的默认值（这正是 sft 侧漏传 torch_compile 的成因）")

    extra = sorted(produced - keys)
    assert not extra, f"as_train_args 产出了 Trainer 从不读取的死键：{extra}"


def test_helper_values_follow_config():
    tc = TrainConfig(epochs=3, batch_size=5, torch_compile=True, num_workers=2,
                     mixed_precision_dtype="none", max_steps=123, extra_steps=7,
                     reset_step=True, seed=99, grad_accumulation_steps=4, grad_clip=0.5)
    pc = PathConfig(output_dir="/tmp/o", last_checkpoint_path="/tmp/c.pth")
    ta = as_train_args(tc, pc)

    assert ta["num_train_epochs"] == 3
    assert ta["train_batch_size"] == 5
    assert ta["gradient_accumulation_steps"] == 4
    assert ta["grad_clip"] == 0.5
    assert ta["max_steps"] == 123
    assert ta["extra_steps"] == 7
    assert ta["reset_step"] is True
    assert ta["seed"] == 99
    # 以前只有 pretrainer 传这三个键，sft 侧传了 flag 也不生效
    assert ta["torch_compile"] is True
    assert ta["num_workers"] == 2
    assert ta["compile_mode"] == TrainConfig().compile_mode
    assert ta["output_dir"] == "/tmp/o"
    assert ta["last_checkpoint_path"] == "/tmp/c.pth"
    # mixed_precision_dtype="none" 必须关掉 AMP，并把 dtype 回落到 fp16（Trainer 只认两种）
    assert ta["use_mixed_precision"] is False
    assert ta["mixed_precision_dtype"] == "float16"


def test_both_entries_use_the_shared_helper():
    """两个生产入口不得再手写 train_args dict（AST 校验）。"""
    for name in ("pretrainer.py", "sft_trainer.py"):
        src = (ROOT / "minigpt" / "train" / name).read_text(encoding="utf-8")
        assert "as_train_args" in src, f"{name} 未使用共享翻译层 minigpt/train/train_args.py"

        hand_written = [
            node for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "train_args" for t in node.targets)
            # 允许 `train_args = as_train_args(tc, pc)`；禁止字面量 dict（那正是漂移的来源）
            and not (isinstance(node.value, ast.Call)
                     and getattr(node.value.func, "id", None) == "as_train_args")
        ]
        assert not hand_written, (
            f"{name}:{hand_written[0].lineno} 仍存在手写的 train_args 赋值，"
            f"请改用 as_train_args(tc, pc)")
