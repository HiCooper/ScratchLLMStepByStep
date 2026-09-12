"""回归护栏：`scripts/run_v5_downstream.sh` 的命令行必须真的能被目标脚本解析。

为什么值得单独测：这个驱动脚本一次训练只跑一次、且排在基座训练 20 小时之后，
写错一个 flag 名字（`--data_max_lines` 写成 `--data-max-lines`）不会在提交时暴露，
只会在第二天交付缺一块时暴露。这里把它变成一次 2 秒的单测。

只做**与数据无关**的检查（CI 里 `dataset/` 是 gitignore 的，断言文件存在会误报）。
"""
from __future__ import annotations

import os
import re
import shlex
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVER = os.path.join(ROOT, "scripts", "run_v5_downstream.sh")


def _driver_src() -> str:
    with open(DRIVER, encoding="utf-8") as f:
        return f.read()


def _iter_invocations(src: str):
    """产出 (目标脚本相对路径, 该次调用的完整命令行文本)——含续行与 `$args` 内联。"""
    lines = src.splitlines()
    args_var = re.search(r'^\s*args="([^"]+)"', src, re.M)
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            continue
        m = re.search(r'python3 (?:-u )?(?:-m ([\w.]+)|(scripts/[\w/]+\.py))', ln)
        if not m:
            continue
        path = m.group(2) or m.group(1).replace(".", "/") + ".py"
        buf, j = ln, i
        while buf.rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            buf += " " + lines[j]
        if args_var and "sft_trainer" in path:
            buf += " " + args_var.group(1)
        yield path, buf
    # sft_stage 的三处调用点自己不带 python3（python3 在函数体里用 $args 展开），
    # 真正写死的 flag 都在调用点，必须单独扫——否则拼错的 flag 会被漏掉。
    for s in _sft_stage_arg_strings():
        yield "minigpt/train/sft_trainer.py", s


def test_driver_target_scripts_exist():
    for path, _ in _iter_invocations(_driver_src()):
        assert os.path.exists(os.path.join(ROOT, path)), f"驱动引用了不存在的脚本：{path}"


def _declared_flags(path: str) -> set[str]:
    """目标脚本认识的 flag 集合。

    `scripts/*.py` 是字面 add_argument；`minigpt/train/*_trainer.py` 还有一批是
    `add_cli_overrides` 按 dataclass 字段**动态**注册的（`--train_<field>`），
    只扫源码会误报，所以这里直接把那个 parser 建出来取 option_strings。
    """
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        flags = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', f.read()))
    if path.startswith("minigpt/"):
        import argparse

        from minigpt.config import DataConfig, ModelConfig, PathConfig, TrainConfig, add_cli_overrides
        ap = argparse.ArgumentParser()
        for section, cls in (("model", ModelConfig), ("data", DataConfig),
                             ("train", TrainConfig), ("paths", PathConfig)):
            add_cli_overrides(ap, section, cls)
        flags |= {o for a in ap._actions for o in a.option_strings}
    return flags


def test_driver_flags_are_declared_by_target_script():
    """每个 `--flag` 必须在目标脚本认识的集合里（允许 `-`/`_` 互换）。"""
    for path, cmd in _iter_invocations(_driver_src()):
        declared = _declared_flags(path)
        for flag in set(re.findall(r"(?<![\w-])--[A-Za-z][\w-]*", cmd)):
            variants = {flag, flag.replace("-", "_"), flag.replace("_", "-")}
            assert variants & declared, f"{path} 不认识 {flag}（驱动里写错了？）"


def _sft_stage_arg_strings():
    """抽出 `sft_stage` 三处调用的公共参数串（第 4 个参数之后的那段）。"""
    src = _driver_src()
    calls = re.findall(r'sft_stage\s+"[^"]+"[\s\\]*"[^"]+"[\s\\]*"[^"]+"[\s\\]*"([^"]+)"', src, re.S)
    assert len(calls) == 3, f"预期 3 个 sft_stage 调用（SFT/easy/hard），实际 {len(calls)}"
    return calls


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_sft_stage_args_parse_including_oom_retry(idx):
    """SFT/easy/hard 的公共参数 + 正常 batch + OOM 降 batch 重试，两套都要能被 argparse 接受。"""
    from minigpt.train.sft_trainer import parse_args

    raw = _sft_stage_arg_strings()[idx]
    for var, val in (("$BASE_CKPT", "/tmp/base.pt"), ("$SFT_CKPT", "/tmp/sft.pt"),
                     ("$TOK", "models/tokenizer_v3"), ("$CP", "models/checkpoints")):
        raw = raw.replace(var, val)
    base = shlex.split(raw.replace("\\\n", " "))
    variants = {
        "first": base + ["--train_batch_size", "8", "--paths_output_dir", "/tmp/out"],
        "retry": base + ["--train_batch_size", "4", "--train_grad_accumulation_steps", "2",
                         "--paths_output_dir", "/tmp/out"],
    }
    for name, argv in variants.items():
        old = sys.argv
        sys.argv = ["sft_trainer"] + argv
        try:
            a = parse_args()
        finally:
            sys.argv = old
        assert a.paths_output_dir == "/tmp/out" and a.data_max_lines and a.data_max_len
        assert a.train_epochs == 2 and a.train_warmup_steps == 50
        if name == "retry":
            assert a.train_batch_size == 4 and a.train_grad_accumulation_steps == 2
