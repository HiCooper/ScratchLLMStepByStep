"""文档护栏：`*.md` 里的示例命令必须真的能对上下面的脚本。

为什么值得一个单测：这个仓库的文档密度很高（AGENTS/SKILL/references/dataset/minigpt/README），
而且文档里的命令是 agent 与用户最可能**照抄**的东西。本 session 里光是"指向已被取代的数据/
不存在的脚本/拼错的 flag"就修了十来处——每一次都是同一个失效模式：
**文档与代码各自演进，没有任何东西会在漂移的那一刻变红**。

这里锁三件事（都无需 GPU、无需真实数据，毫秒级）：
1. `scripts/` 下的每个脚本都在某处文档里被提到（不存在"隐形脚本"）；
2. 文档命令里出现的 `python3 x.py` / `python3 -m x.y` 指向的脚本/模块存在；
3. 文档命令里出现的每个 `--flag`，目标脚本都认识（trainer 的动态 flag 走真 parser 取）。

会跳过含占位符的命令（`<ckpt>`、`$VAR`、`...`、`|`、`[`），因为它们本来就不是可执行样例。
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DOCS = ["README.md", "AGENTS.md", "dataset/README.md", "minigpt/README.md"] + sorted(
    glob.glob(os.path.join(ROOT, "skills", "**", "*.md"), recursive=True))

# 含这些字符的命令是模板而不是可执行样例（占位符/变量/多选/省略）
PLACEHOLDER = re.compile(r"[<>]|\$|\.\.\.|\[|YYYY|\|")
CMD_RE = re.compile(r"(?:^|\s)(python3?|bash)\s+(?:-u\s+)?(?:-m\s+([\w.]+)|(\S+\.(?:py|sh)))")
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*")


def _doc_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _commands():
    """产出 (文档名, 命令行, 目标脚本路径相对 ROOT)。"""
    for doc in DOCS:
        if not os.path.exists(doc):
            continue
        for block in re.findall(r"```(?:bash|sh|console)\n(.*?)```", _doc_text(doc), re.S):
            for raw in block.splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or PLACEHOLDER.search(line):
                    continue
                m = CMD_RE.search(line)
                if not m:
                    continue
                mod, path = m.group(2), m.group(3)
                if mod:
                    if not mod.startswith("minigpt"):
                        continue        # pytest 之类的第三方入口不属于本仓库，跳过
                    path = mod.replace(".", "/") + ".py"
                yield os.path.basename(doc), line, path


def _trainer_flags() -> set:
    """trainer 入口由 add_cli_overrides 按 dataclass 字段动态注册的 flag。"""
    from minigpt.config import (DataConfig, ModelConfig, PathConfig, TrainConfig,
                                add_cli_overrides)
    ap = argparse.ArgumentParser()
    for section, cls in (("model", ModelConfig), ("data", DataConfig),
                         ("train", TrainConfig), ("paths", PathConfig)):
        add_cli_overrides(ap, section, cls)
    return {o for a in ap._actions for o in a.option_strings}


def _declared_flags(path: str) -> set:
    body = _doc_text(os.path.join(ROOT, path))
    flags = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', body))
    if path.endswith(".sh"):
        # shell 脚本手写解析（case/while），没有 argparse：认它源码里出现过的 flag 字面量。
        # 只看非注释行，避免"注释里写过就能过"。
        code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
        flags |= set(FLAG_RE.findall(code))
        if '"$@"' in body or "${@}" in body:
            flags |= _trainer_flags()      # 形如 pretrain_start.sh 把 "$@" 透传给 python 入口
    elif path.endswith("trainer.py"):
        # trainer 入口一批 flag 由 add_cli_overrides 按 dataclass 字段动态注册，
        # 只扫源码会误报；直接把 parser 建出来取 option_strings。
        from minigpt.config import (DataConfig, ModelConfig, PathConfig, TrainConfig,
                                    add_cli_overrides)
        ap = argparse.ArgumentParser()
        for section, cls in (("model", ModelConfig), ("data", DataConfig),
                             ("train", TrainConfig), ("paths", PathConfig)):
            add_cli_overrides(ap, section, cls)
        flags |= {o for a in ap._actions for o in a.option_strings}
    return flags


def test_no_invisible_script():
    """`scripts/` 下每个脚本都要在文档里能被找到（否则后人不知道它存在、能不能用）。"""
    body = "\n".join(_doc_text(d) for d in DOCS if os.path.exists(d))
    # 递归扫描：脚本已按角色分到 scripts/{data,tools,experiments}/ 子目录，
    # 只看顶层会让子目录里新增的脚本"隐形"（这条测试的意义就是防这个）。
    scripts = sorted(os.path.relpath(p, os.path.join(ROOT, "scripts"))
                     for p in glob.glob(os.path.join(ROOT, "scripts", "**", "*"), recursive=True)
                     if os.path.isfile(p) and p.endswith((".py", ".sh")))
    assert scripts, "scripts/ 目录读不到脚本，路径解析有问题"
    missing = [f for f in scripts if os.path.basename(f) not in body]
    assert not missing, f"这些脚本在任何文档里都没被提到：{missing}"


def test_doc_commands_point_at_existing_scripts():
    bad = [f"[{doc}] {line}" for doc, line, path in _commands()
           if not os.path.exists(os.path.join(ROOT, path))]
    assert not bad, "文档命令指向了不存在的脚本：\n" + "\n".join(bad)


def test_doc_commands_use_declared_flags():
    """文档里的 `--flag` 必须是目标脚本认识的（`-`/`_` 允许互换）。"""
    bad = []
    for doc, line, path in _commands():
        if not os.path.exists(os.path.join(ROOT, path)):
            continue                                    # 上一条测试负责报这个
        declared = _declared_flags(path)
        for flag in set(FLAG_RE.findall(line)):
            variants = {flag, flag.replace("-", "_"), flag.replace("_", "-")}
            if not (variants & declared):
                bad.append(f"[{doc}] {path} 不认识 {flag}  ← {line}")
    assert not bad, "文档命令用了目标脚本不认识的 flag：\n" + "\n".join(bad)


def test_doc_command_scan_is_not_vacuous():
    """防止正则失效导致上面几条"永远通过"（本 session 真实踩过：测试和实现一起错）。"""
    cmds = list(_commands())
    assert len(cmds) >= 30, f"只从文档里扫出 {len(cmds)} 条命令，扫描逻辑可能已失效"
    assert any(p.endswith("run_v5_downstream.sh") or p.endswith("pipeline.sh") for _, _, p in cmds)
