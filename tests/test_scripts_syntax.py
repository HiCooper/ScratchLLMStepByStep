"""脚本静态检查：所有 shell/python 脚本语法可编译（防止坏脚本进仓库）。"""
import ast
import glob
import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _shell_scripts():
    # scripts/ 已按角色分目录（scripts/data、scripts/train、scripts/tools…），
    # 用递归 glob：非递归的 "scripts/*.sh" 会让搬走的脚本静默失去语法护栏
    # （build_cot_sft.py 的 sys.path 兜底就是趁这个空档溜进仓库的）。
    pats = ["scripts/**/*.sh", "skills/**/*.sh"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(ROOT, p), recursive=True)
    return sorted(out)


def _python_scripts():
    pats = ["scripts/**/*.py", "skills/**/*.py"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(ROOT, p), recursive=True)
    return sorted(out)


def test_script_globs_are_not_vacuous():
    """护栏本身不能空转：glob 写错时会收不到文件，而 parametrize 收到空列表就"全绿"。

    同时钉住"子目录必须被覆盖"：scripts/ 下只要还有被 glob 漏掉的 .py/.sh，这里就报错。
    """
    found = {os.path.relpath(p, ROOT) for p in _shell_scripts() + _python_scripts()}
    assert len(found) > 10, f"脚本 glob 只收 {len(found)} 个文件，疑似失效：{sorted(found)}"

    on_disk = set()
    for dirpath, _, filenames in os.walk(os.path.join(ROOT, "scripts")):
        for name in filenames:
            if name.endswith((".py", ".sh")):
                on_disk.add(os.path.relpath(os.path.join(dirpath, name), ROOT))
    missed = on_disk - found
    assert not missed, f"以下脚本未被语法检查覆盖：{sorted(missed)}"


@pytest.mark.parametrize("path", _shell_scripts())
def test_shell_syntax(path):
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"{os.path.relpath(path, ROOT)}: {r.stderr}"


@pytest.mark.parametrize("path", _python_scripts())
def test_python_syntax(path):
    with open(path, encoding="utf-8") as f:
        ast.parse(f.read(), filename=path)
