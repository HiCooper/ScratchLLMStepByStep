"""脚本静态检查：所有 shell/python 脚本语法可编译（防止坏脚本进仓库）。"""
import ast
import glob
import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _shell_scripts():
    pats = ["scripts/*.sh", "skills/**/*.sh"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(ROOT, p), recursive=True)
    return sorted(out)


def _python_scripts():
    pats = ["scripts/*.py", "skills/**/*.py"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(ROOT, p), recursive=True)
    return sorted(out)


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
