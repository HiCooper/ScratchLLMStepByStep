"""教学 notebook 的护栏：环境自检必须在、路径口径必须统一、代码必须能编译。

为什么值得测：`notebooks/` 的问题一向是**静默**的 —— 01 课读了已被删除的语料
（`pretrain_t2t_mini.jsonl`）、08/13 课的产物位置随 Jupyter 启动目录变化、09 课把
`model1` 写成 `model` 导致"未训练 vs 训练后"的对比用的是同一份权重。这些都不会报错，
只会给出错误结论或让学习者卡在 FileNotFoundError。

本文件不需要 torch/pytest 之外的重依赖，所以任何环境都能跑（含 CI 与只装了 pytest 的机器）。
"""
import ast
import glob
import json
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOKS = sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb")))

# IPython 魔法/兼容语法行：去掉后剩余部分应当仍是合法 Python
_MAGIC = re.compile(r"^\s*(!|%|%%)")
# 代码里不该出现的相对路径（图片引用在 markdown 里，由 Jupyter 按 notebook 目录解析，不受影响）
_BAD_RELATIVE = re.compile(r"\.\./(?:dataset|models)/")
# 注释行里可以出现（我们自己在 setup 的注释里说明历史漂移），只查真正的代码行
_COMMENT = re.compile(r"^\s*#")


def _cells(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        return json.load(f)["cells"]


def _text(cell):
    return "".join(cell["source"])


def _code_cells(nb_path):
    return [(i, c) for i, c in enumerate(_cells(nb_path)) if c["cell_type"] == "code"]


def test_there_are_notebooks_to_check():
    """护栏不能空转：glob 写错时下面所有 parametrize 都收不到参数、"全绿"。"""
    assert len(NOTEBOOKS) >= 15, f"只找到 {len(NOTEBOOKS)} 个 notebook，路径口径可能变了"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_every_notebook_bootstraps_the_course_env(nb):
    """每课都必须调 `course_env.setup(...)`：统一 chdir 到仓库根 + 校验依赖。

    没有它，路径就依赖"Jupyter 从哪里启动"，而 01/08/13 的产物位置会随之漂移。
    """
    assert any("from course_env import setup" in _text(c) for c in _cells(nb)), \
        f"{os.path.basename(nb)} 缺少环境自检（notebooks/_shared/course_env.py 的 setup 调用）"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_paths_are_repo_root_relative(nb):
    """代码里的数据/模型路径必须相对**仓库根**（与 scripts/、minigpt/ 一致）。

    历史事故：`../dataset/pretrain_t2t_mini.jsonl` 既写了错误的口径，又指向一个已经被
    `download_data.sh` 删除的文件 —— 学习者按 README 下完数据打开第 1 课仍然 FileNotFoundError。
    """
    offenders = []
    for i, cell in _code_cells(nb):
        for line in _text(cell).splitlines():
            if _COMMENT.match(line):
                continue
            if _BAD_RELATIVE.search(line):
                offenders.append(f"cell[{i}]: {line.strip()[:80]}")
    assert not offenders, f"{os.path.basename(nb)} 代码里有非仓库根相对路径：{offenders}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_code_cells_are_syntactically_valid(nb):
    """所有 code cell 去掉魔法行后必须能编译。

    教学代码也是代码：写坏的 cell 会让学习者以为"是自己环境的问题"。
    """
    errors = []
    for i, cell in _code_cells(nb):
        cleaned = "\n".join(ln for ln in _text(cell).splitlines() if not _MAGIC.match(ln))
        if not cleaned.strip():
            continue
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            errors.append(f"cell[{i}] line {exc.lineno}: {exc.msg}")
    assert not errors, f"{os.path.basename(nb)} 有语法错误的 code cell：{errors}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_every_notebook_declares_teaching_vs_production_scope(nb):
    """每课都要有"教学版 vs 生产工程"的边界声明。

    本仓库双定位（notebooks 教学 / minigpt 生产），学生照抄教学代码去跑真实训练是最贵的
    一类错误（如 13 课的简化 loss 掩码对应两次真实事故），边界必须写在每课最前面。
    """
    assert any("教学版 vs 生产工程" in _text(c) for c in _cells(nb)), \
        f"{os.path.basename(nb)} 缺少教学版/生产工程的边界声明"
