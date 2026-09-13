"""教学 notebook 的共享环境自检（唯一实现，所有课程第一个 cell 都调它）。

**为什么需要它**：`notebooks/` 是教学演示，但 15 个 notebook 各自硬编码相对路径，
已经漂移出多套口径 —— 有的写 `../dataset/xxx`（假定 CWD=notebooks/，如 01/08 课），
有的先 `os.chdir(仓库根)` 再用 `dataset/xxx`（如 09-12/14/15 课）；01 课甚至在读一个
**已经被 `scripts/data/download_data.sh` 删除**的语料（`pretrain_t2t_mini.jsonl`）。
结果是"照 README 下载完数据，打开第 1 课直接 FileNotFoundError"——而第 1 课产出的
分词器是 02-15 课的前置依赖，整条课程链第一步就断。

**做法**：每个 notebook 的第一个 cell 调一次 `setup({...})`：

- 向上找到仓库根（同时含 `minigpt/` 与 `scripts/`），把工作目录切过去
  ⇒ 之后所有相对路径统一为「相对仓库根」，与生产脚本（`scripts/`、`minigpt/`）一致；
- 顺带把仓库根放进 `sys.path`，于是教学 notebook 和生产工程**用同一个 `minigpt` 包**
  （不再出现"notebook 里一套实现、`minigpt/` 里另一套"的静默漂移）；
- 校验本课依赖的文件是否存在，缺件时**立刻报错并打印生成它的命令**，
  而不是等到几屏之后某个 cell 抛 FileNotFoundError。

用法（每个 notebook 的第一个 code cell）：

    import sys, pathlib
    _here = pathlib.Path.cwd()
    while _here != _here.parent and not (_here / "minigpt").is_dir():
        _here = _here.parent
    sys.path.insert(0, str(_here / "notebooks" / "_shared"))
    from course_env import setup
    setup({"dataset/pretrain_t2t.jsonl": "bash scripts/data/download_data.sh"})
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def find_repo_root(start=None) -> Path:
    """从 start（默认 CWD）向上找同时含 `minigpt/` 与 `scripts/` 的目录。"""
    p = Path(start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "minigpt").is_dir() and (cand / "scripts").is_dir():
            return cand
    raise RuntimeError(
        f"找不到仓库根：从 {p} 逐级向上都没有同时含 minigpt/ 与 scripts/ 的目录。\n"
        "请从仓库内部启动 jupyter（例如在仓库根执行 `jupyter lab notebooks/`），"
        "或把 notebook 复制回仓库的 notebooks/ 目录后再运行。")


def _present(root: Path, name: str) -> bool:
    """依赖是否满足。名字里带 `*` 时按 glob 匹配（任一命中即算满足）——

    预训练产物这类依赖的路径含运行名（`models/checkpoints/pretrain_<preset>/final.pt`），
    没法写成一个固定文件名，用模式表达才不会"明明有产物却报缺件"。
    """
    if "*" in name or "?" in name:
        return bool(glob.glob(str(root / name)))
    return (root / name).exists()


def require(required: dict, root=None) -> Path:
    """校验 required 里的依赖都在；缺任何一个就抛错并给出生成命令。

    `required` 形如 `{"dataset/pretrain_t2t.jsonl": "bash scripts/data/download_data.sh"}`
    —— 把"这个文件从哪来"写成机器可校验的，而不是正文里的一句说明。
    key 里可以带 `*`（glob），见 `_present`。
    """
    root = Path(root or find_repo_root())
    missing = [(name, how) for name, how in (required or {}).items()
               if not _present(root, name)]
    if missing:
        detail = "\n".join(f"  - {name}\n      生成命令：{how}" for name, how in missing)
        raise FileNotFoundError(
            f"缺少本课需要的文件（仓库根：{root}）：\n{detail}\n"
            "先执行上面的命令，再重新运行本单元。")
    return root


def setup(required: dict | None = None, *, optional: dict | None = None,
          chdir: bool = True, quiet: bool = False, show_env: bool = True) -> Path:
    """统一工作目录到仓库根 + 校验本课依赖（详见模块文档）。返回仓库根 Path。

    `required` 缺件即报错（本课后面一定会崩）；`optional` 只**提示**不阻断——
    用于"正式跑需要、但不影响本课数据/原理演示"的依赖，典型例子是 SFT 课依赖的
    预训练权重（要由生产预训练或 09 课产出，没有它本课仍能读完数据与原理部分）。

    名字里带 `*` 的 key 按 glob 匹配（见 `_present`），用于产物路径含运行名的情况，
    例如 `models/checkpoints/pretrain_*/final.pt`。
    """
    root = find_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if chdir:
        os.chdir(root)
    require(required or {}, root=root)
    missing_opt = [(n, how) for n, how in (optional or {}).items()
                   if not _present(root, n)]
    if not quiet:
        print(f"仓库根：{root}")
        print("依赖检查通过：" + ("、".join(required) if required else "（本课无需外部文件）"))
        if missing_opt:
            print("\n提示：以下**可选**依赖缺失，本课的原理/数据部分仍可照跑，"
                  "但涉及它的单元会失败（正式跑需要它）：")
            for n, how in missing_opt:
                print(f"  - {n}\n      生成方式：{how}")
        if show_env:
            try:                                  # 只做展示：环境不对时能让学习者一眼看到
                import torch
                print(f"python {sys.version.split()[0]} / torch {torch.__version__} / "
                      f"cuda {'可用' if torch.cuda.is_available() else '不可用'}")
            except Exception:                     # noqa: BLE001
                print(f"python {sys.version.split()[0]} / torch 未安装")
    return root
