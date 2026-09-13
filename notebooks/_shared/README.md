# notebooks/_shared/

教学 notebook 共用的**环境自检**（唯一实现）。每个 notebook 的第一个 code cell 都会调它：

```python
import sys, pathlib
_here = pathlib.Path.cwd()
while _here != _here.parent and not (_here / "minigpt").is_dir():
    _here = _here.parent
sys.path.insert(0, str(_here / "notebooks" / "_shared"))
from course_env import setup

setup(
    required={"dataset/pretrain_t2t.jsonl": "bash scripts/data/download_data.sh"},
    optional={"models/checkpoints/pretrain_*/final.pt": "bash skills/minigpt-train/scripts/pipeline.sh --smoke"},
)
```

它做三件事：

1. **统一工作目录到仓库根** —— 于是所有相对路径都是「相对仓库根」，与 `scripts/`、`minigpt/` 完全一致；
   以前各课自己硬编码，漂移出过两套口径（`../dataset/x` 假定从 `notebooks/` 启动 vs `dataset/x` 假定已 chdir）。
2. **校验依赖** —— `required` 缺件**当场报错**并打印生成它的命令；`optional` 只**提示**不阻断
   （用于"正式跑需要、但不影响原理演示"的依赖，例如 SFT 课依赖的预训练权重）。
3. 顺带把仓库根放进 `sys.path`、打印 python/torch 版本（环境不对时能一眼看到）。

约定：

- 依赖写成 `{"文件": "生成它的命令"}`，把「这个文件从哪来」变成机器可校验的，而不是正文里的一句说明；
- `dataset/`、`models/` 都在 `.gitignore` 里（数据与权重不进版本库），所以全新 clone 后这些依赖**必然缺失**，
  `required` 的报错信息就是给这种情况准备的；
- **不依赖** torch，只用到标准库，任何环境都能跑。

护栏见 `tests/test_notebooks_course_env.py`：每课都必须调用 `setup`、代码里不得出现
`../dataset|x/models/` 这类启动目录相关的路径、所有 code cell 必须能编译、每课必须有
「教学版 vs 生产工程」的边界声明。
