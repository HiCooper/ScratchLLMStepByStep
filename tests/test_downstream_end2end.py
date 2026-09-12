"""下游驱动的**编排**端到端测试（不训练、不碰 GPU）。

为什么需要：`run_v5_downstream.sh` 的每个阶段单独都验过（命令行解析、数据存在、
模型加载、报告渲染、交付自检），但"驱动把它们按顺序串起来、失败继续、最后写 SUMMARY
并以自检 rc 退出"这条**编排逻辑**从没跑过——而它正是"交付是否会缺块"的决定因素。

做法：在 tmp 里造一个沙箱仓库，把一个 **python3 影子命令**放到 PATH 最前：
- 训练/评测类调用直接 no-op（产物由 fixture 预置，形状照真实 schema）；
- 只有 report_training.py / verify_v5_delivery.py 转发给真 python3，并注入
  `--dir/--cp/--dataset` 指到沙箱，避免读到（或写到）真实仓库的产物。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_verify_delivery import _make_tree  # noqa: E402

DRIVER = os.path.join(ROOT, "scripts", "run_v5_downstream.sh")

SHIM = """#!/bin/bash
# 影子 python3：训练/评测 no-op；报告与自检转发给真 python3（脚本路径换成真实仓库里的
# 绝对路径——沙箱里没有这两个脚本），并注入沙箱的 --dir/--cp/--dataset，保证隔离。
REAL="__REAL__"
SB="__SB__"
case " $* " in
  *" scripts/report_training.py "*)
    exec __PY__ "$REAL/scripts/report_training.py" --dir "$SB/models/checkpoints" "${@:2}" ;;
  *" scripts/verify_v5_delivery.py "*)
    exec __PY__ "$REAL/scripts/verify_v5_delivery.py" --cp "$SB/models/checkpoints" --dataset "$SB/dataset" "${@:2}" ;;
esac
exit 0
"""


@pytest.fixture
def sandbox(tmp_path):
    sb = tmp_path / "sb"
    (sb / "scripts").mkdir(parents=True)
    (sb / "bin").mkdir()
    shutil.copy(DRIVER, sb / "scripts" / "run_v5_downstream.sh")
    _make_tree(str(sb / "models" / "checkpoints"), str(sb / "dataset"))
    # 基座 run 目录（驱动会优先取 best.pt）+ 一份**真实**的 tensorboard 事件，
    # 否则报告的 §1.5 曲线与 §6 同预算点都是空的——而交付自检正是要检查它们存在。
    from torch.utils.tensorboard import SummaryWriter
    base = sb / "models" / "checkpoints" / "pretrain_v5"
    base.mkdir(parents=True, exist_ok=True)
    (base / "best.pt").write_bytes(b"")
    w = SummaryWriter(str(base / "tensorboard"))
    for step, loss in ((2000, 5.11), (100000, 2.80), (210000, 2.55), (412000, 2.40)):
        w.add_scalar("eval/loss", loss, step)
        w.add_scalar("eval/perplexity", 2.718281828 ** loss, step)
    for step in range(0, 412000, 20000):
        w.add_scalar("train/loss", 2.6, step)
        w.add_scalar("train/tokens_per_sec", 24000.0, step)
    w.close()
    shim = sb / "bin" / "python3"
    shim.write_text(SHIM.replace("__SB__", str(sb)).replace("__REAL__", ROOT)
                    .replace("__PY__", sys.executable), encoding="utf-8")
    shim.chmod(0o755)
    return sb


def _run(sb, env_extra=None):
    env = {**os.environ, "PATH": f"{sb}/bin:{os.environ['PATH']}", **(env_extra or {})}
    return subprocess.run(["bash", str(sb / "scripts" / "run_v5_downstream.sh")],
                          cwd=str(sb), env=env, capture_output=True, text=True, timeout=600)


def test_driver_runs_all_stages_and_passes_selfcheck(sandbox):
    """完整产物齐备时：每个阶段 ✅、SUMMARY 有自检结论、驱动以 0 退出、报告含各章节。"""
    r = _run(sandbox)
    summary = (sandbox / "models/checkpoints/downstream_v5/SUMMARY.md").read_text(encoding="utf-8")
    assert "✅ SFT 指令微调" in summary
    assert "✅ CoT easy 训练" in summary and "✅ CoT hard 训练" in summary
    assert "✅ 交付自检全部通过" in summary
    assert "## 产物路径" in summary and "## 评测口径说明" in summary
    assert r.returncode == 0, f"自检应通过但 rc={r.returncode}\n{r.stdout[-1500:]}"

    report = (sandbox / "models/checkpoints/TRAINING_REPORT_v5.md").read_text(encoding="utf-8")
    for sect in ("## 1.", "## 1.5", "## 4.5", "## 5.", "## 6."):
        assert sect in report, f"报告缺少 {sect}"
    # 曲线来自上面写的真实 tensorboard 事件 → §1.5 有数据点、§6 有同预算点对比
    assert "| 曲线 step:loss |" in report and "2.0k:5.1100" in report
    assert "同预算点对比" in report and "| `pretrain_v5`（本次，同预算点） | 210,000 |" in report
    # 各阶段的日志文件都应有落盘（stage() 的第一个动作就是重定向）
    out = sandbox / "models/checkpoints/downstream_v5"
    for log in ("sft.log", "cot_easy.log", "cot_hard.log", "ppl_v5.log", "chat_probe.log"):
        assert (out / log).exists(), f"缺少阶段日志 {log}"


def test_driver_reports_missing_artifact_but_still_writes_report(sandbox):
    """缺一件产物时：自检 ❌ 写进 SUMMARY、驱动以非 0 退出，但报告仍然产出（不留半成品）。"""
    os.remove(sandbox / "models/checkpoints/sft_v5_cot_hard" / "best.pt")
    r = _run(sandbox)
    summary = (sandbox / "models/checkpoints/downstream_v5/SUMMARY.md").read_text(encoding="utf-8")
    assert "❌ 交付自检未通过" in summary
    assert r.returncode != 0
    assert (sandbox / "models/checkpoints/TRAINING_REPORT_v5.md").exists()
    assert "❌" in (sandbox / "models/checkpoints/downstream_v5/verify.log").read_text(encoding="utf-8")


def test_driver_is_isolated_from_real_repo(sandbox):
    """沙箱跑完后，真实仓库的产物目录不应被写入（防测试污染正在跑的交付）。"""
    real = os.path.join(ROOT, "models", "checkpoints", "TRAINING_REPORT_v5.md")
    before = os.path.exists(real)
    _run(sandbox)
    assert os.path.exists(real) == before, "沙箱运行写到了真实仓库的报告路径"
