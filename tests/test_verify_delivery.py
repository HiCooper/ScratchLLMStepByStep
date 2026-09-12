"""交付自检脚本的单测：完整产物树必须通过，缺一件必须失败，--allow-partial 不失败。

用最小 fixture 构造"看起来像真产物"的目录树（`.pt` 只要存在、JSON 只要字段齐全——
校验器刻意只读元数据，不做权重解析，所以不需要真模型）。
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.verify_v5_delivery import Checker  # noqa: E402


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, (dict, list)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(obj)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def _make_tree(cp: str, ds: str, eval_loss: float = 2.70) -> None:
    ev = eval_loss
    _w(os.path.join(cp, "pretrain_v5", "metrics.json"), {
        "step": 412000, "epoch": 0, "train_loss": 2.6, "eval_loss": ev,
        "perplexity": math.exp(ev), "best_eval_loss": ev, "best_step": 410000,
        "data_split": {"split": "blocked-document-aligned", "n_blocks": 500, "eval_rows": 500},
        "bin_meta": {"tokens": 1691885203, "bin_vocab": 32000},
    })
    _w(os.path.join(cp, "pretrain_v5", "config.json"), {
        "model": {"model_emb_dim": 512, "model_n_layers": 10, "model_n_heads": 8,
                  "model_context_length": 512},
        "train": {"max_steps": 412000, "mixed_precision_dtype": "float16"},
    })
    _touch(os.path.join(cp, "pretrain_v5", "final.pt"))
    _touch(os.path.join(cp, "pretrain_v5", "best.pt"))
    _w(os.path.join(ds, "bins", "pretrain_v5_full.bin.meta.json"),
       {"tokens": 1691885203, "vocab": 32000})
    for run, step in (("sft_v5_chat", 10000), ("sft_v5_cot_easy", 15000), ("sft_v5_cot_hard", 20000)):
        _w(os.path.join(cp, run, "metrics.json"),
           {"step": step, "train_loss": 1.0, "eval_loss": 1.1, "best_eval_loss": 1.0,
            "best_step": step - 500})
        _touch(os.path.join(cp, run, "best.pt"))
    _w(os.path.join(cp, "ppl_v5_general.json"),
       {"eval_loss": 2.7, "perplexity": math.exp(2.7), "split": "val",
        "bin": "dataset/bins/pretrain_v5_full.bin", "rows": 500, "max_rows": 0})
    for name in ("eval_v5_cot_easy", "eval_v5_cot_hard"):
        _w(os.path.join(cp, f"{name}.json"),
           {"summary": {"n": 200, "accuracy": {"plain": 0.31, "single": 0.28}}})
    _w(os.path.join(cp, "samples_v5_chat_probe.txt"), "场景 1\n回复：" + "好" * 400)
    _w(os.path.join(cp, "TRAINING_REPORT_v5.md"),
       "# MiniGPT 训练报告\n## 1.5 曲线\n| 曲线 step:loss | 2.0k:5.1 |\n## 6. 历史基线对照\nbest.pt\n")
    _w(os.path.join(cp, "downstream_v5", "SUMMARY.md"), "- ✅ 全部")


def test_complete_tree_passes(tmp_path):
    cp, ds = str(tmp_path / "cp"), str(tmp_path / "dataset")
    _make_tree(cp, ds)
    assert Checker(cp, ds).run() == 0


def test_missing_one_artifact_fails(tmp_path):
    cp, ds = str(tmp_path / "cp"), str(tmp_path / "dataset")
    _make_tree(cp, ds)
    os.remove(os.path.join(cp, "eval_v5_cot_hard.json"))
    assert Checker(cp, ds).run() == 1


def test_wrong_arch_is_detected(tmp_path):
    cp, ds = str(tmp_path / "cp"), str(tmp_path / "dataset")
    _make_tree(cp, ds)
    p = os.path.join(cp, "pretrain_v5", "config.json")
    cfg = json.load(open(p, encoding="utf-8"))
    cfg["model"]["model_n_layers"] = 6          # 架构对不上（不是 v5 预设）必须报错
    _w(p, cfg)
    assert Checker(cp, ds).run() == 1


def test_partial_mode_never_fails(tmp_path):
    cp, ds = str(tmp_path / "cp"), str(tmp_path / "dataset")
    _make_tree(cp, ds)
    os.remove(os.path.join(cp, "ppl_v5_general.json"))
    assert Checker(cp, ds, partial=True).run() == 0
