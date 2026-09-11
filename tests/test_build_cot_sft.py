"""CoT 数据构造的回归测试：训练/评测题目必须零重合。

真实事故：easy profile 的操作数只有 1..9，全部唯一题目仅约 1064 个，而训练集有
数万行——旧实现"评测集换个 seed"根本无效，实测旧评测集 156 条题目 **100%** 出现在
60k 训练集里，于是 README 的"easy CoT 90%→100%"测的是记忆而不是泛化。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import build_cot_sft as cot  # noqa: E402


def test_generate_unique_pool_dedups():
    pool, _ = cot.generate_unique_pool(cot.EASY_GENERATORS, seed=0, target=500)
    questions = [q["instruction"] for q in pool]
    assert len(questions) == len(set(questions)), "题目池必须按题面去重"
    assert len(pool) == 500, "空间足够时必须凑满 target"


def test_easy_question_space_is_small_that_is_the_root_cause():
    """把根因固定成断言：默认 easy 空间小到无法切出不相交评测集。"""
    pool, unique_total = cot.generate_unique_pool(cot.EASY_GENERATORS, seed=1, target=5000)
    assert len(pool) == unique_total
    assert len(pool) < 1500, "若此断言失败说明 easy 题目空间已扩大，可放宽 disjoint 限制"


def test_build_disjoint_guarantees_no_overlap():
    train, eval_rows, unique_total = cot.build_disjoint(
        2000, 100, seed=7, profile="hard")
    train_q = {r["instruction"] for r in train}
    eval_q = {r["instruction"] for r in eval_rows}
    assert not (train_q & eval_q), f"评测集与训练集重合 {len(train_q & eval_q)} 条"
    assert len(eval_rows) == 100


def test_build_disjoint_with_exclude():
    """--exclude-jsonl 的语义：排除题面绝不能出现在生成的池里。"""
    banned = [r["instruction"] for r in
              cot.build_easy(50, seed=3)] if hasattr(cot, "build_easy") else []
    if not banned:
        pool, _ = cot.generate_unique_pool(cot.EASY_GENERATORS, seed=3, target=50)
        banned = [q["instruction"] for q in pool]
    train, eval_rows, _ = cot.build_disjoint(
        200, 20, seed=11, profile="hard", exclude=banned)
    all_q = {r["instruction"] for r in train} | {r["instruction"] for r in eval_rows}
    assert not (all_q & set(banned)), "被排除的题面仍然出现在结果里"


def test_build_disjoint_shrinks_eval_when_space_is_tight():
    """空间不足时必须缩小评测集而不是与训练集重叠。"""
    pool, unique_total = cot.generate_unique_pool(cot.EASY_GENERATORS, seed=5, target=5000)
    train, eval_rows, _ = cot.build_disjoint(900, 200, seed=5, profile="easy")
    train_q = {r["instruction"] for r in train}
    eval_q = {r["instruction"] for r in eval_rows}
    assert eval_q and not (train_q & eval_q)
    # 评测集规模不得超过剩余空间的一个安全比例，保证训练侧仍有足够题目
    assert len(eval_rows) <= min(200, max(1, len(pool) // 5))


def test_written_files_are_disjoint(tmp_path, monkeypatch):
    """端到端：跑一次 main()，落盘的两份文件题面零重合。"""
    out_train = tmp_path / "train.jsonl"
    out_eval = tmp_path / "eval.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "build_cot_sft.py", "--profile", "hard", "--n-train", "300", "--n-eval", "50",
        "--mix-general", "0", "--out-train", str(out_train), "--out-eval", str(out_eval),
    ])
    cot.main()

    def qs(p):
        return {json.loads(l)["instruction"] for l in open(p, encoding="utf-8") if l.strip()}

    tr, ev = qs(out_train), qs(out_eval)
    assert ev and not (tr & ev), f"落盘文件仍有 {len(tr & ev)} 条重合"
