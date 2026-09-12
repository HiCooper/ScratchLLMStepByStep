"""真实 SFT 数据上的**语义**预检：监督掩码给模型的 target 到底是不是 assistant 回复。

为什么不放在 fixture 里：真实文件有 12 个字段、带 `history` 多轮（前 200 条里 51 条多轮），
fixture 很难复现"多轮 + 长 prompt 截断 + qwen 模板"这三者叠加的情况；而这正是
"右截断把整段回复切掉 ⇒ 零监督"老问题会露头的地方。

只读分词器与前 4 万行数据（约 2 秒，不需要 GPU/权重），data/ 不存在时自动 skip（CI 里 dataset/ 被 ignore）。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
JSONL = os.path.join(ROOT, "dataset", "sft", "sft_data_zh.jsonl")
TOK_DIR = os.path.join(ROOT, "models", "tokenizer_v3")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(JSONL) and os.path.isdir(TOK_DIR)),
    reason="需要真实 SFT 数据与分词器（dataset/ 与 models/ 不入库）",
)

TURN_END = "<|im_end|>"


def _batch(n=8, max_len=512, max_lines=2000):
    from transformers import AutoTokenizer

    from minigpt.data.sft_dataset import InstructionDataset, collate

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = InstructionDataset(JSONL, tok, max_len=max_len, max_lines=max_lines)
    items = [ds[i] for i in range(min(n, len(ds)))]
    return tok, items, collate(items, tok.pad_token_id, tok)


def test_every_sample_has_supervision():
    """零监督样本必须为 0：否则那条样本对 SFT 完全无贡献（白跑）。"""
    _, items, (ids, labels, attn) = _batch()
    sup = (labels != -100).sum(dim=1)
    assert (sup > 0).all(), f"存在零监督样本：{sup.tolist()}"
    assert ids.shape == labels.shape == attn.shape
    assert ids.shape[1] <= 512 and (attn.sum(dim=1) <= 512).all()


def test_supervised_targets_are_assistant_replies():
    """语义口径：把被监督的 target 解码出来，必须是回复（以回合结束符收尾），而不是用户提问。"""
    tok, items, (ids, labels, attn) = _batch()
    lines = open(JSONL, encoding="utf-8").readlines()
    checked = 0
    for r, it in enumerate(items):
        seq = it[0] if isinstance(it, tuple) else it
        sup_pos = (labels[r] != -100).nonzero().flatten().tolist()
        sup_txt = tok.decode([int(labels[r][p]) for p in sup_pos])
        row = json.loads(lines[r])
        # 用户侧文本 = instruction + input（部分 BelleGroup 行的 instruction 为空、正文在 input）
        prompt_txt = ((row.get("instruction") or "") + (row.get("input") or "")).strip()
        assert sup_txt.rstrip().endswith(TURN_END), f"样本 {r} 监督段没有以 {TURN_END} 收尾：{sup_txt[-40:]!r}"
        if len(prompt_txt) >= 12:
            checked += 1
            assert prompt_txt[:12] not in sup_txt, f"样本 {r} 把用户提问也算进了 loss"
        # 每个监督位置都必须落在某个 assistant 区间附近（±1 是 labels↔input_ids 的移位约定）
        from minigpt.data.sft_dataset import assistant_spans
        spans = assistant_spans(seq, tok)
        allowed = {p + d for s, e in spans for p in range(s, e) for d in (-1, 0)}
        assert set(sup_pos) <= allowed, f"样本 {r} 监督位置跑到 assistant 区间外：{sorted(set(sup_pos) - allowed)[:5]}"
    assert checked >= 5, f"只有 {checked} 条样本参与了'用户提问未进 loss'的检查，覆盖不足"


def test_multi_turn_history_is_supervised_multiple_spans():
    """多轮样本应当有多个监督段（只监督最后一轮会丢掉 history 里的回复）。"""
    tok, items, (ids, labels, _) = _batch(n=40, max_lines=200)
    multi = 0
    for r, it in enumerate(items):
        seq = it[0] if isinstance(it, tuple) else it
        pos = (labels[r] != -100).nonzero().flatten().tolist()
        runs = sum(1 for a, b in zip(pos, pos[1:]) if b != a + 1) + (1 if pos else 0)
        if runs > 1:
            multi += 1
    assert multi > 0, "40 条里一条多轮都没命中，样本选取或 history 处理可能有问题"


# ─────────────────────── CoT 数据契约（标记协议 + 通用指令比例）───────────────────────
COT_EASY = os.path.join(ROOT, "dataset", "sft", "sft_cot_easy_em50_60k.jsonl")
COT_HARD = os.path.join(ROOT, "dataset", "sft", "sft_cot_hard_80k.jsonl")
COT_EVAL = [os.path.join(ROOT, "dataset", "sft", "cot_eval_easy_em50_disjoint.jsonl"),
            os.path.join(ROOT, "dataset", "sft", "cot_eval_hard_disjoint.jsonl")]

cot_needed = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [COT_EASY, COT_HARD, *COT_EVAL]),
    reason="需要真实 CoT 数据（dataset/ 不入库）",
)


def _cot_stats(path):
    """按「是不是 CoT 行」分类统计。

    数据布局（实测，两个文件一致）：
      - CoT 行：`instruction` 非空（题面），`input` 为空，`output` = THINK…MARK…数字；
      - 通用指令行（BelleGroup 混入的 25%）：`instruction` **为空**，题面在 `input` 里，
        output 是普通回答、不带 CoT 标记。
    所以「`instruction` 是否非空」就是可靠的判据。

    第一版按「有没有 MARK」分类，变异测试当场证明它是空的：把某条 CoT 行的
    `最终答案：` 改掉后，它只是从"已标记"变成"未标记"，通用占比 25%→26% 仍在容差内，
    检查全绿。第二版又错在把"非模板算式题"当成通用行（CoT 里也有应用题），
    导致真数据被判红——判据必须是 `instruction` 空不空，不是题面长什么样。
    """
    import re

    from minigpt.cot_format import ANSWER_MARK, THINK_PREFIX
    num = re.compile(r"-?\d+(?:\.\d+)?")
    n = cot = cot_bad = general = general_marked = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            o = json.loads(line)
            out = o.get("output") or ""
            if (o.get("instruction") or "").strip():
                cot += 1
                if (ANSWER_MARK not in out or not out.startswith(THINK_PREFIX)
                        or not num.search(out.split(ANSWER_MARK)[-1])):
                    cot_bad += 1
            else:
                general += 1
                if ANSWER_MARK in out:
                    general_marked += 1      # 通用行不该出现 CoT 标记
    return n, cot, cot_bad, general, general_marked


@cot_needed
@pytest.mark.parametrize("path", [COT_EASY, COT_HARD])
def test_cot_train_data_contract(path):
    """CoT 训练集：CoT 行必须"THINK 开头 + MARK 后是数字"，通用行不得带标记，比例 ~25%。

    真实隐患：`build_cot_sft.py` 是可重新生成的脚本，一旦改坏（漏拼 THINK、答案格式变化、
    混入比例跑偏、标记被吞），训练照样能跑，但模型会学出不一致的输出格式——评测的
    extract_answer 只能退回"取最后一个数字"的兜底路径，准确率数字就不可信了。
    """
    n, cot, bad, general, general_marked = _cot_stats(path)
    assert n > 0 and cot > 0
    assert bad == 0, f"{os.path.basename(path)} 有 {bad}/{cot} 条 CoT 行格式不符（THINK/MARK/数字）"
    assert general_marked == 0, f"{os.path.basename(path)} 有 {general_marked} 条通用行混进了 CoT 标记"
    ratio = general / n
    assert 0.20 <= ratio <= 0.30, f"通用指令占比 {ratio:.1%} 偏离设计的 25%"


@cot_needed
@pytest.mark.parametrize("path", COT_EVAL)
def test_cot_eval_sets_are_pure_arithmetic_with_markers(path):
    """留出集必须 100% 是"带标记的 CoT 行"（要用来算分档准确率，格式错一条就少一条有效样本）。"""
    n, cot, bad, general, _ = _cot_stats(path)
    assert cot == n, f"{os.path.basename(path)} 有 {n - cot} 条是通用行（留出集不该混通用指令）"
    assert bad == 0
