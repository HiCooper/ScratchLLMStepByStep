"""`eval_thinking.py` / `chat_probe.py` 的 CPU 端到端冒烟。

为什么单独一个文件：这两个脚本此前**只被间接引用**（报告要读它们的产物），
没有任何测试真正把它们跑起来。而它们是交付链路里最后两个没被执行过的环节——
一旦 argparse/输出字段/落盘路径出问题，只有等基座跑完 20 小时后才会暴露。

做法：用真实分词器词表（32000）建一个 4M 参数的**随机** tiny checkpoint，
在 CPU 上（`CUDA_VISIBLE_DEVICES=""`，绝不碰正在训练的 GPU）只跑 1~2 条、
`--max-new-tokens 8`，重点验证"整条管子通不通"而不是数值好不好看
（随机权重当然算不对，准确率 0% 是预期）。
依赖真实 dataset/models 时自动 skip（CI 里两者都在 .gitignore）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(ROOT, "scripts")
TOK_DIR = os.path.join(ROOT, "models", "tokenizer_v3")
EVAL_JSONL = os.path.join(ROOT, "dataset", "sft", "cot_eval_easy_em50_disjoint.jsonl")
SCENARIOS = os.path.join(ROOT, "eval_sets", "chat_scenarios_zh.jsonl")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(TOK_DIR) and os.path.exists(EVAL_JSONL) and os.path.exists(SCENARIOS)),
    reason="需要真实分词器 / CoT 留出集 / 对话场景（均不入库）",
)


@pytest.fixture(scope="module")
def tiny_probe_ckpt(tmp_path_factory):
    """词表必须与分词器一致（32000），否则加载期就会因 embedding 形状不符报错。"""
    import torch
    from transformers import AutoTokenizer

    from minigpt.model.transformer import GPTConfig, MiniGPT

    vocab = len(AutoTokenizer.from_pretrained(TOK_DIR))
    cfg = GPTConfig(emb_dim=64, n_layers=2, n_heads=2, context_length=512, vocab_size=vocab)
    path = tmp_path_factory.mktemp("probe") / "tiny.pt"
    torch.save({"model_state": MiniGPT(cfg).state_dict(), "config": cfg.to_dict(), "step": 0}, path)
    return str(path)


def _run(args, extra_env=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **(extra_env or {})}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def test_eval_thinking_runs_end_to_end_on_cpu(tiny_probe_ckpt, tmp_path):
    out = tmp_path / "eval.json"
    r = _run([os.path.join(SCRIPTS, "eval_thinking.py"), "--checkpoint", tiny_probe_ckpt,
              "--tokenizer-dir", TOK_DIR, "--eval-jsonl", EVAL_JSONL,
              "--n", "2", "--strategies", "plain,single",
              "--max-new-tokens", "8", "--thinking-max-tokens", "8",
              "--repetition-penalty", "1.0", "--output", str(out)])
    assert r.returncode == 0, r.stderr[-1500:]
    js = json.load(open(out, encoding="utf-8"))
    s = js["summary"]
    assert s["n"] == 2 and {"plain", "single"} <= set(s["accuracy"])
    # 报告 §4.5 与交付自检都依赖 by_difficulty
    assert s["by_difficulty"] and all("n" in v for v in s["by_difficulty"].values())
    assert len(js["details"]) == 2 and "plain_ok" in js["details"][0]


def test_chat_probe_runs_end_to_end_on_cpu(tiny_probe_ckpt, tmp_path):
    out = tmp_path / "probe.txt"
    r = _run([os.path.join(SCRIPTS, "chat_probe.py"), "--checkpoint", tiny_probe_ckpt,
              "--tokenizer-dir", TOK_DIR, "--scenarios", SCENARIOS, "--limit", "2",
              "--modes", "greedy", "--max-new-tokens", "8", "--output", str(out)])
    assert r.returncode == 0, r.stderr[-1500:]
    txt = out.read_text(encoding="utf-8")
    assert "对话能力探针报告" in txt and "meta-01" in txt
    # 同目录会落一份 json 侧车（报告不读它，但接口稳定性值得锁住）
    assert out.with_suffix(".json").exists()
