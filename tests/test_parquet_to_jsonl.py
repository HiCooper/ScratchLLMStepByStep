"""parquet→jsonl 转换器单测（无 pyarrow 时自动跳过）。"""
import json
import os
import subprocess
import sys

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


def _make_parquet(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table({
        "text": [r["text"] for r in rows],
        "quality_score": [r.get("quality_score", 4.0) for r in rows],
        "max_line_length": [r.get("max_line_length", 60) for r in rows],
    }), path)


def test_filters_and_mix(tmp_path):
    src = tmp_path / "src"
    rows = [{"text": f"x{i}" + "x" * 500} for i in range(10)]      # 10 条保留
    rows += [{"text": "y" * 100},                                  # 太短，丢弃
             {"text": "z" * 500, "quality_score": 1.0},            # 质量低，丢弃
             {"text": "w" * 500, "max_line_length": 900}]          # 行太长，丢弃
    _make_parquet(str(src / "a" / "part.parquet"), rows)
    mix = tmp_path / "mix.jsonl"
    mix.write_text("\n".join(json.dumps({"text": f"通用语料{i}" * 50}, ensure_ascii=False)
                             for i in range(50)), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "parquet_to_jsonl.py"),
                        "--src", str(src), "--out", str(out),
                        "--min-chars", "300", "--max-line-length", "500", "--min-quality", "3.0",
                        "--mix-jsonl", str(mix), "--mix-ratio", "0.15", "--tokenizer", ""],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    domain = [x for x in rows if x["text"].startswith("x")]
    general = [x for x in rows if x["text"].startswith("通用")]
    assert len(domain) == 10                     # 只有 10 条通过全部过滤
    assert len(general) >= 1                     # 15% 混入生效


def test_mix_sampled_from_whole_file_not_prefix(tmp_path):
    """混料必须**全文件均匀**抽，不能只读前 N 行。

    真实问题：`pretrain_t2t.jsonl` 按领域排序（前 120 万行中文短文本、最后约 7 万行英文长文本），
    旧实现顺序读前 `--mix-limit` 行当混料池，于是代码领域续训的"防遗忘"重放 **100% 落在中文头部、
    0% 落在英文尾部**（实测 14,232 条混料全部命中前 30 万行），与"领域续训后通用 ppl +41%"吻合。
    """
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                                    "scripts"))
    from parquet_to_jsonl import sample_jsonl_texts

    n = 10_000
    mix = tmp_path / "mix.jsonl"
    with open(mix, "w", encoding="utf-8") as f:
        for i in range(n):                      # 前半 HEAD_*、后半 TAIL_*（模拟按领域排序）
            tag = "HEAD" if i < n // 2 else "TAIL"
            f.write(json.dumps({"text": f"{tag}_{i}" + "x" * 50}) + "\n")

    pool = sample_jsonl_texts(str(mix), "text", limit=n // 10, seed=0)
    assert len(pool) == n // 10, "抽样池大小应等于 limit"
    head = sum(1 for t in pool if t.startswith("HEAD"))
    tail = sum(1 for t in pool if t.startswith("TAIL"))
    assert tail > 0, "英文尾部一条都没抽到——说明又退化成只读前缀了"
    # 均匀抽样下两半应大致各占一半（旧实现是 head=100%/tail=0%）
    assert 0.35 < head / len(pool) < 0.65, f"分布不均: head={head} tail={tail}"
