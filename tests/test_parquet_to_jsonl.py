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
