"""`mix_corpora.py` 多源配比器的回归测试。

背景：分布问题的根源是"取到了哪一部分"不可见。旧实现（`--mix-limit` 顺序读前 N 行）
在本仓库按领域排序的语料上等于只取中文头部，于是领域续训的"防遗忘"重放 100% 落在
前 30 万行、英文尾部一条没进（实测）。这些测试锁住三件事：
  1) 权重配额按**字符**生效（不是按条数平摊）；
  2) 每个来源内部是**整文件均匀抽样**（不是前缀）；
  3) manifest 落盘，配额/实际占比可复核。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write(path, tag, n, body_len):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"{tag}{i}-" + "x" * body_len}) + "\n")


def _run(args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "mix_corpora.py")] + args,
                          capture_output=True, text=True, cwd=ROOT)


def test_char_weighted_quota(tmp_path):
    """字符配额按权重分配：70/30 时 A 的字符数应约为 B 的 7/3 倍。"""
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, "A", 2000, 300)          # 平均 ~305 字符
    _write(b, "B", 2000, 900)          # 平均 ~905 字符
    out = tmp_path / "mix.jsonl"
    r = _run(["--source", f"{a}:0.7", "--source", f"{b}:0.3",
              "--target-chars", "200000", "--out", str(out)])
    assert r.returncode == 0, r.stderr

    texts = [json.loads(l)["text"] for l in out.read_text(encoding="utf-8").splitlines()]
    ca = sum(len(t) for t in texts if t.startswith("A"))
    cb = sum(len(t) for t in texts if t.startswith("B"))
    assert ca + cb > 0
    assert 0.65 < ca / (ca + cb) < 0.75, f"字符占比 {ca/(ca+cb):.2f} 偏离 70%"


def test_sampling_is_uniform_not_prefix(tmp_path):
    """每个来源内部必须整文件均匀抽：抽到的下标应横跨首尾，而不是只落在开头。"""
    a = tmp_path / "a.jsonl"
    _write(a, "A", 4000, 300)
    out = tmp_path / "mix.jsonl"
    r = _run(["--source", str(a), "--target-chars", "60000", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    idx = sorted(int(json.loads(l)["text"].split("-")[0][1:])
                 for l in out.read_text(encoding="utf-8").splitlines())
    assert idx[-1] > 3000, f"最大下标只有 {idx[-1]}：又退化成只读前缀了"
    assert idx[0] < 1000, f"最小下标 {idx[0]}：分布偏后"


def test_manifest_records_quota(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, "A", 500, 300)
    _write(b, "B", 500, 300)
    out = tmp_path / "mix.jsonl"
    r = _run(["--source", f"{a}:0.5", "--source", f"{b}:0.5",
              "--target-chars", "100000", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "mix.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert len(m["sources"]) == 2
    assert m["total"]["rows"] == sum(s["picked"] for s in m["sources"])
    for s in m["sources"]:
        assert s["lines_scanned"] == 500          # 扫过整个来源（而不是提前停）
        assert 0.4 < s["actual_share"] < 0.6


def test_rejects_mixed_weight_styles(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, "A", 10, 100)
    _write(b, "B", 10, 100)
    r = _run(["--source", f"{a}:0.7", "--source", str(b),
              "--target-chars", "1000", "--out", str(tmp_path / "o.jsonl")])
    assert r.returncode != 0, "部分带权重、部分不带的写法必须直接报错"
