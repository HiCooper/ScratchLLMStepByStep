"""并行分词（`build_pretrain_bin.py --nproc N`）的等价性测试。

这是**逐字节**等价性测试：并行路径按行首把语料切成 N 段各自 tokenize 再按序拼接，
必须与单进程顺序 tokenize 的产物完全一致——差一个 token 都会让 bin 与 meta 对不上。
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "scripts", "build_pretrain_bin.py")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


@pytest.fixture()
def corpus(tmp_path):
    """含空行、坏行、长短行、中文与 emoji 的混合语料（覆盖各种切分边界）。"""
    p = tmp_path / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(400):
            body = ("中文测试文本" * (i % 7 + 1)) + f" line={i} ✅"
            f.write(json.dumps({"text": body}, ensure_ascii=False) + "\n")
            if i % 37 == 0:
                f.write("\n")                                  # 空行
            if i == 123:
                f.write('{"text": \x02坏行\n')                  # JSON 非法行
    return p


def _run(corpus, out, nproc, tokenizer_dir):
    r = subprocess.run([sys.executable, SCRIPT, "build", "--corpus-jsonl", str(corpus),
                        "--tokenizer-dir", tokenizer_dir, "--out-bin", str(out),
                        "--nproc", str(nproc)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_parallel_matches_sequential_byte_for_byte(corpus, tmp_path, tiny_tokenizer):
    tok_dir = str(tmp_path / "tok")
    tiny_tokenizer.save_pretrained(tok_dir)

    seq, par = tmp_path / "seq.bin", tmp_path / "par.bin"
    _run(corpus, seq, 1, tok_dir)
    _run(corpus, par, 4, tok_dir)

    assert _sha(seq) == _sha(par), "并行分词与串行分词产物不一致"
    m1 = json.loads((tmp_path / "seq.meta.json").read_text(encoding="utf-8"))
    m2 = json.loads((tmp_path / "par.meta.json").read_text(encoding="utf-8"))
    assert (m1["lines"], m1["tokens"], m1["bad_lines"]) == \
           (m2["lines"], m2["tokens"], m2["bad_lines"])
    assert m2["parts"] == 4 and len(m2["parallel_bounds"]) == 5


def test_parallel_rejects_max_lines(corpus, tmp_path, tiny_tokenizer):
    tok_dir = str(tmp_path / "tok")
    tiny_tokenizer.save_pretrained(tok_dir)
    r = subprocess.run([sys.executable, SCRIPT, "build", "--corpus-jsonl", str(corpus),
                        "--tokenizer-dir", tok_dir, "--out-bin", str(tmp_path / "x.bin"),
                        "--nproc", "3", "--max-lines", "10"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "互斥" in (r.stdout + r.stderr)


def test_line_aligned_bounds_cover_file_exactly(corpus):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from build_pretrain_bin import line_aligned_bounds

    size = os.path.getsize(corpus)
    for n in (1, 2, 3, 7, 16):
        bounds = line_aligned_bounds(str(corpus), n)
        assert bounds[0] == 0 and bounds[-1] == size
        assert bounds == sorted(set(bounds)), f"边界必须严格单调：{bounds}"
        # 每个内部边界都必须落在行首（前一字节是换行）
        with open(corpus, "rb") as f:
            for b in bounds[1:-1]:
                f.seek(b - 1)
                assert f.read(1) == b"\n", f"边界 {b} 不在行首"
