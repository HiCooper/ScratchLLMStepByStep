"""长文档清洗器（`clean_long_docs.py`）的回归测试。

背景：长文档语料（arXiv 论文，中位 28,651 字符）在 ctx512 下**一篇要占 25–35 个窗口**，
窗口彼此高度同质 → 单位 token 信息量极低。清洗的关键不是"截断"，而是**按结构切成独立样本**，
并把这类语料特有的噪声（YAML 元数据头、`$$…$$`/`\\label{}` 脚手架、参考文献段、公式与表格堆）
挡在外面。这些测试逐条锁住那几种噪声。
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))


@pytest.fixture()
def cleaner():
    import clean_long_docs as C
    return C


def _args(**kw):
    class A:
        min_chunk = 50
        max_chunk = 4000
        min_letters = 0.45
        max_latex = 25.0
        max_digits_sym = 0.35
        max_repeat = 0.30
        min_unique_lines = 0.55
        min_words_per_100 = 8.0
        max_citation_lines = 0.4
        lang = "any"
    for k, v in kw.items():
        setattr(A, k, v)
    return A


def test_splits_long_document_into_structural_chunks(cleaner):
    """长文档必须被切成多条独立样本，而不是留下一条巨型记录。"""
    doc = "\n\n".join(f"这是第 {i} 段正文，用来测试按结构切块的效果。" * 8 for i in range(40))
    chunks = cleaner.chunk_document(cleaner.normalize(doc), chunk_chars=800, min_chunk=100)
    assert len(chunks) >= 5, f"应切出多块，实际 {len(chunks)}"
    assert all(len(c) <= 4000 for c, _ in chunks)
    # 清洗前一篇 40 段 → 清洗后多块，单位字符的样本数大幅提高
    assert sum(len(c) for c, _ in chunks) > len(doc) * 0.8


def test_drops_reference_section_even_without_header(cleaner):
    """参考文献段必须整段丢弃：只有第一块首行是 "References"，
    后续块看起来是普通文本（实测第一版清洗漏放了整张文献表）。"""
    doc = ("正文段落。" * 60) + "\n\nReferences\n\n" + \
          "\n".join(f"Author {i}, [*Book title {i}*]{{}}, Springer, 200{i%10}." for i in range(40))
    chunks = cleaner.chunk_document(cleaner.normalize(doc), chunk_chars=600, min_chunk=50)
    kept = [c for c, boiler in chunks if not boiler]
    dropped = [c for c, boiler in chunks if boiler]
    assert dropped, "参考文献段应被标记"
    assert not any("Book title" in c for c in kept), "文献条目漏进了正文"
    assert all("Book title" not in c for c in kept)


def test_drops_formula_and_table_dumps(cleaner):
    """公式堆/矩阵/交换图不是语言信号：词密度低、LaTeX 密度高，必须丢。"""
    formula = ("$$\\begin{smallmatrix} \\morphism|b|<1200,0>[P(f)`P_1Y;f] \\end{xy}$$\n" * 20)
    ok, why = cleaner.clean_chunk(formula, False, _args())
    assert ok is None and ("LaTeX" in why or "自然语言密度" in why or "字母" in why)


def test_keeps_math_prose_with_inline_latex(cleaner):
    """但要保留"数学行文"——带行内公式的正常句子不能一起被洗掉。"""
    prose = ("Let $f$ be a problem, $T$ a time constructible function. The problem $f$ is said "
             "to be computable in forced measurement $T(n)$-time, if for every input $x$ of "
             "size $n$, there is a sequence of adaptive measurements of length at most $T(n)$. ") * 4
    ok, why = cleaner.clean_chunk(prose, False, _args())
    assert ok is not None, f"数学行文被误杀：{why}"


def test_strips_yaml_frontmatter_and_separators(cleaner):
    doc = "---\nabstract: 'This paper presents a collection of useful formulas.'\n---\n\n" + \
          ("正文段落内容。" * 40) + "\n--------------------\n" + ("结尾段落。" * 40)
    s = cleaner.normalize(doc)
    assert "--------------------" not in s, "分隔线应被去掉"
    assert "\n\n\n" not in s, "连续空行应被折叠"


def test_cli_end_to_end_and_report(tmp_path, cleaner):
    """最小端到端：parquet 进、jsonl 出、报告落盘且统计自洽。"""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    src = tmp_path / "src" / "chinese" / "high"
    src.mkdir(parents=True)
    docs = [f"第{i}篇：这是一段足够长的中文正文，句子内容各不相同，用于测试清洗器端到端流程。"
            f"这里再补一句不同的内容，避免被判成复读。编号 {i}。" * 6 for i in range(5)]
    docs.append("$$\\label{} \\frac{a}{b} $$ " * 60)                       # 公式堆，应丢
    docs.append("References\n" + "\n".join(f"Author {i}, [*T{i}*]{{}}, Springer, 2001." for i in range(30)))
    pq.write_table(pa.table({"text": docs, "quality_score": [4.0] * len(docs)}),
                   str(src / "rank_00000.parquet"))
    out = tmp_path / "clean.jsonl"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "data", "clean_long_docs.py"),
                        "--src", str(tmp_path / "src"), "--out", str(out),
                        "--chunk-chars", "600", "--min-chunk", "100"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l)["text"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows, "应产出清洗后的块"
    assert not any("Springer" in t for t in rows), "参考文献条目必须被丢掉"
    rep = json.loads((tmp_path / "clean.jsonl.report.json").read_text(encoding="utf-8"))
    assert rep["docs_in"] == 7 and rep["chunks_out"] == len(rows)
    assert rep["filter_dropped"] > 0 and rep["docs_per_mchar"] > 0
