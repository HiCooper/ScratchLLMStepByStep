#!/usr/bin/env python3
"""长文档语料的清洗器：**按结构切块** + 噪声过滤 + 去重，把"篇"变成"条"。

为什么需要单独一个清洗器（而不是复用 parquet_to_jsonl 的长度过滤）：
  - 长文档语料（如 arXiv 论文，中位 28,651 字符）在 ctx512 下**一篇要占 25–35 个窗口**，
    窗口之间高度同质 → 单位 token 的信息量极低。单纯"截断到 N 字符"只是把同一篇的前 N 字符
    留下，同质问题没解决；**按结构切成独立样本**才是对症的清洗。
  - 这类语料的脏东西也特殊：YAML/abstract 元数据头、`$$...$$` 与 `\\label{}` 脚手架、
    `--------------------` 分隔线、References/Acknowledgements 段、公式与表格堆。
    它们是"合法文本"但不是语言建模想要的信号。

处理链（每篇文档）：
  1) 归一化：去行尾空白、折叠 3+ 连续空行、丢掉纯分隔线行（`---`/`===`/`***`）；
  2) 切块：先按空行/标题/分隔线切段落，再**贪心打包**成 ≤`--chunk-chars` 的块，
     遇到标题优先起新块（保证块的语义完整）；
  3) 过滤（逐块）：长度、字母占比、LaTeX 密度、重复字符率、重复行率、参考文献/致谢等样板段；
  4) 去重：归一化后文本的精确去重（跨文档，全局）；
  5) 输出 `{"text": ...}` jsonl + 清洗报告（各过滤器的丢弃量、长度分布、语言占比）。

用法：
    python scripts/data/clean_long_docs.py --src dataset/IndustryCorpus2_mathematics_statistics_high \\
        --out dataset/domain/math_clean.jsonl --chunk-chars 2000 --target-chars 500000000 \\
        --shuffle-files --seed 20260912
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SEP_LINE = re.compile(r"^\s*([-=*_~#]{3,}|-{2,}\s*)\s*$")
HEADER_LINE = re.compile(r"^\s{0,3}(#{1,6}\s+\S|[-=]{4,}\s*$)")
LATEX_CMD = re.compile(r"\\[a-zA-Z]{2,}")
LATEX_ENV = re.compile(r"\\(begin|end)\{[a-zA-Z*]+\}")
BOILERPLATE = re.compile(
    r"^\s*(references|bibliography|acknowledg(e)?ments?|appendix|"
    r"author contributions?|funding|conflicts? of interest|data availability|"
    r"supplementary material)\s*:?\s*$", re.I)
ARXIV_ID = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}", re.I)
CJK = re.compile(r"[\u4e00-\u9fff]")
WORD = re.compile(r"[A-Za-z\u4e00-\u9fff]{2,}")
# 参考文献行：`[*书名*]{}`、et al.、@引用键、年份+出版社/卷期
CITATION_LINE = re.compile(
    r"(\[\*[^\]]*\*\]\{\})|(\bet al\.)|(@[A-Za-z]+\d{2,})|"
    r"(\b(19|20)\d{2}\b[^\n]{0,60}\b(press|publish|springer|wiley|elsevier|"
    r"academic|kluwer|vol\.|no\.|pp\.|eds\.))", re.I)


def normalize(text: str) -> str:
    """行尾空白、连续空行、纯分隔线行。"""
    out = []
    for ln in text.splitlines():
        ln = ln.rstrip()
        if SEP_LINE.match(ln):
            continue
        out.append(ln)
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_paragraphs(text: str) -> list[str]:
    """按空行切段；标题行单独成段（便于后续"遇标题起新块"）。"""
    paras = []
    for raw in re.split(r"\n\s*\n", text):
        p = raw.strip()
        if not p:
            continue
        lines = p.splitlines()
        if not lines:                      # 理论上不会发生，但绝不能让空段流到下游
            continue
        if HEADER_LINE.match(lines[0]) and len(p) > 200:
            paras.append(lines[0])
            tail = "\n".join(lines[1:]).strip()
            if tail:                       # 关键：单行标题段落时 tail 为空，不能追加空段
                paras.append(tail)
        else:
            paras.append(p)
    return paras


def chunk_document(text: str, chunk_chars: int, min_chunk: int) -> list[tuple[str, bool]]:
    """贪心打包段落成块，返回 [(chunk, in_boilerplate)]。

    `in_boilerplate`：一旦遇到 References/Acknowledgements/Appendix 标题，**后续所有块**都标记为
    样板段，直到出现下一个正常章节标题。这是必须的——参考文献往往被切成好几块，
    只有第一块的首行是 "References"，后面几块看起来就是普通文本（实测第一版清洗就漏放了整段文献表）。
    """
    chunks: list[tuple[str, bool]] = []
    cur, boiler = "", False
    for p in split_paragraphs(text):
        first = p.splitlines()[0].strip()
        if BOILERPLATE.match(first):
            if cur.strip():
                chunks.append((cur.strip(), boiler))
            cur, boiler = "", True
            continue
        is_header = bool(HEADER_LINE.match(first)) and len(first) < 120
        if is_header:
            if cur.strip():
                chunks.append((cur.strip(), boiler))
            cur, boiler = p, False          # 进入新章节 → 退出样板段
            continue
        if len(cur) + len(p) + 1 > chunk_chars and len(cur) >= min_chunk:
            chunks.append((cur.strip(), boiler))
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur.strip():
        chunks.append((cur.strip(), boiler))
    return [(c, b) for c, b in chunks if len(c) >= min_chunk]


def max_char_repeat(s: str, n: int = 12) -> float:
    """最长重复子串的粗略代理：n-gram 重复占比（>0.3 视为复读/表格堆）。"""
    if len(s) < n * 2:
        return 0.0
    grams = collections.Counter(s[i:i + n] for i in range(0, len(s) - n, max(1, n // 2)))
    top = grams.most_common(1)[0][1]
    return top * n / len(s)


NOT_LETTER = re.compile(r"[\W\d_]", re.UNICODE)   # 非字母（含数字/下划线）
SPACE_RE = re.compile(r"\s", re.UNICODE)


def junk_ratio(s: str) -> tuple[float, float, float]:
    """返回 (字母占比, LaTeX 密度/千字符, 数字+符号占比)。

    用 C 层正则替换 Python 逐字符循环：实测 genexpr 求和占清洗 CPU 的 ~90%，
    是唯一热点；换成 `re.sub` 计数后同机吞吐提升数倍。
    """
    n = max(1, len(s))
    # NOT_LETTER.sub("") 去掉的是**非字母**，剩下的长度就是字母数（别写成 n - len(...)）
    letters = len(NOT_LETTER.sub("", s))
    digits_sym = n - letters - (n - len(SPACE_RE.sub("", s)))
    latex_n = len(LATEX_CMD.findall(s)) + 2 * len(LATEX_ENV.findall(s)) + 2 * s.count("$$")
    return letters / n, latex_n / n * 1000, digits_sym / n


def lang_of(s: str) -> str:
    probe = s[:1500]
    c = len(CJK.findall(probe))
    a = sum(1 for ch in probe if ch.isascii() and ch.isalpha())
    return "zh" if c > a else "en"


def clean_chunk(chunk: str, in_boilerplate: bool, args) -> tuple[str | None, str]:
    """返回 (清洗后的块 or None, 丢弃原因)。"""
    if in_boilerplate:
        return None, "参考文献/致谢/附录段"
    if len(chunk) < args.min_chunk or len(chunk) > args.max_chunk:
        return None, "长度"
    first = chunk.splitlines()[0]
    if BOILERPLATE.match(first.strip()):
        return None, "参考文献/致谢/附录段"
    if ARXIV_ID.search(chunk[:200]):
        return None, "arXiv 元数据头"
    letters, latex, digits_sym = junk_ratio(chunk)
    if letters < args.min_letters:
        return None, "字母占比低"
    if latex > args.max_latex:
        return None, "LaTeX/公式密度高"
    if digits_sym > args.max_digits_sym:
        return None, "数字符号占比高"
    if max_char_repeat(chunk) > args.max_repeat:
        return None, "复读"
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if len(lines) >= 6 and len(set(lines)) / len(lines) < args.min_unique_lines:
        return None, "重复行多"
    # 自然语言密度：公式堆/矩阵/交换图的"词/百字符"极低
    words = len(WORD.findall(chunk))
    if words / max(1, len(chunk) / 100) < args.min_words_per_100:
        return None, "自然语言密度低（公式/表格堆）"
    # 参考文献块：很多行像引用条目
    if len(lines) >= 4:
        cites = sum(1 for ln in lines if CITATION_LINE.search(ln))
        if cites / len(lines) > args.max_citation_lines:
            return None, "参考文献条目多"
    if args.lang != "any" and lang_of(chunk) != args.lang:
        return None, "语种不符"
    return chunk, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="长文档语料清洗（结构切块 + 噪声过滤 + 去重）")
    ap.add_argument("--src", required=True, help="parquet 根目录或 jsonl 文件")
    ap.add_argument("--out", required=True)
    ap.add_argument("--content-key", default="text")
    ap.add_argument("--chunk-chars", type=int, default=2000, help="目标块大小（字符）")
    ap.add_argument("--min-chunk", type=int, default=300)
    ap.add_argument("--max-chunk", type=int, default=4000)
    ap.add_argument("--target-chars", type=float, default=0, help="输出字符预算（0=全量）")
    ap.add_argument("--shuffle-files", action="store_true", help="打乱分片顺序（预算截断时防领域偏差）")
    ap.add_argument("--shards", type=int, default=0,
                    help="最多使用几个分片（0=自动按预算均摊到 min(8,分片数) 个；"
                         "1 会把预算全花在单个分片上，是本仓库明令避免的单源偏差）")
    ap.add_argument("--lang", choices=["any", "zh", "en"], default="any")
    ap.add_argument("--min-letters", type=float, default=0.45, help="字母占比下限")
    ap.add_argument("--max-latex", type=float, default=25.0,
                    help="LaTeX 命令/环境密度上限（每千字符）；实测论文 p50=8、p90=33")
    ap.add_argument("--max-digits-sym", type=float, default=0.35, help="数字+符号占比上限")
    ap.add_argument("--max-repeat", type=float, default=0.30, help="12-gram 重复占比上限")
    ap.add_argument("--min-unique-lines", type=float, default=0.55)
    ap.add_argument("--min-words-per-100", type=float, default=8.0,
                    help="自然语言密度下限（词/百字符）；公式堆/矩阵通常 <5")
    ap.add_argument("--max-citation-lines", type=float, default=0.4,
                    help="像参考文献条目的行占比上限")
    ap.add_argument("--min-quality", type=float, default=0.0, help="源 quality_score 下限")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--report", default=None, help="清洗报告落盘（默认 <out>.report.json）")
    args = ap.parse_args()

    if os.path.isdir(args.src):
        files = sorted(glob.glob(os.path.join(args.src, "**", "*.parquet"), recursive=True))
    else:
        files = [args.src]
    if not files:
        raise SystemExit(f"没找到输入: {args.src}")
    if args.shuffle_files:
        random.Random(args.seed).shuffle(files)

    n_shards = args.shards or (min(8, len(files)) if args.target_chars else len(files))
    n_shards = max(1, min(n_shards, len(files)))
    per_shard = (args.target_chars / n_shards) if args.target_chars else 0
    files = files[:n_shards]
    print(f"[clean] 使用 {len(files)}/{len(glob.glob(os.path.join(args.src, '**', '*.parquet'), recursive=True)) if os.path.isdir(args.src) else 1}"
          f" 个分片，每片预算 {per_shard/1e6:.1f}M 字符" if args.target_chars else f"[clean] 使用 {len(files)} 个分片")

    drop = collections.Counter()
    seen: set = set()
    n_in = n_out = chars_out = dup = 0
    lens, langs, srcs = [], collections.Counter(), collections.Counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stop = False

    with open(args.out, "w", encoding="utf-8") as fo:
        for path in files:
            if stop:
                break
            shard_chars = 0
            if path.endswith(".parquet"):
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                cols = [args.content_key] + (["quality_score"] if "quality_score" in pf.schema.names else [])
                stream = (r for b in pf.iter_batches(batch_size=256, columns=cols) for r in b.to_pylist())
            else:
                def _lines():
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                yield json.loads(line)
                stream = _lines()
            src = os.path.relpath(path, args.src) if os.path.isdir(args.src) else os.path.basename(path)
            for row in stream:
                n_in += 1
                if args.min_quality and float(row.get("quality_score") or 0) < args.min_quality:
                    drop["源质量分低"] += 1
                    continue
                text = row.get(args.content_key) or ""
                if not text.strip():
                    drop["空文本"] += 1
                    continue
                srcs[src] += 1
                for chunk, boiler in chunk_document(normalize(text), args.chunk_chars, args.min_chunk):
                    ok, why = clean_chunk(chunk, boiler, args)
                    if ok is None:
                        drop[why] += 1
                        continue
                    h = hashlib.blake2b(ok.encode("utf-8"), digest_size=8).digest()
                    if h in seen:
                        dup += 1
                        continue
                    seen.add(h)
                    fo.write(json.dumps({"text": ok}, ensure_ascii=False) + "\n")
                    n_out += 1
                    chars_out += len(ok)
                    shard_chars += len(ok)
                    lens.append(len(ok))
                    if n_out % 7 == 0:
                        langs[lang_of(ok)] += 1
                    if args.target_chars and chars_out >= args.target_chars:
                        stop = True
                        break
                    if per_shard and shard_chars >= per_shard:
                        break
                if stop or (per_shard and shard_chars >= per_shard):
                    break
            print(f"[clean] {src}: 累计 {n_in:,} 篇 → {n_out:,} 块 / {chars_out/1e6:.1f}M 字符",
                  flush=True)

    import numpy as np
    L = np.array(lens) if lens else np.zeros(1, dtype=np.int64)
    zh = langs.get("zh", 0) / max(1, sum(langs.values()))
    report = {
        "src": args.src, "out": args.out, "docs_in": n_in, "chunks_out": n_out,
        "chars_out": int(chars_out), "dup_dropped": dup,
        "filter_dropped": sum(drop.values()),
        "dropped": dict(drop), "files_used": len(srcs),
        "chunk_chars_p50": int(np.percentile(L, 50)), "chunk_chars_p90": int(np.percentile(L, 90)),
        "zh_share": round(zh, 4),
        "docs_per_mchar": round(n_out / max(1e-9, chars_out / 1e6), 1),
        "params": vars(args),
    }
    rp = args.report or (args.out + ".report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[clean] 输入 {n_in:,} 篇 → 输出 {n_out:,} 块 / {chars_out/1e6:.1f}M 字符")
    print(f"[clean] 块长 p50={report['chunk_chars_p50']:,} p90={report['chunk_chars_p90']:,}；"
          f"块级过滤丢弃 {sum(drop.values()):,}；精确重复丢弃 {dup:,}")
    for k, v in drop.most_common():
        print(f"           - {k}: {v:,}")
    print(f"[clean] 中文块占比 {zh:.0%}；使用分片 {len(srcs)} 个")
    print(f"[clean] **每 M 字符的不同样本数 = {report['docs_per_mchar']:,}**"
          f"（原始语料清洗前约 26 个/百万字符）")
    print(f"[clean] 报告 → {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
