#!/usr/bin/env python3
"""数据集体检：下一次训练前，确认「语料 / 分词器 / .bin / SFT 集」四者自洽且没有静默缺口。

与 `skills/minigpt-train/scripts/preflight.py` 的分工：
  - preflight：**存在性与词表一致性**，秒级、只读，产出硬件预设；
  - 本脚本：**内容级审计**——jsonl 单遍扫描（行数/JSON 合法性/空文本/重复率/长度分布/
    语种占比）、`.bin` 字节数与 `.meta.json` 精确对账、训练-验证切分预览、
    SFT 训练集与评测集泄漏、CoT 协议埋点、parquet 下载完整性、磁盘与 token 预算。

用法：
    python scripts/data/audit_dataset.py                 # 全量（含 1.2GB 语料单遍扫描，约 1-2 分钟）
    python scripts/data/audit_dataset.py --quick         # 跳过语料扫描，只查 bin/切分/SFT/parquet
    python scripts/data/audit_dataset.py --json out.json # 落盘机器可读报告

退出码：存在 BLOCKER 时返回 1（可直接作为训练前的门禁）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from minigpt.config import ROOT, DataConfig, ModelConfig  # noqa: E402

RESULTS: list[dict] = []
BLOCKERS = 0
WARNINGS = 0


def add(level: str, name: str, detail: str, hint: str = "") -> None:
    global BLOCKERS, WARNINGS
    RESULTS.append({"level": level, "name": name, "detail": detail, "hint": hint})
    if level == "BLOCKER":
        BLOCKERS += 1
    elif level == "WARN":
        WARNINGS += 1
    icon = {"OK": "✅", "WARN": "⚠️ ", "BLOCKER": "❌"}.get(level, "•")
    print(f"{icon} {name:<28} {detail}" + (f"\n      ↳ {hint}" if hint else ""))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# ---------------------------------------------------------------- tokenizer
def audit_tokenizer() -> int | None:
    from transformers import AutoTokenizer
    tok_dir = Path(DataConfig().tokenizer_dir)
    if not tok_dir.exists():
        add("BLOCKER", "tokenizer", f"不存在 {tok_dir}", "先跑 scripts/data/train_tokenizer.py")
        return None
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    add("OK", "tokenizer", f"{tok_dir.name} vocab={len(tok)} eos={tok.eos_token_id} "
                           f"bos={tok.bos_token_id} unk={tok.unk_token_id} pad={tok.pad_token_id}")
    if tok.pad_token_id is None:
        add("WARN", "tokenizer:pad", "未定义 pad_token（SFT collator 会回退到 eos）",
            "预训练不受影响；SFT 用 eos 做 padding 时靠真实长度掩码，loss 不受污染")
    return len(tok)


# --------------------------------------------------------------------- bins
def audit_bins(vocab: int | None) -> list[dict]:
    bins = []
    for path in sorted(glob.glob(str(ROOT / "dataset" / "bins" / "*.bin"))):
        p = Path(path)
        meta_path = p.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        size = p.stat().st_size
        item = {"path": str(p.relative_to(ROOT)), "bytes": size, "meta": meta}
        bins.append(item)

        if meta is None:
            add("WARN", f"bin:{p.name}", "缺 .meta.json（dtype 只能按 uint16 猜）",
                "用 scripts/data/build_pretrain_bin.py 重建才有 meta")
            dtype = "uint16"
        else:
            dtype = meta.get("dtype", "uint16")
        itemsize = np.dtype(dtype).itemsize
        if size % itemsize:
            add("BLOCKER", f"bin:{p.name}", f"大小 {size}B 不是 {dtype}({itemsize}B) 的整数倍",
                "文件被截断或 dtype 记录错误：必须重建")
            continue
        tokens = size // itemsize
        item["tokens"] = tokens
        if meta is not None and meta.get("tokens") is not None:
            if int(meta["tokens"]) != tokens:
                add("BLOCKER", f"bin:{p.name}",
                    f"meta.tokens={meta['tokens']} 与实际 {tokens} 不符",
                    "语料/分词器变过但没重建 bin：先重建再训练")
                continue
        if vocab is not None and meta is not None and int(meta.get("vocab_size", 0)) != vocab:
            add("BLOCKER", f"bin:{p.name}",
                f"meta.vocab_size={meta.get('vocab_size')} ≠ tokenizer {vocab}",
                "换 --data_tokenizer_dir 后必须重建 bin（否则 embedding 静默错配）")
            continue
        add("OK", f"bin:{p.name}", f"{human(size)} tokens={tokens:,} dtype={dtype} "
                                   f"vocab={meta.get('vocab_size') if meta else '?'}")
    return bins


def audit_split(bins: list[dict]) -> None:
    """按训练入口同一口径预览 train/eval 切分（窗口数、验证 token 量、代表性问题）。"""
    from minigpt.data.pretrain_dataset import TokenBinDataset, split_train_eval_blocks
    dc = DataConfig()
    for b in bins:
        meta = b.get("meta") or {}
        for ctx in sorted({ModelConfig().context_length, 512, 1024}):
            if b.get("tokens", 0) < ctx * 8:
                continue
            ds = TokenBinDataset(str(ROOT / b["path"]), ctx)
            try:
                train, val, info = split_train_eval_blocks(
                    ds, eval_ratio=dc.eval_ratio, eos_id=meta.get("eos_id"), n_blocks=dc.eval_blocks)
            except ValueError as exc:
                add("WARN", f"split:{b['path'].split('/')[-1]}@ctx{ctx}", f"切分失败：{exc}")
                continue
            val_tok = info["eval_rows"] * (ctx - 1)
            note = (f"ctx{ctx}: 窗口 {len(ds):,} → train {len(train):,} / eval {len(val):,} "
                    f"({info['actual_eval_ratio']:.4%}, {val_tok:,} tokens)")
            # 验证 token 太少时 loss 噪声大（跨 run 比较会失真）
            if val_tok < 200_000:
                add("WARN", f"split:{b['path'].split('/')[-1]}@ctx{ctx}", note,
                    f"验证集仅 {val_tok:,} tokens，ppl 噪声约 ±{1/np.sqrt(max(val_tok,1)/ (ctx-1)):.3f}；"
                    f"跨 run 对比建议同时看 best_eval_loss 与多个 eval 点")
            else:
                add("OK", f"split:{b['path'].split('/')[-1]}@ctx{ctx}", note)


# ------------------------------------------------------------------- corpus
def audit_corpus(path: Path, content_key: str, do_scan: bool, meta_lines: int | None = None) -> dict:
    st = path.stat()
    info = {"path": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
            "bytes": st.st_size, "lines": None}
    if not do_scan:
        return info
    lines = blank = bad_json = missing = empty = 0
    dup = 0
    seen: set[bytes] = set()
    lens: list[int] = []
    cjk = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            lines += 1
            if not line.strip():
                blank += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            text = obj.get(content_key)
            if text is None:
                missing += 1
                continue
            if not text.strip():
                empty += 1
                continue
            lens.append(len(text))
            if lines % 7 == 0:                      # 抽样估语种占比（省时间）
                cjk += sum(1 for ch in text[:200] if "\u4e00" <= ch <= "\u9fff") > 20
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            if h in seen:
                dup += 1
            else:
                seen.add(h)
    kept = len(lens)
    arr = np.asarray(lens) if lens else np.zeros(1, dtype=np.int64)
    info.update(lines=lines, blank=blank, bad_json=bad_json, missing=missing,
                empty=empty, dup=dup, kept=kept,
                chars=int(arr.sum()), p50=int(np.percentile(arr, 50)),
                p99=int(np.percentile(arr, 99)), mx=int(arr.max()),
                cjk_ratio=round(cjk / max(1, lines // 7), 3))
    name = f"corpus:{path.name}"
    if bad_json or missing or empty:
        add("BLOCKER" if missing else "WARN", name,
            f"{lines:,} 行：JSON 非法 {bad_json} / 缺 {content_key} {missing} / 空文本 {empty}",
            "非法行会被 tokenize 阶段静默跳过或直接报错，先清洗")
    else:
        add("OK", name, f"{lines:,} 行 / {human(st.st_size)} / 字符 {arr.sum():,} "
                        f"(p50={info['p50']}, p99={info['p99']}, max={info['mx']:,}) "
                        f"中文占比≈{info['cjk_ratio']:.0%}")
    if dup:
        rate = dup / max(1, kept)
        (add("WARN", f"dup:{path.name}", f"重复文本 {dup:,} 行（{rate:.2%}）",
             "重复样本会被多次计入 loss，等价于给该内容加权；训练前建议去重")
         if rate > 0.01 else
         add("OK", f"dup:{path.name}", f"重复文本 {dup:,} 行（{rate:.3%}，可接受）"))
    if meta_lines is not None and lines != meta_lines:
        add("BLOCKER", f"corpus↔bin:{path.name}",
            f"语料 {lines:,} 行 ≠ bin 的 meta.lines={meta_lines:,}",
            "语料改过但 bin 没重建：训练前必须重建 bin")
    elif meta_lines is not None:
        add("OK", f"corpus↔bin:{path.name}", f"行数与 meta.lines 一致（{lines:,}）")
    return info


# ---------------------------------------------------------------------- SFT
def audit_sft(tok, do_scan: bool) -> None:
    from minigpt.data.sft_dataset import (DEFAULT_ASSISTANT_MARKER, assistant_spans,
                                          create_batch_collator)
    sft_dir = ROOT / "dataset" / "sft"
    if not sft_dir.exists():
        add("WARN", "sft-data", "dataset/sft 不存在（只能做预训练）")
        return
    files = sorted(sft_dir.glob("*.jsonl"))
    add("OK", "sft-data", f"{len(files)} 个 jsonl 文件")

    def key_of(obj: dict) -> str:
        return re.sub(r"\s+", "", (obj.get("instruction") or "") + (obj.get("input") or ""))

    for p in files:
        n = 0
        no_output = 0
        no_assistant = 0
        keys: Counter = Counter()
        sample_keys: list[str] = []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                n += 1
                obj = json.loads(line)
                keys.update(obj.keys())
                if not (obj.get("output") or "").strip():
                    no_output += 1
                if do_scan and i < 20:
                    sample_keys.append(key_of(obj))
                    ids = tok.apply_chat_template(
                        [{"role": "user", "content": key_of(obj)},
                         {"role": "assistant", "content": obj.get("output") or ""}],
                        tokenize=True, add_generation_prompt=False)
                    if not isinstance(ids, list):
                        ids = ids["input_ids"]
                    if not assistant_spans(ids, tok, DEFAULT_ASSISTANT_MARKER):
                        no_assistant += 1
        flag = "OK" if not (no_output or no_assistant) else "WARN"
        add(flag, f"sft:{p.name}",
            f"{n:,} 行 / 空 output {no_output}" +
            (f" / 前 20 条中 assistant 标记缺失 {no_assistant}" if do_scan else "") +
            ("" if do_scan else "（--quick 未做模板渲染检查）"),
            "含 'history' 多轮字段" if "history" in keys else "")
        if do_scan and no_assistant:
            add("BLOCKER", f"sft-template:{p.name}",
                f"{no_assistant}/20 条渲染不出 assistant 标记",
                "chat template 与 assistant_marker 不匹配，SFT 会在 collate 阶段直接报错")
    audit_cot_disjoint(do_scan)


def audit_cot_disjoint(do_scan: bool) -> None:
    """CoT 训练集 vs 评测集的题目重合度（仓库声称 disjoint 版本为 0）。

    两个口径都要过：

    1. **题面级**：instruction+input 归一化后完全相同；
    2. **算式级**：抽出 `请计算 <算式> 等于多少` 里的算式再比。只比题面是不够的——
       题面模板只要改一个字（"请计算"→"计算一下"）题面级就"零重合"，但题目还是同一道；
       实测旧留出集 vs 新训练集在题面级是 38.5% 重合，说明这个口径确实抓得住污染。
       反过来只用"把所有数字抠出来"这种糙口径会误报：通用指令里出现的 "35" 会和
       应用题里的 3 和 5 撞车（实测就产生过 1 条假重合），所以按模板抽算式。

    只审**链路实际使用**的两对。旧的 `sft_cot_easy_disjoint60k / cot_eval_easy_disjoint`
    （`--easy-max 9`）已被 `_em50_` 版本取代，不再审：继续审它只会让体检长期变红，
    反而把真信号淹掉（它已被 `dataset/README.md` 记为"已取代，仅作来源留档"）。
    """
    sft_dir = ROOT / "dataset" / "sft"
    pairs = [("sft_cot_easy_em50_60k.jsonl", "cot_eval_easy_em50_disjoint.jsonl"),
             ("sft_cot_hard_80k.jsonl", "cot_eval_hard_disjoint.jsonl")]
    if not do_scan:
        return
    expr_re = re.compile(r"^请计算\s*(.+?)\s*等于多少")

    def keys(path):
        """返回 (题面级集合, 算式级集合)。"""
        full, expr = set(), set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                s = (o.get("instruction") or "") + (o.get("input") or "")
                full.add(re.sub(r"\s+", "", s))
                m = expr_re.match((o.get("instruction") or "").strip())
                if m:
                    expr.add(re.sub(r"\s+", "", m.group(1)))
        return full, expr

    for train_name, eval_name in pairs:
        tp, ep = sft_dir / train_name, sft_dir / eval_name
        if not (tp.exists() and ep.exists()):
            continue
        tr_full, tr_expr = keys(tp)
        ev_full, ev_expr = keys(ep)
        hit = len(ev_full & tr_full)
        expr_hit = len(ev_expr & tr_expr)
        rate = hit / max(1, len(ev_full))
        detail = (f"与 {train_name}：题面级 {hit}/{len(ev_full)}（{rate:.1%}），"
                  f"算式级 {expr_hit}/{len(ev_expr)}")
        if "disjoint" in eval_name:
            (add("OK", f"cot-disjoint:{eval_name}", detail)
             if hit == 0 and expr_hit == 0 else
             add("BLOCKER", f"cot-disjoint:{eval_name}", detail,
                 "评测集被训练集污染：重跑 scripts/data/build_cot_sft.py 重建"))
        else:
            add("WARN" if rate > 0.3 else "OK", f"cot-overlap:{eval_name}",
                f"与 {train_name} 重合 {rate:.1%}（历史非 disjoint 口径，仅作对照）")


# ------------------------------------------------------------------ parquet
def audit_parquet() -> None:
    import glob as _glob

    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    for d in sorted(glob.glob(str(ROOT / "dataset" / "IndustryCorpus2_*"))):
        root = Path(d)
        on_disk = sorted(root.rglob("*.parquet"))
        if not on_disk:
            continue
        visible = sorted(_glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        msc = root / ".msc"
        recorded = 0
        if msc.exists():
            blob = msc.read_bytes()
            recorded = sum(1 for p in on_disk
                           if str(p.relative_to(root)).encode() in blob)
        temp = [p for p in on_disk if "._____temp" in str(p)]
        size = sum(p.stat().st_size for p in on_disk)

        # 以「能否解析」为唯一硬证据：ModelScope 的半截下载连 parquet footer 都没有，
        # 只看文件名/.msc 登记都会误判（.msc 不更新不代表文件坏，文件坏也不一定在 temp 里）。
        corrupt: list[str] = []
        if pq is not None:
            for p in on_disk:
                try:
                    pq.ParquetFile(p).metadata          # 只读 footer
                except Exception:                        # noqa: BLE001
                    corrupt.append(str(p.relative_to(root)))
        corrupt_in_temp = [p for p in corrupt if "._____temp" in p]

        detail = (f"{len(on_disk)} 个 parquet / {human(size)}；可解析 "
                  f"{len(on_disk) - len(corrupt)}；glob 可见 {len(visible)}；"
                  f"隐藏目录 {len(temp)}；.msc 登记 {recorded}")
        if corrupt and not corrupt_in_temp:
            add("BLOCKER", f"parquet:{root.name}", detail,
                f"有 {len(corrupt)} 个分片无法解析（下载截断）：{corrupt[:3]}。"
                f"先删除后重新下载，否则转换会在半路产出截断的 jsonl")
        elif corrupt_in_temp:
            add("BLOCKER", f"parquet:{root.name}", detail,
                f"隐藏目录 ._____temp 里有 {len(corrupt_in_temp)} 个**损坏**分片（半截下载），"
                f"且 glob(**/*.parquet) 不匹配隐藏目录 → 直接转换会静默少这些文件。"
                f"续传：modelscope download --dataset <repo> --local_dir <该子目录> "
                f"--include <缺失分片>（本轮已按此修复 8 个，可用 scripts/data/audit_dataset.py 复核）")
        elif temp:
            add("WARN", f"parquet:{root.name}", detail,
                f"隐藏目录里有 {len(temp)} 个可解析的 parquet，glob 看不到它们 → "
                f"转换会静默漏掉；确认可用后移出 ._____temp 再转换")
        else:
            add("OK", f"parquet:{root.name}", detail)


def audit_domain_jsonl() -> None:
    d = ROOT / "dataset" / "domain"
    if not d.exists():
        return
    for p in sorted(d.glob("*.jsonl")):
        add("OK", "domain-jsonl", f"{p.name} {human(p.stat().st_size)}（可 build_pretrain_bin 后做领域续训）")


# --------------------------------------------------------------- 磁盘/预算
def audit_budget(bins: list[dict]) -> None:
    import shutil
    from minigpt.config import estimate_params
    free = shutil.disk_usage(ROOT).free
    # derived：由其它 bin 的语料**派生**出来的混合产物（算进去就是重复计数）
    # superseded：被新版本取代的旧 bin（v3/mini、code_domain 已在清理中删除，这里只留兜底）
    derived = {"domain_code70_general30.bin", "domain_code50_general50.bin"}
    # v4 是 v5 的 A/B 对照组（同 token 预算、纯通用），与 v5 同源，不能重复计入预算
    superseded = {"pretrain_v3_full.bin", "code_domain.bin", "pretrain_v4_full.bin"}
    usable = [b for b in bins if Path(b["path"]).name not in superseded | derived]
    total_tokens = sum(b.get("tokens", 0) for b in usable)
    # 只列**磁盘上真实存在**而被排除的（旧版删掉后不该再出现在报告里）
    skipped = sorted(n for n in superseded | derived
                     if (ROOT / "dataset" / "bins" / n).exists())
    add("OK", "disk", f"可用 {human(free)} / 可用语料（已排除取代/对照/派生）{total_tokens/1e6:.1f}M tokens"
        + (f"；未计入 {', '.join(skipped)}" if skipped else ""))
    for mc in (ModelConfig(), ModelConfig(emb_dim=768, n_layers=12, n_heads=12)):
        params = estimate_params(mc.emb_dim, mc.n_layers, mc.vocab_size or 32000,
                                 use_swiglu=mc.use_swiglu, tie_word_embeddings=mc.tie_word_embeddings)
        ckpt = params * 16  # fp16 权重2 + fp32 主权重4 + 动量4 + 方差4 + 峰值梯度2
        ratio = total_tokens / params
        epochs = 20 / ratio if ratio else 0
        add("OK", f"budget:{mc.emb_dim}/{mc.n_layers}/{mc.n_heads}",
            f"{params/1e6:.1f}M params；单 checkpoint≈{human(ckpt)}；"
            f"单遍 tokens/params={ratio:.1f}×"
            + (f"（达到 Chinchilla 20× 需约 {epochs:.1f} 个 epoch，或补语料）" if ratio < 15
               else "（已达 Chinchilla 20×）"))


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="MiniGPT 数据集体检")
    ap.add_argument("--quick", action="store_true", help="跳过 1.2GB 语料单遍扫描")
    ap.add_argument("--json", default=None, help="报告落盘路径")
    ap.add_argument("--corpus", default=None, help="额外审计一个 jsonl（如新的领域语料）")
    args = ap.parse_args()

    print(f"=== MiniGPT 数据集体检 @ {__import__('time').strftime('%Y-%m-%d %H:%M:%S')} ===")
    vocab = audit_tokenizer()
    bins = audit_bins(vocab)

    # 语料 ↔ bin 的行数对账（约定映射，见 AGENTS.md）
    mapping = [("dataset/mixed/pretrain_v5_general80_structured20.jsonl", "pretrain_v5_full.bin"),
               ("dataset/pretrain_t2t.jsonl", "pretrain_v4_full.bin"),
               ("dataset/domain/code_corpus.dedup.jsonl", "code_domain_dedup.bin"),
               ("dataset/mixed/domain_code70_general30.jsonl", "domain_code70_general30.bin"),
               ("dataset/mixed/domain_code50_general50.jsonl", "domain_code50_general50.bin"),
               ("dataset/domain/finance_corpus.jsonl", "finance_corpus.bin"),
               ("dataset/domain/math_clean.jsonl", "math_clean.bin")]
    by_name = {Path(b["path"]).name: b for b in bins}
    for corpus, bin_name in mapping:
        cp = ROOT / corpus
        if not cp.exists():
            add("WARN", "corpus", f"缺失 {corpus}")
            continue
        meta_lines = (by_name.get(bin_name, {}).get("meta") or {}).get("lines")
        audit_corpus(cp, "text", not args.quick, meta_lines=int(meta_lines) if meta_lines else None)
    if args.corpus:
        audit_corpus(Path(args.corpus), "text", not args.quick)

    audit_domain_jsonl()
    audit_parquet()
    tok = None
    if not args.quick:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(DataConfig().tokenizer_dir)
    audit_sft(tok, do_scan=not args.quick)
    audit_split(bins)
    audit_budget(bins)

    print(f"\n汇总: OK={sum(1 for r in RESULTS if r['level']=='OK')} "
          f"WARN={WARNINGS} BLOCKER={BLOCKERS}")
    print("结论: " + ("存在阻塞项，先修数据再训练" if BLOCKERS else
                     "数据可用，可以开始下一次训练"))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": {"blockers": BLOCKERS, "warnings": WARNINGS}, "results": RESULTS},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 报告: {args.json}")
    return 1 if BLOCKERS else 0


if __name__ == "__main__":
    raise SystemExit(main())
