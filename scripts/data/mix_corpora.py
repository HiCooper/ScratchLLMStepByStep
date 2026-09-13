#!/usr/bin/env python3
"""多源语料配比器：把 N 份 jsonl 按**指定权重**混成一份，权重与抽样口径全部可复现。

为什么需要它（真实问题）：分布失衡往往不是"数据不够"，而是"取到了哪一部分"不可见。
本仓库踩过的坑：
  - `pretrain_t2t.jsonl` 按领域排序（前 120 万行中文、最后约 7 万行英文），
    任何"顺序读前 N 行"的取法都等于只取一个领域；
  - 领域续训的"防遗忘"混料曾 100% 来自通用语料前 30 万行，英文尾部一条没被重放；
  - 领域语料比通用语料大一到两个数量级，"全量混进去"会把分布压成单一领域。
因此本工具强制三件事：**整文件均匀抽样**（不是前缀）、**按权重配额**（不是全量堆叠）、
**落盘 manifest**（配额/来源/字符数可复核）。

用法：
    # 领域:通用 = 70:30，总量 3000 万字符
    python scripts/data/mix_corpora.py \
        --source dataset/domain/code_corpus.dedup.jsonl:0.70 \
        --source dataset/pretrain_t2t.jsonl:0.30 \
        --target-chars 30000000 --out dataset/mixed/code70_general30.jsonl

    # 不给权重时按文件大小等比例混合
    python scripts/data/mix_corpora.py --source a.jsonl --source b.jsonl \
        --target-chars 1e8 --out mix.jsonl

设计要点：
  - 每个来源独立做**蓄水池抽样**（单遍、O(配额) 内存），8GB 来源也能在有限内存里抽；
  - 字符配额用该来源的**平均字符长度**（前 2 万行抽样估）换算成条数配额；
  - 输出前整体 shuffle（seed 固定），避免"前一半是 A、后一半是 B"；
  - manifest 记录每个来源的权重/抽中条数/字符数/实际占比，便于验收。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def _fmt(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def parse_source(spec: str) -> tuple[str, float | None]:
    """`path[:weight]`；权重省略返回 None（调用方按文件大小等比例）。"""
    path, _, tail = spec.rpartition(":")
    if path:
        try:
            return path, float(tail)
        except ValueError:
            pass
    return spec, None


def rough_avg_chars(path: str, content_key: str, sample_lines: int = 20000) -> float:
    """**粗略**估平均字符长度，仅用于"超采定容"，不用于最终配额。

    不能只读前 N 行当真实均值：本仓库语料按领域排序（`pretrain_t2t.jsonl` 头部 274 字符、
    全文件真实均值 382 字符），按头部估会把 70:30 混成 62:38（实测踩到）。
    真正的配额由 `reservoir_sample` 扫全文得到的**精确均值**决定，见 main()。
    """
    n = chars = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                t = json.loads(line).get(content_key) or ""
            except json.JSONDecodeError:
                continue
            chars += len(t)
            n += 1
            if n >= sample_lines:
                break
    return chars / max(1, n)


def trim_to_lines(pool: list, want: int, seed: int) -> tuple[list, int]:
    """随机裁到 `want` 条（保持均匀性）。返回 (pool, 丢弃条数)。"""
    if want <= 0 or len(pool) <= want:
        return pool, 0
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:want], len(pool) - want


def reservoir_sample(path: str, content_key: str, want: int | None, seed: int,
                     dedup: bool = True) -> tuple[list, dict]:
    """整文件均匀抽 `want` 条（蓄水池）；`want=None` 表示全收（内存需放得下）。"""
    rng = random.Random(seed)
    pool: list = []
    seen: set = set()
    stats = {"lines": 0, "dup": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                t = json.loads(line).get(content_key)
            except json.JSONDecodeError:
                continue
            if not t or not t.strip():
                continue
            stats["lines"] += 1
            stats["total_chars"] = stats.get("total_chars", 0) + len(t)
            if dedup:
                h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
                if h in seen:
                    stats["dup"] += 1
                    continue
                seen.add(h)
            if want is None:
                pool.append(t)
            elif len(pool) < want:
                pool.append(t)
            else:
                j = rng.randrange(stats["lines"])
                if j < want:
                    pool[j] = t
    stats["kept"] = len(pool)
    stats["chars"] = sum(len(t) for t in pool)
    return pool, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="多源语料配比器（均匀抽样 + 权重配额 + manifest）")
    ap.add_argument("--source", action="append", required=True,
                    help="`path[:weight]`，可重复。全部省略权重时按文件大小等比例混合")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-chars", type=float, default=0,
                    help="目标字符总量（0=各来源全量合并，内存与体积都会很大）")
    ap.add_argument("--content-key", default="text")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-dedup", action="store_true", help="关闭每来源精确去重（默认开启）")
    ap.add_argument("--manifest", default=None, help="manifest 路径（默认 <out>.manifest.json）")
    args = ap.parse_args()

    srcs = []
    for spec in args.source:
        path, w = parse_source(spec)
        if not os.path.exists(path):
            raise SystemExit(f"来源不存在: {path}")
        srcs.append({"path": path, "weight": w})
    if any(s["weight"] is None for s in srcs):
        if any(s["weight"] is not None for s in srcs):
            raise SystemExit("权重必须全部给出或全部省略")
        sizes = [os.path.getsize(s["path"]) for s in srcs]
        tot = sum(sizes)
        for s, sz in zip(srcs, sizes):
            s["weight"] = sz / max(1, tot)
    total_w = sum(s["weight"] for s in srcs)

    print(f"[mix] 目标 {_fmt(args.target_chars) if args.target_chars else '全量'} 字符，"
          f"{len(srcs)} 个来源")
    manifest = {"target_chars": args.target_chars, "seed": args.seed, "sources": []}
    picked = []
    for i, s in enumerate(srcs):
        share = s["weight"] / max(1e-9, total_w)
        rough = rough_avg_chars(s["path"], args.content_key)
        char_quota = args.target_chars * share if args.target_chars else 0
        # 探针只用来"超采定容"（1.5 倍余量）；真实配额由扫全文后的精确均值决定
        want = max(1, int(char_quota * 1.5 / max(1.0, rough))) if char_quota else None
        texts, st = reservoir_sample(s["path"], args.content_key, want,
                                     seed=args.seed + i, dedup=not args.no_dedup)
        avg_exact = st.get("total_chars", 0) / max(1, st["lines"])
        need = int(char_quota / max(1.0, avg_exact)) if char_quota else 0
        texts, dropped = trim_to_lines(texts, need, args.seed + i)
        secs = sum(len(t) for t in texts)
        print(f"[mix] w={share:>5.1%}  {Path(s['path']).name}  精确均值 {avg_exact:.0f} 字符/条"
              + (f"（探针 {rough:.0f}，字符目标 {char_quota/1e6:.1f}M）" if char_quota else "")
              + f" → 抽 {len(texts):,} 条 / {secs/1e6:.1f}M 字符（裁掉 {dropped:,}）")
        if char_quota and secs < char_quota * 0.95:
            print(f"        ⚠️  实际 {secs/1e6:.1f}M 字符 < 目标的 95%：该来源总量不足，"
                  f"或把 --target-chars 调小")
        picked.append(texts)
        entry = {"path": s["path"], "weight": round(share, 4),
                 "avg_chars": round(avg_exact, 1), "avg_chars_probe": round(rough, 1),
                 "line_quota": want, "lines_scanned": st["lines"], "dup_dropped": st["dup"],
                 "trimmed_to_chars": dropped, "picked": len(texts), "chars": secs}
        manifest["sources"].append(entry)
        print(f"        扫过 {st['lines']:,} 行（去重丢 {st['dup']:,}）→ "
              f"抽中 {len(texts):,} 条 / {st['chars']/1e6:.1f}M 字符")

    rows = [t for group in picked for t in group]
    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total_chars = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            total_chars += len(t)
    manifest["total"] = {"rows": len(rows), "chars": total_chars,
                         "bytes": os.path.getsize(args.out)}
    for e in manifest["sources"]:
        e["actual_share"] = round(e["chars"] / max(1, total_chars), 4)
    mpath = args.manifest or (args.out + ".manifest.json")
    Path(mpath).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[mix] 写出 {args.out}: {len(rows):,} 行 / {total_chars/1e6:.1f}M 字符 / "
          f"{_fmt(os.path.getsize(args.out))}")
    print("[mix] 实际占比: " + ", ".join(
        f"{Path(e['path']).name} {e['actual_share']:.1%}" for e in manifest["sources"]))
    print(f"[mix] manifest → {mpath}")
    print(f"[mix] 下一步: python scripts/data/build_pretrain_bin.py build --corpus-jsonl {args.out} "
          f"--tokenizer-dir models/tokenizer_v3 --out-bin <out.bin>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
