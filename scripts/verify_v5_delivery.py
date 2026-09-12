#!/usr/bin/env python3
"""v5 全链路交付自检：一条命令回答"这次交付到底齐了没有、数字合不合理"。

为什么要有它：整条链路的产物散在 5 个阶段、十来个文件里，靠人挨个 `ls` 很容易漏；
而"基座跑完但下游没接上"这种失败**不会报错**（驱动是 `set -uo pipefail` 但不 abort）。
这个脚本把交付标准写成可执行断言，缺失/异常直接 exit 1。

用法：
  python3 scripts/verify_v5_delivery.py                     # 默认 models/checkpoints
  python3 scripts/verify_v5_delivery.py --cp <dir> --dataset <dir>
  python3 scripts/verify_v5_delivery.py --allow-partial     # 训练途中看进度：缺件标 ⏭ 且不算失败
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PASS, FAIL, SKIP = "✅", "❌", "⏭"


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _exists(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


class Checker:
    def __init__(self, cp: str, dataset: str, partial: bool = False):
        self.cp = cp
        self.dataset = dataset
        self.partial = partial
        self.rows: list[tuple[str, str, str]] = []

    def add(self, ok: bool, name: str, detail: str = "") -> None:
        # --allow-partial：训练途中查看用，未产出的东西标 ⏭ 而不是 ❌（退出码也不失败）
        if not ok and self.partial:
            self.rows.append((SKIP, name, detail))
            return
        self.rows.append((PASS if ok else FAIL, name, detail))

    def note(self, name: str, detail: str = "") -> None:
        self.rows.append((SKIP, name, detail))

    # ---------- 各阶段 ----------
    def check_pretrain(self) -> dict:
        run = os.path.join(self.cp, "pretrain_v5")
        ck = {"best": os.path.join(run, "best.pt"), "final": os.path.join(run, "final.pt")}
        self.add(os.path.exists(ck["final"]), "基座 final.pt 存在", ck["final"])
        self.add(os.path.exists(ck["best"]), "基座 best.pt 存在", ck["best"])
        m = _load(os.path.join(run, "metrics.json"))
        self.add(m is not None, "基座 metrics.json 存在", os.path.join(run, "metrics.json"))
        if not m:
            return {}
        cfg = _load(os.path.join(run, "config.json")) or {}
        model, train = cfg.get("model", {}), cfg.get("train", {})
        arch = {k: model.get(f"model_{k}") for k in ("emb_dim", "n_layers", "n_heads", "context_length")}
        self.add(arch == {"emb_dim": 512, "n_layers": 10, "n_heads": 8, "context_length": 512},
                 "基座架构 = 512/10/8 ctx512", str(arch))
        self.add(str(train.get("mixed_precision_dtype")) == "float16", "基座 fp16",
                 str(train.get("mixed_precision_dtype")))
        target = int(train.get("max_steps") or 0)
        step = int(m.get("step") or 0)
        self.add(target > 0 and step >= target * 0.99, "基座步数达到目标",
                 f"{step}/{target}")
        ev = m.get("eval_loss")
        self.add(isinstance(ev, (int, float)) and math.isfinite(ev) and ev < 5.0,
                 "基座 eval_loss 合理（<5.0）", str(ev))
        if isinstance(ev, (int, float)):
            self.add(abs(math.exp(min(ev, 80.0)) - float(m.get("perplexity") or 0)) < 1e-3,
                     "基座 perplexity = exp(eval_loss)", f"{m.get('perplexity')}")
        sample = os.path.join(run, "sample.txt")
        self.add(os.path.exists(sample) and os.path.getsize(sample) > 20,
                 "基座 sample.txt 非空（关键样例）",
                 f"{os.path.getsize(sample) if os.path.exists(sample) else 0}B")
        split = (m.get("data_split") or {}).get("split")
        self.add(bool(split), "基座记录评估口径", str(split))
        self.add((m.get("best_eval_loss") is not None) and (m.get("best_step") is not None),
                 "基座记录 best_eval_loss@step",
                 f"{m.get('best_eval_loss')}@{m.get('best_step')}")
        # .bin 对账：metrics 里记的 bin 信息必须与磁盘 meta 一致
        # （注意真实键名：meta 文件是 `*.meta.json`（不是 `*.bin.meta.json`），里面是
        #   `tokens`/`vocab_size`；metrics.bin_meta 来自 validate_bin_tokenizer，
        #   键是 `bin_tokens`/`bin_vocab`。fixture 与真实产物不一致会让自检假红。）
        meta_path = os.path.join(self.dataset, "bins", "pretrain_v5_full.meta.json")
        meta = _load(meta_path)
        bm = m.get("bin_meta") or {}
        if meta and bm.get("checked"):
            self.add(meta.get("tokens") == bm.get("bin_tokens")
                     and meta.get("vocab_size") == bm.get("bin_vocab"),
                     ".bin 与 metrics 记录的 token/vocab 对账",
                     f"disk={meta.get('tokens')}/{meta.get('vocab_size')} "
                     f"metrics={bm.get('bin_tokens')}/{bm.get('bin_vocab')}")
        else:
            self.note("跳过 .bin 对账", f"缺少 {meta_path} 或 bin_meta 未校验")
        return m

    def check_run(self, run: str, label: str, expect_step: int | None = None) -> None:
        d = os.path.join(self.cp, run)
        has = _exists(os.path.join(d, "best.pt")) or _exists(os.path.join(d, "final.pt"))
        self.add(has, f"{label} 有 best.pt/final.pt", d)
        m = _load(os.path.join(d, "metrics.json"))
        self.add(m is not None, f"{label} metrics.json 存在", d)
        if not m:
            return
        ev = m.get("eval_loss")
        self.add(isinstance(ev, (int, float)) and math.isfinite(ev), f"{label} eval_loss 有限", str(ev))
        sample = os.path.join(d, "sample.txt")
        self.add(os.path.exists(sample) and os.path.getsize(sample) > 20,
                 f"{label} sample.txt 非空（关键样例）",
                 f"{os.path.getsize(sample) if os.path.exists(sample) else 0}B")
        if expect_step:
            st = int(m.get("step") or 0)
            self.add(st >= expect_step * 0.9, f"{label} 步数接近预期", f"{st}/{expect_step}")

    def check_evals(self) -> None:
        ppl = _load(os.path.join(self.cp, "ppl_v5_general.json"))
        self.add(ppl is not None, "ppl_v5_general.json 存在")
        if ppl:
            # evaluate_pretrain.py 写的 `split` 是 split_train_eval_blocks 返回的 **dict**
            # （{"split": "blocked-document-aligned", "n_blocks": ...}），不是字符串
            sp = ppl.get("split")
            sp_name = sp.get("split") if isinstance(sp, dict) else sp
            self.add(sp_name in ("val", "blocked-document-aligned"),
                     "ppl 用 --split val 口径", f"{sp_name}（{sp.get('n_blocks') if isinstance(sp, dict) else '-'} 块）")
            self.add("pretrain_v5_full.bin" in str(ppl.get("bin")), "ppl 评的是 v5 bin", str(ppl.get("bin")))
            self.add(isinstance(ppl.get("perplexity"), (int, float))
                     and math.isfinite(ppl["perplexity"]), "ppl 有限", str(ppl.get("perplexity")))
        for name, label in (("eval_v5_cot_easy", "CoT easy 分档"),
                            ("eval_v5_cot_hard", "CoT hard 分档")):
            js = _load(os.path.join(self.cp, f"{name}.json"))
            self.add(js is not None, f"{label} json 存在")
            if not js:
                continue
            acc = (js.get("summary") or js).get("accuracy") or {}
            n = (js.get("summary") or js).get("n")
            self.add(bool(acc.get("plain") is not None), f"{label} 有 plain 准确率", str(acc.get("plain")))
            self.add(bool(acc.get("single") is not None), f"{label} 有 single 准确率", str(acc.get("single")))
            self.add(bool(n), f"{label} 有样本数 n", str(n))
            self.add(bool((js.get("summary") or js).get("by_difficulty")),
                     f"{label} 有分档准确率 by_difficulty",
                     "平均准确率看不出容量墙")
        sample = os.path.join(self.cp, "samples_v5_chat_probe.txt")
        self.add(os.path.exists(sample) and os.path.getsize(sample) > 500,
                 "chat_probe 对话样例非空", f"{sample} {os.path.getsize(sample) if os.path.exists(sample) else 0}B")

    def check_report(self) -> None:
        rep = os.path.join(self.cp, "TRAINING_REPORT_v5.md")
        self.add(os.path.exists(rep), "交付报告存在", rep)
        if not os.path.exists(rep):
            return
        txt = open(rep, encoding="utf-8").read()
        for need, why in (("## 1.5", "eval_loss 曲线/吞吐/ETA 章节"),
                          ("## 3.5", "SFT/CoT 阶段曲线章节"),
                          ("## 4.5", "CoT 分档准确率章节"),
                          ("## 6.", "历史基线对照章节"),
                          ("step:loss", "曲线数据点"),
                          ("best.pt", "各阶段产物路径")):
            self.add(need in txt, f"报告含{why}", need)
        summ = os.path.join(self.cp, "downstream_v5", "SUMMARY.md")
        self.add(os.path.exists(summ), "下游 SUMMARY.md 存在", summ)

    # ---------- 输出 ----------
    def run(self) -> int:
        self.check_pretrain()
        self.check_run("sft_v5_chat", "SFT", expect_step=10000)
        self.check_run("sft_v5_cot_easy", "CoT easy", expect_step=15000)
        self.check_run("sft_v5_cot_hard", "CoT hard", expect_step=20000)
        self.check_evals()
        self.check_report()
        n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
        n_skip = sum(1 for s, _, _ in self.rows if s == SKIP)
        width = max(len(n) for _, n, _ in self.rows)
        for s, n, d in self.rows:
            print(f"{s} {n.ljust(width)}  {d}")
        if n_fail:
            tail = f"，失败 {n_fail} 项"
        elif n_skip:
            tail = f"，未产出 {n_skip} 项（尚未完成）"
        else:
            tail = "，交付完整 ✅"
        print(f"\n通过 {len(self.rows) - n_fail - n_skip}/{len(self.rows)}" + tail)
        return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", default=os.path.join(ROOT, "models", "checkpoints"))
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--allow-partial", action="store_true",
                    help="训练途中查看：缺件标 ⏭，不算失败（退出码 0）")
    args = ap.parse_args()
    return Checker(args.cp, args.dataset, partial=args.allow_partial).run()


if __name__ == "__main__":
    raise SystemExit(main())
