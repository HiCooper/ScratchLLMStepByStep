#!/usr/bin/env python3
"""汇总训练/评测产物，生成 Markdown 对比报告（供 agent 与用户查看）。

自动收集：
  - 预训练 run：models/checkpoints/pretrain_*/metrics.json（eval_loss / perplexity）
  - SFT/CoT run：models/checkpoints/sft_*/metrics.json
  - 思考模式评测：models/checkpoints/eval_v2_cot_{easy,hard}.json（plain/single/two-phase 准确率）
  - 同切分 ppl 对比：models/checkpoints/ppl_*.json
  - 问答样例：models/checkpoints/samples_v2_*.txt、各 run 的 sample.txt

用法：
  python3 scripts/report_training.py [--out models/checkpoints/TRAINING_REPORT.md] [--json /tmp/report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def read_text(path, limit=1200):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()[:limit]
    except Exception:  # noqa: BLE001
        return ""


def _pick(cfg: dict, name: str, default=None):
    """同时兼容 flat(emb_dim) 与 nested(model_emb_dim / 嵌套 dict) 两种 config 结构。"""
    if not isinstance(cfg, dict):
        return default
    if "model" in cfg and isinstance(cfg["model"], dict) and f"model_{name}" in cfg["model"]:
        return cfg["model"][f"model_{name}"]
    for key in (name, f"model_{name}"):
        if key in cfg:
            return cfg[key]
    return default


def estimate_params(cfg: dict) -> str:
    try:
        emb = int(_pick(cfg, "emb_dim", 512))
        layers = int(_pick(cfg, "n_layers", 10))
        vocab = int(_pick(cfg, "vocab_size", 0) or 32000)
        tied = bool(_pick(cfg, "tie_word_embeddings", True))
        per_layer = 12 * emb * emb          # attn(4) + FFN(8, 4x 扩展)
        total = vocab * emb * (1 if tied else 2) + layers * per_layer
        return f"{total/1e6:.1f}M"
    except Exception:  # noqa: BLE001
        return "?"


def collect(cp: str | None = None):
    """扫描 checkpoints 目录并汇总（cp 可指定，便于测试）。"""
    cp = cp or os.path.join(ROOT, "models", "checkpoints")
    data = {"time": datetime.now().strftime("%F %T"), "pretrain": [], "sft": [],
            "thinking": {}, "ppl": {}, "samples": {}, "success_rate": {}}
    for path in sorted(glob.glob(os.path.join(cp, "pretrain_*", "metrics.json"))):
        m = load(path) or {}
        cfg = load(os.path.join(os.path.dirname(path), "config.json")) or {}
        run = os.path.basename(os.path.dirname(path))
        data["pretrain"].append({"run": run, "step": m.get("step"), "epoch": m.get("epoch"),
                                 "train_loss": m.get("train_loss"), "eval_loss": m.get("eval_loss"),
                                 "perplexity": m.get("perplexity"),
                                 "best_eval_loss": m.get("best_eval_loss"),
                                 "params": estimate_params(cfg),
                                 "final": os.path.exists(os.path.join(os.path.dirname(path), "final.pt"))})
    for path in sorted(glob.glob(os.path.join(cp, "sft_*", "metrics.json"))):
        m = load(path) or {}
        data["sft"].append({"run": os.path.basename(os.path.dirname(path)),
                            "step": m.get("step"), "train_loss": m.get("train_loss"),
                            "eval_loss": m.get("eval_loss"),
                            "final": os.path.exists(os.path.join(os.path.dirname(path), "final.pt"))})
    for name in ("eval_v2_cot_easy", "eval_v2_cot_hard"):
        js = load(os.path.join(cp, name + ".json"))
        if js:
            data["thinking"][name] = js.get("summary", js)
    for path in sorted(glob.glob(os.path.join(cp, "ppl_*.json"))):
        js = load(path) or {}
        data["ppl"][os.path.basename(path)[4:-5]] = {k: js.get(k) for k in
                                                     ("eval_loss", "perplexity", "rows", "tokens", "checkpoint")}
    for path in sorted(glob.glob(os.path.join(cp, "samples_v2_*.txt")) + glob.glob(os.path.join(cp, "*", "sample.txt"))):
        data["samples"][os.path.relpath(path, cp)] = read_text(path)
    # 评测中的正确率明细（若存在 details 且不含 ok 字段，则统计 success_rate）
    for name, js in list(data["thinking"].items()):
        det = js.get("details") if isinstance(js, dict) else None
        if det:
            for strat in ("plain", "single", "two-phase"):
                vals = [d.get(f"{strat}_ok") for d in det if f"{strat}_ok" in d]
                if vals:
                    data["success_rate"][f"{name}:{strat}"] = round(sum(bool(v) for v in vals) / len(vals), 4)
    return data


def render(d: dict) -> str:
    L = [f"# MiniGPT 训练报告", "", f"生成时间：{d['time']}", ""]
    L += ["## 1. 预训练 run", "",
          "> 注：各 run 的 eval_loss 来自各自的验证切分（语料/比例可能不同），趋势可比；"
          "严格的同口径对比见 §2（同一 bin、同一 `--max-rows`）。", "" "| run | step | params | train_loss | eval_loss | perplexity | final.pt |",
          "|---|---|---|---|---|---|---|"]
    for r in d["pretrain"]:
        L.append(f"| {r['run']} | {r['step']} | {r['params']} | "
                 f"{r['train_loss']:.4f} | **{r['eval_loss']:.4f}** | {r['perplexity']:.2f} | "
                 f"{'✅' if r['final'] else '—'} |")
    if not d["pretrain"]:
        L.append("| （暂无） | | | | | | |")

    L += ["", "## 2. 同切分 perplexity 对比（同一 bin / 同一 --max-rows）", ""]
    if d["ppl"]:
        L += ["| run | eval_loss | perplexity | 评估窗口 |", "|---|---|---|---|"]
        for k, v in d["ppl"].items():
            L.append(f"| {k} | {v.get('eval_loss')} | {v.get('perplexity')} | {v.get('rows')} |")
    else:
        L.append("（尚未产出：等待 run_downstream.sh 的 4c 阶段）")

    L += ["", "## 3. 下游 SFT / CoT run", "", "| run | step | train_loss | eval_loss | final.pt |",
          "|---|---|---|---|---|"]
    for r in d["sft"]:
        L.append(f"| {r['run']} | {r['step']} | {r['train_loss']:.4f} | {r['eval_loss']:.4f} | "
                 f"{'✅' if r['final'] else '—'} |")
    if not d["sft"]:
        L.append("| （暂无） | | | | |")

    L += ["", "## 4. 思考模式准确率（留出集，贪心 + repetition_penalty=1.0）", ""]
    if d["thinking"]:
        L += ["| 评测集 | n | plain | single | two-phase |", "|---|---|---|---|---|"]
        for name, js in d["thinking"].items():
            acc = js.get("accuracy", {}) if isinstance(js, dict) else {}
            L.append(f"| {name} | {js.get('n')} | {acc.get('plain')} | {acc.get('single')} | "
                     f"{acc.get('two-phase')} |")
    else:
        L.append("（尚未产出：等待 4a/4b 阶段）")

    L += ["", "## 5. 问答 / 生成样例", ""]
    if d["samples"]:
        for name, text in d["samples"].items():
            L += [f"### {name}", "", "```", text, "```", ""]
    else:
        L.append("（尚未产出：等待 4d 阶段）")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "checkpoints", "TRAINING_REPORT.md"))
    ap.add_argument("--json", default=None, help="同时输出结构化 JSON（供 agent 解析）")
    ap.add_argument("--dir", default=None, help="checkpoints 目录（默认 models/checkpoints）")
    args = ap.parse_args()
    d = collect(args.dir)
    md = render(d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[report] saved -> {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"[report] json  -> {args.json}")


if __name__ == "__main__":
    main()
