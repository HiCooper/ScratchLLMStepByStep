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
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from minigpt.config import estimate_params_from_config  # noqa: E402
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
    """参数量展示（调用 config.py 的唯一实现，含 SwiGLU/8-3·d/tie/head-bias 开关）。

    旧实现自带一份公式且忽略 SwiGLU —— 对 SwiGLU run 会多报 50% 前馈参数。
    """
    try:
        return f"{estimate_params_from_config(cfg) / 1e6:.1f}M"
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
                                 "best_step": m.get("best_step"),
                                 # 评估口径：不同切分/不同 bin 的 eval_loss 不可直接比较
                                 "eval_split": (m.get("data_split") or {}).get("split"),
                                 "eval_rows": (m.get("data_split") or {}).get("eval_rows"),
                                 "eval_blocks": (m.get("data_split") or {}).get("n_blocks"),
                                 "params": estimate_params(cfg),
                                 "final": os.path.exists(os.path.join(os.path.dirname(path), "final.pt")),
                                 "best": os.path.exists(os.path.join(os.path.dirname(path), "best.pt"))})
    for path in sorted(glob.glob(os.path.join(cp, "sft_*", "metrics.json"))):
        m = load(path) or {}
        data["sft"].append({"run": os.path.basename(os.path.dirname(path)),
                            "step": m.get("step"), "train_loss": m.get("train_loss"),
                            "eval_loss": m.get("eval_loss"),
                            "best_eval_loss": m.get("best_eval_loss"),
                            "best_step": m.get("best_step"),
                            "final": os.path.exists(os.path.join(os.path.dirname(path), "final.pt")),
                            "best": os.path.exists(os.path.join(os.path.dirname(path), "best.pt"))})
    for path in sorted(glob.glob(os.path.join(cp, "eval_*_cot_*.json"))):
        js = load(path)
        if js:
            data["thinking"][os.path.basename(path)[:-5]] = js.get("summary", js)
    for path in sorted(glob.glob(os.path.join(cp, "ppl_*.json"))):
        js = load(path) or {}
        data["ppl"][os.path.basename(path)[4:-5]] = {k: js.get(k) for k in
                                                     ("eval_loss", "perplexity", "rows", "tokens",
                                                      "checkpoint", "bin", "max_rows")}
    for path in sorted(glob.glob(os.path.join(cp, "samples_*.txt")) + glob.glob(os.path.join(cp, "*", "sample.txt"))):
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


def _best(r: dict) -> str:
    """best_eval_loss(@step)：老产物没有 best_step 字段时只显示 loss。"""
    v = r.get("best_eval_loss")
    if v is None:
        return "—"
    st = r.get("best_step")
    return f"{float(v):.4f}" if st is None else f"{float(v):.4f}@{st}"


def _split_label(r) -> str:
    """把 metrics.json 里记录的评估口径压缩成短标签（区分"训练集口径"与"留出集口径"）。"""
    name = r.get("eval_split")
    if not name:
        return "—(旧产物)"
    if name == "blocked-document-aligned":
        return f"留出集/{r.get('eval_blocks') or '-'}块"
    return str(name)


def _num(v, nd: int = 4) -> str:
    """统一的数值格式化：None/非数值安全降级为 —。"""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def render(d: dict) -> str:
    L = [f"# MiniGPT 训练报告", "", f"生成时间：{d['time']}", ""]
    L += ["## 1. 预训练 run", "",
          "> 注：各 run 的 eval_loss 来自各自的验证切分（语料/比例/块数可能不同），**只有「评估口径」一致的 run 才可直接比较**；"
          "严格的同口径对比见 §2（同一 bin、同一 `--split val`）。",
          "`best.pt` 是 eval_loss 历史最优时的权重（小模型后期易过拟合，做下游/评测通常优于 `final.pt`）。", "",
          "| run | step | params | train_loss | eval_loss | perplexity | 评估口径 | best_eval_loss@step | final.pt | best.pt |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in d["pretrain"]:
        best = _best(r)
        L.append(f"| {r['run']} | {r['step']} | {r['params']} | "
                 f"{_num(r.get('train_loss'))} | **{_num(r.get('eval_loss'))}** | {_num(r.get('perplexity'), 2)} | "
                 f"{_split_label(r)} | "
                 f"{best} | {'✅' if r['final'] else '—'} | {'✅' if r.get('best') else '—'} |")
    if not d["pretrain"]:
        L.append("| （暂无） | | | | | | | | |")

    L += ["", "## 2. 同切分 perplexity 对比（同一 bin / 同一 --max-rows）", ""]
    if d["ppl"]:
        # 按评估语料分组：只有同 bin + 同窗口数的数字才可直接比较，不同语料混在一张表里会误导
        groups = {}
        for k, v in d["ppl"].items():
            groups.setdefault((v.get("bin") or "(未记录)", v.get("max_rows")), []).append((k, v))
        for (bin_path, max_rows), items in sorted(groups.items()):
            items.sort(key=lambda kv: (kv[1].get("eval_loss") is None, kv[1].get("eval_loss")))
            base = items[0][1].get("eval_loss") if items else None
            L += [f"**语料 `{bin_path}`（max-rows={max_rows}）**", "",
                  "| run | eval_loss | perplexity | 相对最优降低 | 评估窗口 |", "|---|---|---|---|---|"]
            for k, v in items:
                el = v.get("eval_loss")
                delta = "—" if (el is None or base in (None, 0)) else f"{(1 - el / base) * 100:+.1f}%"
                L.append(f"| {k} | {_num(el)} | {_num(v.get('perplexity'), 2)} | {delta} | {v.get('rows')} |")
            L.append("")
    else:
        L.append("（尚未产出：等待 run_downstream.sh 的 4c 阶段）")

    L += ["", "## 3. 下游 SFT / CoT run", "",
          "| run | step | train_loss | eval_loss | best_eval_loss@step | final.pt | best.pt |",
          "|---|---|---|---|---|---|---|"]
    for r in d["sft"]:
        best = _best(r)
        L.append(f"| {r['run']} | {r['step']} | {_num(r.get('train_loss'))} | {_num(r.get('eval_loss'))} | "
                 f"{best} | {'✅' if r['final'] else '—'} | {'✅' if r.get('best') else '—'} |")
    if not d["sft"]:
        L.append("| （暂无） | | | | | | |")

    L += ["", "## 4. 思考模式准确率（留出集，贪心 + repetition_penalty=1.0）", ""]
    if d["thinking"]:
        groups = {}
        for name, js in d["thinking"].items():
            m = re.match(r"eval_(?P<tag>.+)_cot_(?P<set>easy|hard)$", name)
            tag = m.group("tag") if m else name
            key = m.group("set") if m else "all"
            groups.setdefault(key, []).append((tag, js))
        for set_name, rows in sorted(groups.items()):
            L += [f"### {set_name}", "", "| run(tag) | n | plain | single | two-phase |", "|---|---|---|---|---|"]
            for tag, js in sorted(rows):
                acc = js.get("accuracy", {}) if isinstance(js, dict) else {}
                L.append(f"| {tag} | {js.get('n')} | {acc.get('plain')} | {acc.get('single')} | "
                         f"{acc.get('two-phase')} |")
            L.append("")
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
