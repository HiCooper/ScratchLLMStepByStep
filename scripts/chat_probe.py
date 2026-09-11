#!/usr/bin/env python3
"""对话能力探针：跑一批覆盖多场景的中文/英文 prompt，自动标记可疑输出，产出可读的 txt 报告。

为什么要单独一个脚本（而不是拿 `generate.py --prompt` 凑）：真实测评需要
  1) 场景分组（身份/知识/数学/代码/指令遵循/安全/幻觉/长文…），否则看不出"哪一类坏"；
  2) 同一 prompt 同时跑**贪心**与**采样**，区分"模型不知道"和"模型知道但说飘了"；
  3) **自动质检**：空回复、复读、语言漂移、prompt 回声、特殊 token 泄漏、拒答、过短；
  4) 固定 seed + 记录解码参数，报告可复现。

用法：
  python3 scripts/chat_probe.py --checkpoint models/checkpoints/sft_v2_chat/final.pt \
      --scenarios eval_sets/chat_scenarios_zh.jsonl \
      --output models/checkpoints/chat_probe_sft_v2_chat.txt

产物：
  - <output>（txt，人类可读：分组场景 + 贪心/采样回复 + 质检标记）
  - <output 同名 .json>（机器可读：原始回复 + flags，便于回归对比）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minigpt.model.checkpoint import build_model_from_checkpoint  # noqa: E402

# ---- 质检规则 ----
# 拒答必须"道歉/身份 + 拒绝动词"同时出现，否则会把正常道歉（"对不起，我迟到了"）误判。
# 这是被自己的单测抓出来的坑：早期版本用子串 "对不起，我" 直接匹配，误报率极高。
REFUSAL_RE = re.compile(
    r"(?:抱歉|对不起|很遗憾|不好意思)[^。\n！!？?]{0,6}"
    r"(?:我|作为)[^。\n]{0,14}?(?:无法|不能|不便|不会|没办法|没有能力|不具备)"
    r"|作为(?:一个|一名)?\s*(?:AI|人工智能|语言模型|助手)"
    r"|i\s+(?:cannot|can't|can not|am unable to|won't|will not)"
    r"|as an ai",
    re.IGNORECASE)
BOILERPLATE = ["作为一个AI语言模型", "作为一个人工智能", "我没有情感", "我没有自我意识", "没有个人身份"]
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def repeated_ngram(text: str, n: int = 6, times: int = 4) -> bool:
    """检测复读：长度 n 的子串连续/高频重复。"""
    t = re.sub(r"\s+", "", text)
    if len(t) < n * times:
        return False
    counts = Counter(t[i:i + n] for i in range(len(t) - n + 1))
    return counts.most_common(1)[0][1] >= times


def language_drift(prompt: str, resp: str) -> bool:
    """中文提问却用英文作答（英文单词占比过半且几乎无中文）。"""
    if not CJK_RE.search(prompt):
        return False
    cjk = len(CJK_RE.findall(resp))
    en = sum(len(w) for w in ASCII_WORD_RE.findall(resp))
    return cjk == 0 and en >= 12


def prompt_echo(prompt: str, resp: str) -> bool:
    """把用户问题原样吐回来（常见于未对齐好的续写模型）。"""
    p = re.sub(r"\s+", "", prompt)
    r = re.sub(r"\s+", "", resp)
    return len(p) >= 8 and p[: min(len(p), 24)] in r and len(r) <= len(p) * 2.5


def quality_flags(prompt: str, resp: str) -> list[str]:
    flags = []
    r = resp.strip()
    if not r:
        flags.append("空回复")
        return flags
    if len(r) < 6:
        flags.append("回复过短")
    if "<|im_end|>" in r or "<|im_start|>" in r:
        flags.append("特殊token泄漏")
    if repeated_ngram(r):
        flags.append("复读/循环")
    if language_drift(prompt, r):
        flags.append("语言漂移(中文问英文答)")
    if prompt_echo(prompt, r):
        flags.append("prompt回声")
    low = r.lower()
    if REFUSAL_RE.search(r) or REFUSAL_RE.search(low):
        flags.append("拒答模板")
    if any(b in r for b in BOILERPLATE):
        flags.append("AI套话")
    if r.count("\n") > 40:
        flags.append("超长/未收敛")
    return flags


def build_chat(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                        tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_one(model, tokenizer, prompt, device, *, do_sample, temperature,
                 top_k, top_p, repetition_penalty, max_new_tokens, seed):
    torch.manual_seed(seed)
    text = build_chat(tokenizer, prompt)
    ids = torch.tensor([tokenizer.encode(text)], device=device)
    t0 = time.time()
    out = model.generate(ids, max_new_tokens, tokenizer.eos_token_id,
                         do_sample=do_sample, temperature=temperature,
                         top_k=top_k, top_p=top_p,
                         repetition_penalty=repetition_penalty, use_kv_cache=False)
    gen = out[0][ids.shape[1]:].tolist()
    resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return resp, time.time() - t0, len(gen)


def load_scenarios(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_v3")
    ap.add_argument("--n-heads", type=int, default=None, help="覆盖 config-less checkpoint 反推出的注意力头数（n_heads 无法从权重形状反推；有 config 时以 config 为准）")
    ap.add_argument("--scenarios", default="eval_sets/chat_scenarios_zh.jsonl")
    ap.add_argument("--output", default=None, help="txt 报告路径；默认与 checkpoint 同目录")
    ap.add_argument("--modes", default="greedy,sampled",
                    help="greedy=温度0；sampled=温度0.8/top-k50/top-p0.9")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0=用每条场景自己的 max_new_tokens")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    ap.add_argument("--notes-file", default=None,
                    help="人工复核结论文档（Markdown），嵌入报告开头；便于把'原始记录+结论'放在同一个 txt 里")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model, ckpt, kws = build_model_from_checkpoint(args.checkpoint, tokenizer=tokenizer,
                                              device=device, n_heads=args.n_heads)
    model.eval()

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    out_path = args.output or os.path.join(os.path.dirname(args.checkpoint), "chat_probe.txt")
    print(f"[probe] checkpoint={args.checkpoint}")
    print(f"[probe] arch={kws.get('emb_dim')}/{kws.get('n_layers')} vocab={kws.get('vocab_size')} "
          f"device={device} scenarios={len(scenarios)} modes={modes}")

    results = []
    for i, row in enumerate(scenarios, 1):
        max_new = args.max_new_tokens or int(row.get("max_new_tokens", 128))
        rec = {"id": row["id"], "category": row.get("category", "未分类"),
               "prompt": row["prompt"], "max_new_tokens": max_new, "runs": {}}
        for mode in modes:
            greedy = mode == "greedy"
            resp, secs, ntok = generate_one(
                model, tokenizer, row["prompt"], device,
                do_sample=not greedy,
                temperature=0.0 if greedy else args.temperature,
                top_k=None if greedy else args.top_k,
                top_p=None if greedy else args.top_p,
                repetition_penalty=args.repeat_penalty,
                max_new_tokens=max_new, seed=args.seed + i)
            flags = quality_flags(row["prompt"], resp)
            rec["runs"][mode] = {"resp": resp, "seconds": round(secs, 2),
                                 "tokens": ntok, "flags": flags}
        results.append(rec)
        mark = " ".join(sorted({f for m in modes for f in rec["runs"][m]["flags"]}))
        print(f"[probe] {i}/{len(scenarios)} {rec['id']:<10} {rec['category']:<8} "
              f"{'⚠ ' + mark if mark else 'ok'}")

    # ---------- 渲染 txt 报告 ----------
    stats = Counter()
    per_cat = defaultdict(Counter)
    per_mode = defaultdict(Counter)
    for rec in results:
        for mode, r in rec["runs"].items():
            for f in r["flags"]:
                stats[f] += 1
                per_cat[rec["category"]][f] += 1
                per_mode[mode][f] += 1

    L = []
    L.append("=" * 78)
    L.append("MiniGPT 对话能力探针报告")
    L.append("=" * 78)
    L.append(f"生成时间   : {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"checkpoint : {args.checkpoint}")
    L.append(f"模型结构   : {kws.get('emb_dim')}/{kws.get('n_layers')}/{kws.get('n_heads')} "
             f"ctx={kws.get('context_length')} vocab={kws.get('vocab_size')}")
    L.append(f"场景文件   : {args.scenarios}（{len(results)} 条）")
    L.append(f"解码设置   : greedy=temperature 0；"
             f"sampled=temperature {args.temperature}/top-k {args.top_k}/top-p {args.top_p}；"
             f"repeat_penalty {args.repeat_penalty}；seed {args.seed}")
    if args.notes_file and os.path.exists(args.notes_file):
        with open(args.notes_file, encoding="utf-8") as f:
            notes = f.read().rstrip()
        L += ["", "~" * 78,
              f"~ 人工复核结论（来自 {args.notes_file}）",
              "~" * 78]
        L += notes.splitlines()
        L.append("~" * 78)
    L.append("")
    L.append("【自动质检汇总】")
    if stats:
        for f, n in stats.most_common():
            L.append(f"  ⚠ {f:<22} {n:>3} 次 / {len(results) * len(modes)} 条生成")
    else:
        L.append("  未检测到问题")
    L.append("")
    L.append("【按场景类别】")
    for cat, c in sorted(per_cat.items(), key=lambda kv: -sum(kv[1].values())):
        detail = "，".join(f"{f}×{n}" for f, n in c.most_common())
        L.append(f"  {cat:<12} {detail}")
    L.append("")
    L.append("【按解码模式】")
    for mode, c in per_mode.items():
        detail = "，".join(f"{f}×{n}" for f, n in c.most_common()) or "无问题"
        L.append(f"  {mode:<8} {detail}")
    L.append("")

    by_cat = defaultdict(list)
    for rec in results:
        by_cat[rec["category"]].append(rec)

    for cat, recs in by_cat.items():
        L.append("")
        L.append("#" * 78)
        L.append(f"## {cat}（{len(recs)} 条）")
        L.append("#" * 78)
        for rec in recs:
            L.append("")
            L.append(f"--- [{rec['id']}] " + "-" * max(0, 60 - len(rec['id'])))
            L.append("【提问】")
            for ln in rec["prompt"].splitlines():
                L.append("  " + ln)
            for mode, r in rec["runs"].items():
                tag = "贪心" if mode == "greedy" else "采样"
                fl = ("  ⚠ " + "，".join(r["flags"])) if r["flags"] else ""
                L.append(f"【回答·{tag}】{fl}  ({r['seconds']}s / {r['tokens']} tokens)")
                if not r["resp"]:
                    L.append("  <空>")
                for ln in r["resp"].splitlines():
                    L.append("  " + ln)
    L.append("")
    L.append("=" * 78)
    L.append("报告结束")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"time": datetime.now().strftime("%F %T"), "checkpoint": args.checkpoint,
                   "decoding": {"modes": modes, "temperature": args.temperature,
                                "top_k": args.top_k, "top_p": args.top_p,
                                "repeat_penalty": args.repeat_penalty, "seed": args.seed},
                   "summary": {"flags": dict(stats), "by_category": {k: dict(v) for k, v in per_cat.items()}},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[probe] txt  -> {out_path}")
    print(f"[probe] json -> {json_path}")
    print(f"[probe] 质检汇总: {dict(stats) if stats else '无问题'}")


if __name__ == "__main__":
    main()
