"""思考模式评测：在可验证的算术/应用题留出集上对比"关闭思考"与"开启思考"的准确率。

用法：
    python scripts/eval_thinking.py --checkpoint models/checkpoints/sft_cot_v1/final.pt \
        --tokenizer-dir models/tokenizer_v3 --eval-jsonl dataset/sft/cot_eval_hard_disjoint.jsonl \
        --n 60 [--temperature 0] [--thinking-max-tokens 120] [--output result.json]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from minigpt.cot_format import ANSWER_MARK
from minigpt.data.sft_dataset import build_user_content  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minigpt.model.generation import generate_with_thinking  # noqa: E402
from minigpt.model.checkpoint import build_model_from_checkpoint  # noqa: E402

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text, default=""):
    """优先取 ANSWER_MARK（`最终答案：`）之后的数字，否则取最后一个数字。"""
    if ANSWER_MARK in text:
        tail = text.split(ANSWER_MARK)[-1]
        m = NUM_RE.search(tail)
        if m:
            return m.group(0)
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else default


def norm(x):
    try:
        return f"{float(x):.1f}"
    except (TypeError, ValueError):
        return str(x)


def row_prompt(row: dict) -> str:
    """评测用的 user 内容：直接委托 `build_user_content`（训练的**唯一**格式实现）。

    以前这里只传 `instruction`，prompt 比训练时少一个换行（实测 18 vs 19 token，
    差异落在 `<|im_end|>` 之前）——评测量到的必须是与训练同分布下的能力。
    """
    return build_user_content(row.get("instruction"), row.get("input"))


def bucket_of(question: str) -> str:
    """把题目按**规模+题型**分桶，用于分档报告准确率。

    为什么必须分档：单一平均准确率会把"简单题全对、难题全错"平均成一个看不出问题的数字。
    47.9M 参数在多位数乘加上是**容量墙**（继续堆同分布数据无用），只有分档才看得出来。
    这是**报告用**的启发式（判分仍用数值精确匹配），规则与 `build_cot_sft.py` 的生成器对齐。
    """
    digits = [int(x) for x in re.findall(r"\d+", question)]
    scale = "big" if (max(digits) if digits else 0) >= 100 else "small"
    if any(k in question for k in ("苹果", "平均", "剩下", "原有", "多少元", "多少本")):
        return f"word_{scale}"
    if "×" in question or "乘" in question:
        return f"mul_{scale}"
    if "÷" in question or "除" in question:
        return f"div_{scale}"
    ops = sum(question.count(op) for op in "+-")
    if ops >= 2:
        return f"mixed_{scale}"
    if "-" in question or "减" in question:
        return f"sub_{scale}"
    if "+" in question or "加" in question:
        return f"add_{scale}"
    return "other"


def plain_generate(model, tokenizer, question, device, max_new_tokens, temperature, seed,
                   repetition_penalty=1.0):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                           tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    torch.manual_seed(seed)
    kw = dict(repetition_penalty=repetition_penalty)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_k=50, top_p=0.92)
    out = model.generate(ids, max_new_tokens, tokenizer.eos_token_id, use_kv_cache=False, **kw)
    return tokenizer.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_v3")
    ap.add_argument("--n-heads", type=int, default=None, help="覆盖 config-less checkpoint 反推出的注意力头数（n_heads 无法从权重形状反推；有 config 时以 config 为准）")
    ap.add_argument("--eval-jsonl", default="dataset/sft/cot_eval_hard_disjoint.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--thinking-max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0, help="0=贪心（推荐用于准确率对比）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="贪心做算术评测建议 1.0；>1 会惩罚重复数字而破坏结果")
    ap.add_argument("--strategies", default="plain,single,two-phase",
                    help="要评测的思考策略（逗号分隔）")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    # 优先用 checkpoint 自带 config，缺失（如老版本 best.pt / 周期 checkpoint）则按权重形状反推
    model, ck, kws = build_model_from_checkpoint(args.checkpoint, tokenizer=tokenizer,
                                            device=device, n_heads=args.n_heads)
    print(f"[ckpt] {args.checkpoint} arch={kws.get('emb_dim')}/{kws.get('n_layers')}/"
          f"{kws.get('n_heads')} ctx={kws.get('context_length')}")

    rows = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[: args.n]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    stats = {s: 0 for s in strategies}
    buckets = {}                      # 桶 -> {"n": 题数, "<策略>": 正确数}
    details = []
    t0 = time.time()
    for i, row in enumerate(rows):
        q = row["instruction"]
        q_prompt = row_prompt(row)
        gold = extract_answer(row["output"])
        bkey = bucket_of(q)
        bslot = buckets.setdefault(bkey, {"n": 0, **{s: 0 for s in strategies}})
        bslot["n"] += 1
        plain = plain_generate(model, tokenizer, q_prompt, device, args.max_new_tokens,
                               args.temperature, args.seed + i, args.repetition_penalty)
        p_pred = extract_answer(plain)
        row_detail = {"q": q, "gold": gold, "plain": plain[:120], "plain_pred": p_pred,
                      "plain_ok": norm(p_pred) == norm(gold)}
        if "plain" in strategies:
            stats["plain"] += row_detail["plain_ok"]
            bslot["plain"] += row_detail["plain_ok"]
        for strat in [s for s in strategies if s != "plain"]:
            res = generate_with_thinking(
                model, tokenizer, q_prompt, strategy=strat,
                max_new_tokens=args.max_new_tokens,
                thinking_max_tokens=args.thinking_max_tokens, hide_thinking=True,
                do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-6),
                top_k=50, top_p=0.92, repetition_penalty=args.repetition_penalty,
                device=device)
            pred = extract_answer(res["answer"])
            ok = norm(pred) == norm(gold)
            stats[strat] += ok
            bslot[strat] += ok
            row_detail[f"{strat}_pred"] = pred
            row_detail[f"{strat}_ok"] = ok
            row_detail[f"{strat}_thinking"] = res["thinking"][:120]
            row_detail[f"{strat}_answer"] = res["answer"][:120]
        details.append(row_detail)
        if (i + 1) % 10 == 0:
            print(f"[eval] {i+1}/{len(rows)} | " + " | ".join(f"{k} {v}" for k, v in stats.items()))

    n = len(rows)
    result = {
        "n": n,
        "accuracy": {k: round(v / n, 4) for k, v in stats.items()},
        "accuracy_plain": round(stats.get("plain", 0) / n, 4),
        "accuracy_thinking": round(stats.get("single", stats.get("two-phase", 0)) / n, 4),
        "temperature": args.temperature,
        "seconds": round(time.time() - t0, 1),
        "checkpoint": args.checkpoint,
        # 分档准确率：单一平均会掩盖"简单题全对、难题全错"（容量墙）
        "by_difficulty": {
            b: {"n": v["n"], **{k: round(v[k] / max(1, v["n"]), 3)
                                for k in strategies if k in v}}
            for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]["n"])
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n--- 分档准确率（按题量排序）---")
    print(f"{'档位':<14}{'题数':>6}  " + "  ".join(f"{k:>10}" for k in strategies))
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
        accs = "  ".join(f"{v[k]/max(1, v['n']):>10.1%}" for k in strategies if k in v)
        flag = ""
        if v["n"] >= 5 and all(v[k] / max(1, v["n"]) < 0.5 for k in strategies if k in v):
            flag = "   ← 全档低于 50%，优先怀疑容量/数据分布，而不是再加步数"
        print(f"{b:<14}{v['n']:>6}  {accs}{flag}")
    print("\n--- 样例（前 3 条）---")
    for d in details[:3]:
        extra = " ".join(f"{k}={d.get(k)}" for k in d if k.endswith("_pred"))
        print(f"题目: {d['q']}\n  金标: {d['gold']} | {extra}\n"
              f"  思考: {d.get('single_thinking') or d.get('two-phase_thinking','')}\n"
              f"  回答: {d.get('single_answer') or d.get('two-phase_answer','')}\n")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": result, "details": details}, f, ensure_ascii=False, indent=2)
        print(f"[eval] saved -> {args.output}")


if __name__ == "__main__":
    main()
