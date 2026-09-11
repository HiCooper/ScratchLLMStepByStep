"""生产推理 CLI：加载 checkpoint + tokenizer，按采样参数生成文本。

用法：
    python scripts/generate.py \
        --checkpoint models/checkpoints/minigpt_pretrain/checkpoint-2000.pth \
        --tokenizer-dir models/tokenizer_v3 \
        --prompt "什么是AI？" --chat --max-new-tokens 128 \
        --do-sample --temperature 0.7 --top-k 50 --top-p 0.9 --repeat-penalty 1.1
    # 或 --prompt-file prompts.txt（每行一条）；或 --interactive
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from minigpt.model.generation import generate_with_thinking  # noqa: E402
from minigpt.model.checkpoint import build_model_from_checkpoint  # noqa: E402


def load_model(checkpoint_path, tokenizer, n_heads=None):
    """加载 checkpoint：优先用其自带 config，缺失则按权重形状反推（兼容老产物）。"""
    model, ckpt, kws = build_model_from_checkpoint(checkpoint_path, tokenizer=tokenizer,
                                                 n_heads=n_heads)
    print(f"[ckpt] arch={kws.get('emb_dim')}/{kws.get('n_layers')}/{kws.get('n_heads')} "
          f"ctx={kws.get('context_length')} vocab={kws.get('vocab_size')}")
    return model, ckpt.get("step"), ckpt.get("eval_loss")


def build_prompt(tokenizer, text, chat):
    if not chat:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_v3")
    ap.add_argument("--n-heads", type=int, default=None, help="覆盖 config-less checkpoint 反推出的注意力头数（n_heads 无法从权重形状反推；有 config 时以 config 为准）")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--chat", action="store_true", help="用 chat 模板包装用户输入")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repeat-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", action="store_true",
                    help="思考模式：两阶段『先逐步分析，再给最终答案』")
    ap.add_argument("--thinking-strategy", choices=["single", "two-phase"], default="single",
                    help="single=一次生成思考+答案（推荐）；two-phase=先思考到标记再作答")
    ap.add_argument("--thinking-max-tokens", type=int, default=120,
                    help="思考段最大 token 数（思考预算）")
    ap.add_argument("--hide-thinking", action="store_true",
                    help="只显示最终答案，隐藏思考过程（思考仍在内部生成）")
    ap.add_argument("--keep-special-tokens", action="store_true",
                    help="保留 <|im_end|> 等特殊 token（默认去除，仅用于调试）")
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed) if args.seed else None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model, ckpt_step, ckpt_eval = load_model(args.checkpoint, tokenizer,
                                         n_heads=getattr(args, "n_heads", None))
    model.to(device)
    print(f"[generate] loaded {args.checkpoint} step={ckpt_step} eval_loss={ckpt_eval} "
          f"vocab={len(tokenizer)} device={device}")

    prompts = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts.extend(line.rstrip("\n") for line in f if line.strip())
    if not prompts and not args.interactive:
        ap.error("请提供 --prompt / --prompt-file / --interactive")

    lines = []
    def run_one(text):
        t0 = time.time()
        if args.thinking:   # 思考模式：两阶段解码（先思考后作答）
            res = generate_with_thinking(
                model, tokenizer, text, chat=args.chat,
                strategy=args.thinking_strategy,
                max_new_tokens=args.max_new_tokens,
                thinking_max_tokens=args.thinking_max_tokens,
                hide_thinking=args.hide_thinking,
                do_sample=args.do_sample, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repeat_penalty,
                use_kv_cache=False, device=device)
            print(f"[generate] 耗时 {time.time()-t0:.1f}s (thinking)")
            return res["display"]
        prompt = build_prompt(tokenizer, text, args.chat)
        ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        out = model.generate(ids, args.max_new_tokens, tokenizer.eos_token_id,
                             do_sample=args.do_sample, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.repeat_penalty,
                             use_kv_cache=False)
        gen = out[0][ids.shape[1]:]
        # 默认去掉 <|im_end|> 等特殊 token（模型用它在 eos 处结束回合，是预期行为）；
        # 调试时可用 --keep-special-tokens 查看原始序列
        resp = tokenizer.decode(gen.tolist(),
                                skip_special_tokens=not args.keep_special_tokens).strip()
        print(f"[generate] 耗时 {time.time()-t0:.1f}s")
        return resp

    def flush(text, resp):
        lines.append(f"用户: {text}\n模型: {resp}\n")

    for text in prompts:
        resp = run_one(text)
        flush(text, resp)
        print(f"用户: {text}\n模型: {resp}\n")

    if args.interactive:
        try:
            while True:
                text = input("用户> ").strip()
                if not text:
                    continue
                resp = run_one(text)
                flush(text, resp)
                print(f"模型: {resp}\n")
        except (EOFError, KeyboardInterrupt):
            print("\n[generate] bye")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[generate] saved -> {args.output_file}")


if __name__ == "__main__":
    main()
