"""生产推理 CLI：加载 checkpoint + tokenizer，按采样参数生成文本。

用法：
    python scripts/generate.py \
        --checkpoint models/checkpoints/minigpt_pretrain/checkpoint-2000.pth \
        --tokenizer-dir models/tokenizer_qwen2 \
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

from minigpt.model.transformer import GPTConfig, MiniGPT  # noqa: E402


def load_model(checkpoint_path, tokenizer):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"checkpoint 缺少 model_state 字段: {checkpoint_path}")
    cfg_dict = ckpt.get("config")
    if not cfg_dict:
        raise ValueError(f"checkpoint 缺少 config 字段（生产 checkpoint 由 pretrainer.py 产出）")
    keys = {k: v for k, v in cfg_dict.items()}
    keys["vocab_size"] = len(tokenizer)
    gpt = GPTConfig(**{k: keys[k] for k in
                       ("emb_dim", "n_layers", "n_heads", "context_length", "vocab_size",
                        "drop_rate", "qkv_bias", "flash_attn", "tie_word_embeddings",
                        "use_swiglu", "use_checkpoint") if k in keys})
    model = MiniGPT(gpt)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
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
    ap.add_argument("--tokenizer-dir", default="models/tokenizer_qwen2")
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
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed) if args.seed else None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model, ckpt_step, ckpt_eval = load_model(args.checkpoint, tokenizer)
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
        prompt = build_prompt(tokenizer, text, args.chat)
        ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        t0 = time.time()
        out = model.generate(ids, args.max_new_tokens, tokenizer.eos_token_id,
                             do_sample=args.do_sample, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.repeat_penalty,
                             use_kv_cache=False)
        gen = out[0][ids.shape[1]:]
        resp = tokenizer.decode(gen.tolist(), skip_special_tokens=False).strip()
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
