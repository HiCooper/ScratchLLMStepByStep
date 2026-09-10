"""带停止序列与"思考模式"（CoT 两阶段解码）的生成工具。

设计
- `generate_ids`：在固定采样策略（复用 model/transformer.py 的 top-k/top-p/温度/重复惩罚）下逐 token 解码，
  支持 `stop_strings`（按解码文本后缀判定停止）与 eos 提前结束；用于需要"边生成边判定停止"的场景。
- `generate_with_thinking`：把一次回答拆成两阶段：
    阶段1 思考：在 assistant 起始后强制前缀 `让我逐步分析：\n`，生成到 `最终答案：` 或思考预算耗尽；
    阶段2 作答：以"思考内容 + 最终答案："为前缀继续生成，得到面向用户的答案。
  这样"思考"是显式可训练、可截断、可隐藏的协议，而不是靠模型自觉。

协议约定（与 scripts/build_cot_sft.py 构造的数据一致）：
    <|im_start|>assistant
    让我逐步分析：
    1. ...
    2. ...
    最终答案：<答案><|im_end|>
"""
from __future__ import annotations

from typing import List, Optional

import torch

from minigpt.model.transformer import (apply_repetition_penalty, sample_next_token)

THINK_PREFIX = "让我逐步分析：\n"
ANSWER_MARK = "最终答案："


@torch.inference_mode()
def generate_ids(model, input_ids: torch.Tensor, max_new_tokens: int,
                 eos_token_id: Optional[int] = None,
                 stop_strings: Optional[List[str]] = None,
                 tokenizer=None, do_sample: bool = False, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 repetition_penalty: float = 1.0, use_kv_cache: bool = True,
                 context_length: Optional[int] = None) -> torch.Tensor:
    """逐 token 生成；命中 eos 或任一 stop_strings 即停止（停止串本身保留在序列里）。"""
    model.eval()
    context_length = context_length or getattr(model, "context_length", 512)
    eos_reached = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
    past_kvs = None
    for _ in range(max_new_tokens):
        if use_kv_cache:
            step_input = input_ids if past_kvs is None else input_ids[:, -1:]
        else:
            step_input = input_ids[:, -context_length:]
        out = model(step_input, use_kv_cache=use_kv_cache, past_kvs=past_kvs, return_dict=True)
        past_kvs = out["past_key_values"] if use_kv_cache else None
        logits = out["logits"][:, -1, :]
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        next_id = sample_next_token(logits, do_sample=do_sample, temperature=temperature,
                                    top_k=top_k, top_p=top_p)
        input_ids = torch.cat((input_ids, next_id), dim=1)
        if eos_token_id is not None:
            eos_reached |= (next_id.squeeze(-1) == eos_token_id)
        if eos_reached.all():
            break
        if stop_strings and tokenizer is not None:
            text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            if any(s and s in text for s in stop_strings):
                break
    return input_ids


def _new_text(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(full_ids[0][prompt_len:].tolist(), skip_special_tokens=True)


def generate_with_thinking(model, tokenizer, user_text: str, *,
                           chat: bool = True,
                           strategy: str = "single",
                           max_new_tokens: int = 160,
                           thinking_max_tokens: int = 120,
                           hide_thinking: bool = False,
                           do_sample: bool = False, temperature: float = 1.0,
                           top_k: Optional[int] = None, top_p: Optional[float] = None,
                           repetition_penalty: float = 1.0,
                           use_kv_cache: bool = False,
                           eos_token_id: Optional[int] = None,
                           device: str = "cuda:0"):
    """两阶段"先思考、后作答"。返回 dict(thinking=..., answer=..., display=..., full=...)。"""
    eos_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    if chat:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                               tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"用户：{user_text}\n助手："

    if strategy == "single":
        # 单通道 CoT：一次生成"思考 + 最终答案"，再按标记切分（小模型最稳的思考模式）
        ids = torch.tensor([tokenizer.encode(prompt + THINK_PREFIX)]).to(device)
        out = generate_ids(model, ids, thinking_max_tokens + max_new_tokens,
                           eos_token_id=eos_id, tokenizer=tokenizer,
                           do_sample=do_sample, temperature=temperature,
                           top_k=top_k, top_p=top_p,
                           repetition_penalty=repetition_penalty,
                           use_kv_cache=use_kv_cache)
        text = _new_text(tokenizer, out, ids.shape[1])
        head, tail = split_thinking(text)
        thinking, answer = (head, tail) if tail else ("", text.strip())
        display = answer if hide_thinking else f"【思考】\n{thinking}\n【回答】\n{answer}"
        return {"thinking": thinking, "answer": answer, "display": display,
                "full": f"{THINK_PREFIX}{text}", "marker_hit": bool(tail), "strategy": "single"}

    # ---------------- 阶段 1：思考 ----------------
    think_prompt_ids = torch.tensor([tokenizer.encode(prompt + THINK_PREFIX)]).to(device)
    phase1 = generate_ids(model, think_prompt_ids, thinking_max_tokens,
                          eos_token_id=eos_id, stop_strings=[ANSWER_MARK],
                          tokenizer=tokenizer, do_sample=do_sample, temperature=temperature,
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                          use_kv_cache=use_kv_cache)
    thinking_all = _new_text(tokenizer, phase1, think_prompt_ids.shape[1])
    marker_hit = ANSWER_MARK in thinking_all
    thinking = thinking_all.replace(ANSWER_MARK, "").strip()

    # 思考段没能收敛到「最终答案：」时，直接按普通方式作答（避免强行拼接导致的漂移）
    if not marker_hit:
        plain_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        plain = generate_ids(model, plain_ids, max_new_tokens, eos_token_id=eos_id,
                             tokenizer=tokenizer, do_sample=do_sample, temperature=temperature,
                             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                             use_kv_cache=use_kv_cache)
        plain_text = _new_text(tokenizer, plain, plain_ids.shape[1])
        head, tail = split_thinking(plain_text)
        thinking = head or thinking
        answer = tail
        display = answer if hide_thinking else f"【思考】\n{thinking}\n【回答】\n{answer}"
        return {"thinking": thinking, "answer": answer, "display": display,
                "full": f"{THINK_PREFIX}{thinking}\n{ANSWER_MARK}{answer}", "marker_hit": False,
                "strategy": "two-phase"}

    # ---------------- 阶段 2：作答 ----------------
    phase2_prompt = prompt + THINK_PREFIX + thinking_all
    phase2_ids = torch.tensor([tokenizer.encode(phase2_prompt)]).to(device)
    phase2 = generate_ids(model, phase2_ids, max_new_tokens, eos_token_id=eos_id,
                          tokenizer=tokenizer, do_sample=do_sample, temperature=temperature,
                          top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                          use_kv_cache=use_kv_cache)
    answer = _new_text(tokenizer, phase2, phase2_ids.shape[1]).strip()

    display = answer if hide_thinking else f"【思考】\n{thinking}\n【回答】\n{answer}"
    return {
        "thinking": thinking,
        "answer": answer,
        "display": display,
        "full": f"{THINK_PREFIX}{thinking}\n{ANSWER_MARK}{answer}",
        "marker_hit": True,
        "strategy": "two-phase",
    }


def split_thinking(text: str) -> tuple[str, str]:
    """从一段完整回答里切出 (思考, 答案)；没有标记时思考为空。"""
    if ANSWER_MARK in text:
        head, _, tail = text.partition(ANSWER_MARK)
        head = head.replace(THINK_PREFIX, "").strip()
        return head, tail.strip()
    return "", text.strip()
