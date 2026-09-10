"""思考模式（CoT）单测：停止序列生成、两阶段解码返回结构、思考段切分。"""
import torch

from minigpt.model.generation import (ANSWER_MARK, THINK_PREFIX, generate_ids,
                                      generate_with_thinking, split_thinking)
from minigpt.model.transformer import GPTConfig, MiniGPT


def _tiny_model():
    return MiniGPT(GPTConfig(emb_dim=32, n_layers=1, n_heads=2, context_length=64,
                             vocab_size=100, tie_word_embeddings=True)).eval()


def test_split_thinking():
    text = f"{THINK_PREFIX}1. 先算加法\n{ANSWER_MARK}42"
    think, ans = split_thinking(text)
    assert think == "1. 先算加法"
    assert ans == "42"
    # 没有标记时全部视为答案
    think2, ans2 = split_thinking("直接回答")
    assert think2 == "" and ans2 == "直接回答"


def test_generate_ids_respects_budget():
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    out = generate_ids(model, ids, max_new_tokens=5, eos_token_id=99, use_kv_cache=False)
    assert out.shape[0] == 1
    assert out.shape[1] <= 3 + 5


def test_generate_with_thinking_structure(monkeypatch):
    model = _tiny_model()

    class _Tok:
        eos_token_id = 99

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

        def encode(self, text):
            return [(ord(c) % 97) + 1 for c in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" for _ in ids)

    res = generate_with_thinking(model, _Tok(), "1+1=?", chat=True,
                                 max_new_tokens=4, thinking_max_tokens=4,
                                 hide_thinking=True, use_kv_cache=False, device="cpu")
    assert {"thinking", "answer", "display", "full"} <= set(res)
    assert isinstance(res["answer"], str) and res["display"] == res["answer"]
