"""SFT 标签掩码与 padding 的回归测试。

覆盖两个真实事故：
1. **多轮 history 全部进 loss**：旧 `calc_label` 只定位**第一个** assistant 标记，
   于是前面所有 history（含用户提问）都参与交叉熵。实测一条 3 轮样本 55 token 里
   41 个被监督，模型被训练去生成用户提问。
2. **EOS 被当作 padding 屏蔽**：collator 用 `unk_token_id` 当 pad，而 qwen2 下
   pad/unk/eos 同为 151643，`item == pad_token_id` 把真实的 eos 一并置 -100，
   模型永远学不会停止。

测试自带 chat template，不依赖仓库里被 .gitignore 的词表目录。
"""
import pytest
from transformers import AutoTokenizer

from minigpt.data.sft_dataset import (DEFAULT_ASSISTANT_MARKER, DEFAULT_TURN_END_MARKER,
                                      assistant_spans, calc_label, collate,
                                      create_batch_collator, find_all_sublist_index,
                                      resolve_stop_token_ids)

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
)


@pytest.fixture(scope="module")
def chat_tokenizer(tiny_tokenizer):
    """给 mini BPE 加上 Qwen2 风格的 chat template 与特殊 token。"""
    tk = tiny_tokenizer
    tk.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tk.chat_template = CHAT_TEMPLATE
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    return tk


def _render(tk, turns):
    msgs = []
    for role, content in turns:
        msgs.append({"role": role, "content": content})
    ids = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return ids


def test_find_all_sublist_index():
    assert find_all_sublist_index([1, 2, 1, 2, 1, 2], [1, 2]) == [0, 2, 4]
    assert find_all_sublist_index([1, 2, 3], []) == []
    assert find_all_sublist_index([1, 2, 3], [9]) == []


def test_assistant_spans_finds_every_turn(chat_tokenizer):
    ids = _render(chat_tokenizer, [("user", "第一问"), ("assistant", "第一答"),
                                   ("user", "第二问"), ("assistant", "第二答"),
                                   ("user", "第三问"), ("assistant", "第三答")])
    spans = assistant_spans(ids, chat_tokenizer)
    assert len(spans) == 3, f"应定位到 3 轮 assistant，实际 {len(spans)}"
    # 每个区间都应包含该轮的回答内容
    texts = [chat_tokenizer.decode(ids[s:e]) for s, e in spans]
    assert "第一答" in texts[0] and "第二答" in texts[1] and "第三答" in texts[2]


def test_multi_turn_labels_only_supervise_assistant(chat_tokenizer):
    """核心回归：多轮样本里用户提问绝不能进 loss。"""
    ids = _render(chat_tokenizer, [("user", "第一问"), ("assistant", "第一答"),
                                   ("user", "第二问"), ("assistant", "第二答"),
                                   ("user", "第三问"), ("assistant", "第三答")])
    labels, found = calc_label(ids, None, chat_tokenizer)
    assert found
    supervised_text = chat_tokenizer.decode([t for t in labels if t != -100])
    assert "第一问" not in supervised_text
    assert "第二问" not in supervised_text
    assert "第三问" not in supervised_text
    for ans in ("第一答", "第二答", "第三答"):
        assert ans in supervised_text
    # 应监督的 token 数远小于序列长度（旧实现会监督几乎全部）
    n_sup = sum(1 for t in labels if t != -100)
    assert n_sup < len(ids) * 0.6, f"监督了 {n_sup}/{len(ids)} 个 token，疑似把所有 history 都算进 loss"


def test_turn_end_marker_is_supervised(chat_tokenizer):
    """模型必须学到何时停止：每轮的 <|im_end|> 都要在监督范围内。"""
    ids = _render(chat_tokenizer, [("user", "问"), ("assistant", "答"),
                                   ("user", "再问"), ("assistant", "再答")])
    labels, _ = calc_label(ids, None, chat_tokenizer)
    mark_id = chat_tokenizer.convert_tokens_to_ids(DEFAULT_TURN_END_MARKER)
    assert sum(1 for t in labels if t == mark_id) == 2, "两轮的 <|im_end|> 都应被监督"


def test_labels_never_contain_padding_positions(chat_tokenizer):
    seqs = [_render(chat_tokenizer, [("user", "短"), ("assistant", "答")]),
            _render(chat_tokenizer, [("user", "很长的一个问题需要更多 token"), ("assistant", "答")])]
    X, Y, M = collate(seqs, chat_tokenizer.pad_token_id, chat_tokenizer)
    assert X.shape == Y.shape == M.shape
    for i, s in enumerate(seqs):
        assert M[i, :len(s)].all(), "真实 token 的 attention_mask 必须为 1"
        assert (M[i, len(s):] == 0).all(), "padding 的 attention_mask 必须为 0"
        assert (Y[i, len(s):] == -100).all(), "padding 的 label 必须是 -100"


def test_padding_does_not_mask_real_eos(chat_tokenizer):
    """pad_token_id == eos_token_id 时，真实 eos 仍必须被监督（旧实现会屏蔽掉）。"""
    tk = chat_tokenizer
    if tk.pad_token_id != tk.eos_token_id:
        tk.pad_token = tk.eos_token
    assert tk.pad_token_id == tk.eos_token_id
    seqs = [_render(tk, [("user", "问"), ("assistant", "答")]),
            _render(tk, [("user", "更长的问"), ("assistant", "更长的答")])]
    _, Y, _ = collate(seqs, tk.pad_token_id, tk)
    # batch 内每一条都要在真实长度内保留非 -100 的监督信号
    for i, s in enumerate(seqs):
        assert (Y[i, :len(s)] != -100).any(), "真实区间内没有监督信号"


def test_collate_raises_on_template_mismatch(tiny_tokenizer):
    """模板不匹配必须显式失败，而不是静默退化成"在 prompt 上算 loss"。"""
    tk = tiny_tokenizer
    tk.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    # 故意不给 chat_template，构造一段没有 assistant 标记的 token
    ids = tk.encode("这里完全没有 assistant 标记")[:16]
    assert assistant_spans(ids, tk) == []
    with pytest.raises(RuntimeError, match="assistant 标记"):
        collate([ids], tk.pad_token_id, tk)


def test_create_batch_collator_uses_pad_token_not_unk(tiny_tokenizer):
    tk = tiny_tokenizer
    tk.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tk.chat_template = CHAT_TEMPLATE
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    col = create_batch_collator(tk)
    assert col.keywords["pad_token_id"] == tk.pad_token_id
    assert col.keywords["assistant_marker"] == DEFAULT_ASSISTANT_MARKER


def test_resolve_stop_token_ids_prefers_turn_end(tiny_tokenizer):
    tk = tiny_tokenizer
    tk.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tk.chat_template = CHAT_TEMPLATE
    ids = resolve_stop_token_ids(tk)
    mark_id = tk.convert_tokens_to_ids(DEFAULT_TURN_END_MARKER)
    assert ids[0] == mark_id, "回合结束符必须排在停止列表首位"
    assert tk.eos_token_id in ids, "eos 必须作为兜底保留"
    assert len(ids) == len(set(ids)), "停止 id 不应重复"


def test_generate_accepts_multiple_stop_ids(tiny_tokenizer):
    """qwen2 下 im_end != eos，只传 eos 会导致生成永不停止。"""
    import torch
    from minigpt.model.transformer import GPTConfig, MiniGPT

    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(vocab_size=64, emb_dim=32, n_layers=2, n_heads=2,
                              context_length=16, drop_rate=0.0))
    model.eval()
    prompt = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        out_list = model.generate(prompt, 10, eos_token_id=[0, 1], use_kv_cache=True)
        out_none = model.generate(prompt, 10, eos_token_id=None, use_kv_cache=True)
    assert out_list.shape[1] <= 4 + 10
    assert out_none.shape[1] == 4 + 10      # 无停止符时跑满
