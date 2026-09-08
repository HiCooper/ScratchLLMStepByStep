"""分词器的单元测试：编解码 round-trip、特殊 token 配置。"""


def test_tiny_tokenizer_roundtrip(tiny_tokenizer):
    zh = "在查处虚开增值税专用发票案件中，常常涉及进项留抵税额。"
    en = "The transformer uses self-attention."
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(zh)) == zh
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(en)) == en


def test_tiny_tokenizer_special_tokens(tiny_tokenizer):
    assert tiny_tokenizer.bos_token == "<|im_start|>"
    assert tiny_tokenizer.eos_token == "<|im_end|>"
    assert tiny_tokenizer.unk_token == "<|endoftext|>"
    # bos / eos 应有各自独立的 id
    assert tiny_tokenizer.bos_token_id != tiny_tokenizer.eos_token_id
