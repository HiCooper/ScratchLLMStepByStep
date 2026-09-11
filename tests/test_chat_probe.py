"""对话探针质检规则的单元测试（防止自动标记本身引入误报/漏报）。"""
from scripts.chat_probe import (language_drift, prompt_echo, quality_flags,
                                repeated_ngram)


def test_repeated_ngram_detects_loop():
    assert repeated_ngram("我们点了一份意大利面和一杯红酒，" * 6)
    assert repeated_ngram("不要在陌生人面前大声喊叫" * 5)


def test_repeated_ngram_ignores_normal_text():
    assert not repeated_ngram("光合作用是植物利用光能将二氧化碳和水转化为有机物质的过程。")
    assert not repeated_ngram("")


def test_language_drift_only_for_chinese_prompt():
    assert language_drift("什么是机器学习？", "Machine learning is a field of AI.")
    # 英文提问出现英文回答不算漂移
    assert not language_drift("What is ML?", "Machine learning is a field of AI.")
    # 中文回答不算漂移
    assert not language_drift("什么是机器学习？", "机器学习是人工智能的一个分支。")


def test_prompt_echo():
    q = "请把这句话翻译成英文：今天天气很好。"
    assert prompt_echo(q, q + " 好的。")
    assert not prompt_echo(q, "The weather is nice today, let's take a walk.")


def test_quality_flags_clean_response():
    assert quality_flags("中国的首都是哪里？", "中国的首都是北京。") == []


def test_quality_flags_empty_and_short():
    assert "空回复" in quality_flags("你好", "")
    assert "回复过短" in quality_flags("你好", "好")


def test_quality_flags_special_tokens_and_refusal():
    assert "特殊token泄漏" in quality_flags("你好", "你好<|im_end|>")
    flags = quality_flags("你是谁", "抱歉，我无法回答这个问题。")
    assert "拒答模板" in flags
    assert "AI套话" in quality_flags("你是谁", "作为一个AI语言模型，我没有情感。")


def test_quality_flags_no_false_positive_on_legit_apology():
    """正常道歉（非拒答）不应被判成拒答模板。"""
    flags = quality_flags("请帮我写道歉信", "对不起，我迟到了，以后一定注意。")
    assert "拒答模板" not in flags
