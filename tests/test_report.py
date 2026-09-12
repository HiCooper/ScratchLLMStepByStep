"""训练报告生成器单测：扫描/汇总/渲染，覆盖 flat 与 nested 两种 config 结构。"""
import json
import os

from scripts.report_training import collect, estimate_params, render, _loss_noise


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def test_estimate_params_flat_and_nested():
    flat = {"emb_dim": 512, "n_layers": 10, "vocab_size": 32000, "tie_word_embeddings": True}
    nested = {"model": {"model_emb_dim": 512, "model_n_layers": 10, "model_vocab_size": 32000,
                        "model_tie_word_embeddings": True}}
    assert estimate_params(flat) == estimate_params(nested)
    assert estimate_params(flat).endswith("M")
    # vocab 缺失时用默认 32000，不应崩
    assert estimate_params({"emb_dim": 128, "n_layers": 2}) != "?"


def test_collect_and_render(tmp_path):
    cp = str(tmp_path)
    _write(os.path.join(cp, "pretrain_demo", "metrics.json"),
           {"step": 100, "epoch": 0, "train_loss": 1.5, "eval_loss": 2.5, "perplexity": 12.2,
            "best_eval_loss": 2.4})
    _write(os.path.join(cp, "pretrain_demo", "config.json"),
           {"emb_dim": 512, "n_layers": 10, "vocab_size": 32000, "tie_word_embeddings": True})
    open(os.path.join(cp, "pretrain_demo", "final.pt"), "w").close()
    _write(os.path.join(cp, "sft_demo", "metrics.json"),
           {"step": 50, "train_loss": 0.9, "eval_loss": 1.2})
    _write(os.path.join(cp, "eval_v2_cot_easy.json"),
           {"summary": {"n": 60, "accuracy": {"plain": 0.9, "single": 0.88, "two-phase": 0.85}}})
    _write(os.path.join(cp, "ppl_pretrain_demo.json"),
           {"eval_loss": 2.5, "perplexity": 12.2, "rows": 512, "tokens": 261632})
    with open(os.path.join(cp, "samples_v2_chat.txt"), "w", encoding="utf-8") as f:
        f.write("用户: 什么是AI？\n模型: 人工智能。")

    d = collect(cp)
    assert d["pretrain"][0]["run"] == "pretrain_demo" and d["pretrain"][0]["params"].endswith("M")
    assert d["sft"][0]["run"] == "sft_demo"
    assert d["thinking"]["eval_v2_cot_easy"]["accuracy"]["plain"] == 0.9
    assert "pretrain_demo" in d["ppl"]
    assert "samples_v2_chat.txt" in d["samples"]

    md = render(d)
    for expect in ("MiniGPT 训练报告", "pretrain_demo", "sft_demo", "two-phase", "人工智能。"):
        assert expect in md


def test_loss_noise_is_one_over_sqrt_windows():
    """噪声口径 ≈ 1/√(验证窗口数)：511 个窗口 → ±0.044；退化输入返回 None 而不是抛异常。"""
    assert abs(_loss_noise(511) - 0.0442) < 5e-4
    assert _loss_noise(0) is None and _loss_noise(None) is None and _loss_noise("x") is None


def test_curve_section_renders_noise_note():
    """§1.5 的曲线表要写出"相邻两点差多少才算真收敛"。

    曲线本身来自 tensorboard（fixture 里没有事件文件），所以这里直接注入 curves 字段——
    被测的是**渲染**逻辑，扫描逻辑另有 test_collect 覆盖。
    """
    d = {"time": "2026-01-01 00:00:00", "pretrain": [], "sft": [], "thinking": {},
         "ppl": {}, "samples": {}, "success_rate": {},
         "curves": {"pretrain_demo": {"eval": [(2000, 5.11), (4000, 4.03), (412000, 2.70)],
                                      "tail_tok_s": 24000.0, "tail_train_loss": 2.6,
                                      "eta_hours": None, "n_eval": 3,
                                      "eval_rows": 511, "noise_sigma": _loss_noise(511)}}}
    md = render(d)
    assert "评估噪声" in md and "±0.0442" in md and "511 个验证窗" in md
    assert "2.0k:5.1100" in md and "412.0k:2.7000" in md


def test_difficulty_buckets_are_rendered():
    """§4.5 必须把 by_difficulty 摊开——平均准确率看不出"简单题全对、难题全错"。"""
    d = {"time": "2026-01-01 00:00:00", "pretrain": [], "sft": [], "ppl": {}, "samples": {},
         "success_rate": {}, "curves": {},
         "thinking": {"eval_v5_cot_easy": {
             "n": 200, "accuracy": {"plain": 0.42, "single": 0.55},
             "by_difficulty": {"个位数加减": {"n": 120, "plain": 0.9, "single": 0.95},
                               "两位数加减": {"n": 60, "plain": 0.1, "single": 0.2},
                               "多步混合": {"n": 20, "plain": 0.0, "single": 0.0}}}}}
    md = render(d)
    assert "## 4.5" in md and "个位数加减" in md
    assert "| 个位数加减 | 120 | 90.0% | 95.0% |" in md
    assert "| 多步混合 | 20 | 0.0% | 0.0% |" in md


def test_difficulty_section_degrades_without_buckets():
    """没有 by_difficulty（老产物/未分档评测）时给提示，而不是崩或留空表。"""
    d = {"time": "t", "pretrain": [], "sft": [], "ppl": {}, "samples": {}, "success_rate": {},
         "curves": {}, "thinking": {"eval_old_cot_easy": {"n": 60, "accuracy": {"plain": 1.0}}}}
    md = render(d)
    assert "## 4.5" in md and "尚未产出" in md
