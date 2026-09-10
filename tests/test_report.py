"""训练报告生成器单测：扫描/汇总/渲染，覆盖 flat 与 nested 两种 config 结构。"""
import json
import os

from scripts.report_training import collect, estimate_params, render


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
