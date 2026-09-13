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


def _curve(eval_pts):
    return {"pretrain_v5": {"eval": eval_pts, "tail_tok_s": 24000.0, "tail_train_loss": 2.6,
                            "eta_hours": None, "n_eval": len(eval_pts)}}


def _base_dict(curves):
    return {"time": "2026-01-01 00:00:00", "pretrain": [], "sft": [], "ppl": {}, "samples": {},
            "success_rate": {}, "thinking": {}, "curves": curves}


def test_historical_baseline_matches_readme():
    """§6 里引用的历史数字必须与根 README「实测结果」表一致（两处漂移=报告在编数字）。"""
    import re
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    txt = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    row = re.search(r"\|\s*`pretrain_v2_full`\s*\|([^\n]+)\|", txt)
    assert row, "README 里找不到 pretrain_v2_full 的结果行"
    cells = [c.strip().strip("*") for c in row.group(1).split("|")]
    from scripts.report_training import HISTORICAL_BASELINE as HB
    assert int(cells[0].replace(",", "")) == HB["step"]
    assert abs(float(cells[2]) - HB["train_loss"]) < 1e-6
    assert abs(float(cells[3]) - HB["perplexity"]) < 1e-6


def test_token_matched_comparison_requires_close_budget():
    """只跑到 76k 步时**不能**冒充"同预算点"去和 211k 步的历史基线并列。"""
    early = render(_base_dict(_curve([(2000, 5.11), (76000, 2.7174)])))
    assert "尚未到 211,000 步" in early and "同预算点对比" not in early

    reached = render(_base_dict(_curve([(2000, 5.11), (210000, 2.55), (212000, 2.54)])))
    assert "| `pretrain_v5`（本次，同预算点） | 210,000 | 8.60 亿 |" in reached
    assert "同预算点对比" in reached
    # 历史行与说明必须在
    assert "| `pretrain_v2_full`（历史，文档记录） | 211,000 |" in reached
    assert "这不是 A/B 结论" in reached and "同一 bin、同一 `--split val`" in reached


def test_sft_stage_curves_are_rendered():
    """交付要求里的"各阶段 eval_loss 曲线"——不能只有基座有曲线。"""
    d = _base_dict({})
    d["sft_curves"] = {
        "sft_v5_chat": {"eval": [(200, 1.5), (2000, 1.2), (9800, 0.95)],
                        "tail_train_loss": 0.93, "n_eval": 3},
        "sft_v5_cot_easy": {"eval": [(200, 1.1), (14700, 0.62)],
                            "tail_train_loss": 0.60, "n_eval": 2},
    }
    md = render(d)
    assert "## 3.5" in md
    assert "**sft_v5_chat** — 最新 step 9,800" in md and "最低 0.9500 @step 9800" in md
    assert "| 曲线 step:loss | 200:1.5000 → 2.0k:1.2000 → 9.8k:0.9500 |" in md
    assert "**sft_v5_cot_easy**" in md
    # 没有下游曲线时给出提示而不是空表
    assert "尚未产出：等待下游 SFT/CoT 阶段" in render(_base_dict({}))


# ───────────── 曲线来源：tensorboard 过大时改用日志重建（实测 2GB/286k 步） ─────────────
_LOG_SAMPLE = """\
2026-09-13 08:00:00 lr=0.0006, train_loss: 5.20, eval_loss: 5.11, grad_norm=1.10, steps: 2000/412000
2026-09-13 08:05:00 lr=0.0006, train_loss: 3.30, eval_loss: 3.20, grad_norm=0.70, steps: 4000/412000
2026-09-13 08:10:00 lr=0.0006, train_loss: 3.00, eval_loss: 2.95, grad_norm=0.60, steps: 6000/412000
不是日志行的一行
2026-09-13 08:15:00 lr=0.0005, train_loss: 2.80, eval_loss: 2.80, grad_norm=0.55, steps: 8000/412000
"""


def test_curve_from_log_rebuilds_eval_curve(tmp_path):
    """日志重建：每秒级可读，点、吞吐、ETA 都要对。"""
    from scripts.report_training import curve_from_log
    p = tmp_path / "run.log"
    p.write_text(_LOG_SAMPLE, encoding="utf-8")
    c = curve_from_log(str(p), tokens_per_step=4096)
    assert [s for s, _ in c["eval"]] == [2000, 4000, 6000, 8000]
    assert c["eval"][0][1] == 5.11 and c["eval"][-1][1] == 2.80
    assert c["tail_train_loss"] == 2.80
    # 最后三个 eval 点：4000→8000 步用 10 分钟 ⇒ 4000*4096/600 ≈ 27.3k tok/s
    assert 26000 < c["tail_tok_s"] < 29000
    assert c["max_steps"] == 412000 and c["eta_hours"] > 0
    assert c["source"] == "log"


def test_pick_curve_falls_back_to_log_when_tb_is_huge(tmp_path, monkeypatch):
    """事件目录超过阈值时必须走日志，而不是去解析几 GB 直方图。"""
    import scripts.report_training as rt
    run = tmp_path / "pretrain_demo"
    (run / "tensorboard").mkdir(parents=True)
    (run / "tensorboard" / "events.out").write_bytes(b"x" * 1024)
    (tmp_path / "pretrain_demo.log").write_text(_LOG_SAMPLE, encoding="utf-8")

    monkeypatch.setattr(rt, "TB_MAX_BYTES", 1)          # 任何 tb 都算"过大"
    c = rt._pick_curve(str(run), {}, str(tmp_path / "pretrain_demo.log"))
    assert c["source"] == "log" and c["n_eval"] == 4 and c["tb_bytes"] == 1024

    monkeypatch.setattr(rt, "TB_MAX_BYTES", 10**9)      # 阈值放宽 → 回到 tensorboard 分支
    c2 = rt._pick_curve(str(run), {}, str(tmp_path / "pretrain_demo.log"))
    assert c2["source"] == "log", "假 events 文件读不出标量时应回退到日志而不是空曲线"
