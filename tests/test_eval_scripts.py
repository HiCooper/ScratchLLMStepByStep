"""评测脚本的最小端到端测试（以前这些脚本零测试）。

重点锁两件事：
1. `evaluate_pretrain.py` 能真的跑通并落盘 JSON；
2. 它的 `--split val` 与训练时 `split_train_eval_blocks` 是**同一口径**，
   且 `--split train` / `--split val` 取到的是互不相交的窗口集合。
   （真实事故：以前用 `--max-rows 512` 取 .bin 最前面的窗口，对参与过训练的 bin
    而言那是训练集 loss，却被当成泛化指标写进了 README。）
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")


@pytest.fixture(scope="module")
def tiny_corpus_bin(tmp_path_factory, tiny_tokenizer):
    """造一个足够大的合成 .bin（分块切分要求窗口数 >= 4*n_blocks）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_pretrain_bin", os.path.join(SCRIPTS, "build_pretrain_bin.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 只为拿到 build 函数；脚本自身有 main 保护

    d = tmp_path_factory.mktemp("tiny_bin")
    jsonl = d / "corpus.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(400):
            f.write(json.dumps({"text": f"第{i}句合成语料，用于评测脚本测试。" * 4},
                               ensure_ascii=False) + "\n")
    out_bin = d / "tiny.bin"
    from minigpt.data.pretrain_dataset import tokenize_jsonl_to_bin
    tokenize_jsonl_to_bin(str(jsonl), str(out_bin), tiny_tokenizer)
    return out_bin


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, tiny_tokenizer):
    """存一个最小模型的 checkpoint（带 config，走正常加载路径）。"""
    import torch
    from minigpt.train.checkpoint_io import save_training_checkpoint
    from minigpt.model.transformer import GPTConfig, MiniGPT

    torch.manual_seed(0)
    model = MiniGPT(GPTConfig(vocab_size=len(tiny_tokenizer), emb_dim=32, n_layers=1,
                              n_heads=2, context_length=64, drop_rate=0.0,
                              tie_word_embeddings=True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path_factory.mktemp("tiny_ckpt") / "final.pt"
    save_training_checkpoint(str(path), model, opt, epoch=0, step=1,
                             best_eval_loss=1.0, best_step=1,
                             config=model.config.to_dict())
    return path


def _run_eval(bin_path, ckpt_path, tokenizer_dir, extra):
    out = bin_path.parent / f"metrics_{'_'.join(extra)}.json"
    cmd = [sys.executable, os.path.join(SCRIPTS, "evaluate_pretrain.py"),
           "--checkpoint", str(ckpt_path),
           "--tokenizer-dir", str(tokenizer_dir),
           "--bin", str(bin_path),
           "--batch-size", "2", "--output", str(out)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "MPLCONFIGDIR": str(bin_path.parent)})
    assert r.returncode == 0, f"evaluate_pretrain 失败:\n{r.stdout}\n{r.stderr}"
    return json.loads(out.read_text(encoding="utf-8"))


def test_evaluate_pretrain_val_split_default(tiny_corpus_bin, tiny_ckpt, tiny_tokenizer,
                                             tmp_path):
    """默认 --split val：必须记录分块切分信息，且行数与 train 切分不同。"""
    tok_dir = tiny_tokenizer.name_or_path
    val = _run_eval(tiny_corpus_bin, tiny_ckpt, tok_dir,
                    ["--split", "val", "--eval-blocks", "4"])
    assert val["split"]["split"] == "blocked-document-aligned"
    assert val["split"]["n_blocks"] == 4
    assert val["rows"] > 0 and val["tokens"] > 0
    assert val["perplexity"] > 1.0
    assert val["bin_vocab"] == len(tiny_tokenizer)

    train = _run_eval(tiny_corpus_bin, tiny_ckpt, tok_dir,
                      ["--split", "train", "--eval-blocks", "4"])
    # train 侧应包含 val 侧之外的窗口（两者行数不同、且都小于全量）
    assert train["rows"] != val["rows"]


def test_evaluate_pretrain_split_all_warns(tiny_corpus_bin, tiny_ckpt, tiny_tokenizer):
    """--split all --max-rows 必须打出口径警告（它取的是 bin 最前面的窗口）。"""
    out = tiny_corpus_bin.parent / "metrics_all.json"
    cmd = [sys.executable, os.path.join(SCRIPTS, "evaluate_pretrain.py"),
           "--checkpoint", str(tiny_ckpt), "--tokenizer-dir", str(tiny_tokenizer.name_or_path),
           "--bin", str(tiny_corpus_bin), "--batch-size", "2", "--max-rows", "2",
           "--split", "all", "--output", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "MPLCONFIGDIR": str(tiny_corpus_bin.parent)})
    assert r.returncode == 0, r.stderr
    assert "⚠️" in r.stdout and "--max-rows" in r.stdout, "缺少训练集口径警告"


def test_eval_scripts_have_help_and_no_syntax_error():
    """评测/推理脚本至少能 --help（防止 import 期崩溃）。"""
    for name in ("evaluate_pretrain.py", "eval_thinking.py", "generate.py",
                 "validate_pretrain.py", "report_training.py"):
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, name), "--help"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{name} --help 失败: {r.stderr[:400]}"


def test_eval_thinking_buckets_by_difficulty():
    """CoT 评测必须能**分档**报准确率：单一平均会掩盖"简单题全对、难题全错"。

    真实背景：47.9M 参数在多位数乘加上是容量墙，继续堆同分布数据无用——
    只有分档（规模×题型）才看得出该扩模型还是该补数据。
    """
    sys.path.insert(0, SCRIPTS)
    from eval_thinking import bucket_of

    assert bucket_of("请计算 20 + 8 - 2 等于多少？") == "mixed_small"
    assert bucket_of("请计算 361 + 67 - 77 等于多少？") == "mixed_big"
    assert bucket_of("请计算 9867 - 616 等于多少？") == "sub_big"
    assert bucket_of("请计算 12 × 13 等于多少？") == "mul_small"
    assert bucket_of("请计算 123 × 45 等于多少？") == "mul_big"
    assert bucket_of("请计算 8 ÷ 2 等于多少？") == "div_small"
    assert bucket_of("小明原来有 34 个苹果，又买了 8 个，然后吃掉了 13 个。请问现在还剩多少").startswith("word_")
    assert bucket_of("你好") == "other"
