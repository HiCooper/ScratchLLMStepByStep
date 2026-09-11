from types import SimpleNamespace

import argparse
import json
import warnings

import pytest

from minigpt.config import (DataConfig, ModelConfig, PathConfig, RunConfig, TrainConfig,
                            add_cli_overrides, apply_mapping, build_run_config,
                            dump_run_config, str2bool)


def _parser():
    ap = argparse.ArgumentParser()
    # 各入口自己加的参数（不在 add_cli_overrides 里）
    ap.add_argument("--config-file", default=None)
    for section, cls in (("model", ModelConfig), ("data", DataConfig),
                         ("train", TrainConfig), ("paths", PathConfig)):
        add_cli_overrides(ap, section, cls)
    return ap


def test_build_run_config_overrides():
    args = SimpleNamespace(
        model_emb_dim=256,
        train_batch_size=4,
        paths_output_dir="/tmp/x",
        data_tokenized_bin="/tmp/d.bin",
        model_n_layers=None,
        data_max_lines=None,
    )
    cfg = build_run_config(args)
    assert isinstance(cfg, RunConfig)
    assert cfg.model.emb_dim == 256
    assert cfg.train.batch_size == 4
    assert cfg.paths.output_dir == "/tmp/x"
    assert cfg.data.tokenized_bin == "/tmp/d.bin"
    assert cfg.model.n_layers == ModelConfig().n_layers  # 未覆盖时保留默认


def test_apply_mapping_ignores_unknown():
    cfg = RunConfig()
    apply_mapping(cfg, {"train_unknown": 1, "nope": 2})
    assert not hasattr(cfg.train, "unknown")


# ---------------------------------------------------------------------------
# P0：配置文件里的键名拼错必须出声
# （训练会把 config.json 落到输出目录、鼓励用户改完再用 --config-file 读回，
#   静默忽略等于"某个超参压根没生效"却毫无提示）
# ---------------------------------------------------------------------------
def test_unknown_config_file_key_warns(tmp_path):
    cfg = build_run_config(None)
    path = tmp_path / "config.json"
    dump_run_config(cfg, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"]["emb_dimd"] = 768          # 嵌套形式拼错
    data["train_batchsize"] = 4              # 扁平形式拼错
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="无法对应到任何配置项") as record:
        back = build_run_config(None, str(bad))

    assert back.model.emb_dim == ModelConfig().emb_dim, "拼错的键不得被静默应用"
    msg = str(record[0].message)
    assert "model.emb_dimd" in msg and "train_batchsize" in msg


def test_entry_specific_flags_do_not_warn():
    """入口自有 flag（--pretrain/--sft-jsonl/--data_max_len/--n-heads…）不属于任何
    Config section，不能因为加了它们就在每次训练时刷告警。"""
    args = SimpleNamespace(
        model_emb_dim=512, config_file="/tmp/c.json", pretrain="/tmp/x.pt",
        sft_jsonl="/tmp/s.jsonl", data_max_len=256, n_heads=8,
        model_n_layers=None, paths_output_dir=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = build_run_config(args)
    assert cfg.model.emb_dim == 512


def test_apply_mapping_returns_unknown_keys():
    cfg = RunConfig()
    with pytest.warns(UserWarning, match="无法对应到任何配置项"):
        unknown = apply_mapping(cfg, {"model": {"nope": 1}, "train_batchsize": 2},
                                source="config-file")
    assert unknown == ["model.nope", "train_batchsize"]


# ---------------------------------------------------------------------------
# 回归：CLI 布尔必须能真正关闭开关
# （旧实现用 argparse type=bool，bool("False") == True，导致
#   --train_torch_compile False / --train_reset_step False 全被解析成 True）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("True", True), ("true", True), ("1", True), ("yes", True), ("ON", True),
    ("False", False), ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_str2bool(text, expected):
    assert str2bool(text) is expected


def test_str2bool_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


def test_cli_bool_flags_can_be_disabled():
    """shell 脚本/SKILL 文档统一使用 `--flag False` 形式，必须解析为 False。"""
    args = _parser().parse_args(["--train_torch_compile", "False",
                                 "--train_reset_step", "False",
                                 "--train_save_best", "False",
                                 "--model_tie_word_embeddings", "False"])
    cfg = build_run_config(args)
    assert cfg.train.torch_compile is False
    assert cfg.train.reset_step is False
    assert cfg.train.save_best is False
    assert cfg.model.tie_word_embeddings is False


def test_cli_bool_flags_can_be_enabled():
    args = _parser().parse_args(["--train_torch_compile", "True",
                                 "--train_reset_step", "true"])
    cfg = build_run_config(args)
    assert cfg.train.torch_compile is True
    assert cfg.train.reset_step is True


def test_cli_bool_absent_keeps_dataclass_default():
    assert vars(_parser().parse_args([]))["train_torch_compile"] is None


# ---------------------------------------------------------------------------
# 回归：dump 出来的 config.json 必须能被 --config-file 读回
# （旧实现 dump 写嵌套、apply_mapping 只认扁平键 => 读回是 no-op，
#   emb_dim 512 会静默退回默认值，与 pretrainer 的文档承诺矛盾）
# ---------------------------------------------------------------------------
def test_dump_then_load_config_roundtrip(tmp_path):
    cfg = build_run_config(None)
    cfg.model.emb_dim = 512
    cfg.model.n_layers = 10
    cfg.model.n_heads = 8
    cfg.model.tie_word_embeddings = False
    cfg.data.tokenized_bin = "/tmp/x.bin"
    cfg.train.batch_size = 3
    cfg.train.torch_compile = True
    cfg.paths.output_dir = "/tmp/out"
    p = tmp_path / "config.json"
    dump_run_config(cfg, p)

    back = build_run_config(None, str(p))
    assert back.model.emb_dim == 512
    assert back.model.n_layers == 10
    assert back.model.n_heads == 8
    assert back.model.tie_word_embeddings is False
    assert back.data.tokenized_bin == "/tmp/x.bin"
    assert back.train.batch_size == 3
    assert back.train.torch_compile is True
    assert back.paths.output_dir == "/tmp/out"


def test_config_file_then_cli_override_wins(tmp_path):
    cfg = build_run_config(None)
    cfg.model.emb_dim = 512
    p = tmp_path / "config.json"
    dump_run_config(cfg, p)
    args = _parser().parse_args(["--config-file", str(p), "--model_emb_dim", "768"])
    back = build_run_config(args, args.config_file)
    assert back.model.emb_dim == 768, "CLI 必须覆盖 config-file"


def test_apply_mapping_supports_nested_and_flat():
    cfg = RunConfig()
    apply_mapping(cfg, {"model": {"emb_dim": 111}, "train_batch_size": 7})
    assert cfg.model.emb_dim == 111
    assert cfg.train.batch_size == 7


def test_apply_mapping_ignores_nested_unknown():
    cfg = RunConfig()
    apply_mapping(cfg, {"model": {"nope": 1}, "train": "not-a-dict"})
    assert not hasattr(cfg.model, "nope")


# ---------------------------------------------------------------------------
# 默认值必须与文档声称的生产基线一致
# ---------------------------------------------------------------------------
def test_defaults_match_documented_baseline():
    cfg = RunConfig()
    assert (cfg.model.emb_dim, cfg.model.n_layers, cfg.model.n_heads) == (512, 10, 8)
    assert cfg.model.context_length == 512
    assert cfg.model.tie_word_embeddings is True
    assert cfg.data.tokenizer_dir.endswith("tokenizer_v3"), \
        "默认分词器必须是 32k 的 tokenizer_v3（qwen2 的 151k 词表会让 70%+ 参数花在 embedding 上）"
    assert cfg.data.max_lines == 0, "默认应取全量语料"
    assert cfg.data.eval_blocks >= 1


def test_removed_dead_fields_stay_removed():
    """train_ratio/log_every/paths.pretrain_dataset/paths.tokenizer_dir 是无引用死字段。"""
    cfg = RunConfig()
    for section, attr in (("train", "train_ratio"), ("train", "log_every"),
                          ("paths", "pretrain_dataset"), ("paths", "tokenizer_dir")):
        assert not hasattr(getattr(cfg, section), attr)




# ---------------------------------------------------------------------------
# 参数量估算：唯一实现（此前 estimate_resources.py 与 report_training.py 各有一份，
# 后者忽略 SwiGLU/新的 8/3·d 宽度，会对 SwiGLU run 多报 50% 前馈参数）
# ---------------------------------------------------------------------------
def test_estimate_params_matches_real_model():
    import torch
    from minigpt.config import estimate_params, estimate_params_from_config
    from minigpt.model.transformer import GPTConfig, MiniGPT

    for kw in (dict(tie_word_embeddings=True),
               dict(tie_word_embeddings=False),
               dict(tie_word_embeddings=True, use_swiglu=True),
               dict(tie_word_embeddings=False, use_swiglu=True),
               dict(tie_word_embeddings=True, norm_type="rmsnorm", lm_head_bias=True)):
        m = MiniGPT(GPTConfig(vocab_size=4096, emb_dim=128, n_layers=3, n_heads=4,
                              context_length=32, **kw))
        real = sum(p.numel() for p in m.parameters())
        est = estimate_params_from_config(m.config)
        assert abs(est - real) / real < 0.01, f"{kw}: 估算 {est} vs 实际 {real}"


def test_estimate_params_swiglu_uses_eight_thirds_width():
    """SwiGLU 的中间维是 8/3·d（不是 4·d）：参数量应与 GELU 版基本持平。"""
    from minigpt.config import estimate_params
    gelu = estimate_params(512, 10, 32000, use_swiglu=False, tie_word_embeddings=True)
    swiglu = estimate_params(512, 10, 32000, use_swiglu=True, tie_word_embeddings=True)
    assert abs(swiglu - gelu) / gelu < 0.02
    old_wrong = estimate_params(512, 10, 32000, use_swiglu=True, ffn_hidden_dim=2048,
                                tie_word_embeddings=True)
    assert old_wrong > swiglu * 1.2, "旧的 4d 口径应显著更大，确保不会被误用回去"


def test_estimate_params_accepts_dict_and_dataclass():
    from minigpt.config import estimate_params_from_config
    from minigpt.model.transformer import GPTConfig
    a = estimate_params_from_config(ModelConfig())
    b = estimate_params_from_config(vars(ModelConfig()))          # 扁平 dict（前缀已剥）
    c = estimate_params_from_config(GPTConfig(vocab_size=32000, emb_dim=512,
                                              n_layers=10, n_heads=8, context_length=512,
                                              tie_word_embeddings=True))
    assert a == b == c, f"{a} / {b} / {c} 应一致"
