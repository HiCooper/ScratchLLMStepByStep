from types import SimpleNamespace

from minigpt.config import RunConfig, apply_mapping, build_run_config


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
    assert cfg.model.n_layers == 8  # 未覆盖时保留默认


def test_apply_mapping_ignores_unknown():
    cfg = RunConfig()
    apply_mapping(cfg, {"train_unknown": 1, "nope": 2})
    assert not hasattr(cfg.train, "unknown")
