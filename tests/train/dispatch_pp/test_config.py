"""Config parsing + validation tests for parallelism=dispatch_pp."""

import pytest

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault

MODEL = "ro/train/out_opmix_wordform_s50_a010_eval"


def _cfg(data):
    cfg = SFTConfig(DictDefault(data))
    # __init__ parses fields but does not resolve the model; full __post_init__
    # would download the tokenizer/config. These tests target only the dispatch_pp
    # config contract, so they call the dedicated validator directly (it is invoked
    # from __post_init__ before _validate_ep_config in the real path).
    cfg._validate_dispatch_pp_config()
    return cfg


def _base(tmp_path, **over):
    cfg = {
        "model": MODEL,
        "output_dir": str(tmp_path / "out"),
        "gpus": 1,
        "parallelism": "dispatch_pp",
        "dispatch_pp": {"min_stages": 4, "upper_threshold": 1.1},
    }
    cfg.update(over)
    return cfg


def test_dispatch_pp_parses_subblock(tmp_path):
    cfg = _cfg(_base(tmp_path))
    assert cfg.parallelism == "dispatch_pp"
    assert cfg.dispatch_pp["min_stages"] == 4


def test_default_parallelism_is_unset_and_path_unchanged(tmp_path):
    cfg = _cfg({"model": MODEL, "output_dir": str(tmp_path / "out"), "gpus": 1})
    assert cfg.parallelism in (None, "ddp")


def test_dispatch_pp_rejects_zero_sharding(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*ZeRO|zero_level"):
        _cfg(_base(tmp_path, zero_level=3, shard_weights=True))


def test_dispatch_pp_rejects_cpu_training_mode(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*cpu_training"):
        _cfg(_base(tmp_path, cpu_training=True))


def test_dispatch_pp_rejects_moe_ep(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*MoE|ep_size"):
        _cfg(_base(tmp_path, ep_size=2))


def test_dispatch_pp_rejects_fp8_recipe_in_v1(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*BF16|recipe"):
        _cfg(_base(tmp_path, recipe="fp8_hybrid"))


def test_dispatch_pp_disables_cuda_graphs(tmp_path):
    cfg = _cfg(_base(tmp_path, use_cuda_graphs=True))
    assert cfg.use_cuda_graphs is False
