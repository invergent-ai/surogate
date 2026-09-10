"""GRPOInferenceConfig -> the engine command line, for the plumbed scalars."""

import pytest

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.grpo.inference import surogate_engine
from surogate.utils.dict import DictDefault


@pytest.fixture(autouse=True)
def _cli(monkeypatch):
    # build_argv resolves the `surogate` entry point beside the interpreter; the
    # mapping under test does not need one to exist.
    monkeypatch.setattr(surogate_engine, "_cli", lambda: "/usr/bin/surogate")


def _argv(cfg: dict) -> list[str]:
    return surogate_engine.build_argv(GRPOInferenceConfig(DictDefault(cfg)))


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_max_num_seqs_and_kv_cache_dtype_reach_the_engine():
    argv = _argv({"model": "m", "max_num_seqs": 16, "kv_cache_dtype": "fp8"})
    assert _value(argv, "--max-num-seqs") == "16"
    assert _value(argv, "--kv-dtype") == "fp8"


def test_unset_scalars_are_left_to_the_engine():
    """None is not a value for either flag -- the flag must be absent."""
    argv = _argv({"model": "m"})
    assert "--max-num-seqs" not in argv
    assert "--kv-dtype" not in argv


def test_lora_flags_follow_enable_lora():
    argv = _argv({"model": "m", "enable_lora": True, "max_loras": 4, "max_lora_rank": 32})
    assert "--enable-lora" in argv
    assert _value(argv, "--max-loras") == "4"
    assert _value(argv, "--max-lora-rank") == "32"
    assert "--enable-lora" not in _argv({"model": "m", "enable_lora": False})


def test_the_served_name_is_the_checkpoint_the_orchestrator_expects():
    argv = _argv({"model": "Qwen/Qwen3-0.6B", "port": 8007})
    assert argv[1:3] == ["serve", "Qwen/Qwen3-0.6B"]
    assert _value(argv, "--served-model-name") == "Qwen/Qwen3-0.6B"
    assert _value(argv, "--port") == "8007"


def test_shared_decode_cache_budget_defaults_and_validation():
    assert GRPOInferenceConfig(DictDefault({"model": "m"})).decode_cache_bytes == 0
    assert GRPOInferenceConfig(DictDefault({"model": "m", "decode_cache_bytes": 1 << 30})).decode_cache_bytes == 1 << 30
    with pytest.raises(ValueError, match="nonnegative"):
        GRPOInferenceConfig(DictDefault({"model": "m", "decode_cache_bytes": -1}))


def test_shared_decode_memory_budget_defaults_and_validation():
    assert GRPOInferenceConfig(DictDefault({"model": "m"})).decode_memory_bytes == 0
    assert GRPOInferenceConfig(DictDefault({"model": "m", "decode_memory_bytes": 1 << 30})).decode_memory_bytes == 1 << 30
    with pytest.raises(ValueError, match="nonnegative"):
        GRPOInferenceConfig(DictDefault({"model": "m", "decode_memory_bytes": -1}))


def test_shared_prefill_and_prefix_cache_settings():
    defaults = GRPOInferenceConfig(DictDefault({"model": "m"}))
    assert defaults.decode_prefill_chunk == 256 and defaults.decode_prefix_entries == 32
    custom = GRPOInferenceConfig(DictDefault({"decode_prefill_chunk": 64, "decode_prefix_entries": 0}))
    assert custom.decode_prefill_chunk == 64 and custom.decode_prefix_entries == 0
    for config in ({"decode_prefill_chunk": 0}, {"decode_prefix_entries": -1}):
        with pytest.raises(ValueError):
            GRPOInferenceConfig(DictDefault(config))
