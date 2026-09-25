"""GRPOInferenceConfig -> the engine command line, for the plumbed scalars."""

import pytest
import yaml

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.core.config.loader import load_config
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


def test_offload_yaml_reaches_the_launched_server(tmp_path, monkeypatch):
    path = tmp_path / "infer.yaml"
    path.write_text(
        "model: fixture\ngpu_layers: ${OFFLOAD_GPU_LAYERS}\nhost_moe_layers: all\n"
        "expert_slots: 8\nhost_expert_bank: w8\ncpu_moe_share: auto\n"
        "cpu_moe_prefill_share: 0\ncpu_moe_min_tokens: 1\n"
    )
    monkeypatch.setenv("OFFLOAD_GPU_LAYERS", "0")
    launched = []
    monkeypatch.setattr(surogate_engine.os, "execv", lambda executable, argv: launched.append((executable, argv)))
    surogate_engine.server(load_config(GRPOInferenceConfig, str(path)))
    executable, argv = launched[0]
    assert executable == argv[0] == "/usr/bin/surogate"
    assert argv[1:3] == ["serve", "fixture"]
    assert _value(argv, "--gpu-layers") == "0"
    assert _value(argv, "--host-moe-layers") == "all"
    assert _value(argv, "--expert-slots") == "8"
    assert _value(argv, "--host-expert-bank") == "w8"
    assert _value(argv, "--cpu-moe-share") == "auto"
    assert _value(argv, "--cpu-moe-prefill-share") == "0.0"
    assert _value(argv, "--cpu-moe-min-tokens") == "1"
    assert "--enable-lora" in argv
    assert _value(argv, "--kv-capacity") == "auto"


@pytest.mark.parametrize("settings", [
    {"gpu_layers": "all", "host_moe_layers": "auto", "expert_slots": 0, "host_expert_bank": "auto",
     "cpu_moe_share": 0, "cpu_moe_prefill_share": 1, "cpu_moe_min_tokens": 0},
    {"gpu_layers": 12, "host_moe_layers": 4, "expert_slots": 16, "host_expert_bank": "q4",
     "cpu_moe_share": 0.25, "cpu_moe_prefill_share": 0.5, "cpu_moe_min_tokens": 8},
])
def test_offload_values_and_explicit_null_preserve_engine_semantics(tmp_path, settings):
    path = tmp_path / "infer.yaml"
    path.write_text(yaml.safe_dump({"model": "fixture", **settings}))
    argv = surogate_engine.build_argv(load_config(GRPOInferenceConfig, str(path)))
    for name, expected in settings.items():
        actual = _value(argv, "--" + name.replace("_", "-"))
        if isinstance(expected, (int, float)):
            assert float(actual) == expected
        else:
            assert actual == expected
    null_argv = _argv({"model": "fixture", **dict.fromkeys(settings)})
    assert null_argv == _argv({"model": "fixture"})
    assert not any("--" + name.replace("_", "-") in null_argv for name in settings)


@pytest.mark.parametrize("name,value", [
    ("gpu_layers", -1), ("gpu_layers", 0.5), ("gpu_layers", False), ("gpu_layers", "auto"),
    ("gpu_layers", 2**31), ("host_moe_layers", -1), ("host_moe_layers", True),
    ("expert_slots", 1.5), ("expert_slots", "all"), ("expert_slots", -1),
    ("cpu_moe_min_tokens", -1), ("cpu_moe_min_tokens", False),
    ("host_expert_bank", "bf16"), ("host_expert_bank", False),
    ("cpu_moe_share", -0.1), ("cpu_moe_share", 1.1), ("cpu_moe_share", True),
    ("cpu_moe_share", "nan"), ("cpu_moe_share", float("inf")),
    ("cpu_moe_prefill_share", "auto"), ("cpu_moe_prefill_share", float("nan")),
    ("cpu_moe_prefill_share", -1), ("cpu_moe_prefill_share", 2),
])
def test_invalid_offload_settings_fail_before_launch(name, value):
    with pytest.raises(ValueError, match=name):
        _argv({"model": "fixture", name: value})


# ── Tool calling ──────────────────────────────────────────────────────
#
# Why these exist is recorded on the config fields; what is pinned here is that
# they reach the engine command line, and in an order it accepts.


def test_a_run_that_asks_for_tool_calling_gets_both_flags():
    argv = _argv({
        "model": "m", "enable_auto_tool_choice": True, "tool_call_parser": "hermes",
    })
    assert _value(argv, "--tool-call-parser") == "hermes"
    assert "--enable-auto-tool-choice" in argv


def test_a_run_that_never_calls_a_tool_is_launched_exactly_as_before():
    """With the gate open the engine parses a tool call out of every
    completion, which a run with no tools has no reason to carry."""
    argv = _argv({"model": "m"})
    assert "--enable-auto-tool-choice" not in argv
    assert "--tool-call-parser" not in argv


def test_a_parser_can_be_named_without_opening_the_gate():
    """Naming a parser is not the same as permitting `auto`; a request may
    still name a function explicitly."""
    argv = _argv({"model": "m", "tool_call_parser": "llama3_json"})
    assert _value(argv, "--tool-call-parser") == "llama3_json"
    assert "--enable-auto-tool-choice" not in argv


def test_the_gate_without_a_parser_is_refused_before_launch():
    """The engine refuses this pair, but only after execv, so the run would
    show a server that never turns healthy instead of its error message."""
    with pytest.raises(ValueError, match="turns parsing off"):
        _argv({"model": "m", "enable_auto_tool_choice": True, "tool_call_parser": "none"})


def test_an_unfamiliar_parser_name_is_left_to_the_engine():
    """The parser registry grows between releases; a copy of it here would go
    stale and refuse a name the shipped engine accepts."""
    # With the gate on, so the validation actually runs: without it the guard
    # is never reached and widening it to a stale allowlist would go unnoticed.
    argv = _argv({
        "model": "m", "tool_call_parser": "muse_glimmer",
        "enable_auto_tool_choice": True,
    })
    assert _value(argv, "--tool-call-parser") == "muse_glimmer"
    assert "--enable-auto-tool-choice" in argv


def test_rollouts_are_not_pinned_to_one_seed_by_default():
    """The default must leave `--seed` off the command line.

    The engine applies `--seed` as a process-level override on every request, so
    a seed set here makes all 8 rollouts of a group sample identically. Identical
    rollouts score identically, and both functions in
    `grpo/orchestrator/advantage.py` return exactly zero for a group with no
    reward spread, so the run trains on a zero gradient while looking healthy.
    This defaulted to 0 and silently did that to every split run from the commit
    that first passed it to the engine.

    Omitted, the engine draws a fresh seed per request, which is what colocate
    already gets by never setting one.
    """
    assert "--seed" not in _argv({"model": "m"})


def test_an_explicit_seed_still_reaches_the_engine():
    """Changing the default must not remove the knob, only stop presuming it."""
    assert _value(_argv({"model": "m", "seed": 1234}), "--seed") == "1234"


def test_the_admission_bounds_reach_the_engine():
    """Sized per run in `grpo/utils/capacity.py`; both must survive the trip.

    Written out rather than derived from the field names: a test that builds the
    flag with the same `replace("_", "-")` the builder uses would agree with it
    however wrong it was.
    """
    argv = _argv({"model": "m", "max_pending_requests": 4242, "pending_timeout_ms": 999})
    assert _value(argv, "--max-pending-requests") == "4242"
    assert _value(argv, "--pending-timeout-ms") == "999"


def test_unset_admission_bounds_are_left_to_the_engine():
    argv = _argv({"model": "m"})
    assert "--max-pending-requests" not in argv
    assert "--pending-timeout-ms" not in argv
