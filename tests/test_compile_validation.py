"""Tests for compile-time validation passes in py_compiler.py.

Covers:
1. Shape validation: zero-dimension detection and unresolved string dimension warnings
2. Activation slot validation: graph output / activation slot name cross-referencing
3. Config flow roundtrip: all @hf_config mapped fields appear in compiled IR config
"""

import json

import pytest

from surogate.dsl import (
    Param,
    Tensor,
    compile_model,
    forward,
    graph,
    model,
)
from surogate.dsl.errors import DSLError, ErrorCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_first_error_code(payload: str) -> str:
    doc = json.loads(payload)
    assert doc["success"] is False, f"Expected failure, got success: {payload[:200]}"
    assert "errors" in doc and doc["errors"]
    return doc["errors"][0]["code"]


def _get_warning_codes(payload: str) -> set[str]:
    doc = json.loads(payload)
    warnings = doc.get("warnings", [])
    return {w["code"] for w in warnings}


def _get_warning_messages(payload: str) -> list[str]:
    doc = json.loads(payload)
    warnings = doc.get("warnings", [])
    return [w["message"] for w in warnings]


# ---------------------------------------------------------------------------
# 1. Shape validation — zero-dimension detection
# ---------------------------------------------------------------------------


def test_zero_dimension_param_raises_e004():
    """A param with a zero dimension should raise DSLShapeError (E004)."""

    @model
    class ZeroDimModel:
        weight = Param(Tensor["d_model", "head_size"])

        @forward
        def forward(self, x: Tensor["B", "T", "d_model"]) -> Tensor["B", "T", "d_model"]:
            with graph() as g:
                return g.matmul(x, "weight")

    # d_model=0 should trigger the zero-dimension check
    payload = compile_model(
        ZeroDimModel,
        {"d_model": 0, "head_size": 64},
        raise_on_error=False,
    )
    assert _get_first_error_code(payload) == "E004"


def test_zero_dimension_param_raises_with_raise_on_error():
    """Zero-dim param raises DSLShapeError when raise_on_error=True."""

    @model
    class ZeroDimModel2:
        weight = Param(Tensor["d_model", "head_size"])

        @forward
        def forward(self, x: Tensor["B", "T", "d_model"]) -> Tensor["B", "T", "d_model"]:
            with graph() as g:
                return g.matmul(x, "weight")

    with pytest.raises(DSLError) as excinfo:
        compile_model(
            ZeroDimModel2,
            {"d_model": 0, "head_size": 64},
            raise_on_error=True,
        )
    assert excinfo.value.code == ErrorCode.E004


# ---------------------------------------------------------------------------
# 2. Shape validation — unresolved string dimension warning
# ---------------------------------------------------------------------------


def test_unresolved_string_dim_warns_w006():
    """A param with an unresolved string dim should emit W006."""

    @model
    class UnresolvedModel:
        weight = Param(Tensor["d_model", "unknown_dim"])

        @forward
        def forward(self, x: Tensor["B", "T", "d_model"]) -> Tensor["B", "T", "d_model"]:
            with graph() as g:
                return g.matmul(x, "weight")

    # d_model=128 resolves, but "unknown_dim" won't resolve to a concrete value
    payload = compile_model(
        UnresolvedModel,
        {"d_model": 128},
        raise_on_error=False,
    )
    doc = json.loads(payload)
    # Should succeed (warning, not error)
    assert doc["success"] is True, f"Expected success, got: {payload[:300]}"
    assert "W006" in _get_warning_codes(payload)
    # Check the warning message mentions the unresolved dimension
    messages = _get_warning_messages(payload)
    assert any("unknown_dim" in m for m in messages)


# ---------------------------------------------------------------------------
# 3. Config flow roundtrip — hf_config keys appear in IR config
# ---------------------------------------------------------------------------


def test_config_roundtrip_qwen3():
    """All @hf_config mapped fields for Qwen3 should appear in the IR config."""
    from surogate.dsl.models.qwen3 import Qwen3Model

    spec = Qwen3Model._dsl_spec
    hf_mapping = spec.hf_config.param_mapping

    # Build a config that maps all HF keys
    hf_config = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
    }

    # Build DSL config from HF config
    config = {}
    for dsl_key, hf_key in hf_mapping.items():
        if hf_key in hf_config and hf_config[hf_key] is not None:
            config[dsl_key] = hf_config[hf_key]

    payload = compile_model(Qwen3Model, config, raise_on_error=True)
    doc = json.loads(payload)
    assert doc["success"] is True

    ir_config = doc["modules"][0]["config"]

    # Every @hf_config mapped DSL key should be in the compiled IR config
    for dsl_key in hf_mapping:
        assert dsl_key in ir_config, (
            f"DSL key '{dsl_key}' (mapped from HF '{hf_mapping[dsl_key]}') "
            f"missing from compiled IR config"
        )


def test_config_roundtrip_nemotron_h():
    """All @hf_config mapped fields for NemotronH should appear in the IR config."""
    from surogate.dsl.models.nemotron_h import NemotronHModel

    spec = NemotronHModel._dsl_spec
    hf_mapping = spec.hf_config.param_mapping

    # Minimal HF config for Nemotron-H (tiny model)
    hf_config = {
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "mamba_num_heads": 8,
        "mamba_head_dim": 32,
        "ssm_state_size": 128,
        "n_groups": 4,
        "conv_kernel": 4,
        "chunk_size": 128,
        "time_step_limit": [0.0, 1e9],
        "time_step_min": 0.001,
        "time_step_max": 0.1,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "layer_norm_epsilon": 1e-5,
        # 4 layers: M, A, P, M
        "hybrid_override_pattern": "M*-M",
        "n_routed_experts": 0,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "moe_shared_expert_intermediate_size": 0,
        "routed_scaling_factor": 1.0,
        "mlp_hidden_act": "relu2",
    }

    config = {}
    for dsl_key, hf_key in hf_mapping.items():
        if hf_key in hf_config and hf_config[hf_key] is not None:
            config[dsl_key] = hf_config[hf_key]

    payload = compile_model(NemotronHModel, config, raise_on_error=True)
    doc = json.loads(payload)
    assert doc["success"] is True

    ir_config = doc["modules"][0]["config"]

    for dsl_key in hf_mapping:
        assert dsl_key in ir_config, (
            f"DSL key '{dsl_key}' (mapped from HF '{hf_mapping[dsl_key]}') "
            f"missing from compiled IR config"
        )


def test_config_roundtrip_qwen3_moe():
    """All @hf_config mapped fields for Qwen3MoE should appear in the IR config."""
    from surogate.dsl.models.qwen3_moe import Qwen3MoEModel

    spec = Qwen3MoEModel._dsl_spec
    hf_mapping = spec.hf_config.param_mapping

    # Minimal Qwen3MoE config
    hf_config = {
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "moe_intermediate_size": 512,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 256,
        "norm_topk_prob": True,
    }

    config = {}
    for dsl_key, hf_key in hf_mapping.items():
        if hf_key in hf_config and hf_config[hf_key] is not None:
            config[dsl_key] = hf_config[hf_key]

    payload = compile_model(Qwen3MoEModel, config, raise_on_error=True)
    doc = json.loads(payload)
    assert doc["success"] is True

    ir_config = doc["modules"][0]["config"]

    for dsl_key in hf_mapping:
        assert dsl_key in ir_config, (
            f"DSL key '{dsl_key}' (mapped from HF '{hf_mapping[dsl_key]}') "
            f"missing from compiled IR config"
        )
