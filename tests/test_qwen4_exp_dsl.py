"""CPU-only DSL compile tests for Qwen3.8-Flash-Next (qwen4_exp).

No GPU, no weights, no C++ extension: compiles a mini config through the Python DSL and
asserts on the emitted IR — block schedule, hyper-connection wiring, the sigmoid GDN
gate, MoE renormalisation, and the HF weight mapping (including that there is NO final
norm: hyper-connections replace every layer norm).

Run: pytest tests/test_qwen4_exp_dsl.py -v --no-header
"""

from __future__ import annotations

import json

import pytest

from surogate.dsl.py_compiler import compile_model_for_hf


def _mini_text_config() -> dict:
    """4 layers of the real 3x(linear)+1x(full) pattern, scaled down."""

    return {
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 512,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 6,  # ratio 3, like the real 48/16
        "hc_count": 4,
        "hc_lowrank": 8,
        "output_gate_type": "sigmoid",
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_conv_kernel_size": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "mtp_num_hidden_layers": 1,
    }


def _compile(arch: str, config: dict) -> dict:
    raw = compile_model_for_hf(arch, config)
    result = json.loads(raw) if isinstance(raw, str) else raw
    assert result.get("success"), f"DSL compilation failed: {result.get('errors')}"
    return result


def _model_config(ir: dict) -> dict:
    if ir.get("config"):
        return ir["config"]
    mods = ir.get("modules") or []
    return (mods[0].get("config") or {}) if mods else {}


def _hf_mapping(ir: dict) -> dict:
    if ir.get("hf_mapping"):
        return ir["hf_mapping"]
    mods = ir.get("modules") or []
    for m in mods:
        if m.get("hf_mapping"):
            return m["hf_mapping"]
    return {}


@pytest.fixture(scope="module")
def causal_ir() -> dict:
    cfg = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text", **_mini_text_config()}
    return _compile("Qwen4ExpForCausalLM", cfg)


def _ir_text(ir: dict) -> str:
    return json.dumps(ir)


class TestQwen4ExpCompile:
    def test_block_schedule(self, causal_ir):
        config = _model_config(causal_ir)
        assert config.get("n_mamba_blocks") == 3
        assert config.get("n_attn_blocks") == 1
        # The sigmoid GDN gate decision is visible in the runtime config.
        assert bool(config.get("gdn_gate_sigmoid"))

    def test_deferred_subsystem_config_captured(self, causal_ir):
        """PLE/indexer/MTP hyperparameters must reach the runtime config even though the
        subsystems are not in the training graph yet — the serve-spec generator reads
        them from here."""

        config = _model_config(causal_ir)
        assert bool(config.get("has_ple"))
        assert config.get("indexer_budget") == 2048
        assert config.get("indexer_compress_ratio") == 4
        assert config.get("mtp_num_hidden_layers") == 1

    def test_sigmoid_gdn_gate_in_graph(self, causal_ir):
        text = _ir_text(causal_ir)
        assert "mamba_gated_rmsnorm" in text
        assert '"gate_activation": "sigmoid"' in text
        assert "chunk_gated_delta_rule" in text

    def test_hyper_connection_graph(self, causal_ir):
        text = _ir_text(causal_ir)
        # Mix + combine wiring exists for both sublayers and the output head.
        assert "hc_attn_normed" in text
        assert "hc_ffn_normed" in text
        assert "hc_attn_inject_proj" in text
        assert "output_hc_" in text
        # No pre-norm modules: hyper-connections replace them.
        assert "attn_norm_weight" not in text
        assert "ln1_weight" not in text

    def test_hf_mapping(self, causal_ir):
        mapping = _hf_mapping(causal_ir)
        assert mapping.get("hc_attn_norm") == "model.layers.{layer}.attn_hyper_connection.hc_norm.weight"
        assert mapping.get("hc_ffn_inject") == "model.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight"
        assert mapping.get("output_hc_norm") == "model.hyper_connection_mixer.hc_norm.weight"
        assert mapping.get("experts_gate_up") == "model.layers.{layer}.mlp.experts.gate_up_proj"
        assert mapping.get("shared_expert_gate_proj_weight") == "model.layers.{layer}.mlp.shared_expert_gate.weight"
        assert mapping.get("lin_in_proj_qkv_weight") == "model.layers.{layer}.linear_attn.in_proj_qkv.weight"
        assert mapping.get("q_norm_weight") == "model.layers.{layer}.self_attn.q_norm.weight"
        assert mapping.get("embedding") == "model.embed_tokens.weight"
        assert mapping.get("lm_head") == "lm_head.weight"
        # There is no final norm in this architecture.
        assert "final_norm" not in mapping

    def test_conditional_variant(self):
        cfg = {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {"model_type": "qwen4_exp_text", **_mini_text_config()},
            "vision_config": {"model_type": "qwen4_exp", "depth": 1},
        }
        ir = _compile("Qwen4ExpForConditionalGeneration", cfg)
        mapping = _hf_mapping(ir)
        assert mapping.get("hc_attn_norm") == "model.language_model.layers.{layer}.attn_hyper_connection.hc_norm.weight"
        assert mapping.get("embedding") == "model.language_model.embed_tokens.weight"
        assert mapping.get("output_hc_up") == "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"

    def test_rejects_bad_value_head_ratio(self):
        cfg = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text", **_mini_text_config()}
        cfg["linear_num_value_heads"] = 5  # not divisible by 2 key heads
        raw = compile_model_for_hf("Qwen4ExpForCausalLM", cfg)
        result = json.loads(raw) if isinstance(raw, str) else raw
        assert not result.get("success")

    def test_rejects_bad_gate_type(self):
        cfg = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text", **_mini_text_config()}
        cfg["output_gate_type"] = "gelu"
        raw = compile_model_for_hf("Qwen4ExpForCausalLM", cfg)
        result = json.loads(raw) if isinstance(raw, str) else raw
        assert not result.get("success")
