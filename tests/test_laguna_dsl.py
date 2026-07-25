"""CPU-only DSL compile tests for the Laguna model (LagunaForCausalLM)."""

from __future__ import annotations

import json

import pytest

import surogate.dsl.models  # noqa: F401 - registers nn-style models
from surogate.dsl.py_compiler import compile_model_for_hf


def _mini_laguna_config(**overrides):
    config = {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 192,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
        "gating": "per-head",
        "sliding_window": 128,
        "tie_word_embeddings": False,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse"],
        "num_attention_heads_per_layer": [4, 8, 8, 8, 4],
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 512,
                "beta_slow": 1.0,
                "beta_fast": 64.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    }
    config.update(overrides)
    return config


def _compile(config=None):
    result = json.loads(compile_model_for_hf("LagunaForCausalLM", config or _mini_laguna_config()))
    assert result["success"], result.get("errors")
    return result["modules"][0]


def _ops(module):
    return [op["kernel_type"] for op in module["forward"]["operations"]]


def test_laguna_compiles_as_hybrid_dense_sparse_model():
    ir = _compile()

    assert ir["hf_config"]["architecture"] == "LagunaForCausalLM"
    assert ir["hf_config"]["model_type"] == "laguna"
    assert ir["config"]["n_full_dense_blocks"] == 1
    assert ir["config"]["n_full_sparse_blocks"] == 1
    assert ir["config"]["n_sliding_dense_blocks"] == 0
    assert ir["config"]["n_sliding_sparse_blocks"] == 3

    forward_ops = _ops(ir)
    assert "softplus" in forward_ops
    assert "moe_sigmoid" in forward_ops
    assert "moe_topk" in forward_ops
    assert "flash_attention" in forward_ops
    assert "swiglu" in forward_ops
    # one softplus attention gate per layer
    assert forward_ops.count("softplus") == 5
    # sigmoid routing only on the 4 sparse layers
    assert forward_ops.count("moe_sigmoid") == 4


def test_laguna_per_layer_param_shapes():
    ir = _compile()
    params = ir["forward"]["params"]

    # Layer 0: full attention (4 heads x 16) + dense MLP (fused up+gate)
    assert params["blocks[0].q_proj_weight"]["shape"] == [64, 64]
    assert params["blocks[0].g_proj_weight"]["shape"] == [4, 64]
    assert params["blocks[0].mlp_up_weight"]["shape"] == [384, 64]
    assert params["blocks[0].mlp_down_weight"]["shape"] == [64, 192]
    # full-attention rope: partial_rotary_factor 0.5 → rotary_dim 8 → 4 angle pairs
    assert params["blocks[0].rope_freqs"]["shape"] == [1024, 4, 2]

    # Layer 1: sliding attention (8 heads x 16) + sparse MoE
    assert params["blocks[1].q_proj_weight"]["shape"] == [128, 64]
    assert params["blocks[1].g_proj_weight"]["shape"] == [8, 64]
    assert params["blocks[1].k_proj_weight"]["shape"] == [32, 64]
    assert params["blocks[1].router_weight"]["shape"] == [8, 64]
    assert params["blocks[1].e_score_correction_bias"]["shape"] == [8]
    assert params["blocks[1].experts_gate_up"]["shape"] == [8, 64, 64]
    assert params["blocks[1].experts_down"]["shape"] == [8, 64, 32]
    assert params["blocks[1].shared_expert_gate"]["shape"] == [32, 64]
    # sliding rope: full-rotary → 8 angle pairs
    assert params["blocks[1].rope_freqs"]["shape"] == [1024, 8, 2]


def test_laguna_topk_uses_correction_bias_and_scaling():
    ir = _compile()
    topk_ops = [op for op in ir["forward"]["operations"] if op["kernel_type"] == "moe_topk"]
    assert topk_ops, "expected moe_topk ops"
    for op in topk_ops:
        assert any("e_score_correction_bias" in inp for inp in op["inputs"])
        attrs = op.get("attributes") or op.get("attrs") or {}
        assert attrs.get("normalize") is True
        assert attrs.get("scaling_factor") == pytest.approx(2.5)


def test_laguna_sliding_window_only_on_sliding_layers():
    ir = _compile()
    fa_ops = [op for op in ir["forward"]["operations"] if op["kernel_type"] == "flash_attention"]
    assert len(fa_ops) == 5
    windows = {}
    for op in fa_ops:
        layer = next(inp for inp in op["inputs"] if inp.startswith("blocks[")).split(".")[0]
        attrs = op.get("attributes") or op.get("attrs") or {}
        windows[layer] = attrs.get("window_size")
    assert windows["blocks[0]"] is None
    assert windows["blocks[4]"] is None
    assert windows["blocks[1]"] == 128
    assert windows["blocks[2]"] == 128
    assert windows["blocks[3]"] == 128


def test_laguna_hf_weight_mappings_match_transformers_names():
    ir = _compile()
    mappings = ir["hf_mapping"]

    assert mappings["embedding"] == "model.embed_tokens.weight"
    assert mappings["final_norm"] == "model.norm.weight"
    assert mappings["lm_head"] == "lm_head.weight"

    assert mappings["blocks[0].q_proj_weight"] == "model.layers.0.self_attn.q_proj.weight"
    assert mappings["blocks[0].g_proj_weight"] == "model.layers.0.self_attn.g_proj.weight"
    assert mappings["blocks[0].q_norm_weight"] == "model.layers.0.self_attn.q_norm.weight"
    assert mappings["blocks[0].ln1_weight"] == "model.layers.0.input_layernorm.weight"
    assert mappings["blocks[0].ln2_weight"] == "model.layers.0.post_attention_layernorm.weight"

    # Dense MLP: fused [up; gate]
    assert mappings["blocks[0].mlp_up_weight"]["sources"] == [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]
    assert mappings["blocks[0].mlp_down_weight"] == "model.layers.0.mlp.down_proj.weight"

    # Sparse MoE
    assert mappings["blocks[1].router_weight"] == "model.layers.1.mlp.gate.weight"
    assert mappings["blocks[1].e_score_correction_bias"] == "model.layers.1.mlp.experts.e_score_correction_bias"
    assert mappings["blocks[1].experts_gate_up"]["pattern"] == "model.layers.1.mlp.experts.{expert}.gate_proj.weight"
    assert mappings["blocks[1].experts_gate_up"]["fuse_gate_up"] is True
    assert mappings["blocks[1].experts_down"]["pattern"] == "model.layers.1.mlp.experts.{expert}.down_proj.weight"
    assert mappings["blocks[1].shared_expert_gate"] == "model.layers.1.mlp.shared_expert.gate_proj.weight"
    assert mappings["blocks[1].shared_expert_up"] == "model.layers.1.mlp.shared_expert.up_proj.weight"
    assert mappings["blocks[1].shared_expert_down"] == "model.layers.1.mlp.shared_expert.down_proj.weight"


def test_laguna_rejects_mixed_heads_within_layer_type():
    config = _mini_laguna_config(num_attention_heads_per_layer=[4, 8, 4, 8, 4])
    result = json.loads(compile_model_for_hf("LagunaForCausalLM", config))
    assert not result["success"]


def test_laguna_rejects_router_softcapping():
    config = _mini_laguna_config(moe_router_logit_softcapping=30.0)
    result = json.loads(compile_model_for_hf("LagunaForCausalLM", config))
    assert not result["success"]


def test_laguna_per_element_gating():
    ir = _compile(_mini_laguna_config(gating="per-element"))
    params = ir["forward"]["params"]
    # per-element gate: g_proj outputs Hq * D
    assert params["blocks[0].g_proj_weight"]["shape"] == [64, 64]
    assert params["blocks[1].g_proj_weight"]["shape"] == [128, 64]
