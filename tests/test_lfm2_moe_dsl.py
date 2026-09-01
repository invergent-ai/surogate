from __future__ import annotations

import json

import surogate.dsl.models  # noqa: F401 - registers nn-style models
from surogate.dsl.py_compiler import compile_model_for_hf


def _mini_lfm2_moe_config(**overrides):
    config = {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 192,
        "moe_intermediate_size": 96,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "num_dense_layers": 2,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "use_expert_bias": True,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": True,
        "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention"],
    }
    config.update(overrides)
    return config


def _compile(config=None):
    result = json.loads(compile_model_for_hf("Lfm2MoeForCausalLM", config or _mini_lfm2_moe_config()))
    assert result["success"], result.get("errors")
    return result["modules"][0]


def test_lfm2_moe_splits_layers_over_operator_and_ffn_axes():
    ir = _compile()

    assert ir["hf_config"]["architecture"] == "Lfm2MoeForCausalLM"
    assert ir["hf_config"]["model_type"] == "lfm2_moe"
    config = ir["config"]
    # Layers 0-1 keep a dense feed-forward, 2-5 are sparse; the operator axis is
    # independent of that split.
    # C/A are dense conv/attention, c/a their MoE counterparts.
    assert config["hybrid_pattern"] == "CCacca"
    assert config["n_conv_blocks"] == 2
    assert config["n_attn_blocks"] == 0
    assert config["n_conv_moe_blocks"] == 2
    assert config["n_attention_moe_blocks"] == 2
    assert config["n_conv_blocks"] + config["n_attn_blocks"] == config["num_dense_layers"]


def test_lfm2_moe_uses_raw_intermediate_sizes():
    # LFM2-MoE drops LFM2's ff-dim adjustment: dense layers use
    # intermediate_size verbatim and the experts their own width.
    config = _compile()["config"]
    assert config["d_ff"] == 192
    assert config["moe_d_ff"] == 96


def test_lfm2_moe_maps_per_expert_checkpoint_tensors():
    mappings = _compile()["hf_mapping"]

    experts = mappings["blocks[2].experts_gate_up"]
    assert experts["type"] == "stack_experts"
    assert experts["pattern"] == "model.layers.2.feed_forward.experts.{expert}.w1.weight"
    # The checkpoint names the gate/up pair w1/w3, so the up side must be stated
    # explicitly -- the default derivation only knows gate_proj/up_proj.
    assert experts["fuse_gate_up"] is True
    assert experts["up_pattern"] == "model.layers.2.feed_forward.experts.{expert}.w3.weight"
    assert mappings["blocks[2].experts_down"]["pattern"] == "model.layers.2.feed_forward.experts.{expert}.w2.weight"
    assert mappings["blocks[2].router_weight"] == "model.layers.2.feed_forward.gate.weight"


def test_lfm2_moe_dense_layers_keep_the_fused_mlp():
    mappings = _compile()["hf_mapping"]

    dense = mappings["blocks[0].mlp_up_weight"]
    assert dense["type"] == "fuse"
    assert dense["sources"] == [
        "model.layers.0.feed_forward.w3.weight",
        "model.layers.0.feed_forward.w1.weight",
    ]
    assert mappings["blocks[0].mlp_down_weight"] == "model.layers.0.feed_forward.w2.weight"
    # A dense layer has no router, and a sparse layer no fused MLP.
    assert "blocks[0].router_weight" not in mappings
    assert "blocks[2].mlp_up_weight" not in mappings


def test_lfm2_moe_shares_the_lfm2_operator_mappings():
    mappings = _compile()["hf_mapping"]

    attention = mappings["blocks[5].qkv_weight"]
    assert attention["type"] == "fuse"
    assert attention["sources"][0] == "model.layers.5.self_attn.q_proj.weight"
    assert mappings["blocks[5].out_weight"] == "model.layers.5.self_attn.out_proj.weight"
    assert mappings["blocks[4].conv_weight"] == "model.layers.4.conv.conv.weight"
    assert mappings["final_norm"] == "model.embedding_norm.weight"
    assert mappings["lm_head"]["target"] == "embedding"


def test_lfm2_moe_accepts_full_attn_idxs_instead_of_layer_types():
    # HF's Lfm2MoeConfig requires layer_types, but a checkpoint carrying only the
    # older full_attn_idxs key still describes the same model.
    config = _mini_lfm2_moe_config()
    del config["layer_types"]
    config["full_attn_idxs"] = [2, 5]
    assert _compile(config)["config"]["hybrid_pattern"] == "CCacca"


def test_lfm2_moe_rejects_unknown_layer_type():
    config = _mini_lfm2_moe_config(layer_types=["conv", "conv", "mamba", "conv", "conv", "full_attention"])
    result = json.loads(compile_model_for_hf("Lfm2MoeForCausalLM", config))
    assert not result["success"]
