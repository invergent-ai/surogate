from __future__ import annotations

import json

import surogate.dsl.models  # noqa: F401 - registers nn-style models
from surogate.dsl.py_compiler import compile_model_for_hf


def _mini_lfm2_vl_config(**overrides):
    config = {
        "architectures": ["Lfm2VlForConditionalGeneration"],
        "model_type": "lfm2_vl",
        "image_token_id": 396,
        "downsample_factor": 2,
        "projector_hidden_size": 128,
        "tie_word_embeddings": False,
        "vision_config": {
            "model_type": "siglip2_vision_model",
            "hidden_size": 48,
            "num_hidden_layers": 2,
        },
        "text_config": {
            "model_type": "lfm2",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 192,
            "num_hidden_layers": 3,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 256,
            "norm_eps": 1e-5,
            "conv_bias": False,
            "conv_L_cache": 3,
            "block_multiple_of": 32,
            "block_ffn_dim_multiplier": 1.0,
            "block_auto_adjust_ff_dim": True,
            "layer_types": ["conv", "full_attention", "conv"],
        },
    }
    config.update(overrides)
    return config


def _compile(config=None):
    result = json.loads(compile_model_for_hf("Lfm2VlForConditionalGeneration", config or _mini_lfm2_vl_config()))
    assert result["success"], result.get("errors")
    return result["modules"][0]


def test_lfm2_vl_reads_its_geometry_from_the_nested_text_config():
    ir = _compile()

    assert ir["hf_config"]["architecture"] == "Lfm2VlForConditionalGeneration"
    config = ir["config"]
    assert config["hybrid_pattern"] == "CAC"
    assert config["n_attn_blocks"] == 1
    assert config["n_conv_blocks"] == 2
    # The text stack keeps LFM2's ff-dim adjustment (192 -> 128 at multiple 32).
    assert config["d_ff"] == 128


def test_lfm2_vl_carries_the_projector_geometry():
    config = _compile()["config"]

    assert config["image_token_id"] == 396
    assert config["downsample_factor"] == 2
    # Pixel-unshuffle by f widens the vision features by f**2 before the connector.
    assert config["projector_in_features"] == 48 * 2 * 2
    assert config["projector_hidden_size"] == 128


def test_lfm2_vl_scatters_image_features_into_the_embedding_stream():
    ir = _compile()

    inputs = ir["forward"]["inputs"] if "inputs" in ir["forward"] else []
    names = [i["name"] if isinstance(i, dict) else i for i in inputs]
    if names:
        assert "visual_embeds" in names and "visual_pos_masks" in names
    ops = [op["kernel_type"] for op in ir["forward"]["operations"]]
    assert any("scatter" in op for op in ops), ops


def test_lfm2_vl_weights_live_under_the_language_model_prefix():
    mappings = _compile()["hf_mapping"]

    assert mappings["embedding"] == "model.language_model.embed_tokens.weight"
    assert mappings["final_norm"] == "model.language_model.embedding_norm.weight"
    # Unlike Lfm2ForCausalLM, the VL head is a real tensor rather than tied.
    assert mappings["lm_head"] == "lm_head.weight"

    qkv = mappings["blocks[1].qkv_weight"]
    assert qkv["type"] == "fuse"
    assert qkv["sources"] == [
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.k_proj.weight",
        "model.language_model.layers.1.self_attn.v_proj.weight",
    ]
    assert mappings["blocks[0].conv_weight"] == "model.language_model.layers.0.conv.conv.weight"
    assert mappings["blocks[0].mlp_down_weight"] == "model.language_model.layers.0.feed_forward.w2.weight"
