from __future__ import annotations

import json

import pytest

import surogate.dsl.models  # noqa: F401 - registers nn-style models
from surogate.dsl.py_compiler import compile_model_for_hf


def _mini_lfm2_config(**overrides):
    config = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 192,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "conv_bias": False,
        "conv_L_cache": 3,
        "block_multiple_of": 32,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": True,
        "layer_types": ["full_attention", "conv", "full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    config.update(overrides)
    return config


def _compile(config=None):
    result = json.loads(compile_model_for_hf("Lfm2ForCausalLM", config or _mini_lfm2_config()))
    assert result["success"], result.get("errors")
    return result["modules"][0]


def _module_by_name(ir, name):
    for module in [ir, *ir.get("submodules", [])]:
        if module.get("name") == name:
            return module
    raise AssertionError(f"Missing module {name}; got {[m.get('name') for m in [ir, *ir.get('submodules', [])]]}")


def _ops(module):
    return [op["kernel_type"] for op in module["forward"]["operations"]]


def test_lfm2_compiles_as_hybrid_attention_and_conv_model():
    ir = _compile()

    assert ir["hf_config"]["architecture"] == "Lfm2ForCausalLM"
    assert ir["hf_config"]["model_type"] == "lfm2"
    assert ir["config"]["n_attn_blocks"] == 2
    assert ir["config"]["n_conv_blocks"] == 1
    assert ir["config"]["d_ff"] == 128
    assert ir["config"]["M"] == 128

    forward_ops = _ops(ir)
    assert "qkv_qk_norm_rope" in forward_ops
    assert "mamba_conv1d" in forward_ops
    assert "swiglu" in forward_ops


def test_lfm2_ir_uses_adjusted_ffn_width_for_runtime_config():
    ir = _compile(
        _mini_lfm2_config(
            hidden_size=1024,
            intermediate_size=6656,
            num_attention_heads=16,
            num_key_value_heads=8,
            block_multiple_of=256,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=True,
        )
    )

    assert ir["config"]["d_ff"] == 4608
    assert ir["config"]["M"] == 4608


def test_lfm2_hf_weight_mappings_match_transformers_names():
    ir = _compile()
    mappings = ir["hf_mapping"]

    assert mappings["embedding"] == "model.embed_tokens.weight"
    assert mappings["final_norm"] == "model.embedding_norm.weight"
    assert mappings["lm_head"]["target"] == "embedding"

    attn_qkv = mappings["blocks[0].qkv_weight"]
    assert attn_qkv["type"] == "fuse"
    assert attn_qkv["sources"] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    assert mappings["blocks[0].q_norm_weight"] == "model.layers.0.self_attn.q_layernorm.weight"
    assert mappings["blocks[0].mlp_up_weight"]["sources"] == [
        "model.layers.0.feed_forward.w3.weight",
        "model.layers.0.feed_forward.w1.weight",
    ]

    assert mappings["blocks[1].conv_in_proj_weight"] == "model.layers.1.conv.in_proj.weight"
    assert mappings["blocks[1].conv_weight"] == "model.layers.1.conv.conv.weight"
    assert mappings["blocks[1].conv_out_proj_weight"] == "model.layers.1.conv.out_proj.weight"
    assert mappings["blocks[1].mlp_down_weight"] == "model.layers.1.feed_forward.w2.weight"


def test_lfm2_layer_types_default_from_full_attention_indices():
    ir = _compile(_mini_lfm2_config(layer_types=None, full_attn_idxs=[1]))

    assert ir["config"]["n_attn_blocks"] == 1
    assert ir["config"]["n_conv_blocks"] == 2
    assert ir["config"]["hybrid_pattern"] == "CAC"


def test_lfm2_pretrained_config_reads_norm_eps(tmp_path):
    try:
        import surogate._surogate as _surogate
    except ImportError:
        pytest.skip("surogate._surogate C++ extension not built")

    config_dir = tmp_path / "lfm2_config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(_mini_lfm2_config(norm_eps=7e-6)) + "\n")

    cfg = _surogate.PretrainedConfig.from_pretrained(str(config_dir), "bf16")

    assert cfg.rms_norm_eps == pytest.approx(7e-6)


def test_lfm2_pretrained_config_defaults_to_tied_embeddings(tmp_path):
    try:
        import surogate._surogate as _surogate
    except ImportError:
        pytest.skip("surogate._surogate C++ extension not built")

    config = _mini_lfm2_config()
    config.pop("tie_word_embeddings")
    config_dir = tmp_path / "lfm2_config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(config) + "\n")

    cfg = _surogate.PretrainedConfig.from_pretrained(str(config_dir), "bf16")

    assert cfg.tie_word_embeddings is True
