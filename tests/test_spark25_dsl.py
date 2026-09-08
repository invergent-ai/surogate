"""Spark training graph and its shared serving declaration."""

import json

import pytest

from surogate.dsl.models.spark2_5 import Spark2_5Model
from surogate.dsl.py_compiler import compile_model_for_hf
from surogate.serve.convert.spark2_5.recipe import build_recipes, geometry_from_config


def config():
    return dict(
        architectures=["Spark2_5ForCausalLM"],
        model_type="spark2_5",
        hidden_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=192,
        vocab_size=512,
        num_hidden_layers=4,
        max_position_embeddings=2048,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=64,
        rms_norm_eps=1e-6,
        hidden_act="gelu",
        attention_bias=False,
        mlp_bias=False,
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        tie_word_embeddings=True,
        rope_parameters={
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
        },
    )


def compile_config(cfg):
    result = json.loads(compile_model_for_hf("Spark2_5ForCausalLM", cfg))
    assert result["success"], result.get("errors")
    return result["modules"][0]


def test_graph_preserves_head_gate_exact_gelu_and_mixed_residuals():
    model = compile_config(config())
    ops = model["forward"]["operations"]
    attention = [op for op in ops if op["kernel_type"] == "flash_attention"]
    assert [op["attrs"].get("window_size", 0) for op in attention] == [64, 64, 64, 0]
    rope = [op for op in ops if op["kernel_type"] == "rope"]
    assert [op["attrs"]["rotary_dim"] for op in rope] == ["D", "D", "D", 8]
    assert len([op for op in ops if op["kernel_type"] == "sigmoid"]) == 4
    gelu = [op for op in ops if op["kernel_type"] == "gelu"]
    assert len(gelu) == 4 and all(op["attrs"]["approximate"] == "none" for op in gelu)
    assert not any(op["kernel_type"] in ("gelu_glu", "swiglu", "qkv_qk_norm") for op in ops)
    slots = {slot["name"]: slot for slot in model["activation_layout"]["slots"]}
    for name in ("res_ffn", "res_att", "residual_final"):
        assert slots[name]["dtype"] == "fp32"
    # The raw attention must survive the gate for its weight gradient.
    assert "att_flat" not in slots["att"].get("aliases", [])
    assert model["config"]["full_rope_theta"] == 5000000
    assert model["config"]["sliding_rope_theta"] == 10000
    assert model["config"]["residual_fp32"] is True


def test_serving_sources_and_shapes_are_derived_from_training():
    cfg = config()
    geometry = geometry_from_config(cfg)
    params = geometry.declared.params
    assert geometry.declared.dims(params["blocks[0].qkv_weight"]) == (256, 96)
    assert geometry.declared.dims(params["blocks[0].out_weight"]) == (96, 128)
    recipes = {item.object_name: item.expression for item in build_recipes(geometry)}
    assert recipes["text/layers/0/attention/output_gate"].name == "model.layers.0.self_attn.g_proj.weight"
    assert recipes["text/output_head"].name == "model.embedding.weight"
    assert Spark2_5Model._serve_block_schedule_(geometry.declared.config) == ["sliding"] * 3 + ["full"]


@pytest.mark.parametrize("key", ["hidden_size", "head_dim", "layer_types", "rope_parameters", "sliding_window"])
def test_missing_geometry_cannot_fall_back_to_a_model_size(key):
    cfg = config()
    cfg.pop(key)
    result = json.loads(compile_model_for_hf("Spark2_5ForCausalLM", cfg))
    assert not result["success"]


@pytest.mark.parametrize("hidden,layers,heads,kv,ffn", [(2048, 28, 8, 2, 6656), (2560, 36, 16, 4, 10240)])
def test_published_shapes_compile(hidden, layers, heads, kv, ffn):
    cfg = config()
    cfg.update(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=256,
        intermediate_size=ffn,
        layer_types=(["sliding_attention"] * 3 + ["full_attention"]) * (layers // 4),
    )
    model = compile_config(cfg)
    assert model["config"]["d_model"] == hidden
    assert model["config"]["rotary_dim"] == 64
