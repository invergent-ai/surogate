"""LFM2 variants must preserve both feed-forward widths and the complete vision checkpoint."""

import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.recipe import source_requirements, validate_recipe_coverage
from surogate.serve.convert.lfm2.convert import _geometry_block
from surogate.serve.convert.lfm2_moe import inventory as moe, recipe as moe_recipe
from surogate.serve.convert.lfm2_vl import inventory as vl, recipe as vl_recipe
from surogate.serve.ingest import converter_for_config, _flatten_text_config


def text_config():
    return dict(hidden_size=256, vocab_size=256, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, intermediate_size=512, norm_eps=1e-5, conv_L_cache=3,
                max_position_embeddings=8192, rope_parameters={"rope_type": "default", "rope_theta": 234567.0},
                layer_types=["conv", "full_attention", "conv", "full_attention"],
                block_auto_adjust_ff_dim=False, tie_word_embeddings=True, eos_token_id=7)


def moe_config():
    return dict(**text_config(), architectures=["Lfm2MoeForCausalLM"], model_type="lfm2_moe",
                moe_intermediate_size=128, num_experts=8, num_experts_per_tok=2, num_dense_layers=2,
                norm_topk_prob=True, use_expert_bias=True, routed_scaling_factor=1.0)


def vl_config(*, projector_norm=True):
    return dict(architectures=["Lfm2VlForConditionalGeneration"], model_type="lfm2_vl",
                text_config={**text_config(), "model_type": "lfm2"},
                vision_config=dict(model_type="siglip2_vision_model", hidden_size=128,
                    intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                    num_patches=256, num_channels=3, patch_size=16, layer_norm_eps=1e-6,
                    hidden_act="gelu_pytorch_tanh"),
                image_token_id=255, downsample_factor=2, projector_hidden_size=384,
                projector_hidden_act="gelu", projector_bias=True, projector_use_layernorm=projector_norm)


def test_moe_preserves_dense_and_expert_widths_on_both_mixers():
    config = moe_config()
    g = moe.geometry_from_config(config)
    recipes = moe_recipe.build_recipes(g)
    specs = moe.tensor_specs(moe.declared_objects(g))
    validate_recipe_coverage(recipes, specs)
    shapes = {s.name: s.shape for s in specs}
    assert shapes["text/layers/0/mlp/down"] == (256, 512)
    assert shapes["text/layers/1/mlp/gate_up"] == (1024, 256)
    assert shapes["text/layers/2/moe/routed_gate_up"] == (8 * 256, 256)
    assert shapes["text/layers/3/moe/routed_down"] == (8 * 256, 128)
    sources = source_requirements(recipes)
    assert sources["model.layers.0.feed_forward.w2.weight"].shape == (256, 512)
    assert sources["model.layers.3.feed_forward.experts.0.w1.weight"].shape == (128, 256)
    assert sources["model.layers.2.feed_forward.expert_bias"].shape == (8,)
    metadata = _geometry_block(g, token_domain=256)
    assert (metadata["dense_intermediate"], metadata["intermediate"]) == (512, 128)
    assert metadata["leading_dense_layers"] == 2 and metadata["rope_theta"] == 234567.0
    assert converter_for_config(config).key == "lfm2_moe"


@pytest.mark.parametrize("norm", [True, False])
def test_vl_preserves_tower_projector_and_text_geometry(norm):
    config = vl_config(projector_norm=norm)
    g = vl.geometry_from_config(config)
    specs = vl.tensor_specs(vl.declared_objects(g))
    recipes = vl_recipe.build_recipes(g)
    validate_recipe_coverage(recipes, specs)
    shapes = {s.name: s.shape for s in specs}
    assert shapes["vision/patch_embedding"] == (128, 768)
    assert shapes["vision/merger/fc1"] == (384, 512)
    assert shapes["vision/merger/fc2"] == (256, 384)
    assert ("vision/merger/norm/weight" in shapes) is norm
    assert shapes["vision/post_norm/weight"] == (128,)
    assert g.vision["projector_norm"] == int(norm) and g.vision["rotary_dim"] == 0
    assert _geometry_block(g, token_domain=256)["rope_theta"] == 234567.0
    assert converter_for_config(_flatten_text_config(config)).key == "lfm2_vl"


@pytest.mark.parametrize("key", ["hidden_size", "num_hidden_layers", "num_experts",
    "moe_intermediate_size", "num_dense_layers", "norm_eps", "max_position_embeddings"])
def test_moe_missing_metadata_never_selects_a_preset(key):
    config = moe_config(); config.pop(key)
    with pytest.raises(ValueError): moe.geometry_from_config(config)


@pytest.mark.parametrize("key,value", [("num_experts_per_tok", 9), ("num_dense_layers", True),
    ("conv_bias", True), ("norm_topk_prob", False), ("use_expert_bias", False),
    ("routed_scaling_factor", 2), ("layer_types", ["conv"]), ("norm_eps", float("nan"))])
def test_moe_rejects_inconsistent_or_unimplemented_configuration(key, value):
    config = moe_config(); config[key] = value
    with pytest.raises(ValueError): moe.geometry_from_config(config)


@pytest.mark.parametrize("family", ["lfm2_moe", "lfm2_vl"])
def test_complete_variant_conversion(tmp_path, family):
    from importlib import import_module
    inventory = moe if family == "lfm2_moe" else vl
    recipe = moe_recipe if family == "lfm2_moe" else vl_recipe
    config = moe_config() if family == "lfm2_moe" else vl_config()
    geometry = inventory.geometry_from_config(config)
    requirements = source_requirements(recipe.build_recipes(geometry))
    # Store the flattened text names accepted by the common reader; every source has a distinct value.
    save_file({name: torch.full(source.shape, (i % 9 + 1) / 32, dtype=torch.bfloat16)
               for i, (name, source) in enumerate(requirements.items())}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    (tmp_path / "tokenizer_config.json").write_text('{}')
    if family == "lfm2_vl":
        (tmp_path / "processor_config.json").write_text(json.dumps({
            "image_processor": {"image_processor_type": "Lfm2VlImageProcessorFast", "resample": 2}}))
    output = tmp_path / "model.sinfer"
    import_module(f"surogate.serve.convert.{family}.convert").convert(tmp_path, output, device="cpu")
    with Artifact(output) as artifact:
        assert artifact.identity.architecture == family
        assert artifact.geometry["hidden"] == 256
        if family == "lfm2_vl":
            assert artifact.vision_geometry["siglip2"] == 1
            assert artifact.find("vision/merger/fc1").shape == (384, 512)
        else:
            assert artifact.geometry["experts"] == 8
