"""A checkpoint's dimensions and identity are independent serving inputs."""

import importlib

import pytest

from surogate.serve.convert.common.recipe import validate_recipe_coverage


def config_for(family, *, layers=3, hidden=256, head_dim=64):
    return {
        "architectures": ["Qwen3ForCausalLM" if family == "qwen3" else "LlamaForCausalLM"],
        "model_type": family,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": 2 * hidden,
        "vocab_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": head_dim,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 123456.0,
    }


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("layers,hidden,head_dim", [(3, 256, 64), (5, 512, 128)])
def test_checkpoint_shapes_and_rename_invariance(family, layers, hidden, head_dim):
    inventory = importlib.import_module(f"surogate.serve.convert.{family}.inventory")
    recipe = importlib.import_module(f"surogate.serve.convert.{family}.recipe")
    config = config_for(family, layers=layers, hidden=hidden, head_dim=head_dim)
    geometry = recipe.geometry_from_config(config)
    specs = inventory.build_tensor_specs(geometry)
    recipes = recipe.build_recipes(geometry, tied_output_head=False)
    validate_recipe_coverage(recipes, specs)
    by_name = {spec.name: spec for spec in specs}
    assert by_name["text/token_embedding"].shape == (512, hidden)
    assert by_name[f"text/layers/{layers - 1}/attention/query_key_value"].shape == (
        8 * head_dim, hidden,
    )
    assert not any(spec.name.startswith(f"text/layers/{layers}/") for spec in specs)
    renamed = recipe.geometry_from_config({**config, "_name_or_path": "a-completely-new-name"})
    assert inventory.build_tensor_specs(renamed) == specs
    assert recipe.build_recipes(renamed, tied_output_head=False) == recipes
    assert geometry.declared.hf_config["rope_theta"] == 123456.0
    with pytest.raises(TypeError):
        inventory.build_tensor_specs()


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("key", ["hidden_size", "num_hidden_layers", "intermediate_size",
                                 "vocab_size", "num_attention_heads", "num_key_value_heads"])
def test_missing_checkpoint_dimensions_are_not_defaulted(family, key):
    recipe = importlib.import_module(f"surogate.serve.convert.{family}.recipe")
    config = config_for(family)
    del config[key]
    with pytest.raises(ValueError, match=key):
        recipe.geometry_from_config(config)


def test_llama_derives_head_width_only_when_division_is_exact():
    from surogate.serve.convert.llama.recipe import geometry_from_config

    config = config_for("llama")
    del config["head_dim"]
    assert geometry_from_config(config).head_dim == 64
    assert "head_dim" not in config
    with pytest.raises(ValueError, match="divisible"):
        geometry_from_config({**config, "hidden_size": 257})


def test_llama_explicit_head_width_and_optional_bias():
    from surogate.serve.convert.llama import convert, inventory

    config = {**config_for("llama", hidden=384, head_dim=128),
              "hidden_act": "silu", "rope_scaling": None, "tie_word_embeddings": False}
    geometry, report = convert.validate_config(config)
    specs = {spec.name: spec for spec in inventory.build_tensor_specs(geometry)}
    assert geometry.head_dim == 128
    assert geometry.query_size == 512
    assert report["text"]["head_dim"] == 128
    assert specs["text/layers/0/attention/output"].shape == (384, 512)
    with pytest.raises(ValueError, match="attention_bias"):
        convert.validate_config({**config, "attention_bias": True})


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_runtime_metadata_uses_checkpoint_execution_parameters(family):
    from surogate.serve.convert.common.checkpoint import dense_geometry
    from surogate.serve.artifact.geometry import REQUIRED_TEXT_FIELDS, validate_resolved_geometry

    recipe = importlib.import_module(f"surogate.serve.convert.{family}.recipe")
    geometry = recipe.geometry_from_config(config_for(family))
    metadata = dense_geometry(geometry, token_domain=500)
    assert metadata["attention_scale"] == 0.125
    assert metadata["max_context"] == 8192
    assert metadata["rms_epsilon"] == 1e-5
    assert metadata["rope_theta"] == 123456.0
    assert metadata["token_domain"] == 500
    for name in REQUIRED_TEXT_FIELDS:
        incomplete = dict(metadata)
        del incomplete[name]
        with pytest.raises(ValueError, match=f"missing geometry.{name}"):
            validate_resolved_geometry(incomplete)


def test_architecture_and_weight_profile_do_not_depend_on_artifact_label():
    from dataclasses import replace
    from surogate.serve.artifact.container import ArtifactIdentity, ResourceObject, encode_directory, parse_directory
    from surogate.serve.convert.qwen3_5.inventory import (
        NVFP4_MIXED_BF16, NVFP4_MLP_ONLY, profile_for, weights_id_for,
    )

    identity = ArtifactIdentity("my-checkpoint", "groupwise-int", architecture="qwen3")
    objects = (ResourceObject("fixture", "raw-bytes-v1", 0, 1),)
    parsed, _ = parse_directory(encode_directory(replace(identity, model_id="renamed"), objects))
    assert parsed.architecture == identity.architecture
    assert parsed.weights_id == identity.weights_id
    for profile in (NVFP4_MIXED_BF16, NVFP4_MLP_ONLY):
        assert profile_for(weights_id_for(profile)) == profile
    with pytest.raises(ValueError, match="unknown weights profile"):
        profile_for("nvfp4")


@pytest.mark.parametrize("layers,hidden", [(3, 384), (5, 768)])
@pytest.mark.parametrize("tied", [False, True])
def test_gemma_inventory_and_schedule_follow_the_checkpoint(layers, hidden, tied):
    from types import SimpleNamespace
    from surogate.serve.convert.gemma3 import convert, inventory, recipe

    config = {
        **config_for("gemma3", layers=layers, hidden=hidden),
        "architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text",
        "layer_types": ["sliding_attention"] + ["full_attention"] * (layers - 1),
        "query_pre_attn_scalar": 16, "sliding_window": 256,
        "rope_local_base_freq": 10000.0,
        "tie_word_embeddings": tied,
    }
    geometry = recipe.geometry_from_config(config)
    assert geometry.layer_types == tuple(config["layer_types"])
    specs, _ = inventory.active_specs(geometry=geometry, tied_output_head=tied)
    validate_recipe_coverage(recipe.build_recipes(geometry), specs)
    assert any(spec.name == "text/output_head" for spec in specs) != tied
    assert specs[0].shape == (512, hidden)
    metadata = convert.geometry_block(SimpleNamespace(geometry=geometry), token_domain=geometry.vocab - 1)
    assert metadata["attention_scale"] == 0.25  # checkpoint scalar, independent of head width
    assert metadata["sliding_window"] == 256
    assert metadata["max_context"] == 8192


def test_layer_schedule_is_serialized_and_validated():
    import json
    from surogate.serve.artifact.container import ArtifactIdentity, ArtifactError, ResourceObject, encode_directory

    identity = ArtifactIdentity("custom", "groupwise-int", architecture="gemma3")
    objects = (ResourceObject("fixture", "raw-bytes-v1", 0, 1),)
    metadata = {"layers": 3, "sliding_window": 256}
    types = ["full_attention", "sliding_attention", "full_attention"]
    encoded = encode_directory(identity, objects, geometry=metadata, layer_types=types)
    assert json.loads(encoded)["layer_types"] == types
    for invalid in ([], types[:2], ["full_attention", "typo", "full_attention"]):
        with pytest.raises(ArtifactError, match="layer_types"):
            encode_directory(identity, objects, geometry=metadata, layer_types=invalid)


@pytest.mark.parametrize("layers,hidden,head_dim", [(3, 384, 64), (5, 512, 128)])
def test_embedding_metadata_is_resolved_from_gguf(layers, hidden, head_dim):
    from types import SimpleNamespace
    from surogate.serve.convert.gemma_embedding import convert, inventory, recipe

    metadata = {
        "block_count": layers, "feed_forward_length": hidden * 2,
        "attention.head_count": 4, "attention.head_count_kv": 1,
        "attention.key_length": head_dim, "attention.sliding_window_pattern": 2,
        "context_length": 4096, "attention.layer_norm_rms_epsilon": 1e-5,
        "attention.sliding_window": 128, "rope.freq_base": 123456.0,
        "rope.freq_base_swa": 1234.0, "pooling_type": 1,
    }
    fields = {"general.architecture": SimpleNamespace(contents=lambda: "gemma-embedding")}
    for key, value in metadata.items():
        fields[f"gemma-embedding.{key}"] = SimpleNamespace(contents=lambda value=value: value)
    source = SimpleNamespace(fields=fields, tensor=lambda name: SimpleNamespace(shape=(512, hidden)))
    config = inventory.config_from_gguf(source)
    geometry = recipe.geometry_from_config(config)
    recipe.validate_recipe_coverage(recipe.build_recipes(geometry), geometry)
    assert inventory.declared_objects(geometry)[0]["shape"] == (512, hidden)
    serialized = convert.geometry_block(geometry, token_domain=geometry.vocab - 1)
    assert serialized["max_context"] == 4096
    assert serialized["attention_scale"] == head_dim ** -0.5
    assert serialized["rms_epsilon"] == 1e-5
    assert serialized["sliding_rope_theta"] == 1234.0
    assert geometry.layer_types[1] == "full_attention"
    assert geometry.layer_types[2] == "sliding_attention"
    with pytest.raises(ValueError, match="exactly one key/value head"):
        recipe.geometry_from_config({**config, "num_key_value_heads": 2})


def test_lfm_resolution_preserves_adjusted_ffn_schedule_and_untied_head():
    from surogate.serve.convert.lfm2 import convert, inventory, recipe
    from surogate.serve.convert.common.recipe import expression_sources

    config = {
        "architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2",
        "hidden_size": 384, "num_hidden_layers": 3, "num_attention_heads": 6,
        "num_key_value_heads": 2, "vocab_size": 512, "conv_L_cache": 5,
        "block_ff_dim": 640, "block_auto_adjust_ff_dim": True, "block_multiple_of": 128,
        "full_attn_idxs": [0, 2], "max_position_embeddings": 8192,
        "norm_eps": 1e-5, "rope_theta": 123456.0, "tie_word_embeddings": False,
    }
    geometry = inventory.geometry_from_config(config)
    assert geometry.intermediate == 512
    assert geometry.layer_types == ("full_attention", "linear_attention", "full_attention")
    specs = inventory.tensor_specs(inventory.declared_objects(geometry))
    recipes = recipe.build_recipes(geometry)
    validate_recipe_coverage(recipes, specs)
    head = next(item for item in recipes if item.object_name == "text/output_head")
    assert "lm_head.weight" in {item.name for item in expression_sources(head.expression)}
    metadata = convert._geometry_block(geometry, token_domain=geometry.vocab - 1)
    assert metadata["gdn_conv_kernel"] == 5
    assert metadata["attention_scale"] == 0.125
    assert metadata["rms_epsilon"] == 1e-5
    del config["block_ff_dim"]
    with pytest.raises(ValueError, match="intermediate_size"):
        inventory.geometry_from_config(config)


def test_moe_inventory_keeps_resolved_expert_dimensions():
    from surogate.serve.convert.qwen3_moe import convert, inventory, recipe

    config = {
        **config_for("qwen3_moe"),
        "architectures": ["Qwen3MoeForCausalLM"], "moe_intermediate_size": 128,
        "num_experts": 32, "num_experts_per_tok": 4,
    }
    geometry = recipe.geometry_from_config(config)
    assert geometry.intermediate == 128
    specs = inventory.build_tensor_specs(geometry)
    validate_recipe_coverage(recipe.build_recipes(geometry), specs)
    metadata = convert._geometry_block(geometry, token_domain=geometry.vocab - 1)
    assert metadata["experts"] == 32 and metadata["experts_per_token"] == 4
    assert metadata["max_context"] == 8192
    renamed = recipe.geometry_from_config({**config, "_name_or_path": "renamed"})
    assert inventory.build_tensor_specs(renamed) == specs
    for key in ("moe_intermediate_size", "num_experts", "num_experts_per_tok", "head_dim"):
        missing = dict(config)
        del missing[key]
        with pytest.raises(ValueError, match=key):
            recipe.geometry_from_config(missing)
