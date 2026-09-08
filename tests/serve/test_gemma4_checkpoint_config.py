"""Gemma 4 shapes, schedules and optional features follow checkpoint configuration."""

from copy import deepcopy
import importlib

import pytest

from surogate.serve.convert.common.recipe import expression_sources, validate_recipe_coverage


def checkpoint(target, *, hidden=256, global_head_dim=64, nested=False, tied=True):
    text = {
        "hidden_size": hidden, "num_hidden_layers": 4, "intermediate_size": hidden * 2,
        "vocab_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 64, "global_head_dim": global_head_dim,
        "max_position_embeddings": 4096, "sliding_window": 128,
        "rms_norm_eps": 1e-4, "tie_word_embeddings": tied,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 20000.0, "rope_type": "default"},
            "full_attention": {"rope_theta": 500000.0, "rope_type": "proportional",
                               "partial_rotary_factor": 0.5},
        },
        "attention_k_eq_v": target != "gemma4_e",
    }
    if target == "gemma4_e":
        text.update(hidden_size_per_layer_input=32, vocab_size_per_layer_input=512,
                    num_kv_shared_layers=2, use_double_wide_mlp=True)
    if target == "gemma4_moe":
        text.update(enable_moe_block=True, num_experts=8, top_k_experts=2,
                    moe_intermediate_size=128)
    if nested:
        return {"architectures": ["Gemma4ForConditionalGeneration"], "text_config": text}
    return {"architectures": ["Gemma4ForCausalLM"], **text}


@pytest.mark.parametrize("target", ["gemma4", "gemma4_e", "gemma4_moe"])
@pytest.mark.parametrize("hidden,global_head_dim", [(256, 64), (384, 128)])
@pytest.mark.parametrize("nested,tied", [(False, True), (True, False)])
def test_shapes_schedule_and_recipes_share_checkpoint(target, hidden, global_head_dim, nested, tied):
    inventory = importlib.import_module(f"surogate.serve.convert.{target}.inventory")
    recipe = importlib.import_module(f"surogate.serve.convert.{target}.recipe")
    convert = importlib.import_module(f"surogate.serve.convert.{target}.convert")
    config = checkpoint(target, hidden=hidden, global_head_dim=global_head_dim,
                        nested=nested, tied=tied)
    original = deepcopy(config)
    geometry = inventory.geometry_from_config(config)
    specs = inventory.tensor_specs(inventory.stored_objects(geometry, tied_output_head=tied))
    recipes = recipe.build_recipes(geometry)
    validate_recipe_coverage(recipes, specs)
    assert config == original
    assert geometry.layer_types == ("sliding_attention", "full_attention") * 2
    by_name = {spec.name: spec for spec in specs}
    assert by_name["text/token_embedding"].shape == (512, hidden)
    assert by_name["text/layers/0/attention/query"].shape == (256, hidden)
    assert by_name["text/layers/1/attention/query"].shape == (4 * global_head_dim, hidden)
    assert ("text/output_head" in by_name) == (not tied)
    if not tied:
        head = next(r for r in recipes if r.object_name == "text/output_head")
        assert [s.name for s in expression_sources(head.expression)] == ["lm_head.weight"]
    metadata = convert._geometry_block(geometry, token_domain=geometry.vocab - 1)
    assert metadata["rms_epsilon"] == 1e-4
    assert metadata["attention_scale"] == 1.0
    assert metadata["max_context"] == 4096
    assert metadata["sliding_rope_theta"] == 20000.0
    assert metadata["rope_theta"] == 500000.0
    assert metadata["global_rotary_angles"] == global_head_dim // 4
    renamed = inventory.geometry_from_config({**config, "_name_or_path": "renamed-checkpoint"})
    assert inventory.tensor_specs(inventory.stored_objects(renamed, tied_output_head=tied)) == specs
    assert recipe.build_recipes(renamed) == recipes
    if target == "gemma4_e":
        assert metadata["kv_shared_layers"] == 2
        assert metadata["shared_kv_intermediate"] == 4 * hidden
        assert "text/layers/2/attention/key" not in by_name
        assert by_name["text/layers/2/mlp/gate"].shape == (4 * hidden, hidden)
    elif target == "gemma4_moe":
        assert metadata["experts"] == 8 and metadata["experts_per_token"] == 2
        assert metadata["intermediate"] == 128 and metadata["dense_intermediate"] == 2 * hidden
    else:
        assert not any("per_layer" in name for name in by_name)


@pytest.mark.parametrize("target", ["gemma4", "gemma4_e", "gemma4_moe"])
@pytest.mark.parametrize("key", ["hidden_size", "intermediate_size", "head_dim", "global_head_dim",
                                 "max_position_embeddings", "rms_norm_eps"])
def test_required_checkpoint_values_cannot_fall_back_to_training_defaults(target, key):
    inventory = importlib.import_module(f"surogate.serve.convert.{target}.inventory")
    config = checkpoint(target)
    del config[key]
    with pytest.raises(ValueError, match=key):
        inventory.geometry_from_config(config)


def test_shared_kv_requires_an_owner_of_each_attention_type():
    from surogate.serve.convert.gemma4_e.inventory import geometry_from_config

    config = checkpoint("gemma4_e")
    config["num_kv_shared_layers"] = 3
    with pytest.raises(ValueError, match="earlier owner"):
        geometry_from_config(config)


def test_expert_count_is_required_and_top_k_cannot_exceed_it():
    from surogate.serve.convert.gemma4_moe.inventory import geometry_from_config

    config = checkpoint("gemma4_moe")
    config["top_k_experts"] = 9
    with pytest.raises(ValueError, match="exceeds"):
        geometry_from_config(config)
    del config["num_experts"]
    with pytest.raises(ValueError, match="num_experts"):
        geometry_from_config(config)
