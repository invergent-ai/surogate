"""Checkpoint-derived Qwen3-VL geometry, source mapping, and complete conversion."""

import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.recipe import source_requirements, validate_recipe_coverage
from surogate.serve.convert.qwen3_vl import inventory, recipe, convert
from surogate.serve.ingest import converter_for_config, _flatten_text_config


def config_for():
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
            "vocab_size": 256, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
            "rope_theta": 5000000, "hidden_act": "silu", "attention_bias": False,
            "rope_scaling": {"rope_type": "default", "mrope_interleaved": True,
                             "mrope_section": [12, 10, 10]},
            "tie_word_embeddings": True, "eos_token_id": 1,
        },
        "vision_config": {
            "depth": 4, "hidden_size": 128, "intermediate_size": 256, "num_heads": 2,
            "in_channels": 3, "temporal_patch_size": 2, "patch_size": 16,
            "spatial_merge_size": 2, "num_position_embeddings": 16, "out_hidden_size": 256,
            "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [0, 2],
        },
    }


def test_dimensions_and_deepstack_follow_config():
    g = inventory.geometry_from_config(config_for())
    specs = inventory.build_tensor_specs(g)
    recipes = recipe.build_recipes(g)
    validate_recipe_coverage(recipes, specs)
    by_name = {s.name: s for s in specs}
    assert g.layers == 3 and g.vision["layers"] == 4
    assert g.vision["deepstack_layers"] == 2
    assert by_name["vision/layers/2/deepstack/norm/weight"].shape == (512,)
    assert by_name["vision/merger/norm/weight"].shape == (128,)
    assert by_name["vision/layers/2/deepstack/fc2"].shape == (256, 512)
    assert not any("layers/1/deepstack" in s.name for s in specs)
    sources = source_requirements(recipes)
    assert "model.visual.deepstack_merger_list.1.norm.weight" in sources
    assert "model.layers.0.self_attn.q_proj.weight" in sources
    assert "lm_head.weight" not in sources
    metadata = inventory.geometry_block(g, token_domain=200)
    assert [metadata[f"mrope_{axis}"] for axis in ("temporal", "height", "width")] == [12, 10, 10]
    assert metadata["token_domain"] == 200 and metadata["output_rows"] == 256
    assert inventory.MODEL_ID == "qwen3_vl"
    assert converter_for_config(_flatten_text_config(config_for())).key == "qwen3_vl"


@pytest.mark.parametrize("name", ["hidden_size", "num_hidden_layers", "intermediate_size", "head_dim",
                                 "num_attention_heads", "num_key_value_heads", "vocab_size"])
def test_missing_text_dimensions_never_use_dsl_defaults(name):
    c = config_for()
    del c["text_config"][name]
    with pytest.raises(ValueError, match=name):
        inventory.geometry_from_config(c)


@pytest.mark.parametrize("indexes", [[2, 0], [1, 1], [4], [True], [0, 1, 2, 3], None])
def test_invalid_deepstack_schedule(indexes):
    c = config_for()
    c["vision_config"]["deepstack_visual_indexes"] = indexes
    with pytest.raises(ValueError, match="deepstack_visual_indexes"):
        inventory.geometry_from_config(c)


@pytest.mark.parametrize("sections", [[10, 10, 10], [10, 11, 11], [12, 10], [True, 10, 21]])
def test_invalid_mrope_sections(sections):
    c = config_for()
    c["text_config"]["rope_scaling"]["mrope_section"] = sections
    with pytest.raises(ValueError, match="mrope_section"):
        inventory.geometry_from_config(c)


def test_untied_and_rope_parameters_config():
    c = config_for()
    c["tie_word_embeddings"] = c["text_config"]["tie_word_embeddings"] = False
    c["text_config"]["rope_parameters"] = c["text_config"].pop("rope_scaling")
    g = inventory.geometry_from_config(c)
    assert "lm_head.weight" in source_requirements(recipe.build_recipes(g))


def test_complete_vl_checkpoint_conversion(tmp_path):
    c = config_for()
    g = inventory.geometry_from_config(c)
    sources = source_requirements(recipe.build_recipes(g))
    # The real checkpoints nest text weights under language_model.
    tensors = {
        name.replace("model.layers.", "model.language_model.layers.")
            .replace("model.embed_tokens.", "model.language_model.embed_tokens.")
            .replace("model.norm.", "model.language_model.norm."):
        torch.full(source.shape, 0.125, dtype=torch.bfloat16)
        for name, source in sources.items()
    }
    save_file(tensors, tmp_path / "model.safetensors")
    for name, data in {
        "config.json": c, "tokenizer.json": {"model": {"vocab": {"a": 0, "b": 1}}},
        "tokenizer_config.json": {"chat_template": "{{ messages }}"},
        "preprocessor_config.json": {"patch_size": 16},
        "video_preprocessor_config.json": {"patch_size": 16},
    }.items():
        (tmp_path / name).write_text(json.dumps(data))
    output = tmp_path / "converted.sinfer"
    convert.convert(tmp_path, output, device="cpu")
    with Artifact(output) as artifact:
        assert artifact.identity.architecture == "qwen3_vl"
        assert artifact.geometry["hidden"] == 256
        assert artifact.geometry["mrope_temporal"] == 12
        assert artifact.vision_geometry["deepstack_layers"] == 2
        assert artifact.find("frontend/chat_template.jinja") is not None
        assert artifact.find("vision/layers/2/deepstack/norm/weight").shape == (512,)
