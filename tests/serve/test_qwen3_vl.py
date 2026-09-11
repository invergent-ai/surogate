"""Checkpoint-derived Qwen3-VL geometry, source mapping, and complete conversion."""

import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.recipe import source_requirements, validate_recipe_coverage
from surogate.serve.convert.qwen3_vl import convert, inventory, recipe
from surogate.serve.ingest import _flatten_text_config, converter_for_config


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


def moe_config_for():
    config = config_for()
    config.update(model_type="qwen3_vl_moe", architectures=["Qwen3VLMoeForConditionalGeneration"])
    config["text_config"].update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
                                 decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True)
    return config


@pytest.mark.parametrize("layout", ["input_major", "output_major", "separate"])
def test_moe_expert_layout_matches_output_major_banks(layout):
    from surogate.serve.convert.common.recipe import materialize_recipe
    g = inventory.geometry_from_config(moe_config_for())
    assert converter_for_config(_flatten_text_config(moe_config_for())).key == "qwen3_vl"
    metadata = inventory.geometry_block(g, token_domain=200)
    assert (metadata["experts"], metadata["experts_per_token"], metadata["intermediate"]) == (4, 2, 128)
    gate = torch.arange(g.experts * g.intermediate * g.hidden).reshape(g.experts, g.intermediate, g.hidden)
    up = -gate
    down = gate.transpose(1, 2).contiguous()
    prefix = "model.layers.0.mlp.experts."
    if layout == "input_major":
        tensors = {prefix + "gate_up_proj": torch.cat((gate, up), 1).transpose(1, 2).contiguous(),
                   prefix + "down_proj": down.transpose(1, 2).contiguous()}
    elif layout == "output_major":
        tensors = {prefix + "gate_up_proj": torch.cat((gate, up), 1), prefix + "down_proj": down}
    else:
        tensors = {prefix + "gate_proj.weight": gate, prefix + "up_proj.weight": up,
                   prefix + "down_proj.weight": down}
    class Reader:
        def has(self, name): return name in tensors
        def get(self, name): return tensors[name]
        def metadata(self, names):
            from types import SimpleNamespace
            return {name: SimpleNamespace(shape=tuple(tensors[name].shape)) for name in names}
    recipes = {r.object_name: r for r in recipe.build_recipes(g, reader=Reader())}
    for suffix, expected in (("routed_gate_up", torch.cat((gate, up), 1)), ("routed_down", down)):
        result = materialize_recipe(recipes["text/layers/0/moe/" + suffix], Reader())
        torch.testing.assert_close(result, expected.reshape(result.shape))


@pytest.mark.parametrize("change,match", [
    ({"num_experts_per_tok": 5}, "exceeds"), ({"norm_topk_prob": False}, "normalized"),
    ({"mlp_only_layers": [0]}, "every decoder layer"), ({"decoder_sparse_step": 2}, "every decoder layer"),
])
def test_moe_invalid_routing_is_rejected(change, match):
    config = moe_config_for()
    config["text_config"].update(change)
    with pytest.raises(ValueError, match=match):
        inventory.geometry_from_config(config)


@pytest.mark.parametrize("layout", ["input_major", "output_major"])
def test_complete_packed_moe_checkpoint_conversion(tmp_path, layout):
    from dataclasses import replace

    from surogate.serve.convert.common.recipe import resolve_options
    config = moe_config_for()
    if layout == "output_major":
        config["text_config"]["num_local_experts"] = config["text_config"].pop("num_experts")
    g = inventory.geometry_from_config(config)
    recipes = recipe.build_recipes(g)
    # Select the packed, transposed names used by the released HF checkpoint.
    def available(name):
        return ".experts." not in name or name.endswith(("experts.gate_up_proj", "experts.down_proj"))
    resolved = [replace(r, expression=resolve_options(r.expression, available)) for r in recipes]
    sources = source_requirements(resolved)
    tensors = {name.replace("model.layers.", "model.language_model.layers.")
                   .replace("model.embed_tokens.", "model.language_model.embed_tokens.")
                   .replace("model.norm.", "model.language_model.norm."):
               torch.full(source.shape, 0.125, dtype=torch.bfloat16) for name, source in sources.items()}
    if layout == "output_major":
        tensors = {name: value.transpose(1, 2).contiguous() if name.endswith(("experts.gate_up_proj", "experts.down_proj")) else value
                   for name, value in tensors.items()}
    save_file(tensors, tmp_path / "model.safetensors")
    for name, data in {"config.json": config, "tokenizer.json": {"model": {"vocab": {"a": 0, "b": 1}}},
                       "tokenizer_config.json": {"chat_template": "{{ messages }}"},
                       "preprocessor_config.json": {"patch_size": 16}}.items():
        (tmp_path / name).write_text(json.dumps(data))
    if layout == "output_major":
        (tmp_path / "preprocessor_config.json").unlink()
        (tmp_path / "processor_config.json").write_text(json.dumps({
            "processor_class": "Qwen3VLProcessor", "image_processor": {"patch_size": 16},
            "video_processor": {"patch_size": 16, "fps": 2}}))
    out = tmp_path / "moe.sinfer"
    convert.convert(tmp_path, out, device="cpu")
    with Artifact(out) as artifact:
        assert artifact.identity.architecture == "qwen3_vl_moe"
        assert artifact.geometry["experts"] == 4
        assert json.loads(bytes(artifact.payload("frontend/preprocessor_config.json")))["patch_size"] == 16
        if layout == "output_major":
            assert json.loads(bytes(artifact.payload("frontend/video_preprocessor_config.json")))["fps"] == 2
        assert artifact.find("text/layers/0/moe/routed_gate_up").shape == (1024, 256)
        assert artifact.vision_geometry["deepstack_layers"] == 2


def test_235b_moe_geometry_uses_its_wider_experts():
    config = moe_config_for()
    config["text_config"].update(hidden_size=4096, num_hidden_layers=94,
                                 num_attention_heads=64, num_key_value_heads=8, head_dim=128,
                                 num_experts=128, num_experts_per_tok=8, moe_intermediate_size=1536)
    config["text_config"]["rope_scaling"]["mrope_section"] = [24, 20, 20]
    config["vision_config"]["out_hidden_size"] = 4096
    g = inventory.geometry_from_config(config)
    specs = {s.name: s for s in inventory.build_tensor_specs(g)}
    assert specs["text/layers/93/moe/routed_gate_up"].shape == (393216, 4096)
    assert specs["text/layers/93/moe/routed_down"].shape == (524288, 1536)
    assert inventory.geometry_block(g, token_domain=256)["experts_per_token"] == 8
