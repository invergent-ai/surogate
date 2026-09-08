"""Hybrid conversion must depend on the checkpoint, including storage and optional towers."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from surogate.serve.convert.qwen3_5 import inventory as inv, recipe
from surogate.serve.convert.common import qwen3_5 as checkpoint
from surogate.serve.convert.common.recipe import expression_shape, expression_sources


def config_for(*, hidden=128, head=128, nested=False, vision=False, tied=False):
    config = {
        "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5",
        "hidden_size": hidden, "num_hidden_layers": 4, "intermediate_size": 256,
        "vocab_size": 512, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": head, "linear_num_key_heads": 2, "linear_key_head_dim": 32,
        "linear_num_value_heads": 4, "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
        "layer_types": ["full_attention", "linear_attention", "full_attention", "linear_attention"],
        "max_position_embeddings": 8192, "rms_norm_eps": 2e-5, "tie_word_embeddings": tied,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 543210.,
                            "partial_rotary_factor": 64 / head,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    }
    if nested:
        config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": config}
    if vision:
        config["vision_config"] = {
            "depth": 2, "hidden_size": 96, "intermediate_size": 192, "num_heads": 3,
            "in_channels": 3, "temporal_patch_size": 1, "patch_size": 4,
            "spatial_merge_size": 2, "num_position_embeddings": 16, "out_hidden_size": hidden,
        }
    return config


@pytest.mark.parametrize("hidden,head", [(128, 128), (384, 256)])
@pytest.mark.parametrize("nested,tied,vision", [(False, False, False), (True, True, True)])
def test_objects_and_recipes_follow_config(hidden, head, nested, tied, vision):
    config = config_for(hidden=hidden, head=head, nested=nested, vision=vision, tied=tied)
    g = inv.geometry_from_config(config, token_domain=500)
    specs = {s.name: s for s in inv.build_tensor_specs(g)}
    recipes = {r.object_name: r for r in recipe.build_recipes(g)}
    assert specs.keys() == recipes.keys()
    assert all(expression_shape(recipes[name].expression) == spec.shape for name, spec in specs.items())
    assert g.full_attention_layers == (0, 2)
    assert specs["text/layers/1/gdn/norm"].shape == (64,)
    assert specs["text/draft_head"].shape == (500, hidden)
    output_sources = expression_sources(recipes["text/output_head"].expression)
    assert output_sources[0].name == ("model.embed_tokens.weight" if tied else "lm_head.weight")
    metadata = checkpoint.geometry_block(g)
    assert metadata["attention_scale"] == pytest.approx(head ** -.5)
    assert metadata["gdn_scale"] == pytest.approx(32 ** -.5)
    assert metadata["rms_epsilon"] == 2e-5
    assert metadata["rope_theta"] == 543210.
    if vision:
        assert specs["vision/patch_embedding"].shape == (96, 48)
        assert specs["vision/merger/fc2"].shape == (hidden, 384)
        assert inv.vision_tower(g)["layers"] == 2
    renamed = deepcopy(config)
    renamed["_name_or_path"] = "my-renamed-checkpoint-with-no-size"
    assert inv.build_tensor_specs(inv.geometry_from_config(renamed, token_domain=500)) == tuple(specs.values())


@pytest.mark.parametrize("missing", ["hidden_size", "head_dim", "linear_value_head_dim", "max_position_embeddings"])
def test_required_checkpoint_dimensions_do_not_fall_back(missing):
    config = config_for()
    del config[missing]
    with pytest.raises(ValueError, match=missing):
        inv.geometry_from_config(config)


def test_interval_starts_at_the_declared_interval():
    config = config_for()
    config.pop("layer_types")
    config["full_attention_interval"] = 2
    assert inv.geometry_from_config(config).full_attention_layers == (1, 3)


def _checkpoint_tensors(g):
    from surogate.serve.convert.common.recipe import source_requirements
    rs = recipe.build_recipes(g)
    return {name: torch.zeros(source.shape, dtype=torch.float32 if source.dtype == "F32" else torch.bfloat16)
            for name, source in source_requirements(rs).items()}


@pytest.mark.parametrize("profile", [inv.NVFP4_UNIFORM, inv.NVFP4_ALL, inv.NVFP4_MIXED_BF16, inv.NVFP4_MLP_ONLY, inv.FP8_BLOCK, inv.FP8_CHANNEL])
def test_quantized_plan_uses_observed_formats_and_dimensions(tmp_path, profile):
    from safetensors.torch import save_file
    from surogate.serve.convert.common.quant_scope import observed_scope
    from surogate.serve.convert.common.safetensors import ShardReader
    from surogate.serve.convert.qwen3_5.exports import quantized, recipe_nvfp4_uniform as matrices
    g = inv.geometry_from_config(config_for())
    tensors = _checkpoint_tensors(g)
    # Arbitrary exception layer: all attention stays BF16, while one MLP is NVFP4.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stem = "model.layers.2.mlp." + proj
        n, k = tensors.pop(stem + ".weight").shape
        tensors[stem + ".weight_packed"] = torch.zeros((n, k // 2), dtype=torch.uint8)
        tensors[stem + ".weight_scale"] = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn)
        tensors[stem + ".weight_global_scale"] = torch.tensor(2.)
        tensors[stem + ".input_global_scale"] = torch.tensor(3.)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with ShardReader.for_directory(tmp_path) as reader:
        g = replace(g, observed_scope=observed_scope(reader.names))
        plan = quantized.build(g, profile, quantized.Sources(reader))
        specs = {s.name: s for s in plan.tensors}
        assert specs["text/layers/2/mlp/gate_up"].format == inv.NVFP4
        assert specs["text/layers/0/attention/query_key_gate_value"].format == inv.BF16
        assert specs["text/layers/1/gdn/norm"].shape == (64,)
        assert len(plan.divisors) == 2
        assert specs["text/layers/2/mlp/gate_up"].shape == (512, 128)


@pytest.mark.parametrize("modelopt", [False, True])
def test_nvfp4_source_words_and_scale_conventions_are_preserved(tmp_path, modelopt):
    from safetensors.torch import save_file
    from surogate.serve.artifact.layouts import decode_nvfp4_words
    from surogate.serve.convert.common.safetensors import ShardReader
    from surogate.serve.convert.qwen3_5.exports import quantized, recipe_nvfp4_uniform as matrices
    stem = "model.layers.0.mlp.gate_proj"
    weight_field = "weight" if modelopt else "weight_packed"
    scale_field = "weight_scale_2" if modelopt else "weight_global_scale"
    input_field = "input_scale" if modelopt else "input_global_scale"
    packed = torch.arange(256, dtype=torch.uint8).repeat(16).reshape(128, 32)
    scales = torch.ones((128, 4), dtype=torch.float8_e4m3fn)
    save_file({stem + "." + weight_field: packed, stem + ".weight_scale": scales,
               stem + "." + scale_field: torch.tensor(.5 if modelopt else 2.),
               stem + "." + input_field: torch.tensor(.25 if modelopt else 4.)},
              str(tmp_path / "model.safetensors"))
    source = matrices.MatrixSource(stem, (128, 64))
    entry = matrices.Nvfp4WeightRecipe("text/layers/0/mlp/gate_up", (128, 64), (matrices._all(source),), (source,))
    with ShardReader.for_directory(tmp_path) as reader:
        sources = quantized.Sources(reader)
        payload = quantized.encode_matrix(entry, sources, "cpu")
        codes, decoded_scales, divisor = decode_nvfp4_words(payload, entry.shape)
        assert torch.equal(codes, packed)
        assert torch.equal(decoded_scales, scales.view(torch.uint8))
        assert float(divisor.reshape(())) == 2.
        assert quantized._same_divisor(sources, entry.parts, "input_scale") == 4.
        # 0x21 encodes 0.5 then 1.0, scaled by one and divided by two.
        assert torch.equal(sources.get(stem + ".weight")[1, 2:4], torch.tensor([.25, .5], dtype=torch.bfloat16))


@pytest.mark.parametrize("mtp,vision,tied", [(True, True, False), (False, False, True)])
def test_quantized_conversion_writes_complete_checkpoint_metadata(tmp_path, monkeypatch, mtp, vision, tied):
    from safetensors.torch import save_file
    import numpy as np
    from surogate.serve.artifact.container import Artifact
    from surogate.serve.convert.qwen3_5 import convert as base_convert
    from surogate.serve.convert.qwen3_5.exports import quantized
    config = config_for(vision=True, tied=tied)
    config["eos_token_id"] = 3
    config["quantization_config"] = {"quant_method": "compressed-tensors", "config_groups": {
        "projection": {"weights": {"num_bits": 4, "type": "float", "symmetric": True,
                                    "strategy": "tensor_group", "group_size": 16}},
    }}
    g = inv.geometry_from_config(config)
    tensors = _checkpoint_tensors(g)
    stem = "model.layers.2.mlp.down_proj"
    n, k = tensors.pop(stem + ".weight").shape
    tensors[stem + ".weight_packed"] = torch.zeros((n, k // 2), dtype=torch.uint8)
    tensors[stem + ".weight_scale"] = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn)
    tensors[stem + ".weight_global_scale"] = torch.tensor(2.)
    tensors[stem + ".input_global_scale"] = torch.tensor(3.)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(500)}}}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    ranking = tmp_path / "freq_corpus/fixtures/ranking/ranking.train.counts.i64"
    ranking.parent.mkdir(parents=True)
    np.arange(g.vocab, dtype="<i8").tofile(ranking)
    monkeypatch.setattr(base_convert, "_tools_root", lambda: tmp_path)
    output = quantized.convert(tmp_path, tmp_path / "artifact.sinfer", profile=inv.NVFP4_ALL,
                               device="cpu", mtp=mtp, vision=vision)
    report = json.loads((tmp_path / "artifact.sinfer.conversion.json").read_text())
    assert report["config_summary"]["mtp_layers"] == int(mtp)
    with Artifact.open(output) as artifact:
        assert artifact.geometry["hidden"] == 128
        assert artifact.geometry["token_domain"] == 500
        assert artifact.geometry["draft_vocab"] == 500
        assert artifact.geometry["mtp_layers"] == int(mtp)
        assert bool(artifact.vision_geometry) == vision
        assert any(obj.name.startswith("mtp/") for obj in artifact.objects) == mtp
        assert any(obj.name.startswith("vision/") for obj in artifact.objects) == vision
        if mtp:
            assert artifact.find("mtp/input_projection").format == inv.BF16
            assert artifact.find("mtp/layer/attention/query_key_gate_value").format == inv.BF16
        if vision:
            assert artifact.find("vision/layers/0/mlp/fc1").format == inv.BF16
        assert tuple(artifact.layer_types) == g.layer_types
        assert artifact.find("text/token_embedding").format == inv.FP8
        assert artifact.find("text/layers/2/mlp/down").format == inv.NVFP4
        assert artifact.find("text/layers/0/mlp/down").format == inv.BF16
        from surogate.serve.convert.qwen3_5.exports.verify_nvfp4_mixed_bf16 import verify_artifact
        verify_artifact(artifact, tmp_path)


def test_unsupported_convolution_is_refused_before_conversion():
    config = config_for()
    config["linear_conv_kernel_dim"] = 5
    with pytest.raises(ValueError, match="linear_conv_kernel_dim=4"):
        inv.geometry_from_config(config)


@pytest.mark.parametrize("component,object_name", [
    ("mtp", "mtp/input_projection"), ("vision", "vision/layers/0/mlp/fc1"),
])
def test_quantized_export_refuses_unsupported_optional_component_storage(tmp_path, component, object_name):
    from safetensors.torch import save_file
    from surogate.serve.convert.common.safetensors import ShardReader
    from surogate.serve.convert.qwen3_5.exports import quantized
    g = inv.geometry_from_config(config_for(vision=True))
    tensors = _checkpoint_tensors(g)
    selected = next(r for r in recipe.build_recipes(g) if r.object_name == object_name)
    source, = expression_sources(selected.expression)
    rows, columns = tensors.pop(source.name).shape
    tensors[source.name.removesuffix(".weight") + ".weight_packed"] = torch.zeros((rows, columns // 2), dtype=torch.uint8)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with ShardReader.for_directory(tmp_path) as reader:
        with pytest.raises(ValueError, match=f"--no-{component}"):
            quantized.build(g, inv.NVFP4_ALL, quantized.Sources(reader))
