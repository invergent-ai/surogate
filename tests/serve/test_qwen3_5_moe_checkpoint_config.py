"""Hybrid MoE and its separate drafter must resolve independently of checkpoint names."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from tests.serve.test_qwen3_5_checkpoint_config import config_for as dense_config
from surogate.serve.convert.qwen3_5_moe import inventory as inv, recipe
from surogate.serve.convert.common import dflash
from surogate.serve.convert.common.recipe import expression_shape, source_requirements
from surogate.serve.artifact.geometry import validate_dflash_geometry


def config_for(**kwargs):
    c = dense_config(**kwargs)
    c["architectures"] = ["Qwen3_5MoeForConditionalGeneration" if "text_config" in c else "Qwen3_5MoeForCausalLM"]
    c["model_type"] = "qwen3_5_moe"
    text = c.get("text_config", c)
    text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
                shared_expert_intermediate_size=128)
    return c


def draft_config(g, *, layers=3, head=64, targets=(0, 2)):
    return dict(hidden_size=g.hidden, vocab_size=g.vocab, num_target_layers=g.layers,
                num_hidden_layers=layers, intermediate_size=192, head_dim=head,
                num_attention_heads=4, num_key_value_heads=2, sliding_window=128,
                max_position_embeddings=4096, layer_types=["sliding_attention"] * (layers - 1) + ["full_attention"],
                rms_norm_eps=3e-5, rope_parameters={"rope_theta": 234567.},
                dflash_config={"block_size": 8, "mask_token_id": 500, "target_layer_ids": list(targets)})


@pytest.mark.parametrize("hidden,experts,width,shared", [(128, 4, 64, 128), (384, 8, 192, 256)])
@pytest.mark.parametrize("nested,vision,tied", [(False, False, False), (True, True, True)])
def test_inventory_and_recipes_follow_checkpoint(hidden, experts, width, shared, nested, vision, tied):
    c = config_for(hidden=hidden, nested=nested, vision=vision, tied=tied)
    c.get("text_config", c).update(num_experts=experts, moe_intermediate_size=width,
                                   shared_expert_intermediate_size=shared)
    g = inv.geometry_from_config(c, token_domain=500)
    specs = {s.name: s for s in inv.build_tensor_specs(g)}
    recipes = {r.object_name: r for r in recipe.build_recipes(g)}
    assert specs.keys() == recipes.keys()
    assert all(s.shape == expression_shape(recipes[n].expression) for n, s in specs.items())
    assert specs["text/layers/0/moe/routed_gate_up"].shape == (experts * 2 * width, hidden)
    assert specs["text/layers/0/moe/shared_down"].shape == (hidden, shared)
    assert specs["text/layers/1/gdn/norm"].shape == (64,)
    assert specs["text/draft_head"].shape == (500, hidden)
    assert g.full_attention_layers == (0, 2)
    assert not any(n.startswith("dflash/") for n in specs)
    c["_name_or_path"] = "a-renamed-checkpoint"
    assert tuple(specs.values()) == inv.build_tensor_specs(inv.geometry_from_config(c, token_domain=500))
    metadata = inv.geometry_block(g)
    assert metadata["experts"] == experts
    assert metadata["shared_intermediate"] == shared


@pytest.mark.parametrize("layers,head,targets", [(3, 64, (0, 2)), (4, 128, (1, 3, 0))])
def test_dflash_uses_separate_config(layers, head, targets):
    g = inv.geometry_from_config(config_for(), token_domain=500)
    d = dflash.geometry_from_config(draft_config(g, layers=layers, head=head, targets=targets), g)
    recipe.validate_recipe_coverage(g, dflash=d)
    specs = {s.name: s for s in inv.build_dflash_specs(g, d)}
    assert specs["dflash/feature_projection"].shape == (g.hidden, g.hidden * len(targets))
    assert specs["dflash/layers/0/attention/query_key_value"].shape == (8 * head, g.hidden)
    assert specs["dflash/layers/0/attention/query_norm"].shape == (head,)
    metadata = dflash.geometry_block(d)
    assert validate_dflash_geometry(metadata, list(targets), inv.geometry_block(g)) == metadata
    for key in metadata:
        bad = dict(metadata)
        del bad[key]
        with pytest.raises(ValueError, match=key):
            validate_dflash_geometry(bad, list(targets), inv.geometry_block(g))
    with pytest.raises(ValueError, match="target_layer"):
        validate_dflash_geometry(metadata, [targets[0]] * len(targets), inv.geometry_block(g))


@pytest.mark.parametrize("missing", ["num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"])
def test_missing_dimensions_never_use_dsl_defaults(missing):
    c = config_for()
    del c[missing]
    with pytest.raises(ValueError, match=missing):
        inv.geometry_from_config(c)


def _save(root, config, recipes, *, frontend=False):
    from safetensors.torch import save_file
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    tensors = {name: torch.zeros(s.shape, dtype=torch.bfloat16)
               for name, s in source_requirements(recipes).items()}
    save_file(tensors, root / "model.safetensors")
    if frontend:
        (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(500)}}}))
        (root / "tokenizer_config.json").write_text('{}')
        (root / "chat_template.jinja").write_text('{{ messages }}')
        (root / "generation_config.json").write_text('{}')


def test_cpu_conversion_preserves_optional_drafter_metadata(tmp_path, monkeypatch):
    from surogate.serve.artifact.container import Artifact
    from surogate.serve.convert.qwen3_5_moe import convert
    from surogate.serve.convert.common.draft_head import compute_shortlist
    c = config_for()
    c["mtp_num_hidden_layers"] = 0
    g = inv.geometry_from_config(c, token_domain=500)
    dc = draft_config(g)
    d = dflash.geometry_from_config(dc, g)
    recipes = recipe.build_recipes(g, dflash=d)
    model, drafter = tmp_path / "renamed", tmp_path / "draft"
    _save(model, c, [r for r in recipes if not r.object_name.startswith("dflash/")], frontend=True)
    _save(drafter, dc, [r for r in recipes if r.object_name.startswith("dflash/")])
    ranking = tmp_path / "counts.i64"
    np.arange(g.vocab, dtype='<i8').tofile(ranking)
    monkeypatch.setattr(convert.draft_head, "compute_shortlist", lambda _path, root, *, geometry:
                        compute_shortlist(ranking, root, n=geometry.draft_vocab, vocab=geometry.vocab,
                                          tokenizer_vocab_size=geometry.token_domain))
    output = tmp_path / "model.sinfer"
    report = convert.convert(model, drafter, output, device="cpu")
    with Artifact(output) as artifact:
        assert artifact.geometry == inv.geometry_block(g)
        assert artifact.dflash_geometry == dflash.geometry_block(d)
        assert artifact.dflash_target_layers == [0, 2]
        assert artifact.vision_geometry == {}
        assert artifact.layer_types == list(g.layer_types)
    assert json.loads(report.read_text())["draft_head"]["rows"] == 500
    from surogate.serve.tools.reference.qwen3_5_moe.bindings import ArtifactBinding
    with ArtifactBinding.open(output) as binding:
        assert binding.mtp is None and binding.vision is None
        assert len(binding.dflash.layers) == 3
        assert binding.dflash.feature_projection.shape == (g.hidden, 2 * g.hidden)
        assert binding.dflash.layers[0].attention.query_norm.shape == (64,)
        assert binding.dflash.mask_embedding.row_begin == 500
        assert binding.dflash_config.kv_heads == 2


def test_routed_nvfp4_uses_actual_expert_count():
    from surogate.serve.convert.qwen3_5_moe.exports import routed_nvfp4 as routed
    g = inv.geometry_from_config(config_for())
    specs = {s.name: s for s in routed.tensor_specs(g)}
    assert specs["text/layers/0/moe/routed_gate_up_scale"].shape == (2 * g.experts,)
    assert len(tuple(routed.source_names(0, g))) == g.experts * 3 * 4


def test_routed_nvfp4_preserves_codes_and_each_experts_calibration():
    from surogate.serve.convert.qwen3_5_moe.exports import routed_nvfp4 as routed
    from surogate.serve.artifact.layouts import decode_nvfp4_words
    g = inv.geometry_from_config(config_for())
    data = {}
    for expert in range(g.experts):
        for projection in ("up_proj", "gate_proj", "down_proj"):
            stem = f"model.layers.0.mlp.experts.{expert}.{projection}."
            rows, columns = ((g.hidden, g.intermediate) if projection == "down_proj"
                             else (g.intermediate, g.hidden))
            code = 0x12 if projection == "up_proj" else 0x34 if projection == "gate_proj" else 0x56
            data[stem + "weight_packed"] = torch.full((rows, columns // 2), code, dtype=torch.uint8)
            data[stem + "weight_scale"] = torch.ones((rows, columns // 16), dtype=torch.float8_e4m3fn)
            data[stem + "weight_global_scale"] = torch.tensor(float(expert + 2))
            data[stem + "input_global_scale"] = torch.tensor(float(expert + 3))
    for build, shape, halves in ((routed.build_gate_up, (g.experts * 2 * g.intermediate, g.hidden), 2),
                                  (routed.build_down, (g.experts * g.hidden, g.intermediate), 1)):
        payload, second, act, alpha = build(data, 0, g)
        codes, scales, divisor = decode_nvfp4_words(payload, shape)
        assert float(divisor) == 1.0
        assert torch.all(scales == torch.tensor(1., dtype=torch.float8_e4m3fn).view(torch.uint8))
        for expert in range(g.experts):
            assert act[expert] == expert + 3
            assert alpha[expert] == pytest.approx(1 / ((expert + 2) * (expert + 3)))
            assert torch.all(second[expert * halves:(expert + 1) * halves] == 1 / (expert + 2))
            if halves == 2:
                start = expert * 2 * g.intermediate
                assert torch.all(codes[start:start + g.intermediate] == 0x12)
                assert torch.all(codes[start + g.intermediate:start + 2 * g.intermediate] == 0x34)
            else:
                assert torch.all(codes[expert * g.hidden:(expert + 1) * g.hidden] == 0x56)
    data["model.layers.0.mlp.experts.0.gate_proj.input_global_scale"] = torch.tensor(9.)
    with pytest.raises(ValueError, match="disagree on their global scales"):
        routed.build_gate_up(data, 0, g)


def test_compressed_shared_experts_require_explicit_requantization():
    from types import SimpleNamespace
    from surogate.serve.convert.qwen3_5_moe.exports.compressed_tensors_source import CompressedTensorsSource
    g = inv.geometry_from_config(config_for(), token_domain=500)
    specs = tuple(s for s in inv.build_tensor_specs(g) if s.name.endswith(("moe/shared_gate_up", "moe/shared_down")))
    names = {s.name for s in specs}
    recipes = {r.object_name: r for r in recipe.build_recipes(g) if r.object_name in names}
    source = object.__new__(CompressedTensorsSource)
    source.logical = {name: s.shape for name, s in source_requirements(tuple(recipes.values())).items()}
    source.resolved = SimpleNamespace(modules={})
    with pytest.raises(ValueError, match="--shared-expert w8"):
        source.plan(specs, recipes, shared_expert="as-stored")
    plan = source.plan(specs, recipes, shared_expert="w8")
    assert {s.name for s in plan.specs} == names
    assert all(obj.requantize_to == inv.W8 for obj in plan.objects.values())
