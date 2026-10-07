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


def test_routed_experts_are_read_as_the_checkpoint_stores_them():
    """The DSL declares the fused routed experts gate_first_experts(), so the trainer reorders
    their halves for its SwiGLU. That is the trainer's layout, not the engine's: the engine reads
    routed experts gate-first, so the artifact takes the checkpoint tensor as stored."""
    g = inv.geometry_from_config(config_for(), token_domain=500)
    assert g.declared.hf_mapping["experts_gate_up"] == "model.layers.{layer}.mlp.experts.gate_up_proj"
    recipes = {r.object_name: r for r in recipe.build_recipes(g)}
    fused, split = recipes["text/layers/0/moe/routed_gate_up"].expression.options
    assert (fused.source.name, fused.source.shape) == ("model.layers.0.mlp.experts.gate_up_proj", (4, 2 * 64, g.hidden))
    assert [part.name for part in split.source.sources] == [
        "model.layers.0.mlp.experts.gate_proj", "model.layers.0.mlp.experts.up_proj",
    ]


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


def _shared_expert_fixture():
    from types import SimpleNamespace
    from surogate.serve.convert.qwen3_5_moe.exports.compressed_tensors_source import CompressedTensorsSource
    g = inv.geometry_from_config(config_for(), token_domain=500)
    specs = tuple(s for s in inv.build_tensor_specs(g) if s.name.endswith(("moe/shared_gate_up", "moe/shared_down")))
    names = {s.name for s in specs}
    recipes = {r.object_name: r for r in recipe.build_recipes(g) if r.object_name in names}
    source = object.__new__(CompressedTensorsSource)
    source.logical = {name: s.shape for name, s in source_requirements(tuple(recipes.values())).items()}
    source.stored_module = {}
    source.resolved = SimpleNamespace(modules={})
    return source, specs, names, recipes


def test_compressed_shared_experts_are_requantized_to_w8_by_default():
    source, specs, names, recipes = _shared_expert_fixture()
    for choice in ({}, {"shared_expert": "auto"}, {"shared_expert": "w8"}):
        plan = source.plan(specs, recipes, **choice)
        assert {s.name for s in plan.specs} == names
        assert all(obj.requantize_to == inv.W8 for obj in plan.objects.values())
        # the MTP block's shared expert keeps its own path; only the text core is planned here
        assert plan.covered == {n for n in names if n.startswith("text/")}
        assert plan.shared_expert == "w8"
    with pytest.raises(ValueError, match="as-stored cannot be served"):
        source.plan(specs, recipes, shared_expert="as-stored")
    with pytest.raises(ValueError, match="must be one of"):
        source.plan(specs, recipes, shared_expert="bf16")


def test_compressed_shared_experts_missing_from_export_is_an_error():
    source, specs, names, recipes = _shared_expert_fixture()
    source.logical = {}  # an export whose shared expert cannot be found under either dialect
    with pytest.raises(ValueError, match="not found in the compressed-tensors export"):
        source.plan(specs, recipes)


def test_compressed_source_folds_nested_text_tower_names(monkeypatch, tmp_path):
    """Official Qwen3.5/3.6 exports nest the text tower under model.language_model.; the
    recipes speak the flat dialect, so the planner must answer flat names from nested files."""
    from types import SimpleNamespace
    from surogate.serve.convert.qwen3_5_moe.exports import compressed_tensors_source as cts
    stored = {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_packed": (16, 4),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_scale": (16, 1),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_global_scale": (1,),
        "model.language_model.layers.0.mlp.gate.weight": (4, 8),
        "lm_head.weight": (500, 8),
    }
    quantized = SimpleNamespace(scheme=SimpleNamespace(weights=SimpleNamespace(num_bits=4)))
    plain = SimpleNamespace(scheme=None)
    resolved = SimpleNamespace(modules={
        "model.language_model.layers.0.mlp.shared_expert.gate_proj": quantized,
        "model.language_model.layers.0.mlp.gate": plain,
        "lm_head": plain,
    })
    monkeypatch.setattr(cts, "read_tensor_shapes", lambda model_dir: stored)
    monkeypatch.setattr(cts, "resolve_checkpoint", lambda model_dir: resolved)
    source = cts.CompressedTensorsSource(tmp_path)
    assert source.logical["model.layers.0.mlp.shared_expert.gate_proj.weight"] == (16, 8)
    assert source.logical["model.layers.0.mlp.gate.weight"] == (4, 8)
    assert source.logical["lm_head.weight"] == (500, 8)
    assert source.is_quantized("model.layers.0.mlp.shared_expert.gate_proj.weight")
    assert not source.is_quantized("model.layers.0.mlp.gate.weight")
    assert (source.module_as_stored("model.layers.0.mlp.shared_expert.gate_proj.weight")
            == "model.language_model.layers.0.mlp.shared_expert.gate_proj")
    # an export spelling one tensor both ways is ambiguous and stays as stored
    both = dict(stored); both["model.layers.0.mlp.gate.weight"] = (4, 8)
    monkeypatch.setattr(cts, "read_tensor_shapes", lambda model_dir: both)
    ambiguous = cts.CompressedTensorsSource(tmp_path)
    assert "model.language_model.layers.0.mlp.gate.weight" in ambiguous.logical


#: The Linears Qwen's block-FP8 exports quantise, by the stored tensor's suffix.
_FP8_LINEARS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight",
                "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
                "shared_expert.down_proj.weight")


def _fp8_linear(generator, rows, columns):
    codes = (torch.rand((rows, columns), generator=generator) * 8 - 4).to(torch.float8_e4m3fn)
    scales = (torch.rand((rows // 128, columns // 128), generator=generator) + 0.5).to(torch.bfloat16)
    return codes, scales


def _save_fp8_export(root, c, g, ranking, monkeypatch, convert):
    """A Qwen-style block-FP8 export: per-expert routed Linears, E4M3 codes and BF16 multipliers."""
    from safetensors.torch import save_file
    from surogate.serve.convert.common.draft_head import compute_shortlist
    generator = torch.Generator().manual_seed(7)
    tensors = {}
    for name, source in source_requirements(recipe.build_recipes(g)).items():
        if ".mlp.experts." in name:
            continue  # the stacked spellings a BF16 export uses; this one stores per expert
        if name.endswith(_FP8_LINEARS):
            codes, scales = _fp8_linear(generator, *source.shape)
            tensors[name], tensors[name + "_scale_inv"] = codes, scales
        else:
            tensors[name] = torch.randn(source.shape, generator=generator).to(torch.bfloat16)
    for prefix in ["model.layers.%d" % layer for layer in range(g.layers)] + ["mtp.layers.0"]:
        for expert in range(g.experts):
            for projection, shape in (("gate_proj", (g.intermediate, g.hidden)),
                                      ("up_proj", (g.intermediate, g.hidden)),
                                      ("down_proj", (g.hidden, g.intermediate))):
                stem = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
                tensors[stem], tensors[stem + "_scale_inv"] = _fp8_linear(generator, *shape)
    root.mkdir()
    c["quantization_config"] = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                "weight_block_size": [128, 128]}
    (root / "config.json").write_text(json.dumps(c))
    save_file(tensors, root / "model.safetensors")
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(500)}}}))
    (root / "tokenizer_config.json").write_text('{}')
    (root / "chat_template.jinja").write_text('{{ messages }}')
    (root / "generation_config.json").write_text('{}')
    np.arange(g.vocab, dtype='<i8').tofile(ranking)
    monkeypatch.setattr(convert.draft_head, "compute_shortlist", lambda _path, root, *, geometry:
                        compute_shortlist(ranking, root, n=geometry.draft_vocab, vocab=geometry.vocab,
                                          tokenizer_vocab_size=geometry.token_domain))
    return tensors


def _fp8_words(artifact, name, shape):
    from surogate.serve.artifact.layouts import block_scale128_geometry
    from surogate.serve.convert.common.inventory import FP8_BLOCK
    obj = artifact.find(name)
    assert obj.format == FP8_BLOCK and tuple(obj.shape) == shape
    geometry = block_scale128_geometry(FP8_BLOCK, shape)
    payload = bytes(artifact.payload(obj))
    codes = torch.frombuffer(bytearray(payload[:geometry.code_plane_bytes]), dtype=torch.uint8)
    scales = torch.frombuffer(bytearray(payload[geometry.scale_plane_offset:geometry.payload_bytes]),
                              dtype=torch.float32)
    return codes.reshape(shape), scales.reshape(shape[0] // 128, shape[1] // 128)


def test_fp8_block_export_keeps_its_codes_where_kernels_read_them(tmp_path, monkeypatch):
    from surogate.serve.artifact.container import Artifact
    from surogate.serve.artifact.layouts import dequantize_row_split
    from surogate.serve.convert.qwen3_5_moe import convert
    c = config_for()
    c.update(moe_intermediate_size=128, shared_expert_intermediate_size=128)
    g = inv.geometry_from_config(c, token_domain=500)
    model = tmp_path / "fp8"
    stored = _save_fp8_export(model, c, g, tmp_path / "counts.i64", monkeypatch, convert)
    output = tmp_path / "model.sinfer"
    report = json.loads(convert.convert(model, None, output, device="cpu").read_text())
    assert report["quantization"]["fp8_block"] == {
        "stored_codes": 2 * g.layers, "routed_stacked": 2 * (g.layers + 1),
        "requantized_to_base_format": 2 * (g.layers + 1) + 2}

    def codes(name):
        return stored[name].view(torch.uint8)

    def scales(name):
        return stored[name + "_scale_inv"].float()

    with Artifact(output) as artifact:
        assert artifact.identity.weights_id == "fp8-block"
        # The routed experts: each expert's [gate; up] rows and its own block multipliers.
        for prefix, source in (("text/layers/1/moe/", "model.layers.1"), ("mtp/layer/moe/", "mtp.layers.0")):
            n = g.experts * 2 * g.intermediate
            gate_up, gate_up_scales = _fp8_words(artifact, prefix + "routed_gate_up", (n, g.hidden))
            down, down_scales = _fp8_words(artifact, prefix + "routed_down", (g.experts * g.hidden, g.intermediate))
            for expert in range(g.experts):
                stem = f"{source}.mlp.experts.{expert}."
                rows = slice(expert * 2 * g.intermediate, (expert + 1) * 2 * g.intermediate)
                expected = torch.cat([codes(stem + "gate_proj.weight"), codes(stem + "up_proj.weight")])
                assert torch.equal(gate_up[rows], expected)
                assert torch.equal(gate_up_scales[2 * expert:2 * expert + 2],
                                   torch.cat([scales(stem + "gate_proj.weight"), scales(stem + "up_proj.weight")]))
                assert torch.equal(down[expert * g.hidden:(expert + 1) * g.hidden], codes(stem + "down_proj.weight"))
                assert torch.equal(down_scales[expert], scales(stem + "down_proj.weight")[0])
        # The fused attention input: [query heads | key | gate heads | value], 128-row blocks
        # of the stored Linears, each with its own multiplier.
        q = "model.layers.0.self_attn.q_proj.weight"
        k = "model.layers.0.self_attn.k_proj.weight"
        v = "model.layers.0.self_attn.v_proj.weight"
        qkgv, qkgv_scales = _fp8_words(artifact, "text/layers/0/attention/query_key_gate_value",
                                       (codes(q).shape[0] + codes(k).shape[0] + codes(v).shape[0], g.hidden))
        blocks = [(q, 0), (q, 2), (k, 0), (q, 1), (q, 3), (v, 0)]
        for index, (name, block) in enumerate(blocks):
            assert torch.equal(qkgv[index * 128:(index + 1) * 128], codes(name)[block * 128:(block + 1) * 128])
            assert torch.equal(qkgv_scales[index], scales(name)[block])
        _fp8_words(artifact, "text/layers/1/gdn/query_key_value_z", (
            codes("model.layers.1.linear_attn.in_proj_qkv.weight").shape[0]
            + codes("model.layers.1.linear_attn.in_proj_z.weight").shape[0], g.hidden))
        # The shared expert is W8 for its kernels, requantised from the values the codes state.
        shared = artifact.find("text/layers/1/moe/shared_down")
        assert shared.format == inv.W8
        values = dequantize_row_split(bytes(artifact.payload(shared)), inv.W8, tuple(shared.shape),
                                      dtype=torch.float32)
        name = "model.layers.1.mlp.shared_expert.down_proj.weight"
        reference = codes(name).view(torch.float8_e4m3fn).float() * scales(name).repeat_interleave(128, 0).repeat_interleave(128, 1)
        assert torch.allclose(values, reference, rtol=0, atol=float(reference.abs().max()) / 100)
        # The MTP block's dense projections stay the base converter's W8.
        assert artifact.find("mtp/layer/attention/output").format == inv.W8


def test_fp8_export_with_other_block_sizes_is_refused():
    from surogate.serve.convert.common.fp8_block_source import is_fp8_block_export
    assert not is_fp8_block_export(config_for())
    assert is_fp8_block_export({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}})
    with pytest.raises(ValueError, match="128 x 128 block scales"):
        is_fp8_block_export({"quantization_config": {"quant_method": "fp8", "weight_block_size": [1, 128]}})


#: The Linears a mixed-precision compressed-tensors export keeps as FP8 rows, by suffix.
_FP8_CHANNEL_LINEARS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                        "in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight")


def _nvfp4_linear(generator, rows, columns):
    packed = torch.randint(0, 256, (rows, columns // 2), generator=generator, dtype=torch.uint8)
    scales = (torch.rand((rows, columns // 16), generator=generator) * 4 + 0.5).to(torch.float8_e4m3fn)
    return {"weight_packed": packed, "weight_scale": scales,
            "weight_global_scale": torch.tensor([448.0 * 6.0 / 3.0]),
            "input_global_scale": torch.tensor([448.0 * 6.0 / 5.0])}


def _save_mixed_export(root, c, g, ranking, monkeypatch, convert):
    """An llm-compressor mixed export: NVFP4 experts and shared expert, FP8 per-channel
    attention, linear attention and output head with BF16 multipliers, the MTP block in BF16."""
    from safetensors.torch import save_file
    from surogate.serve.convert.common.draft_head import compute_shortlist
    generator = torch.Generator().manual_seed(11)
    tensors = {}
    for name, source in source_requirements(recipe.build_recipes(g)).items():
        if name.startswith("model.") and ".mlp.experts." in name:
            continue  # the stacked spelling; this export stores per expert, packed
        stem = name[: -len(".weight")]
        if name.startswith("model.") and ".shared_expert." in name:
            for suffix, tensor in _nvfp4_linear(generator, *source.shape).items():
                tensors[f"{stem}.{suffix}"] = tensor
        elif (name.startswith("model.") and name.endswith(_FP8_CHANNEL_LINEARS)) or name == "lm_head.weight":
            rows, columns = source.shape
            tensors[name] = (torch.rand(source.shape, generator=generator) * 8 - 4).to(torch.float8_e4m3fn)
            # multipliers far from one, so a recipe that read the codes as values would show
            tensors[stem + ".weight_scale"] = (torch.rand((rows, 1), generator=generator) * .01 + .01).to(torch.bfloat16)
        else:
            tensors[name] = torch.randn(source.shape, generator=generator).to(torch.bfloat16)
    for layer in range(g.layers):
        for expert in range(g.experts):
            for projection, shape in (("gate_proj", (g.intermediate, g.hidden)),
                                      ("up_proj", (g.intermediate, g.hidden)),
                                      ("down_proj", (g.hidden, g.intermediate))):
                stem = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for suffix, tensor in _nvfp4_linear(generator, *shape).items():
                    tensors[f"{stem}.{suffix}"] = tensor
    root.mkdir()
    weights = {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False}
    activations = {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True, "dynamic": True}
    nvfp4 = {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16,
             "symmetric": True, "dynamic": False}
    c["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "mixed-precision", "quantization_status": "compressed",
        "ignore": ["re:^mtp.*"] + [f"model.layers.{layer}.mlp.gate" for layer in range(g.layers)],
        "config_groups": {
            "group_0": {"format": "float-quantized", "weights": weights, "input_activations": activations,
                        "targets": [r"re:.*self_attn\.(q|k|v|o)_proj$",
                                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", "re:.*lm_head"]},
            "group_1": {"format": "nvfp4-pack-quantized", "weights": nvfp4,
                        "input_activations": dict(nvfp4, dynamic="local"),
                        "targets": [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",
                                    r"re:.*shared_expert\.(gate|up|down)_proj$"]},
        },
    }
    (root / "config.json").write_text(json.dumps(c))
    save_file(tensors, root / "model.safetensors")
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(500)}}}))
    (root / "tokenizer_config.json").write_text('{}')
    (root / "chat_template.jinja").write_text('{{ messages }}')
    (root / "generation_config.json").write_text('{}')
    np.arange(g.vocab, dtype='<i8')[::-1].copy().tofile(ranking)
    monkeypatch.setattr(convert.draft_head, "compute_shortlist", lambda _path, root, *, geometry:
                        compute_shortlist(ranking, root, n=geometry.draft_vocab, vocab=geometry.vocab,
                                          tokenizer_vocab_size=geometry.token_domain))
    return tensors


def test_mixed_nvfp4_fp8_export_keeps_every_stored_word(tmp_path, monkeypatch):
    from surogate.serve.artifact.container import Artifact
    from surogate.serve.artifact.layouts import decode_fp8_row_scaled_words, dequantize_row_split
    from surogate.serve.convert.common.inventory import FP8
    from surogate.serve.convert.qwen3_5_moe import convert
    c = config_for()
    c.update(moe_intermediate_size=128, shared_expert_intermediate_size=128)
    g = inv.geometry_from_config(c, token_domain=500)
    model = tmp_path / "mixed"
    stored = _save_mixed_export(model, c, g, tmp_path / "counts.i64", monkeypatch, convert)
    output = tmp_path / "model.sinfer"
    report = json.loads(convert.convert(model, None, output, device="cpu").read_text())
    assert report["quantization"]["compressed_tensors"]["shared_expert"] == "w8"

    def fp8_rows(artifact, name):
        obj = artifact.find(name)
        assert obj.format == FP8
        codes, scales = decode_fp8_row_scaled_words(bytes(artifact.payload(obj)), tuple(obj.shape))
        return codes.view(torch.uint8), scales

    def source_rows(name):
        return stored[name].view(torch.uint8), stored[name[: -len(".weight")] + ".weight_scale"].reshape(-1)

    with Artifact(output) as artifact:
        assert artifact.identity.weights_id == "compressed-tensors"
        # Each FP8 Linear's codes and BF16 multipliers, row for row as stored.
        for name, source in (("text/layers/0/attention/key", "model.layers.0.self_attn.k_proj.weight"),
                             ("text/layers/0/attention/value", "model.layers.0.self_attn.v_proj.weight"),
                             ("text/layers/2/attention/output", "model.layers.2.self_attn.o_proj.weight"),
                             ("text/layers/1/gdn/query_key_value", "model.layers.1.linear_attn.in_proj_qkv.weight"),
                             ("text/layers/1/gdn/z", "model.layers.1.linear_attn.in_proj_z.weight"),
                             ("text/layers/3/gdn/output", "model.layers.3.linear_attn.out_proj.weight"),
                             ("text/output_head", "lm_head.weight")):
            codes, scales = fp8_rows(artifact, name)
            expected_codes, expected_scales = source_rows(source)
            assert torch.equal(codes, expected_codes), name
            assert torch.equal(scales, expected_scales), name
        # q_proj holds [query; gate] per head; the split objects take their halves in head order.
        q_codes, q_scales = source_rows("model.layers.2.self_attn.q_proj.weight")
        heads, dim = c["num_attention_heads"], c["head_dim"]
        for name, half in (("text/layers/2/attention/query", 0), ("text/layers/2/attention/gate", 1)):
            rows = torch.cat([torch.arange(h * 2 * dim + half * dim, h * 2 * dim + (half + 1) * dim)
                              for h in range(heads)])
            codes, scales = fp8_rows(artifact, name)
            assert torch.equal(codes, q_codes[rows]) and torch.equal(scales, q_scales[rows]), name
        # The draft head gathers rows of the FP8 output head: the values the codes stand for.
        ids = torch.frombuffer(bytearray(artifact.payload(artifact.find("text/draft_head_token_ids"))),
                               dtype=torch.int32).long()
        draft = artifact.find("text/draft_head")
        values = dequantize_row_split(bytes(artifact.payload(draft)), draft.format, tuple(draft.shape),
                                      dtype=torch.float32)
        head_codes, head_scales = stored["lm_head.weight"], stored["lm_head.weight_scale"]
        reference = (head_codes.float() * head_scales.float())[ids]
        assert torch.allclose(values, reference, rtol=0, atol=float(reference.abs().max()) / 8)
        # The routed experts stay NVFP4 (their words: test_routed_nvfp4_preserves_codes_...); the
        # shared expert is the W8 its kernels read.
        assert artifact.find("text/layers/1/moe/routed_gate_up").format == "NVFP4"
        assert artifact.find("text/layers/1/moe/shared_gate_up").format == inv.W8
        # The MTP block, which the export leaves in BF16, keeps the base converter's W8.
        assert artifact.find("mtp/layer/attention/output").format == inv.W8
