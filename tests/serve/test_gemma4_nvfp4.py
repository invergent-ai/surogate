"""Gemma 4 mixture NVFP4: the numeric helpers, the routed-expert export reader, the converter."""

import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.layouts import decode_nvfp4_words
from surogate.serve.convert.common import nvfp4
from surogate.serve.convert.common.inventory import FP32, NVFP4
from surogate.serve.convert.gemma4_moe import inventory as moe_inventory
from surogate.serve.convert.gemma4_moe.exports import routed_nvfp4 as routed
from tests.serve.test_gemma4_checkpoint_config import checkpoint as gemma4_text


# ---- numeric helpers ---------------------------------------------------------------------


def test_quantize_reproduces_representable_values_and_packs_even_column_low():
    # One block per row whose values are E2M1 points times a power of two: exactly representable.
    points = torch.tensor(nvfp4.E2M1_MAGNITUDES + tuple(-m for m in nvfp4.E2M1_MAGNITUDES))
    values = torch.stack([points * 0.25, points * 2.0])
    divisor = 3.0
    packed, scales = nvfp4.quantize(values, divisor)
    assert packed.shape == (2, 8) and scales.shape == (2, 1)
    back = nvfp4.dequantize(packed, scales, divisor)
    assert torch.equal(back, values)
    codes = nvfp4.unpack_codes(packed)
    assert torch.equal(nvfp4.pack_codes(codes), packed)
    assert int(packed[0, 0]) & 0x0F == int(codes[0, 0]) and int(packed[0, 0]) >> 4 == int(codes[0, 1])


def test_quantize_rounds_ties_to_the_even_code_and_bounds_the_error():
    # 1.25 sits between 1.0 (code 2) and 1.5 (code 3): the even code wins. 5.0 sits between 4 and 6.
    block = torch.tensor([[6.0, 1.25, 5.0, 0.25] + [0.0] * 12])
    packed, scales = nvfp4.quantize(block, 1.0)
    back = nvfp4.dequantize(packed, scales, 1.0)
    assert back[0, 1] == 1.0 and back[0, 2] == 4.0 and back[0, 3] == 0.0
    torch.manual_seed(0)
    values = torch.randn(64, 256)
    divisor = nvfp4.global_divisor(values)
    packed, scales = nvfp4.quantize(values, divisor)
    back = nvfp4.dequantize(packed, scales, divisor)
    # Half a step of the widest E2M1 gap (2 at the top of the range) over the block's scale.
    block_max = values.reshape(64, 16, 16).abs().amax(dim=2, keepdim=True)
    bound = (block_max / 6 * 1.07).expand(64, 16, 16).reshape(64, 256)
    assert torch.all((back - values).abs() <= bound + 1e-6)
    searched, searched_scales = nvfp4.quantize(values, divisor, scale_candidates=(1.0, 0.9, 1.1))
    error = (nvfp4.dequantize(searched, searched_scales, divisor) - values).square().sum()
    assert error <= (back - values).square().sum()


# ---- the routed-expert reader --------------------------------------------------------------


def _geometry(experts=4, hidden=256, intermediate=128):
    return routed.Geometry(layers=1, experts=experts, hidden=hidden, intermediate=intermediate)


class _Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, name):
        return self.tensors[name]


def _modelopt_expert(tensors, g, expert, projection, fill, weight_scale_2, input_scale):
    rows, columns = ((g.hidden, g.intermediate) if projection == "down_proj"
                     else (g.intermediate, g.hidden))
    stem = f"model.layers.0.experts.{expert}.{projection}."
    tensors[stem + "weight"] = torch.full((rows, columns // 2), fill, dtype=torch.uint8)
    tensors[stem + "weight_scale"] = torch.full((rows, columns // 16), 2.0).to(torch.float8_e4m3fn)
    tensors[stem + "weight_scale_2"] = torch.tensor(weight_scale_2, dtype=torch.float32)
    tensors[stem + "input_scale"] = torch.tensor(input_scale, dtype=torch.float32)


def test_convention_follows_the_declaration():
    assert routed.convention_of({}) is None
    assert routed.convention_of({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                                         "group_size": 16}}) is routed.MODELOPT
    ct = {"quant_method": "compressed-tensors", "config_groups": {"g": {"format": "nvfp4-pack-quantized"}}}
    assert routed.convention_of({"quantization_config": ct}) is routed.COMPRESSED_TENSORS
    with pytest.raises(ValueError, match="only NVFP4"):
        routed.convention_of({"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}})
    with pytest.raises(ValueError, match="not served"):
        routed.convention_of({"quantization_config": {"quant_method": "awq"}})


def test_specs_replace_only_the_two_routed_objects_per_layer():
    g = moe_inventory.geometry_from_config(gemma4_text("gemma4_moe"))
    base = moe_inventory.tensor_specs(moe_inventory.stored_objects(g, tied_output_head=True))
    specs = {s.name: s for s in routed.tensor_specs(base, g.experts)}
    gate_up = specs["text/layers/0/moe/routed_gate_up"]
    assert gate_up.format == NVFP4 and gate_up.shape == (g.experts * 2 * g.expert_intermediate, g.hidden)
    assert specs["text/layers/0/moe/routed_gate_up_scale"].shape == (2 * g.experts,)
    assert specs["text/layers/0/moe/routed_down_alpha"].format == FP32
    assert specs["text/layers/3/moe/routed_down"].shape == (g.experts * g.hidden, g.expert_intermediate)
    untouched = {s.name: s for s in base if not routed.is_routed_object(s.name)}
    assert all(specs[name] == spec for name, spec in untouched.items())
    assert len(specs) == len(base) + 6 * g.layers


def test_modelopt_experts_stack_up_before_gate_with_scales_in_the_engine_direction():
    g = _geometry()
    tensors = {}
    for expert in range(g.experts):
        _modelopt_expert(tensors, g, expert, "up_proj", 0x12, 0.5 / (expert + 1), 0.25 / (expert + 1))
        _modelopt_expert(tensors, g, expert, "gate_proj", 0x34, 0.5 / (expert + 1), 0.25 / (expert + 1))
        _modelopt_expert(tensors, g, expert, "down_proj", 0x56, 0.125, 0.0625 * (expert + 1))
    reader = _Reader(tensors)
    stats = routed.LayerStats()
    payload, second, act, alpha = routed.build_gate_up(reader, g, routed.MODELOPT, 0, stats)
    codes, scales, divisor = decode_nvfp4_words(payload, (g.experts * 2 * g.intermediate, g.hidden))
    assert float(divisor) == 1.0 and not stats.requantised_experts
    for expert in range(g.experts):
        begin = expert * 2 * g.intermediate
        assert torch.all(codes[begin:begin + g.intermediate] == 0x12)
        assert torch.all(codes[begin + g.intermediate:begin + 2 * g.intermediate] == 0x34)
        # weight_scale_2 multiplies, so it is the engine's second level as stored.
        assert second[2 * expert] == pytest.approx(0.5 / (expert + 1))
        assert act[expert] == pytest.approx(4.0 * (expert + 1))
        assert alpha[expert] == pytest.approx((0.25 / (expert + 1)) * (0.5 / (expert + 1)))
    payload, second, act, alpha = routed.build_down(reader, g, routed.MODELOPT, 0)
    codes, _, _ = decode_nvfp4_words(payload, (g.experts * g.hidden, g.intermediate))
    assert torch.all(codes == 0x56)
    assert act[2] == pytest.approx(1 / (0.0625 * 3))
    assert alpha[2] == pytest.approx(0.125 * 0.0625 * 3)


def test_gate_and_up_with_two_weight_scales_are_requantised_to_the_common_one():
    g = _geometry(experts=1)
    tensors = {}
    torch.manual_seed(1)
    for projection, weight_scale_2 in (("up_proj", 0.01), ("gate_proj", 0.02), ("down_proj", 0.01)):
        _modelopt_expert(tensors, g, 0, projection, 0, weight_scale_2, 0.1)
        rows = tensors[f"model.layers.0.experts.0.{projection}.weight"].shape[0]
        tensors[f"model.layers.0.experts.0.{projection}.weight"] = torch.randint(
            0, 256, (rows, tensors[f"model.layers.0.experts.0.{projection}.weight"].shape[1]), dtype=torch.uint8)
    reader = _Reader(tensors)
    before = {p: routed.read_projection(reader, routed.MODELOPT, 0, 0, p) for p in ("up_proj", "gate_proj")}
    stats = routed.LayerStats()
    payload, second, _, alpha = routed.build_gate_up(reader, g, routed.MODELOPT, 0, stats)
    assert stats.requantised_experts == [0]
    # The common divisor is the smaller one (the larger weight_scale_2), so neither half clips.
    assert second[0] == pytest.approx(0.02) and second[1] == pytest.approx(0.02)
    codes, scales, _ = decode_nvfp4_words(payload, (2 * g.intermediate, g.hidden))
    for half, projection in enumerate(("up_proj", "gate_proj")):
        rows = slice(half * g.intermediate, (half + 1) * g.intermediate)
        after = nvfp4.dequantize(codes[rows], scales[rows], 1 / 0.02)
        original = nvfp4.dequantize(before[projection].codes, before[projection].scales,
                                    before[projection].weight_divisor)
        if projection == "gate_proj":
            assert torch.equal(after, original)  # already at the common divisor: words kept
        else:
            block_max = original.reshape(-1, 16).abs().amax(dim=1, keepdim=True)
            assert torch.all((after - original).reshape(-1, 16).abs() <= block_max / 6 * 1.07 + 1e-9)


def test_compressed_tensors_scales_already_divide():
    g = _geometry(experts=1)
    tensors = {}
    for projection in ("up_proj", "gate_proj", "down_proj"):
        rows, columns = ((g.hidden, g.intermediate) if projection == "down_proj" else (g.intermediate, g.hidden))
        stem = f"model.layers.0.experts.0.{projection}."
        tensors[stem + "weight_packed"] = torch.zeros((rows, columns // 2), dtype=torch.uint8)
        tensors[stem + "weight_scale"] = torch.ones((rows, columns // 16)).to(torch.float8_e4m3fn)
        tensors[stem + "weight_global_scale"] = torch.tensor(8.0)
        tensors[stem + "input_global_scale"] = torch.tensor(16.0)
    _, second, act, alpha = routed.build_gate_up(_Reader(tensors), g, routed.COMPRESSED_TENSORS, 0,
                                                routed.LayerStats())
    assert second[0] == pytest.approx(1 / 8) and act[0] == pytest.approx(16.0)
    assert alpha[0] == pytest.approx(1 / 128)


# ---- the converter, end to end ---------------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
def test_gemma_vl_converts_a_modelopt_export_to_the_routed_profile(tmp_path, monkeypatch, dense):
    from tests.serve.test_gemma_vl import config_for

    from surogate.serve.artifact.container import Artifact
    from surogate.serve.convert.common.inventory import RESOURCE_SPECS
    from surogate.serve.convert.common.recipe import source_requirements
    from surogate.serve.convert.gemma_vl import convert, inventory

    config = config_for("gemma4_moe")
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
                                     "ignore": ["lm_head", "model.vision_tower*"]}
    g = inventory.geometry_from_config(config)
    _, text_recipes = inventory.text_specs_and_recipes(g)
    _, vision_recipes = inventory.vision_recipes(g)
    torch.manual_seed(2)
    tensors = {}
    for recipe in (*text_recipes, *vision_recipes):
        if routed.is_routed_object(recipe.object_name):
            continue
        for name, source in source_requirements((recipe,)).items():
            dtype = torch.float32 if "per_expert_scale" in name else torch.bfloat16
            tensors[name] = (torch.randn(source.shape) * 0.02).to(dtype)
    rg = routed.geometry_of(g.text)
    for layer in range(rg.layers):
        for expert in range(rg.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows, columns = ((rg.hidden, rg.intermediate) if projection == "down_proj"
                                 else (rg.intermediate, rg.hidden))
                stem = f"model.language_model.layers.{layer}.experts.{expert}.{projection}."
                tensors[stem + "weight"] = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8)
                tensors[stem + "weight_scale"] = torch.full((rows, columns // 16), 1.0).to(torch.float8_e4m3fn)
                tensors[stem + "weight_scale_2"] = torch.tensor(0.001 * (expert + 1))
                tensors[stem + "input_scale"] = torch.tensor(0.01)
    if dense:
        # Layer 0's query projection and its dense feed-forward quantised too, as an export of
        # everything but the ignore list would store them.
        for module in ("self_attn.q_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            stem = next(name for name in tensors if name.endswith(f"layers.0.{module}.weight"))[: -len("weight")]
            rows, columns = tensors[stem + "weight"].shape
            tensors[stem + "weight"] = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8)
            tensors[stem + "weight_scale"] = torch.full((rows, columns // 16), 1.0).to(torch.float8_e4m3fn)
            tensors[stem + "weight_scale_2"] = torch.tensor(0.002 if module == "mlp.up_proj" else 0.001)
            tensors[stem + "input_scale"] = torch.tensor(0.01 if module != "mlp.up_proj" else 0.02)
    model = tmp_path / "model"
    model.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(model / "model.safetensors"))
    (model / "config.json").write_text(json.dumps(config))
    # The frontend is not what this test is about: resources are placeholders in canonical order.
    monkeypatch.setattr(convert, "resources_for",
                        lambda m, geometry: {spec.name: b"{}" for spec in RESOURCE_SPECS})
    monkeypatch.setattr(convert, "tokenizer_domain", lambda m: 512)
    out = tmp_path / "out.sinfer"
    convert.convert(model, out, device="cpu", text_format="bf16")
    report = json.loads((tmp_path / "out.sinfer.conversion.json").read_text())
    assert report["weights_id"] == "routed-nvfp4" and report["routed_nvfp4"]["convention"] == "modelopt"
    with Artifact(out) as artifact:
        assert artifact.identity.weights_id == "routed-nvfp4"
        gate_up = artifact.find("text/layers/0/moe/routed_gate_up")
        assert gate_up.format == NVFP4
        second = torch.frombuffer(bytearray(artifact.payload("text/layers/0/moe/routed_gate_up_scale")),
                                  dtype=torch.float32)
        assert second[2 * 3] == pytest.approx(0.004)
        if dense:
            query = artifact.find("text/layers/0/attention/query")
            assert query.format == NVFP4
            divisor = bytes(artifact.payload("text/layers/0/attention/query/input_scale_divisor"))
            assert torch.frombuffer(bytearray(divisor), dtype=torch.float32)[0] == pytest.approx(100.0)
            fused = artifact.find("text/layers/0/mlp/gate_up")
            assert fused.format == NVFP4 and fused.shape[0] == 2 * g.text.intermediate
            with pytest.raises(KeyError):
                artifact.find("text/layers/0/mlp/gate")
            # One activation divisor for the pair: the smaller one (up's 1 / 0.02).
            divisor = bytes(artifact.payload("text/layers/0/mlp/gate_up/input_scale_divisor"))
            assert torch.frombuffer(bytearray(divisor), dtype=torch.float32)[0] == pytest.approx(50.0)
            _, _, weight_divisor = decode_nvfp4_words(artifact.payload("text/layers/0/mlp/gate_up"),
                                                      tuple(fused.shape))
            assert float(weight_divisor) == pytest.approx(500.0)
            assert artifact.find("text/layers/1/attention/query").format == "BF16"
            assert report["routed_nvfp4"]["dense_nvfp4_objects"] == 3
        else:
            assert artifact.find("text/layers/0/attention/query").format == "BF16"
        assert artifact.find("text/token_embedding").format == "W8G32_F16S"
        codes, _, _ = decode_nvfp4_words(artifact.payload("text/layers/0/moe/routed_down"),
                                         (rg.experts * rg.hidden, rg.intermediate))
        stem = "model.language_model.layers.0.experts.1.down_proj.weight"
        assert torch.equal(codes[rg.hidden:2 * rg.hidden], tensors[stem])


def test_bf16_text_storage_keeps_routed_experts_quantised():
    from tests.serve.test_gemma_vl import config_for
    from surogate.serve.convert.gemma_vl import convert, inventory

    g = inventory.geometry_from_config(config_for("gemma4_moe"))
    specs, _ = inventory.text_specs_and_recipes(g)
    stored = {s.name: s for s in convert._text_storage(specs, "bf16")}
    assert stored["text/layers/0/moe/routed_gate_up"].format == "W8G32_F16S"
    assert stored["text/layers/0/moe/routed_down"].format == "W8G32_F16S"
    assert stored["text/layers/0/mlp/gate"].format == "BF16"
    assert stored["text/layers/0/attention/output"].format == "BF16"
    assert stored["text/token_embedding"].format == "W8G32_F16S"
