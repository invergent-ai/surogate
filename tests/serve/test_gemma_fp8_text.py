"""FP8 text storage for the Gemma converters: the projections stored for the FP8 tensor cores,
from a per-channel FP8 export's own codes or from a checkpoint's BF16 values."""

import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.fp8_block_source import fp8_projection_specs
from surogate.serve.convert.common.inventory import (
    BF16, BLOCK128_LAYOUT, FP8_BLOCK, FP8_ROW_F32, RESOURCE_SPECS, ROW_SCALE_F32_LAYOUT, TensorSpec, W8)
from surogate.serve.convert.common.quantize import quantize_fp8_blocks, quantize_fp8_rows
from surogate.serve.convert.common.recipe import source_requirements


def _e4m3(codes):
    return codes.view(torch.float8_e4m3fn).to(torch.float32)


def test_fp8_quantizers_map_each_group_maximum_to_448_and_bound_the_error():
    torch.manual_seed(0)
    weight = torch.randn(256, 384) * 0.02
    weight[3, 5] = 1.5
    weight[128:, :128] = 0  # an all-zero block keeps a positive multiplier
    codes, multipliers = quantize_fp8_blocks(weight, device="cpu")
    assert codes.dtype == torch.uint8 and multipliers.shape == (2, 3)
    assert multipliers[0, 0] == pytest.approx(1.5 / 448) and multipliers[1, 0] == 1.0
    values = (_e4m3(codes).reshape(2, 128, 3, 128) * multipliers[:, None, :, None]).reshape(256, 384)
    blocks = weight.reshape(2, 128, 3, 128).abs().amax(dim=(1, 3))
    bound = blocks.repeat_interleave(128, 0).repeat_interleave(128, 1) / 16  # half an E4M3 step at the top
    assert torch.all((values - weight).abs() <= bound + 1e-12)
    codes, multipliers = quantize_fp8_rows(weight, device="cpu")
    assert multipliers.shape == (256,) and multipliers[3] == pytest.approx(1.5 / 448)
    assert torch.all((_e4m3(codes) * multipliers[:, None] - weight).abs()
                     <= weight.abs().amax(dim=1, keepdim=True) / 16 + 1e-12)
    with pytest.raises(ValueError):
        quantize_fp8_blocks(torch.zeros(100, 128))


def test_fp8_projection_specs_move_whole_block_projections_only():
    specs = (
        TensorSpec("text/layers/0/attention/query", (256, 256), W8, "row-split-k128-v1"),
        TensorSpec("text/layers/0/mlp/gate", (2112, 256), W8, "row-split-k128-v1"),
        TensorSpec("text/layers/0/mlp/down", (256, 2112), W8, "row-split-k128-v1"),
        TensorSpec("text/layers/0/per_layer/gate", (256, 256), W8, "row-split-k128-v1"),
        TensorSpec("text/token_embedding", (512, 256), W8, "row-split-k128-v1"),
        TensorSpec("text/layers/0/attention/output", (256, 256), BF16, "contiguous-le-v1"),
    )
    stored = {spec.name: spec for spec in fp8_projection_specs(specs)}
    assert (stored["text/layers/0/attention/query"].format, stored["text/layers/0/attention/query"].layout) \
        == (FP8_BLOCK, BLOCK128_LAYOUT)
    # Rows that are not whole 128s take a multiplier per row; a K that is not, nothing.
    assert (stored["text/layers/0/mlp/gate"].format, stored["text/layers/0/mlp/gate"].layout) \
        == (FP8_ROW_F32, ROW_SCALE_F32_LAYOUT)
    assert stored["text/layers/0/mlp/down"].format == W8
    assert stored["text/layers/0/per_layer/gate"].format == W8
    assert stored["text/token_embedding"].format == W8
    assert stored["text/layers/0/attention/output"].format == BF16


def test_mixture_fp8_keeps_its_bf16_fallback_and_quantised_experts():
    from tests.serve.test_gemma_vl import config_for
    from surogate.serve.convert.gemma_vl import convert, inventory

    g = inventory.geometry_from_config(config_for("gemma4_moe"))
    specs, _ = inventory.text_specs_and_recipes(g)
    stored = {s.name: s for s in convert._text_storage(specs, "fp8", mixture=True)}
    assert stored["text/layers/0/moe/routed_gate_up"].format == W8
    assert stored["text/layers/0/attention/query"].format in (FP8_BLOCK, FP8_ROW_F32)
    assert stored["text/layers/0/mlp/gate"].format in (FP8_BLOCK, FP8_ROW_F32)
    assert stored["text/token_embedding"].format == W8
    assert not any(s.format == W8 and s.name.startswith("text/layers/") and "/moe/routed_" not in s.name
                   for s in stored.values())


_PROJECTIONS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight")


def _per_channel_export(root, monkeypatch, convert, *, unquantised=()):
    """A compressed-tensors per-channel FP8 export of the tiny Gemma 3 (Red Hat's layout), with
    the projections in `unquantised` left BF16."""
    from tests.serve.test_gemma_vl import config_for
    from surogate.serve.convert.gemma_vl import inventory

    config = config_for("gemma3")
    g = inventory.geometry_from_config(config)
    _, text_recipes = inventory.text_specs_and_recipes(g)
    _, vision_recipes = inventory.vision_recipes(g)
    generator = torch.Generator().manual_seed(5)
    tensors = {}
    for name, source in source_requirements((*text_recipes, *vision_recipes)).items():
        if (name.startswith("model.layers.") and name.endswith(_PROJECTIONS)
                and not name.endswith(unquantised)):
            rows = source.shape[0]
            tensors[name] = (torch.rand(source.shape, generator=generator) * 8 - 4).to(torch.float8_e4m3fn)
            # multipliers far from one, so a payload that read the codes as values would show
            tensors[name[: -len("weight")] + "weight_scale"] = (
                torch.rand((rows, 1), generator=generator) * .01 + .01).to(torch.bfloat16)
        else:
            tensors[name] = (torch.randn(source.shape, generator=generator) * 0.02).to(torch.bfloat16)
    weights = {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False}
    activations = {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True, "dynamic": True}
    config["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "float-quantized",
        "config_groups": {"group_0": {"weights": weights, "input_activations": activations,
                                      "targets": ["Linear"]}},
        "ignore": ["lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
    }
    root.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(convert, "resources_for", lambda m, geometry: {spec.name: b"{}" for spec in RESOURCE_SPECS})
    monkeypatch.setattr(convert, "tokenizer_domain", lambda m: 512)
    return tensors


def test_per_channel_export_keeps_its_codes_and_row_multipliers(tmp_path, monkeypatch):
    from surogate.serve.convert.gemma_vl import convert

    tensors = _per_channel_export(tmp_path / "model", monkeypatch, convert, unquantised=("o_proj.weight",))
    out = tmp_path / "out.sinfer"
    convert.convert(tmp_path / "model", out, device="cpu")
    report = json.loads((tmp_path / "out.sinfer.conversion.json").read_text())
    assert report["text_format"] == "fp8"
    with Artifact(out) as artifact:
        query = artifact.find("text/layers/0/attention/query")
        assert (query.format, query.layout) == (FP8_ROW_F32, ROW_SCALE_F32_LAYOUT)
        stem = "model.layers.0.self_attn.q_proj."
        rows, columns = tensors[stem + "weight"].shape
        payload = bytes(artifact.payload("text/layers/0/attention/query"))
        codes = torch.frombuffer(bytearray(payload[: rows * columns]), dtype=torch.uint8).reshape(rows, columns)
        assert torch.equal(codes, tensors[stem + "weight"].view(torch.uint8))
        scale_offset = (rows * columns + 255) // 256 * 256
        scales = torch.frombuffer(bytearray(payload[scale_offset: scale_offset + 4 * rows]), dtype=torch.float32)
        assert torch.equal(scales, tensors[stem + "weight_scale"].reshape(-1).to(torch.float32))
        # The projection the export left BF16 is quantised from its values, block by block.
        output = artifact.find("text/layers/0/attention/output")
        assert output.format in (FP8_BLOCK, FP8_ROW_F32)
        assert artifact.find("text/token_embedding").format == W8


def test_per_channel_export_can_still_be_re_encoded_to_w8(tmp_path, monkeypatch):
    from surogate.serve.convert.gemma_vl import convert

    _per_channel_export(tmp_path / "model", monkeypatch, convert)
    out = tmp_path / "out.sinfer"
    convert.convert(tmp_path / "model", out, device="cpu", text_format="w8")
    with Artifact(out) as artifact:
        assert artifact.find("text/layers/0/attention/query").format == W8
        assert artifact.find("text/layers/1/mlp/down").format == W8
