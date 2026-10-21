"""A ModelOpt MIXED_PRECISION export (W4A16_NVFP4 experts, shared expert and head; FP8 attention
and linear attention) converts with every stored word kept: NVFP4 codes and block scales as
stored, ModelOpt's multiplying second level bit for bit, FP8 codes with their per-tensor
multiplier, and no activation scale anywhere."""

import json
import struct

import numpy as np
import pytest
import torch

from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5_moe import inventory as inv, recipe
from surogate.serve.convert.common.recipe import source_requirements


_FP8_LINEARS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight")


def _w4a16(generator, rows, columns, second=None):
    packed = torch.randint(0, 256, (rows, columns // 2), generator=generator, dtype=torch.uint8)
    scales = (torch.rand((rows, columns // 16), generator=generator) * 300 + 0.01).to(torch.float8_e4m3fn)
    if second is None:
        second = float(torch.rand((), generator=generator)) * 1e-3 + 1e-5
    return {"weight": packed, "weight_scale": scales,
            "weight_scale_2": torch.tensor(second, dtype=torch.float32)}


def _fp8(generator, rows, columns):
    return {"weight": (torch.rand((rows, columns), generator=generator) * 8 - 4).to(torch.float8_e4m3fn),
            "weight_scale": torch.tensor(float(torch.rand((), generator=generator)) * .01 + .002),
            "input_scale": torch.tensor(float(torch.rand((), generator=generator)) * .05 + .01)}


def _save_modelopt_export(root, c, g, ranking, monkeypatch, convert, *, mutate=None):
    from safetensors.torch import save_file
    from surogate.serve.convert.common.draft_head import compute_shortlist
    generator = torch.Generator().manual_seed(5)
    tensors, layers = {}, {}
    for name, source in source_requirements(recipe.build_recipes(g)).items():
        if ".mlp.experts." in name:
            continue  # the stacked spellings; the export stores one tensor per expert
        stem = name[: -len(".weight")]
        if name.startswith("model.") and ".shared_expert." in name and name.endswith("_proj.weight"):
            # gate and up share their second level, as ModelOpt exports them
            layer = stem.split(".")[2]
            second = 2e-4 + int(layer) * 1e-5 if not stem.endswith("down_proj") else None
            for suffix, tensor in _w4a16(generator, *source.shape, second).items():
                tensors[f"{stem}.{suffix}"] = tensor
            layers[stem] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        elif name.startswith("model.") and name.endswith(_FP8_LINEARS):
            for suffix, tensor in _fp8(generator, *source.shape).items():
                tensors[f"{stem}.{suffix}"] = tensor
            layers[stem] = {"quant_algo": "FP8"}
        elif name == "lm_head.weight":
            for suffix, tensor in _w4a16(generator, *source.shape).items():
                tensors[f"{stem}.{suffix}"] = tensor
            layers["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        else:
            tensors[name] = torch.randn(source.shape, generator=generator).to(torch.bfloat16)
    for layer in range(g.layers):
        for expert in range(g.experts):
            second = 1e-4 * (1 + expert) + 1e-6 * layer
            for projection, shape in (("gate_proj", (g.intermediate, g.hidden)),
                                      ("up_proj", (g.intermediate, g.hidden)),
                                      ("down_proj", (g.hidden, g.intermediate))):
                stem = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                level = second if projection != "down_proj" else None
                for suffix, tensor in _w4a16(generator, *shape, level).items():
                    tensors[f"{stem}.{suffix}"] = tensor
        layers[f"model.layers.{layer}.mlp.experts"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    # The MTP block, which the export ignores: BF16, one tensor per expert.
    for expert in range(g.experts):
        for projection, shape in (("gate_proj", (g.intermediate, g.hidden)),
                                  ("up_proj", (g.intermediate, g.hidden)),
                                  ("down_proj", (g.hidden, g.intermediate))):
            tensors[f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"] = (
                torch.randn(shape, generator=generator).to(torch.bfloat16))
    if mutate is not None:
        mutate(tensors, layers)
    root.mkdir()
    c["quantization_config"] = {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "modelopt", "version": "0.45.0"},
        "ignore": ["mtp*", "mtp.layers.0*"],
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "quantized_layers": layers,
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


def _geometry():
    c = config_for()
    c.update(moe_intermediate_size=128, shared_expert_intermediate_size=128)
    return c, inv.geometry_from_config(c, token_domain=500)


def _bits(tensor):
    return tensor.reshape(-1).to(torch.float32).view(torch.int32)


def test_modelopt_mixed_export_keeps_every_stored_word(tmp_path, monkeypatch):
    from surogate.serve.artifact.container import Artifact
    from surogate.serve.artifact.layouts import (
        decode_nvfp4_words, dequantize_row_split, row_scale_f32_geometry)
    from surogate.serve.convert.common.inventory import FP8_ROW_F32, NVFP4_A16
    from surogate.serve.convert.qwen3_5_moe import convert
    c, g = _geometry()
    model = tmp_path / "modelopt"
    stored = _save_modelopt_export(model, c, g, tmp_path / "counts.i64", monkeypatch, convert)
    output = tmp_path / "model.sinfer"
    report = json.loads(convert.convert(model, None, output, device="cpu").read_text())
    assert report["quantization"]["modelopt"]["modules"] == {"W4A16_NVFP4": 4 + 4 * 3 + 1,
                                                             "FP8": 2 * 4 + 2 * 3}

    def nvfp4(artifact, name):
        obj = artifact.find(name)
        assert obj.format == NVFP4_A16, name
        codes, scales, word = decode_nvfp4_words(bytes(artifact.payload(obj)), tuple(obj.shape))
        return codes, scales, word.view(torch.int32)

    def fp32(artifact, name):
        return torch.frombuffer(bytearray(artifact.payload(artifact.find(name))), dtype=torch.float32)

    with Artifact(output) as artifact:
        assert artifact.identity.weights_id == "modelopt"
        names = {obj.name for obj in artifact.objects}
        # Weight-only: no activation divisor and no routed activation scale or alpha anywhere.
        assert not any(n.endswith(("/input_scale_divisor", "_act_scale", "_alpha")) for n in names)
        for layer in range(g.layers):
            # The routed banks, gate then up per expert, as stored; the identity word; each
            # expert's second level bit for bit.
            codes, scales, word = nvfp4(artifact, f"text/layers/{layer}/moe/routed_gate_up")
            assert int(word) == struct.unpack("<i", struct.pack("<f", 1.0))[0]
            expected_codes, expected_scales, expected_second = [], [], []
            for expert in range(g.experts):
                for projection in ("gate_proj", "up_proj"):
                    stem = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                    expected_codes.append(stored[stem + ".weight"])
                    expected_scales.append(stored[stem + ".weight_scale"].view(torch.uint8))
                    expected_second.append(stored[stem + ".weight_scale_2"])
            assert torch.equal(codes, torch.cat(expected_codes))
            assert torch.equal(scales, torch.cat(expected_scales))
            assert torch.equal(_bits(fp32(artifact, f"text/layers/{layer}/moe/routed_gate_up_scale")),
                               _bits(torch.stack(expected_second)))
            codes, scales, _ = nvfp4(artifact, f"text/layers/{layer}/moe/routed_down")
            stems = [f"model.layers.{layer}.mlp.experts.{e}.down_proj" for e in range(g.experts)]
            assert torch.equal(codes, torch.cat([stored[s + ".weight"] for s in stems]))
            assert torch.equal(_bits(fp32(artifact, f"text/layers/{layer}/moe/routed_down_scale")),
                               _bits(torch.stack([stored[s + ".weight_scale_2"] for s in stems])))
            # The shared expert's gate|up kept whole: one NVFP4_A16 object, its shared second
            # level in the trailing word.
            codes, scales, word = nvfp4(artifact, f"text/layers/{layer}/moe/shared_gate_up")
            stem = f"model.layers.{layer}.mlp.shared_expert."
            assert torch.equal(codes, torch.cat([stored[stem + "gate_proj.weight"],
                                                 stored[stem + "up_proj.weight"]]))
            assert torch.equal(word, _bits(stored[stem + "gate_proj.weight_scale_2"])[0])
            _, _, word = nvfp4(artifact, f"text/layers/{layer}/moe/shared_down")
            assert torch.equal(word, _bits(stored[stem + "down_proj.weight_scale_2"])[0])
        codes, scales, word = nvfp4(artifact, "text/output_head")
        assert torch.equal(codes, stored["lm_head.weight"])
        assert torch.equal(scales, stored["lm_head.weight_scale"].view(torch.uint8))
        assert torch.equal(word, _bits(stored["lm_head.weight_scale_2"])[0])
        # FP8 Linears: codes as stored, the per-tensor multiplier on every row.
        for name, source in (("text/layers/0/attention/key", "model.layers.0.self_attn.k_proj"),
                             ("text/layers/2/attention/output", "model.layers.2.self_attn.o_proj"),
                             ("text/layers/1/gdn/query_key_value", "model.layers.1.linear_attn.in_proj_qkv"),
                             ("text/layers/1/gdn/z", "model.layers.1.linear_attn.in_proj_z"),
                             ("text/layers/3/gdn/output", "model.layers.3.linear_attn.out_proj")):
            obj = artifact.find(name)
            assert obj.format == FP8_ROW_F32, name
            geometry = row_scale_f32_geometry(obj.format, tuple(obj.shape))
            payload = bytearray(artifact.payload(obj))
            codes = torch.frombuffer(payload[: geometry.code_plane_bytes], dtype=torch.uint8)
            begin = geometry.scale_plane_offset
            rows = torch.frombuffer(payload[begin: begin + geometry.scale_plane_bytes], dtype=torch.float32)
            assert torch.equal(codes.reshape(obj.shape), stored[source + ".weight"].view(torch.uint8)), name
            assert torch.equal(rows, stored[source + ".weight_scale"].expand(obj.shape[0])), name
        # The text core the export left in BF16 stays BF16.
        embedding = artifact.find("text/token_embedding")
        assert embedding.format == inv.BF16
        assert torch.equal(torch.frombuffer(bytearray(artifact.payload(embedding)), dtype=torch.bfloat16)
                           .reshape(embedding.shape), stored["model.embed_tokens.weight"])
        # The draft head gathers rows of the NVFP4 head: e2m1 * block * weight_scale_2.
        ids = torch.frombuffer(bytearray(artifact.payload(artifact.find("text/draft_head_token_ids"))),
                               dtype=torch.int32).long()
        draft = artifact.find("text/draft_head")
        values = dequantize_row_split(bytes(artifact.payload(draft)), draft.format, tuple(draft.shape),
                                      dtype=torch.float32)
        from surogate.serve.convert.common import nvfp4 as words
        head = words.dequantize(stored["lm_head.weight"], stored["lm_head.weight_scale"], 1.0) * float(
            stored["lm_head.weight_scale_2"])
        assert torch.allclose(values, head[ids], rtol=0, atol=float(head.abs().max()) / 8)
        # The MTP block's per-expert BF16 experts, stacked gate then up, in the base W8.
        gate_up = artifact.find("mtp/layer/moe/routed_gate_up")
        assert gate_up.format == inv.W8
        values = dequantize_row_split(bytes(artifact.payload(gate_up)), gate_up.format,
                                      tuple(gate_up.shape), dtype=torch.float32)
        reference = torch.cat([torch.cat([stored[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"],
                                          stored[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"]])
                               for e in range(g.experts)]).float()
        assert torch.allclose(values, reference, rtol=0, atol=float(reference.abs().max()) / 64)


@pytest.mark.parametrize("problem", ["gate_up_disagree", "activation_scale", "unclaimed_fp8"])
def test_modelopt_export_that_disagrees_with_itself_is_refused(tmp_path, monkeypatch, problem):
    from surogate.serve.convert.qwen3_5_moe import convert
    c, g = _geometry()

    def mutate(tensors, layers):
        if problem == "gate_up_disagree":
            tensors["model.layers.1.mlp.experts.2.up_proj.weight_scale_2"] = torch.tensor(3e-4)
        elif problem == "activation_scale":
            tensors["lm_head.input_scale"] = torch.tensor(0.5)
        else:
            del layers["model.layers.0.self_attn.k_proj"]

    _save_modelopt_export(tmp_path / "modelopt", c, g, tmp_path / "counts.i64", monkeypatch, convert,
                          mutate=mutate)
    with pytest.raises(ValueError, match="gate and up|input_scale|no FP8 entry"):
        convert.convert(tmp_path / "modelopt", None, tmp_path / "model.sinfer", device="cpu")


def test_modelopt_other_algorithms_are_refused():
    from surogate.serve.convert.qwen3_5_moe.exports import modelopt_source
    with pytest.raises(ValueError, match="MIXED_PRECISION"):
        modelopt_source.is_modelopt_export({"quantization_config": {"quant_method": "modelopt",
                                                                    "quant_algo": "NVFP4"}})
    assert not modelopt_source.is_modelopt_export({"quantization_config": {"quant_method": "fp8"}})


def test_a_stale_tokenizer_config_template_yields_to_the_jinja(tmp_path):
    """Transformers loads chat_template.jinja over tokenizer_config.json's key, so where the two
    disagree the artifact keeps the file and drops the stale key; agreeing copies stay as stored."""
    from surogate.serve.convert.common import conversion
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "generation_config.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("{{ new }}")
    stored = json.dumps({"pad_token": "<pad>", "chat_template": "{{ old }}"})
    (tmp_path / "tokenizer_config.json").write_text(stored)
    resources = {r.name: r.data for r in conversion.load_resources(tmp_path, inv.RESOURCE_SPECS)}
    assert resources["frontend/chat_template.jinja"] == b"{{ new }}"
    assert json.loads(resources["frontend/tokenizer_config.json"]) == {"pad_token": "<pad>"}
    agreeing = json.dumps({"pad_token": "<pad>", "chat_template": "{{ new }}"})
    (tmp_path / "tokenizer_config.json").write_text(agreeing)
    resources = {r.name: r.data for r in conversion.load_resources(tmp_path, inv.RESOURCE_SPECS)}
    assert resources["frontend/tokenizer_config.json"] == agreeing.encode()
