"""Exercise converter entry points with complete, small checkpoint directories."""

import importlib
import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.recipe import source_requirements
from tests.serve.test_checkpoint_inventory import config_for


@pytest.mark.parametrize("family", ["qwen3", "qwen3_moe", "llama"])
def test_complete_checkpoint_conversion(tmp_path, family):
    converter = importlib.import_module(f"surogate.serve.convert.{family}.convert")
    recipe = importlib.import_module(f"surogate.serve.convert.{family}.recipe")
    config = {
        **config_for("llama" if family == "llama" else "qwen3",
                     layers=2, hidden=128, head_dim=64 if family == "llama" else 32),
        "vocab_size": 256, "tie_word_embeddings": False, "hidden_act": "silu",
        "attention_bias": False, "rope_scaling": None, "sliding_window": None,
        "use_sliding_window": False,
    }
    if family == "llama":
        del config["attention_bias"]
    if family == "qwen3_moe":
        config.update(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe",
                      moe_intermediate_size=128, num_experts=4, num_experts_per_tok=2)
    geometry = recipe.geometry_from_config(config)
    recipes = recipe.build_recipes(geometry)
    tensors = {
        name: torch.full(source.shape, 0.125, dtype=torch.bfloat16)
        for name, source in source_requirements(recipes).items()
    }
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": [],
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    output = tmp_path / "converted.sinfer"
    converter.convert(tmp_path, output, device="cpu")
    with Artifact(output) as artifact:
        assert artifact.identity.architecture == family
        assert artifact.geometry["hidden"] == 128
        assert artifact.geometry["layers"] == 2
        assert artifact.geometry["token_domain"] == 2
        assert artifact.geometry["output_rows"] == 256
        assert artifact.find("text/token_embedding").shape == (256, 128)
        if family == "qwen3_moe":
            assert artifact.geometry["experts"] == 4


def test_encoder_uses_the_serialized_sentencepiece_vocabulary():
    from sentencepiece import sentencepiece_model_pb2
    from surogate.serve.convert.gemma_embedding.convert import sentencepiece_domain

    model = sentencepiece_model_pb2.ModelProto()
    for token, kind in [("<unk>", 2), ("a", 1), ("b", 1)]:
        piece = model.pieces.add()
        piece.piece, piece.type, piece.score = token, kind, 0.0
    # tokenizer.json can carry additional HF-only tokens; the encoder stores tokenizer.model.
    resources = {"frontend/tokenizer.model": model.SerializeToString(),
                 "frontend/tokenizer.json": b'{"added_tokens":[{"id":999}]}' }
    assert sentencepiece_domain(resources) == 3


def _fp8_block(weight):
    """E4M3 codes and one multiplier per 128 x 128 block, as Qwen's FP8 exports store them."""
    n, k = weight.shape
    blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
    scales = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-8)
    codes = (blocks / scales[:, None, :, None]).reshape(n, k).to(torch.float8_e4m3fn)
    return codes, scales.to(torch.bfloat16)


def _fp8_channel(weight):
    """E4M3 codes and one multiplier per row, compressed-tensors' per-channel FP8."""
    scales = (weight.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-8)
    return (weight.float() / scales).to(torch.float8_e4m3fn), scales


def _write_dense_checkpoint(path, family, quantize):
    recipe = importlib.import_module(f"surogate.serve.convert.{family}.recipe")
    config = {
        **config_for(family, layers=2, hidden=256, head_dim=64),
        "vocab_size": 256, "tie_word_embeddings": False, "hidden_act": "silu",
        "attention_bias": False, "rope_scaling": None, "sliding_window": None,
        "use_sliding_window": False,
    }
    if family == "llama":
        del config["attention_bias"]
    generator = torch.Generator().manual_seed(7)
    tensors = {}
    for name, source in source_requirements(recipe.build_recipes(recipe.geometry_from_config(config))).items():
        value = torch.randn(source.shape, generator=generator).to(torch.bfloat16)
        if quantize is not None and name.endswith("_proj.weight"):
            codes, scales = quantize(value)
            tensors[name] = codes
            tensors[name + ("_scale_inv" if quantize is _fp8_block else "_scale")] = scales
        else:
            tensors[name] = value
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    if quantize is _fp8_block:
        config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                         "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    elif quantize is _fp8_channel:
        config["quantization_config"] = {"quant_method": "compressed-tensors", "config_groups": {
            "group_0": {"targets": ["Linear"], "weights": {
                "num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "dynamic": False},
                "input_activations": {"num_bits": 8, "type": "float", "symmetric": True,
                                      "strategy": "token", "dynamic": True}}},
            "ignore": ["lm_head"]}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": []}))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    return tensors


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_fp8_block_exports_keep_their_codes_in_every_dense_family(tmp_path, family):
    from surogate.serve.artifact.layouts import block_scale128_geometry, dequantize_row_split
    from surogate.serve.convert.common.inventory import FP8_BLOCK, W8
    converter = importlib.import_module(f"surogate.serve.convert.{family}.convert")
    stored = _write_dense_checkpoint(tmp_path / "fp8", family, _fp8_block)
    output = tmp_path / "converted.sinfer"
    report = json.loads(converter.convert(tmp_path / "fp8", output, device="cpu").read_text())
    assert report["quantization"]["fp8"] == {"codes": 2 * 4, "routed": 0, "requantize": 0}

    def words(artifact, name):
        obj = artifact.find(name)
        assert obj.format == FP8_BLOCK
        shape = tuple(obj.shape)
        geometry = block_scale128_geometry(FP8_BLOCK, shape)
        payload = bytes(artifact.payload(obj))
        codes = torch.frombuffer(bytearray(payload[:geometry.code_plane_bytes]), dtype=torch.uint8)
        scales = torch.frombuffer(bytearray(payload[geometry.scale_plane_offset:geometry.payload_bytes]),
                                  dtype=torch.float32)
        return codes.reshape(shape), scales.reshape(shape[0] // 128, shape[1] // 128)

    def codes(*names):
        return torch.cat([stored[n].view(torch.uint8) for n in names])

    def scales(*names):
        return torch.cat([stored[n + "_scale_inv"].float() for n in names])

    stem = "model.layers.1."
    with Artifact(output) as artifact:
        qkv = [stem + f"self_attn.{p}_proj.weight" for p in "qkv"]
        got, got_scales = words(artifact, "text/layers/1/attention/query_key_value")
        assert torch.equal(got, codes(*qkv)) and torch.equal(got_scales, scales(*qkv))
        gate_up = [stem + f"mlp.{p}_proj.weight" for p in ("gate", "up")]
        got, got_scales = words(artifact, "text/layers/1/mlp/gate_up")
        assert torch.equal(got, codes(*gate_up)) and torch.equal(got_scales, scales(*gate_up))
        for name, source in (("attention/output", "self_attn.o_proj"), ("mlp/down", "mlp.down_proj")):
            got, got_scales = words(artifact, "text/layers/1/" + name)
            assert torch.equal(got, codes(stem + source + ".weight"))
        # The vocabulary stays BF16 in these exports, so it takes the converter's own format.
        assert artifact.find("text/output_head").format == W8


def test_fp8_channel_exports_convert_through_their_values(tmp_path):
    from surogate.serve.artifact.layouts import dequantize_row_split
    from surogate.serve.convert.common.inventory import W8
    from surogate.serve.convert.qwen3 import convert as converter
    stored = _write_dense_checkpoint(tmp_path / "fp8", "qwen3", _fp8_channel)
    output = tmp_path / "converted.sinfer"
    report = json.loads(converter.convert(tmp_path / "fp8", output, device="cpu").read_text())
    assert report["quantization"]["fp8"] == {"codes": 0, "routed": 0, "requantize": 2 * 4}
    with Artifact(output) as artifact:
        obj = artifact.find("text/layers/0/mlp/down")
        assert obj.format == W8
        values = dequantize_row_split(bytes(artifact.payload(obj)), W8, tuple(obj.shape), dtype=torch.float32)
        name = "model.layers.0.mlp.down_proj.weight"
        reference = stored[name].float() * stored[name + "_scale"].float()
        assert torch.allclose(values, reference, rtol=0, atol=float(reference.abs().max()) / 100)


def _write_fp8_checkpoint(path, config, recipe, geometry):
    """Every Linear on the 128 x 128 grid as block FP8; embeddings, norms and taps BF16."""
    generator = torch.Generator().manual_seed(11)
    tensors = {}
    for name, source in source_requirements(recipe.build_recipes(geometry)).items():
        value = torch.randn(source.shape, generator=generator).to(torch.bfloat16)
        linear = (value.ndim == 2 and value.shape[0] % 128 == 0 and value.shape[1] % 128 == 0
                  and not any(part in name for part in ("embed", "norm", "lm_head")))
        if linear:
            tensors[name], tensors[name + "_scale_inv"] = _fp8_block(value)
        else:
            tensors[name] = value
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    config = {**config, "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                                "weight_block_size": [128, 128]}}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": []}))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))


def test_fp8_block_lfm2_keeps_codes_on_attention_convolution_and_mlp(tmp_path):
    from surogate.serve.convert.common.inventory import FP8_BLOCK
    from surogate.serve.convert.lfm2 import convert as converter, inventory, recipe
    from tests.serve.test_lfm2_variants import text_config
    config = {**text_config(), "architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2"}
    _write_fp8_checkpoint(tmp_path / "fp8", config, recipe, inventory.geometry_from_config(config))
    output = tmp_path / "converted.sinfer"
    report = json.loads(converter.convert(tmp_path / "fp8", output, device="cpu").read_text())
    assert report["quantization"]["fp8"]["requantize"] == 0
    with Artifact(output) as artifact:
        for name in ("text/layers/0/conv/in_proj", "text/layers/0/conv/out_proj",
                     "text/layers/1/attention/query_key_value", "text/layers/1/attention/output",
                     "text/layers/1/mlp/gate_up", "text/layers/1/mlp/down"):
            assert artifact.find(name).format == FP8_BLOCK, name


def test_fp8_block_gemma3_keeps_codes_on_separate_projections(tmp_path):
    from surogate.serve.convert.common.inventory import FP8_BLOCK
    from surogate.serve.convert.gemma3 import convert as converter
    from tests.serve.test_gemma3_convert import config as gemma3_config
    config = gemma3_config(num_hidden_layers=2, vocab_size=256, _sliding_window_pattern=2,
                           layer_types=["sliding_attention", "full_attention"])
    geometry = converter.geometry_from_config(config)
    _write_fp8_checkpoint(tmp_path / "fp8", config, converter, geometry)
    output = tmp_path / "converted.sinfer"
    report = json.loads(converter.convert(tmp_path / "fp8", output, device="cpu").read_text())
    assert report["quantization"]["fp8"] == {"codes": 2 * 7, "routed": 0, "requantize": 0}
    with Artifact(output) as artifact:
        for role in ("query", "key", "value", "output"):
            assert artifact.find(f"text/layers/1/attention/{role}").format == FP8_BLOCK
        for role in ("gate", "up", "down"):
            assert artifact.find(f"text/layers/1/mlp/{role}").format == FP8_BLOCK
