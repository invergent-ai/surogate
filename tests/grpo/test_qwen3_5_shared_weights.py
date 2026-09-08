"""Hybrid sharing preserves HF/trainer layouts, dtypes and config dimensions."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from surogate.grpo.shared_weights import borrow_weights, qwen3_5_bindings, shared_family, write_shared_artifact
from surogate.serve.artifact.container import Artifact


def config_for(*, nested=True, tied=True, hidden=128, value_heads=4):
    text = dict(architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text",
                hidden_size=hidden, intermediate_size=256, num_hidden_layers=3,
                num_attention_heads=2, num_key_value_heads=1, head_dim=128, vocab_size=512,
                max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=tied,
                linear_num_key_heads=2, linear_key_head_dim=32,
                linear_num_value_heads=value_heads, linear_value_head_dim=64, linear_conv_kernel_dim=4,
                layer_types=["linear_attention", "full_attention", "linear_attention"],
                mtp_num_hidden_layers=1, eos_token_id=1,
                rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=.5,
                                     mrope_section=[11, 11, 10], mrope_interleaved=True))
    return dict(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5", text_config=text) if nested else text


@pytest.mark.parametrize("nested,tied,hidden,value_heads", [(True, True, 128, 4), (False, False, 384, 8)])
def test_index_and_views_preserve_native_storage(tmp_path, nested, tied, hidden, value_heads):
    config = config_for(nested=nested, tied=tied, hidden=hidden, value_heads=value_heads)
    bindings = qwen3_5_bindings(config)
    tensors = {}
    for b in bindings:
        for name, shape in b.sources:
            # Official checkpoints mix the storage dtypes of these small vectors.
            dtype = torch.float32 if name.endswith(("A_log", "linear_attn.norm.weight")) else torch.bfloat16
            tensors[name] = (torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 127).to(dtype) / 128
    names = list(tensors)
    index = {}
    for i, group in enumerate((names[::2], names[1::2])):
        filename = f"model-{i}.safetensors"
        save_file({name: tensors[name] for name in group}, tmp_path / filename)
        index.update({name: filename for name in group})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    output = tmp_path / "shared.sinfer"
    assert write_shared_artifact(tmp_path, output) == bindings
    parameters = {}
    for b in bindings:
        if b.parameter in parameters:
            continue
        source = tensors[b.sources[0][0]]
        if b.parameter.endswith("mlp_up_weight"):
            gate = b.sources[0][0]
            source = torch.cat([tensors[gate.replace("gate_proj", "up_proj")], tensors[gate]])
        parameters[b.parameter] = source.to(torch.float32 if b.dtype == "FP32" else torch.bfloat16)
    views = borrow_weights(SimpleNamespace(get_shared_base_weights=lambda: parameters), bindings)
    with Artifact(output) as artifact:
        assert artifact.identity.architecture == "qwen3_5"
        assert artifact.geometry["mtp_layers"] == 0
        assert list(artifact.layer_types) == config.get("text_config", config)["layer_types"]
        for b in bindings:
            obj = artifact.find(b.name)
            raw = bytearray()
            for source, offset, size in obj.runs:
                with open(artifact.external[source - 1][0], "rb") as stream:
                    stream.seek(offset)
                    raw.extend(stream.read(size))
            if not obj.runs:
                raw.extend(artifact.payload(obj))
            expected = torch.frombuffer(raw, dtype=views[b.name].dtype).reshape(b.shape)
            assert torch.equal(views[b.name], expected), b.name
            assert views[b.name].untyped_storage().data_ptr() == parameters[b.parameter].untyped_storage().data_ptr()
        assert all(not obj.name.startswith(("vision/", "mtp/", "text/draft")) for obj in artifact.objects)
    assert output.stat().st_size < 50000
    assert views["text/layers/0/gdn/convolution_taps"].shape == (128 + 64 * value_heads, 1, 4)
    assert views["text/layers/0/gdn/dt_bias"].dtype == torch.float32
    if tied:
        assert views["text/token_embedding"].data_ptr() == views["text/output_head"].data_ptr()
    else:
        assert views["text/token_embedding"].data_ptr() != views["text/output_head"].data_ptr()


@pytest.mark.parametrize("missing", ["hidden_size", "head_dim", "linear_num_value_heads", "linear_value_head_dim"])
def test_dimensions_are_required(missing):
    config = config_for()
    del config["text_config"][missing]
    with pytest.raises(ValueError, match=missing):
        qwen3_5_bindings(config)


def test_shared_family_rejects_moe_and_nested_quantization():
    config = config_for()
    assert shared_family(config) == "qwen3_5"
    config["text_config"]["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="unquantized dense"):
        shared_family(config)
    config["text_config"].pop("quantization_config")
    config["text_config"]["num_experts"] = 8
    with pytest.raises(ValueError, match="unquantized dense"):
        shared_family(config)
