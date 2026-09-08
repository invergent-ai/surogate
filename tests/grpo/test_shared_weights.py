"""The shared index describes the actual checkpoint and preserves tensor identity."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from surogate.grpo.shared_weights import qwen3_bindings, write_shared_artifact, borrow_weights
from surogate.serve.artifact.container import Artifact


def config_for(*, tied=True, layers=2, hidden=128):
    return dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_act="silu",
                hidden_size=hidden, intermediate_size=256, num_hidden_layers=layers,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=256,
                max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1000000,
                tie_word_embeddings=tied, attention_bias=False, rope_scaling=None,
                sliding_window=None, use_sliding_window=False, eos_token_id=1)


def checkpoint(root, config, *, sharded=False):
    (root / "config.json").write_text(json.dumps(config))
    tensors = {}
    for binding in qwen3_bindings(config):
        for name, shape in binding.sources:
            tensors[name] = torch.full(shape, len(tensors) / 32, dtype=torch.bfloat16)
    if sharded:
        names = list(tensors)
        weight_map = {}
        for i, group in enumerate((names[::2], names[1::2])):
            filename = f"model-{i}.safetensors"
            save_file({name: tensors[name] for name in group}, root / filename)
            weight_map.update({name: filename for name in group})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(tensors, root / "model.safetensors")
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    return tensors


@pytest.mark.parametrize("tied,sharded", [(True, False), (False, True)])
def test_index_points_to_checkpoint_and_training_views(tmp_path, tied, sharded):
    config = config_for(tied=tied)
    tensors = checkpoint(tmp_path, config, sharded=sharded)
    output = tmp_path / "shared.sinfer"
    bindings = write_shared_artifact(tmp_path, output)
    parameters = {}
    for binding in bindings:
        if binding.parameter in parameters:
            continue
        if binding.parameter.endswith("mlp_up_weight"):
            layer = int(binding.parameter.split("[")[1].split("]")[0])
            prefix = f"model.layers.{layer}.mlp."
            parameters[binding.parameter] = torch.cat([tensors[prefix + "up_proj.weight"], tensors[prefix + "gate_proj.weight"]])
        else:
            parts = [tensors[name] for name, _ in binding.sources]
            parameters[binding.parameter] = parts[0] if len(parts) == 1 else torch.cat(parts)
    trainer = SimpleNamespace(get_shared_base_weights=lambda: parameters)
    views = borrow_weights(trainer, bindings)
    with Artifact(output) as artifact:
        for binding in bindings:
            obj = artifact.find(binding.name)
            raw = bytearray()
            for source, offset, size in obj.runs:
                with open(artifact.external[source - 1][0], "rb") as stream:
                    stream.seek(offset)
                    raw.extend(stream.read(size))
            expected = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(binding.shape)
            assert torch.equal(expected, views[binding.name])
            assert views[binding.name].untyped_storage().data_ptr() == parameters[binding.parameter].untyped_storage().data_ptr()
    if tied:
        assert views["text/token_embedding"].data_ptr() == views["text/output_head"].data_ptr()
    assert output.stat().st_size < 30000  # metadata and resources only


@pytest.mark.parametrize("field", ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"])
def test_no_dimension_presets(field):
    config = config_for()
    del config[field]
    with pytest.raises(ValueError, match=field):
        qwen3_bindings(config)


def test_non_bf16_checkpoint_is_rejected(tmp_path):
    config = config_for()
    tensors = checkpoint(tmp_path, config)
    tensors["model.norm.weight"] = tensors["model.norm.weight"].float()
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="BF16 model.norm.weight"):
        write_shared_artifact(tmp_path, tmp_path / "shared.sinfer")
