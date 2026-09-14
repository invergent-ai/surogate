"""Granite 4.2 checkpoint geometry and lossless GGUF frontend/layout handling."""

import json
from types import SimpleNamespace

import pytest
import torch

from surogate.serve.convert.llama.convert import geometry_block, validate_config
from surogate.serve.gguf.bridge import (
    _has_export_transform, _invert_export_transform, _unpermute, synthesised_config,
)
from surogate.serve.gguf.frontend import extract_tokenizer_json
from surogate.serve.ingest import converter_for_config


def config(size="3b"):
    # https://huggingface.co/ibm-granite/granite-4.2-{3b,8b,30b}/blob/main/config.json
    hidden, layers, intermediate, heads, theta = {
        "3b": (2560, 40, 8192, 40, 10_000_000),
        "8b": (4096, 40, 12800, 32, 10_000_000),
        "30b": (4096, 64, 32768, 32, 50_000_000),
    }[size]
    return dict(architectures=["GraniteForCausalLM"], model_type="granite",
                hidden_size=hidden, num_hidden_layers=layers, intermediate_size=intermediate,
                num_attention_heads=heads, num_key_value_heads=8, vocab_size=100352,
                attention_multiplier=heads / hidden, embedding_multiplier=1.0,
                residual_multiplier=1.0, logits_scaling=1.0, max_position_embeddings=131072,
                rms_norm_eps=1e-5, rope_theta=theta, rope_scaling=None,
                rope_parameters={"rope_type": "default", "rope_theta": theta},
                hidden_act="silu", attention_bias=False, mlp_bias=False, tie_word_embeddings=False)


@pytest.mark.parametrize("size", ["3b", "8b", "30b"])
def test_granite_checkpoint_geometry(size):
    cfg = config(size)
    assert converter_for_config(cfg).display == "Granite 4.2"
    geometry, summary = validate_config(cfg)
    actual = geometry_block(SimpleNamespace(geometry=geometry), token_domain=100352)
    assert summary["architecture"] == "GraniteForCausalLM"
    assert actual["attention_scale"] == cfg["attention_multiplier"]
    assert actual["attention_scale"] == 1 / actual["head_dim"]
    assert actual["max_context"] == 131072
    assert actual["rope_theta"] == cfg["rope_theta"]
    assert actual["layers"] == cfg["num_hidden_layers"]


@pytest.mark.parametrize("change", [
    {"attention_multiplier": 0}, {"attention_multiplier": float("nan")},
    {"attention_multiplier": True}, {"residual_multiplier": 0.25},
    {"embedding_multiplier": 2}, {"logits_scaling": 8}, {"attention_bias": True},
    {"mlp_bias": True}, {"rope_parameters": {"rope_type": "linear", "factor": 4}},
    {"rope_parameters": {"rope_type": "default", "rope_theta": 10_000}},
])
def test_granite_does_not_silently_drop_unsupported_arithmetic(change):
    with pytest.raises(ValueError):
        validate_config({**config(), **change})


def gguf_reader():
    fields = {
        "granite.embedding_length": 2560, "granite.block_count": 40,
        "granite.attention.head_count": 40, "granite.attention.head_count_kv": 8,
        "granite.feed_forward_length": 8192, "granite.context_length": 131072,
        "granite.attention.scale": 1 / 64, "granite.attention.layer_norm_rms_epsilon": 1e-5,
        "granite.rope.freq_base": 10_000_000,
        "tokenizer.ggml.tokens": ["a", "b", "<s>", "</s>"],
        "tokenizer.ggml.token_type": [1, 1, 3, 3],
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "granite-docling",
        "tokenizer.ggml.eos_token_id": 3, "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.merges": [],
    }
    return SimpleNamespace(
        kv=lambda key, default=None: fields.get(key, default),
        get_field=lambda key: SimpleNamespace(contents=lambda: fields[key]) if key in fields else None,
        tensor=lambda name: None,
    )


def test_granite_gguf_preserves_model_arithmetic():
    cfg = synthesised_config(gguf_reader(), "granite")
    geometry, _ = validate_config(cfg)
    assert geometry_block(SimpleNamespace(geometry=geometry), token_domain=4)["attention_scale"] == 1 / 64
    assert cfg["model_type"] == "granite"


@pytest.mark.parametrize("projection,heads", [("q", 40), ("k", 8)])
def test_granite_gguf_rotary_row_order(projection, heads):
    # Reproduce llama.cpp's inherited Llama export permutation independently.
    source = torch.arange(heads * 64 * 32).reshape(heads * 64, 32)
    exported = source.reshape(heads, 2, 32, 32).transpose(1, 2).reshape_as(source)
    name = f"model.layers.0.self_attn.{projection}_proj.weight"
    assert _has_export_transform("granite", name)
    restored = _invert_export_transform("granite", name, exported, heads=40, kv_heads=8)
    torch.testing.assert_close(restored, source)
    gather = _unpermute(torch.arange(source.shape[0]), heads)
    torch.testing.assert_close(exported[gather], source)


def test_granite_tokenizer_keeps_gpt2_splits_and_unicode():
    from tokenizers import Tokenizer
    from tokenizers.pre_tokenizers import ByteLevel

    data = extract_tokenizer_json(gguf_reader())
    assert data["normalizer"] is None
    tokenizer = Tokenizer.from_str(json.dumps(data))
    reference = ByteLevel(add_prefix_space=False, use_regex=True)
    for text in ("e\u0301 é", "a  b\n\n\t", "123456 I'M I'm", "日本語🙂"):
        actual = [part for part, _ in tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        expected = [part for part, _ in reference.pre_tokenize_str(text)]
        assert actual == expected


def test_safetensors_index_with_stale_shard_assignments(tmp_path):
    from safetensors.torch import save_file
    from surogate.serve.convert.common.safetensors import ShardReader

    save_file({"model.a.weight": torch.tensor([1.0])}, tmp_path / "a.safetensors")
    save_file({"model.b.weight": torch.tensor([2.0])}, tmp_path / "b.safetensors")
    index = {"weight_map": {"model.a.weight": "b.safetensors", "model.b.weight": "a.safetensors"}}
    source = json.dumps(index)
    (tmp_path / "model.safetensors.index.json").write_text(source)
    with ShardReader(tmp_path) as reader:
        assert reader.metadata(["model.a.weight"])["model.a.weight"].shard == "a.safetensors"
        assert reader.get("model.b.weight").item() == 2
    assert (tmp_path / "model.safetensors.index.json").read_text() == source


@pytest.mark.parametrize("duplicate", [True, False])
def test_safetensors_index_does_not_hide_invalid_shards(tmp_path, duplicate):
    from safetensors.torch import save_file
    from surogate.serve.convert.common.safetensors import ShardReader

    save_file({"a": torch.ones(1)}, tmp_path / "a.safetensors")
    save_file({"a" if duplicate else "b": torch.ones(1)}, tmp_path / "b.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "a.safetensors", "missing": "b.safetensors"}}))
    with pytest.raises(ValueError, match="multiple|missing"):
        ShardReader(tmp_path)
