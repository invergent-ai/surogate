"""LFM GGUF metadata, tensor layout, tokenizer IDs and complete conversion."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from gguf.quants import dequantize, quantize
from surogate.serve import ingest
from surogate.serve.artifact.container import Artifact
from surogate.serve.artifact.layouts import dequantize_row_split
from surogate.serve.convert.lfm2 import convert
from surogate.serve.gguf.bridge import (
    build_hf_dir_from_gguf, gguf_target_key, open_gguf, synthesised_config,
)
from surogate.serve.gguf.frontend import extract_tokenizer_json


def checkpoint(path, *, hidden=256, tied=True, quantized=True, metadata=None, moe=False):
    arch = "lfm2moe" if moe else "lfm2"
    writer = gguf.GGUFWriter(str(path), arch)
    fields = {
        "lfm2.block_count": 3, "lfm2.embedding_length": hidden,
        "lfm2.feed_forward_length": 512, "lfm2.attention.head_count": 4,
        "lfm2.attention.head_count_kv": [0, 2, 0], "lfm2.shortconv.l_cache": 5,
        "lfm2.context_length": 8192, "lfm2.rope.freq_base": 123456.,
        "lfm2.attention.layer_norm_rms_epsilon": 1e-5,
    }
    fields = {key.replace("lfm2.", arch + "."): value for key, value in fields.items()}
    if moe:
        fields.update({f"{arch}.{key}": value for key, value in {
            "expert_count": 8, "expert_used_count": 2, "expert_feed_forward_length": 128,
            "leading_dense_block_count": 1, "expert_gating_func": 2,
        }.items()})
    fields.update(metadata or {})
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_uint32(key, value)
    tokens = ["<bos>", "a", "b", "ab", "<eos>"] + [f"t{i}" for i in range(5, 32)]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("lfm2")
    writer.add_token_list(tokens)
    writer.add_token_types([3, 1, 1, 1, 3] + [1] * 27)
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(4)
    writer.add_add_bos_token(True)
    writer.add_chat_template("{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}")
    rng = np.random.default_rng(41)
    stored = {}

    def add(name, shape, *, matrix=True, kind=None):
        array = rng.normal(0, 0.05, shape).astype(np.float32)
        dtype = kind or (gguf.GGMLQuantizationType.Q8_0 if quantized and matrix else
                         gguf.GGMLQuantizationType.F32)
        data = quantize(array, dtype)
        writer.add_tensor(name, data, raw_dtype=dtype)
        stored[name] = (data, dtype, array)

    add("token_embd.weight", (32, hidden))
    add("token_embd_norm.weight", (hidden,), matrix=False)
    if not tied:
        add("output.weight", (32, hidden))
    for i in range(3):
        p = f"blk.{i}."
        for name in ("attn_norm", "ffn_norm"):
            add(p + name + ".weight", (hidden,), matrix=False)
        if moe and i >= 1:
            add(p + "ffn_gate_inp.weight", (8, hidden), matrix=False)
            add(p + "exp_probs_b.bias", (8,), matrix=False)
            add(p + "ffn_gate_exps.weight", (8, 128, hidden))
            add(p + "ffn_up_exps.weight", (8, 128, hidden))
            add(p + "ffn_down_exps.weight", (8, hidden, 128))
        else:
            for name, shape in (("ffn_gate", (512, hidden)), ("ffn_up", (512, hidden)),
                                ("ffn_down", (hidden, 512))):
                # Gate and up use different GGUF types, both exactly representable in W8.
                kind = gguf.GGMLQuantizationType.Q4_0 if quantized and name == "ffn_up" else None
                add(p + name + ".weight", shape, kind=kind)
        if i == 1:
            for name, rows in (("attn_q", hidden), ("attn_k", hidden // 2),
                               ("attn_v", hidden // 2), ("attn_output", hidden)):
                add(p + name + ".weight", (rows, hidden))
            for name in ("attn_q_norm", "attn_k_norm"):
                add(p + name + ".weight", (hidden // 4,), matrix=False)
        else:
            add(p + "shortconv.conv.weight", (hidden, 5), matrix=False)
            add(p + "shortconv.in_proj.weight", (3 * hidden, hidden))
            add(p + "shortconv.out_proj.weight", (hidden, hidden))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return stored


@pytest.mark.parametrize("hidden,tied", [(256, True), (512, False)])
def test_metadata_defines_geometry_and_head_tying(tmp_path, hidden, tied):
    path = tmp_path / "renamed.gguf"
    checkpoint(path, hidden=hidden, tied=tied)
    assert gguf_target_key(path) == "lfm2"
    with open_gguf(path) as reader:
        config = synthesised_config(reader, "lfm2")
    geometry = convert.validate_config(config)
    assert (geometry.hidden, geometry.intermediate, geometry.conv_kernel) == (hidden, 512, 5)
    assert geometry.attention_layers == (1,)
    assert geometry.kv_heads == 2
    assert config["tie_word_embeddings"] is tied
    assert config["rope_theta"] == 123456.
    assert ingest.converter_for_config(config).gguf_repack


@pytest.mark.parametrize("metadata,match", [
    ({"lfm2.attention.head_count_kv": [0, 2]}, "each layer"),
    ({"lfm2.attention.head_count_kv": [0, 2, 4]}, "uniform"),
    ({"lfm2.shortconv.l_cache": None}, "conv_L_cache"),
    ({"lfm2.attention.sliding_window": 128}, "sliding-window"),
    ({"lfm2.attention.causal": False}, "embedding models"),
    ({"lfm2.rope.dimension_count": 32}, "rope.dimension_count"),
    ({"lfm2.rope.freq_base": None}, "rope.freq_base"),
    ({"lfm2.attention.layer_norm_rms_epsilon": 0.0}, "positive finite"),
])
def test_unsupported_or_incomplete_metadata_is_rejected(tmp_path, metadata, match):
    path = tmp_path / "invalid.gguf"
    checkpoint(path, metadata=metadata)
    with open_gguf(path) as reader, pytest.raises(ValueError, match=match):
        synthesised_config(reader, "lfm2")


def test_special_tokens_keep_their_existing_vocabulary_ids(tmp_path):
    path = tmp_path / "tokenizer.gguf"
    checkpoint(path)
    with open_gguf(path) as reader:
        tokenizer = extract_tokenizer_json(reader)
    assert tokenizer["model"]["vocab"]["<bos>"] == 0
    assert tokenizer["model"]["vocab"]["ab"] == 3
    assert [(t["content"], t["id"]) for t in tokenizer["added_tokens"]] == [("<bos>", 0), ("<eos>", 4)]
    assert tokenizer["normalizer"] is None
    tokenizers = pytest.importorskip("tokenizers")
    parsed = tokenizers.Tokenizer.from_str(json.dumps(tokenizer))
    assert parsed.encode("<bos>ab<eos>", add_special_tokens=False).ids == [0, 3, 4]


@pytest.mark.parametrize("quantized,tied", [(True, False), (False, True)])
def test_complete_gguf_conversion_preserves_weights(tmp_path, monkeypatch, quantized, tied):
    # Even when generic native GGUF splitting is enabled, LFM retains its W8 profile.
    monkeypatch.setenv("SUROGATE_GGUF_SPLIT_HALVES", "1")
    path = tmp_path / "model.gguf"
    stored = checkpoint(path, quantized=quantized, tied=tied)
    model = build_hf_dir_from_gguf(
        path, "lfm2", tmp_path / "bridge",
        repack_planner=ingest._repack_planner(Path(__file__).resolve().parents[2], "lfm2"),
    )
    out = tmp_path / "model.sinfer"
    repack = model / "gguf_repack.json"
    convert.convert(model, out, device="cpu", gguf_repack=repack if repack.exists() else None)
    with Artifact(out) as artifact:
        assert artifact.identity.architecture == "lfm2"
        assert artifact.geometry["intermediate"] == 512
        assert artifact.layer_types == ["linear_attention", "full_attention", "linear_attention"]
        conv = stored["blk.0.shortconv.conv.weight"][2]
        expected = torch.from_numpy(conv).T.contiguous().to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        assert bytes(artifact.payload("text/layers/0/conv/convolution")) == expected
        if quantized:
            for name, sources in {
                "text/layers/0/conv/in_proj": ["blk.0.shortconv.in_proj.weight"],
                "text/layers/1/attention/query_key_value": [f"blk.1.attn_{s}.weight" for s in ("q", "k", "v")],
                "text/layers/0/mlp/gate_up": ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
                "text/output_head": ["output.weight"],
            }.items():
                obj = artifact.find(name)
                assert obj.format == "W8G32_F16S"
                decoded = dequantize_row_split(artifact.payload(obj), obj.format, obj.shape, dtype=torch.float32)
                expected = np.concatenate([dequantize(stored[s][0], stored[s][1]) for s in sources])
                np.testing.assert_array_equal(decoded.numpy(), expected)


@pytest.mark.parametrize("hidden", [256, 512])
def test_complete_moe_gguf_conversion(tmp_path, hidden):
    from surogate.serve.convert.lfm2_moe import convert as moe_convert
    path = tmp_path / "arbitrary-filename.gguf"
    stored = checkpoint(path, hidden=hidden, moe=True)
    assert gguf_target_key(path) == "lfm2_moe"
    model = build_hf_dir_from_gguf(path, "lfm2_moe", tmp_path / "bridge",
        repack_planner=ingest._repack_planner(Path(__file__).resolve().parents[2], "lfm2_moe"))
    config = json.loads((model / "config.json").read_text())
    assert converter_for_moe(config).hidden == hidden
    output = tmp_path / "model.sinfer"
    repack = model / "gguf_repack.json"
    moe_convert.convert(model, output, device="cpu", gguf_repack=repack if repack.exists() else None)
    with Artifact(output) as artifact:
        assert artifact.identity.architecture == "lfm2_moe"
        assert artifact.geometry["intermediate"] == 128
        assert artifact.geometry["dense_intermediate"] == 512
        assert artifact.geometry["leading_dense_layers"] == 1
        bias = stored["blk.2.exp_probs_b.bias"][2]
        assert bytes(artifact.payload("text/layers/2/moe/router_bias")) == bias.tobytes()
        for i in (1, 2):
            for name, source_names in {
                "routed_gate_up": ["ffn_gate_exps", "ffn_up_exps"],
                "routed_down": ["ffn_down_exps"],
            }.items():
                obj = artifact.find(f"text/layers/{i}/moe/{name}")
                actual = dequantize_row_split(artifact.payload(obj), obj.format, obj.shape, dtype=torch.float32)
                expected = np.concatenate([dequantize(stored[f"blk.{i}.{n}.weight"][0],
                    stored[f"blk.{i}.{n}.weight"][1]) for n in source_names], axis=1)
                np.testing.assert_array_equal(actual.numpy(), expected.reshape(obj.shape))


def converter_for_moe(config):
    from surogate.serve.convert.lfm2_moe import inventory
    assert ingest.converter_for_config(config).key == "lfm2_moe"
    return inventory.geometry_from_config(config)
