# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Structural tests for the GGUF ingest bridge (surogate/cli/serve_gguf.py):
# a synthetic GGUF written with gguf-py round-trips through the bridge's
# primitives — KV summary, gguf->HF tensor-name inversion, reversed-dims
# shape restore, and dequantization against gguf-py's own reference.
# CPU-only; no GPU, no model downloads.

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize

from surogate.cli.serve_gguf import _hf_name_map, gguf_target_key, read_gguf_summary


ARCH = "qwen3"  # canonical small dense arch present in gguf-py's mapping table


@pytest.fixture()
def mini_gguf(tmp_path):
    path = tmp_path / "mini.gguf"
    writer = GGUFWriter(str(path), ARCH)
    writer.add_block_count(1)
    writer.add_embedding_length(64)
    writer.add_head_count(4)

    rng = np.random.default_rng(7)
    embd = rng.standard_normal((100, 64), dtype=np.float32)  # HF order [vocab, hidden]
    attnq = rng.standard_normal((64, 64), dtype=np.float32)
    q8 = quantize(attnq, gguf.GGMLQuantizationType.Q8_0)

    writer.add_tensor("token_embd.weight", embd)
    writer.add_tensor(
        "blk.0.attn_q.weight", q8, raw_shape=q8.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path, embd, q8


def test_summary_reads_kv(mini_gguf):
    path, _, _ = mini_gguf
    s = read_gguf_summary(path)
    assert s["architecture"] == ARCH
    assert s["hidden_size"] == 64
    assert s["num_hidden_layers"] == 1
    assert s["tensor_count"] == 2
    assert set(s["quant_types"]) == {"F32", "Q8_0"}


def test_name_map_inverts_to_canonical_hf_names():
    nm = _hf_name_map(ARCH, 1)
    assert nm["token_embd.weight"] == "model.embed_tokens.weight"
    assert nm["blk.0.attn_q.weight"] == "model.layers.0.self_attn.q_proj.weight"
    assert nm["blk.0.ffn_down.weight"] == "model.layers.0.mlp.down_proj.weight"
    assert nm["output_norm.weight"] == "model.norm.weight"


def test_dequant_and_shape_restore(mini_gguf):
    path, embd, q8 = mini_gguf
    reader = GGUFReader(str(path), "r")
    by_name = {t.name: t for t in reader.tensors}

    t = by_name["token_embd.weight"]
    back = np.asarray(dequantize(t.data, t.tensor_type))
    back = back.reshape(tuple(reversed(t.shape.tolist())))
    assert back.shape == (100, 64)  # GGUF dims are innermost-first; HF restored
    np.testing.assert_array_equal(back, embd)

    t = by_name["blk.0.attn_q.weight"]
    back = np.asarray(dequantize(t.data, t.tensor_type))
    back = back.reshape(tuple(reversed(t.shape.tolist())))
    ref = np.asarray(dequantize(q8, gguf.GGMLQuantizationType.Q8_0)).reshape(64, 64)
    np.testing.assert_array_equal(back, ref)


def test_target_key_rejects_unregistered_geometry(mini_gguf):
    path, _, _ = mini_gguf
    # 64-hidden single-layer toy model must not map to any registered target.
    assert gguf_target_key(path) is None
