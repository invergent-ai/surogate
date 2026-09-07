# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Parity tests for the lean GGUF metadata parser (surogate/serve/gguf/lean.py)
# against gguf-py's GGUFReader as the oracle, on a synthetic file carrying
# every KV shape the serve path meets (scalars, strings, fixed arrays, string
# arrays) plus quantized and float tensors. CPU-only.

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import dequantize, quantize

from surogate.serve.gguf.lean import LeanGguf


@pytest.fixture()
def sample(tmp_path):
    path = tmp_path / "lean.gguf"
    writer = GGUFWriter(str(path), "lean-arch")
    writer.add_uint32("lean.block_count", 24)
    writer.add_float32("lean.rope.freq_base", 1.0e7)
    writer.add_bool("lean.flag", True)
    writer.add_string("tokenizer.chat_template", "{{ messages }}")
    writer.add_array("tokenizer.ggml.tokens", ["hello", "wörld", "", "byé"])
    writer.add_array("tokenizer.ggml.token_type", [1, 1, 3, 4])
    writer.add_uint64("lean.big", 1 << 40)

    rng = np.random.default_rng(7)
    f32 = rng.standard_normal((3, 64), dtype=np.float32)
    writer.add_tensor("t.f32", f32)
    q8 = rng.standard_normal((5, 128), dtype=np.float32)
    writer.add_tensor("t.q8", quantize(q8, GGMLQuantizationType.Q8_0),
                      raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_kv_parity(sample):
    lean = LeanGguf(sample)
    ref = GGUFReader(str(sample), "r")
    for key in (
        "general.architecture",
        "lean.block_count",
        "lean.rope.freq_base",
        "lean.flag",
        "tokenizer.chat_template",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "lean.big",
    ):
        expected = ref.get_field(key).contents()
        assert lean.kv(key) == expected, key
        # get_field facade parity
        assert lean.get_field(key).contents() == expected, key
    assert lean.kv("absent.key", "fallback") == "fallback"
    assert lean.get_field("absent.key") is None


def test_tensor_info_parity(sample):
    lean = LeanGguf(sample)
    ref = GGUFReader(str(sample), "r")
    ref_by_name = {t.name: t for t in ref.tensors}
    assert {t.name for t in lean.tensors} == set(ref_by_name)
    for tensor in lean.tensors:
        expected = ref_by_name[tensor.name]
        assert tensor.shape == tuple(int(d) for d in expected.shape), tensor.name
        assert tensor.type_name == expected.tensor_type.name, tensor.name
        assert tensor.data_offset == int(expected.data_offset), tensor.name


def test_payload_view_dequantizes_identically(sample):
    lean = LeanGguf(sample)
    ref = GGUFReader(str(sample), "r")
    mm = np.memmap(sample, dtype=np.uint8, mode="r")
    for expected in ref.tensors:
        tensor = lean.tensor(expected.name)
        view = lean.payload_view(tensor, mm)
        got = dequantize(view, GGMLQuantizationType(tensor.type_id))
        want = dequantize(expected.data, expected.tensor_type)
        assert np.array_equal(np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)), (
            expected.name
        )
