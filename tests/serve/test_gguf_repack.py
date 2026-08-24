# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Bit-exactness tests for the direct Q8_0 -> W8G32_F16S repack
# (csrc/src/serve/ninfer/tools/convert/common/gguf_repack.py, PATCHES.md #14):
# a synthetic Q8_0 GGUF is moved through the recipe row-algebra evaluator and
# the resulting row-split payloads must decode to exactly the values gguf-py's
# own dequantize produces — no requantization anywhere. CPU-only.

import json
import sys

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
torch = pytest.importorskip("torch")

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import dequantize, quantize

from surogate.serve.ingest import _ninfer_root

_ROOT = _ninfer_root()
if _ROOT is None:
    pytest.skip("vendored ninfer tree not found", allow_module_level=True)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.artifact.layouts import dequantize_row_split  # noqa: E402
from tools.convert.common.gguf_repack import GgufRepackSource  # noqa: E402
from tools.convert.qwen3_6.common.recipe import (  # noqa: E402
    Concat,
    GatherRows,
    Reshape,
    Slice,
    SourceTensor,
    TensorRecipe,
)
from tools.convert.qwen3_6.common.inventory import TensorSpec  # noqa: E402

K = 128  # four 32-value groups per row (k128 layout needs k % 128 == 0)


# IQ4_NL codebook (ggml-common.h); gguf-py cannot quantize IQ4_NL, so the
# fixture packs valid blocks by hand: fp16 d + 16 nibble-index bytes.
_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int8,
)


def _pack_iq4_nl(rows):
    rng = np.random.default_rng(rows * 7 + 1)
    groups = K // 32
    d = rng.uniform(0.001, 0.01, size=(rows, groups)).astype(np.float16)
    idx = rng.integers(0, 16, size=(rows, groups, 32), dtype=np.uint8)
    blocks = np.zeros((rows, groups, 18), dtype=np.uint8)
    blocks[:, :, :2] = d.view(np.uint8).reshape(rows, groups, 2)
    blocks[:, :, 2:] = idx[:, :, :16] | (idx[:, :, 16:] << 4)
    reference = (
        _IQ4NL[idx].astype(np.float32) * d.astype(np.float32)[:, :, None]
    ).reshape(rows, K)
    return blocks.reshape(rows, -1), reference


def _write_q8_gguf(path, tensors):
    writer = GGUFWriter(str(path), "test-arch")
    iq4_reference = {}
    for name, (rows, ttype) in tensors.items():
        if ttype == GGMLQuantizationType.IQ4_NL:
            payload, iq4_reference[name] = _pack_iq4_nl(rows)
            writer.add_tensor(name, payload, raw_dtype=ttype)
            continue
        rng = np.random.default_rng(hash(name) % (2**32))
        data = rng.standard_normal((rows, K), dtype=np.float32)
        writer.add_tensor(name, quantize(data, ttype), raw_dtype=ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return iq4_reference


_FIXTURE_TENSORS = {
    "g.a": (8, GGMLQuantizationType.Q8_0),
    "g.b": (4, GGMLQuantizationType.Q8_0),
    "g.q40": (6, GGMLQuantizationType.Q4_0),
    "g.q50": (6, GGMLQuantizationType.Q5_0),
    "g.iq4": (6, GGMLQuantizationType.IQ4_NL),
}
_HF_NAMES = {"g.a": "hf.a", "g.b": "hf.b", "g.q40": "hf.q40",
             "g.q50": "hf.q50", "g.iq4": "hf.iq4"}


@pytest.fixture()
def q8_source(tmp_path):
    path = tmp_path / "mini.gguf"
    iq4_reference = _write_q8_gguf(path, _FIXTURE_TENSORS)
    reader = GGUFReader(str(path), "r")
    candidates = {}
    reference = {}
    for tensor in reader.tensors:
        hf = _HF_NAMES[tensor.name]
        k, rows = (int(d) for d in tensor.shape)
        candidates[hf] = {
            "name": tensor.name,
            "rows": rows,
            "k": k,
            "offset": int(tensor.data_offset),
            "type": tensor.tensor_type.name,
        }
        if tensor.tensor_type == GGMLQuantizationType.IQ4_NL:
            reference[hf] = iq4_reference[tensor.name]
        else:
            reference[hf] = dequantize(tensor.data, tensor.tensor_type).reshape(rows, k)
    del reader
    return path, candidates, reference


def _decoded(source, spec, recipe, token_ids=None):
    payload = source.payload_for(spec, recipe, token_ids)
    return dequantize_row_split(
        payload, spec.format, spec.shape, dtype=torch.float32
    ).numpy()


def test_concat_is_bit_exact(q8_source):
    path, candidates, reference = q8_source
    source = GgufRepackSource.from_sources(path, candidates)
    spec = TensorSpec("obj/fused", (12, K), "W8G32_F16S", "row-split-k128-v1")
    recipe = TensorRecipe(
        "obj/fused",
        Concat((SourceTensor("hf.a", (8, K)), SourceTensor("hf.b", (4, K))), 0),
    )
    expected = np.concatenate([reference["hf.a"], reference["hf.b"]], axis=0)
    assert np.array_equal(_decoded(source, spec, recipe), expected)


def test_head_deinterleave_rows(q8_source):
    # The attention qproj pattern: reshape to heads, slice a half, flatten.
    path, candidates, reference = q8_source
    source = GgufRepackSource.from_sources(path, candidates)
    spec = TensorSpec("obj/half", (4, K), "W8G32_F16S", "row-split-k128-v1")
    recipe = TensorRecipe(
        "obj/half",
        Reshape(
            Slice(Reshape(SourceTensor("hf.a", (8, K)), (2, 4, K)), 1, 2, 4),
            (4, K),
        ),
    )
    expected = reference["hf.a"].reshape(2, 4, K)[:, 2:4].reshape(4, K)
    assert np.array_equal(_decoded(source, spec, recipe), expected)


def test_gather_rows(q8_source):
    path, candidates, reference = q8_source
    source = GgufRepackSource.from_sources(path, candidates)
    ids = torch.tensor([5, 0, 7], dtype=torch.int32)
    spec = TensorSpec("obj/gathered", (3, K), "W8G32_F16S", "row-split-k128-v1")
    recipe = TensorRecipe(
        "obj/gathered",
        GatherRows(SourceTensor("hf.a", (8, K)), token_ids_object="ids", rows=3),
    )
    expected = reference["hf.a"][[5, 0, 7]]
    assert np.array_equal(_decoded(source, spec, recipe, ids), expected)


def test_plan_excludes_non_w8_and_unmapped(q8_source):
    path, candidates, _ = q8_source
    source = GgufRepackSource.from_sources(path, candidates)
    recipes = {
        "obj/w8": TensorRecipe("obj/w8", SourceTensor("hf.a", (8, K))),
        "obj/bf16": TensorRecipe("obj/bf16", SourceTensor("hf.b", (4, K))),
        "obj/stranger": TensorRecipe("obj/stranger", SourceTensor("hf.c", (4, K))),
    }
    specs = (
        TensorSpec("obj/w8", (8, K), "W8G32_F16S", "row-split-k128-v1"),
        TensorSpec("obj/bf16", (4, K), "BF16_CTRL", "contiguous-v1"),
        TensorSpec("obj/stranger", (4, K), "W8G32_F16S", "row-split-k128-v1"),
    )
    assert source.plan(recipes, specs) == ("obj/w8",)


def test_narrow_formats_are_bit_exact(q8_source):
    # Q4_0 / Q5_0 / IQ4_NL codes embed exactly in W8G32 int8 codes.
    path, candidates, reference = q8_source
    source = GgufRepackSource.from_sources(path, candidates)
    for hf in ("hf.q40", "hf.q50", "hf.iq4"):
        rows = candidates[hf]["rows"]
        spec = TensorSpec(f"obj/{hf}", (rows, K), "W8G32_F16S", "row-split-k128-v1")
        recipe = TensorRecipe(f"obj/{hf}", SourceTensor(hf, (rows, K)))
        assert np.array_equal(_decoded(source, spec, recipe), reference[hf]), hf


def test_map_roundtrip_through_json(q8_source, tmp_path):
    path, candidates, reference = q8_source
    map_path = tmp_path / "map.json"
    map_path.write_text(
        json.dumps({"gguf_path": str(path), "sources": candidates})
    )
    source = GgufRepackSource(map_path)
    spec = TensorSpec("obj/plain", (8, K), "W8G32_F16S", "row-split-k128-v1")
    recipe = TensorRecipe("obj/plain", SourceTensor("hf.a", (8, K)))
    assert np.array_equal(_decoded(source, spec, recipe), reference["hf.a"])
