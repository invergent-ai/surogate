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


def _write_q8_gguf(path, tensors):
    writer = GGUFWriter(str(path), "test-arch")
    for name, rows in tensors.items():
        rng = np.random.default_rng(hash(name) % (2**32))
        data = rng.standard_normal((rows, K), dtype=np.float32)
        writer.add_tensor(name, quantize(data, GGMLQuantizationType.Q8_0),
                          raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture()
def q8_source(tmp_path):
    path = tmp_path / "mini.gguf"
    _write_q8_gguf(path, {"g.a": 8, "g.b": 4})
    reader = GGUFReader(str(path), "r")
    candidates = {}
    reference = {}
    for tensor in reader.tensors:
        hf = {"g.a": "hf.a", "g.b": "hf.b"}[tensor.name]
        k, rows = (int(d) for d in tensor.shape)
        candidates[hf] = {
            "name": tensor.name,
            "rows": rows,
            "k": k,
            "offset": int(tensor.data_offset),
        }
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
