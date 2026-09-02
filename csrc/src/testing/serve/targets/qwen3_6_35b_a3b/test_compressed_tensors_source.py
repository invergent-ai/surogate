# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# The compressed-tensors source for the 35B-A3B converter
# (tools/convert/qwen3_6_35b_a3b/compressed_tensors_source.py). One case needs no
# file: the E2M1 nibble table is checked against compressed-tensors' own unpacker
# over every byte value, because a wrong nibble order would corrupt every NVFP4
# weight silently. The rest run against RedHatAI's Qwen3.6-35B-A3B NVFP4 export
# when it is on disk, reading shard headers and a handful of tensors: the plan's
# object count and formats, the split of the fused parents, and one NVFP4 and one
# BF16 payload decoded back and compared bit for bit with the checkpoint.

from __future__ import annotations

from pathlib import Path
import struct

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("compressed_tensors")

from tools.convert.qwen3_6_35b_a3b import compressed_tensors_source as cts  # noqa: E402
from tools.convert.qwen3_6_35b_a3b import recipe, routed_nvfp4  # noqa: E402
from tools.convert.common.safetensors import ShardReader  # noqa: E402
from tools.artifact import layouts  # noqa: E402

REAL_EXPORT = Path("/home/densemax2/work/models/hf/Qwen3.6-35B-A3B-NVFP4-redhat-vllm")
needs_export = pytest.mark.skipif(
    not REAL_EXPORT.is_dir(), reason="RedHatAI 35B-A3B NVFP4 export not on this disk"
)


def test_e2m1_table_matches_the_library_unpacker() -> None:
    import importlib

    nq = importlib.import_module('compressed_tensors.compressors.nvfp4.helpers')

    unpack = getattr(nq, "unpack_fp4_from_uint8", None)
    if unpack is None:
        pytest.skip("compressed_tensors has no unpack_fp4_from_uint8 in this version")
    codes = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
    theirs = unpack(codes, 1, 512).float().reshape(-1)
    ours = torch.empty(512, dtype=torch.float32)
    ours[0::2] = cts._E2M1[(codes[0] & 0x0F).long()]
    ours[1::2] = cts._E2M1[(codes[0] >> 4).long()]
    assert torch.equal(ours, theirs)


@pytest.fixture(scope="module")
def plan():
    source = cts.CompressedTensorsSource(REAL_EXPORT)
    return source, source.plan(routed_nvfp4.tensor_specs(), recipe.BASE_RECIPES_BY_NAME)


@needs_export
def test_the_plan_splits_parents_and_takes_formats_from_the_config(plan) -> None:
    _, result = plan
    by_name = {spec.name: spec for spec in result.specs}
    # every text-core Linear the config quantises is NVFP4 with its divisor sibling;
    # every one it leaves alone is BF16, never requantised
    quantized = [o for o in result.objects.values() if o.quantized]
    plain = [o for o in result.objects.values() if not o.quantized]
    assert len(quantized) == 170 and len(plain) == 92
    for item in quantized:
        assert by_name[item.name].format == "NVFP4"
        assert by_name[item.name + cts.INPUT_DIVISOR_SUFFIX].format == "FP32"
    for item in plain:
        assert by_name[item.name].format == "BF16"
    # the fused parents are gone; their constituents exist at the HF Linear's shape
    assert "text/layers/3/attention/query_key_gate_value" not in by_name
    assert by_name["text/layers/3/attention/query"].shape == (4096, 2048)
    assert by_name["text/layers/3/attention/key"].shape == (512, 2048)
    assert by_name["text/layers/0/gdn/query_key_value"].shape == (8192, 2048)
    assert by_name["text/layers/0/gdn/z"].format == "BF16"
    assert by_name["text/layers/0/moe/shared_gate"].shape == (512, 2048)
    # the export leaves embeddings and lm_head alone, so they stay BF16
    assert by_name["text/token_embedding"].format == "BF16"
    assert by_name["text/output_head"].format == "BF16"
    # nothing outside the text core changed
    assert by_name["vision/merger/fc2"].format == "W8G32_F16S"


@needs_export
def test_payloads_are_the_checkpoints_words(plan) -> None:
    source, result = plan
    with ShardReader.from_index(str(REAL_EXPORT / "model.safetensors.index.json")) as reader:
        item = result.objects["text/layers/3/attention/gate"]  # the strided half of q_proj
        payload = source.payload_for(item.name, result.objects, reader)
        rows = torch.from_numpy(item.rows)
        module = item.source[: -len(".weight")]
        n, k = item.n, item.k
        code_bytes = n * k // 2
        scale_offset = (code_bytes + 255) // 256 * 256
        codes = np.frombuffer(payload[:code_bytes], np.uint8).reshape(n, k // 2)
        assert np.array_equal(
            codes, reader.get(module + ".weight_packed").index_select(0, rows).numpy()
        )
        swizzled = layouts.swizzle_nvfp4_scales(
            reader.get(module + ".weight_scale").view(torch.uint8).index_select(0, rows), (n, k)
        ).numpy()
        assert np.array_equal(np.frombuffer(payload[scale_offset : scale_offset + n * k // 16], np.uint8), swizzled)
        (divisor,) = struct.unpack("<f", payload[scale_offset + n * k // 16 : scale_offset + n * k // 16 + 4])
        assert divisor == float(reader.get(module + ".weight_global_scale"))
        (input_divisor,) = struct.unpack(
            "<f", source.payload_for(item.name + cts.INPUT_DIVISOR_SUFFIX, result.objects, reader)
        )
        assert input_divisor == float(reader.get(module + ".input_global_scale"))

        item = result.objects["text/layers/0/gdn/z"]  # BF16, stored as is
        payload = source.payload_for(item.name, result.objects, reader)
        got = np.frombuffer(payload, np.uint16).reshape(item.n, item.k)
        want = reader.get(item.source).index_select(0, torch.from_numpy(item.rows))
        assert np.array_equal(got, want.view(torch.int16).numpy().view(np.uint16))


@needs_export
def test_the_w8_opt_in_keeps_the_shared_expert_fused(plan) -> None:
    source, _ = plan
    result = source.plan(
        routed_nvfp4.tensor_specs(), recipe.BASE_RECIPES_BY_NAME, shared_expert="w8"
    )
    by_name = {spec.name: spec for spec in result.specs}
    assert by_name["text/layers/0/moe/shared_gate_up"].format == "W8G32_F16S"
    assert by_name["text/layers/0/moe/shared_down"].format == "W8G32_F16S"
    assert "text/layers/0/moe/shared_gate" not in by_name
    item = result.objects["text/layers/0/moe/shared_gate_up"]
    assert item.requantize_to == "W8G32_F16S" and len(item.segments()) == 2 and item.n == 1024
