from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from tools.artifact import ArtifactIdentity
from tools.reference.qwen3_6_27b.bindings import ArtifactBinding
from tools.reference.qwen3_6_27b.weights import WeightStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def binding():
    path = Path(
        os.environ.get(
            "NINFER_QWEN3_6_27B_ARTIFACT",
            PROJECT_ROOT / "out/qwen3_6_27b.ninfer",
        )
    )
    if not path.is_file():
        pytest.skip(f"real Qwen3.6-27B artifact is not available at {path}")
    with ArtifactBinding.open(path) as value:
        yield value


def test_complete_typed_binding_and_vision_policy(binding: ArtifactBinding) -> None:
    assert binding.identity == ArtifactIdentity("qwen3.6-27b", "groupwise-int")
    assert len(binding.tensors) == 1118
    assert len(binding.row_views) == 390
    assert len(binding.axis_views) == 48
    assert len(binding.text.layers) == 64
    assert len(binding.vision.layers) == 27
    assert binding.text.layers[0].gdn.convolution.shape == (10240, 4)
    first_gdn = binding.text.layers[0].gdn
    assert first_gdn is not None
    assert first_gdn.value.block.tensor_id == first_gdn.value_z.tensor_id
    assert (first_gdn.value.row_begin, first_gdn.value.row_end) == (0, 6144)
    assert first_gdn.z.block.tensor_id == first_gdn.value_z.tensor_id
    assert (first_gdn.z.row_begin, first_gdn.z.row_end) == (6144, 12288)

    vision = WeightStore(
        binding,
        "cpu",
        capacity=1,
        text=False,
        vision=True,
        memory_bytes=0,
    )
    try:
        assert vision.plan.fixed_bytes == 0
        assert vision.representation(binding.vision.patch_embedding) == "stream"
        assert vision.representation(binding.vision.position_embedding) == "decoded"
        assert vision.tensor(binding.vision.patch_embedding_bias).shape == (1152,)
    finally:
        vision.close()


def test_streamed_decode_gather_and_chunk_acceptance(
    binding: ArtifactBinding,
) -> None:
    weights = WeightStore(
        binding,
        "cpu",
        capacity=32,
        text=True,
        mtp=True,
        draft_head=True,
        memory_bytes=0,
    )
    try:
        first_gdn = binding.text.layers[0].gdn
        assert first_gdn is not None
        storage = weights.tensor(first_gdn.convolution_storage)
        channel_major = weights.tensor(first_gdn.convolution)
        assert channel_major.shape == (10240, 4)
        assert torch.equal(channel_major, storage.t())

        embedding = weights.rows(
            binding.text.token_embedding,
            torch.tensor([123, 0, 123], dtype=torch.long),
        )
        assert embedding.shape == (3, 5120)
        assert embedding.dtype == torch.bfloat16
        assert torch.equal(embedding[0], embedding[2])

        chunks = weights.chunks(binding.text.output_head, rows=4)
        row_begin, row_end, head = next(chunks)
        chunks.close()
        assert (row_begin, row_end) == (0, 4)
        assert head.shape == (4, 5120)
        assert head.dtype == torch.bfloat16

        draft_ids = weights.tensor(binding.text.draft_head.token_ids)
        assert draft_ids.shape == (131072,)
        assert draft_ids.dtype == torch.int32
    finally:
        weights.close()
