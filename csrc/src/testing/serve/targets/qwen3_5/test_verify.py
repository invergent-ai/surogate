from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from surogate.serve.artifact.container import (
    ArtifactIdentity,
    ResourceSpec,
    TensorSpec,
    plan_objects,
)
from surogate.serve.artifact.layouts import encode_row_split
from surogate.serve.convert.common.quantize import quantize_matrix
from surogate.serve.convert.qwen3_5 import inventory, verify

from tests.serve.test_qwen3_5_checkpoint_config import config_for
GEOMETRY = inventory.geometry_from_config(config_for(vision=True), token_domain=500)
EXPORT = inventory.export_inventory(inventory.GROUPWISE_INT, GEOMETRY)


def _structural_artifact():
    specs = []
    resource_bytes = 1
    for item in EXPORT.OBJECT_SPECS:
        if isinstance(item, inventory.ResourceSpec):
            specs.append(ResourceSpec(item.name, item.encoding, resource_bytes))
            resource_bytes += 1
        else:
            specs.append(TensorSpec(item.name, item.shape, item.format, item.layout))
    objects = plan_objects(specs)
    payload_bytes = objects[-1].offset + objects[-1].bytes
    return SimpleNamespace(
        identity=ArtifactIdentity(EXPORT.MODEL_ID, EXPORT.WEIGHTS_ID, architecture="qwen3_5"),
        objects=objects,
        payload_offset=4096,
        file_bytes=4096 + payload_bytes,
    )


def _changed(artifact, **changes):
    values = vars(artifact).copy()
    values.update(changes)
    return SimpleNamespace(**values)


def test_complete_structure_and_logical_bindings_without_large_payload() -> None:
    artifact = _structural_artifact()
    summary = verify.validate_structure(artifact, GEOMETRY)
    assert summary.objects == len(EXPORT.OBJECT_SPECS)
    assert summary.tensors == len(EXPORT.TENSOR_SPECS)
    assert summary.resources == len(EXPORT.RESOURCE_SPECS)
    assert summary.row_view_bindings > 0 and summary.alias_bindings > 0
    assert summary.payload_bytes == artifact.objects[-1].offset + artifact.objects[-1].bytes


def test_structure_rejects_identity_signature_and_offset_drift() -> None:
    artifact = _structural_artifact()
    with pytest.raises(verify.VerificationError, match="identity"):
        verify.validate_structure(
            _changed(artifact, identity=ArtifactIdentity("wrong-model", "wrong")),
            GEOMETRY,
        )

    objects = list(artifact.objects)
    objects[6] = replace(objects[6], shape=(248319, 5120))
    with pytest.raises(verify.VerificationError, match="signature"):
        verify.validate_structure(_changed(artifact, objects=tuple(objects)), GEOMETRY)

    objects = list(artifact.objects)
    objects[7] = replace(objects[7], offset=objects[7].offset + 256)
    with pytest.raises(verify.VerificationError, match="offset"):
        verify.validate_structure(_changed(artifact, objects=tuple(objects)), GEOMETRY)


@pytest.mark.parametrize(
    ("format_name", "k"),
    (
        ("Q4G64_F16S", 65),
        ("Q5G64_F16S", 130),
        ("Q6G64_F16S", 129),
        ("W8G32_F16S", 33),
    ),
)
def test_representative_quantized_row_group_verifier(format_name: str, k: int) -> None:
    source = torch.linspace(-3.0, 4.0, 5 * k, dtype=torch.float32).reshape(5, k)
    quantized = quantize_matrix(source, format_name, device="cpu")
    payload = encode_row_split(
        quantized.codes,
        quantized.scales,
        format_name,
        source.shape,
    )
    rows = (0, 2, 4)
    compared = verify.verify_quantized_rows(
        payload,
        format_name,
        tuple(source.shape),
        rows,
        source[list(rows)],
        "cpu",
    )
    assert compared in (6, 9)

    damaged = bytearray(payload)
    damaged[0] ^= 1
    with pytest.raises(verify.VerificationError, match="codes differ"):
        verify.verify_quantized_rows(
            damaged,
            format_name,
            tuple(source.shape),
            rows,
            source[list(rows)],
            "cpu",
        )


def test_draft_id_domain_and_uniqueness() -> None:
    token_ids = torch.arange(500, dtype=torch.int32)
    verify.validate_draft_token_ids(token_ids, GEOMETRY)

    duplicate = token_ids.clone()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(verify.VerificationError, match="not unique"):
        verify.validate_draft_token_ids(duplicate, GEOMETRY)

    outside = token_ids.clone()
    outside[-1] = 500
    with pytest.raises(verify.VerificationError, match="outside"):
        verify.validate_draft_token_ids(outside, GEOMETRY)
