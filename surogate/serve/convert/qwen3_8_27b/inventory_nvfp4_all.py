"""Persistent-object contract for the all-NVFP4 Qwen3.8-27B artifact.

`nvfp4` (inventory_nvfp4) keeps the attention and GDN projections FP8-row and
the last eight MLPs FP8, because `unsloth/Qwen3.8-27B-NVFP4` exports them that
way. `sakamakismile/Qwen3.8-27B-MTP-NVFP4` quantises **every** language linear
- its ignore list holds only the vision tower - so this contract carries the
same objects with those matrices as NVFP4 blocks, each with the activation
divisor the W4A4 path needs.

Everything outside the language linears is unchanged: norms, the convolution,
the fused a/b projection, the FP8 endpoints, the draft head, the MTP block and
the vision tower are the same objects with the same formats.
"""

from __future__ import annotations

from surogate.serve.convert.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    I32,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    build_vision_specs,
)

from . import inventory_nvfp4 as base
from .inventory_nvfp4 import (
    BLOCK_SCALE_LAYOUT,
    FORMAT_NAMES,
    FP8,
    FULL_ATTENTION_LAYERS,
    GDN_LAYERS,
    LAYOUT_NAMES,
    NVFP4,
    ROW_SCALE_LAYOUT,
    tensor_spec,
)

MODEL_ID = base.MODEL_ID
WEIGHTS_ID = "nvfp4-all"
TARGET_KEY = base.TARGET_KEY

# Every layer, unlike the mixed contract's `range(56)`.
NVFP4_MLP_LAYERS = tuple(range(64))
FP8_MLP_LAYERS: tuple[int, ...] = ()


def _divisor(name: str) -> TensorSpec:
    return tensor_spec(name, (), FP32)


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (248320, 5120), FP8),
    ]
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        specs.append(tensor_spec(prefix + "input_norm", (5120,), BF16))
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        NVFP4,
                    ),
                    _divisor(
                        prefix + "attention/input_projection/input_scale_divisor"
                    ),
                    tensor_spec(prefix + "attention/query_norm", (256,), BF16),
                    tensor_spec(prefix + "attention/key_norm", (256,), BF16),
                    tensor_spec(prefix + "attention/output", (5120, 6144), NVFP4),
                    _divisor(
                        prefix + "attention/output_projection/input_scale_divisor"
                    ),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "gdn/a_log", (48,), FP32),
                    tensor_spec(prefix + "gdn/dt_bias", (48,), FP32),
                    tensor_spec(prefix + "gdn/convolution", (4, 10240), BF16),
                    tensor_spec(prefix + "gdn/a_b_projection", (96, 5120), BF16),
                    # in_proj_qkv and in_proj_z carry different weight global scales in
                    # this export, and an NVFP4 object holds one divisor, so the two halves
                    # stay apart and the projection runs as two GEMMs (#87).
                    tensor_spec(prefix + "gdn/query_key_value", (10240, 5120), NVFP4),
                    _divisor(prefix + "gdn/input_projection/input_scale_divisor"),
                    tensor_spec(prefix + "gdn/z", (6144, 5120), NVFP4),
                    _divisor(prefix + "gdn/z_projection/input_scale_divisor"),
                    tensor_spec(prefix + "gdn/norm", (128,), BF16),
                    tensor_spec(prefix + "gdn/output", (5120, 6144), NVFP4),
                    _divisor(prefix + "gdn/output_projection/input_scale_divisor"),
                )
            )

        specs.append(tensor_spec(prefix + "post_attention_norm", (5120,), BF16))
        specs.extend(
            (
                tensor_spec(prefix + "mlp/gate_up", (34816, 5120), NVFP4),
                _divisor(prefix + "mlp/gate_up_projection/input_scale_divisor"),
                tensor_spec(prefix + "mlp/down", (5120, 17408), NVFP4),
                _divisor(prefix + "mlp/down_projection/input_scale_divisor"),
            )
        )

    specs.extend(
        (
            tensor_spec("text/final_norm", (5120,), BF16),
            tensor_spec("text/output_head", (248320, 5120), FP8),
        )
    )
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = base.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = base.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = base.VISION_TENSOR_SPECS

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

NVFP4_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == NVFP4)
FP8_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == FP8)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.name.endswith("input_scale_divisor")
)

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = base.LOGICAL_ROW_VIEW_SPECS
LOGICAL_ALIAS_SPECS = getattr(base, "LOGICAL_ALIAS_SPECS", ())


def validate_inventory() -> None:
    """Every language linear is NVFP4 and every one of them carries a divisor."""

    # per layer: mlp gate_up + down, then attention in/out or gdn qkv + z + out
    quantised = 64 * 2 + len(FULL_ATTENTION_LAYERS) * 2 + len(GDN_LAYERS) * 3
    if len(NVFP4_TENSOR_SPECS) != quantised:
        raise ValueError(
            f"expected {quantised} NVFP4 matrices, found {len(NVFP4_TENSOR_SPECS)}"
        )
    if len(INPUT_SCALE_DIVISOR_SPECS) != quantised:
        raise ValueError(
            "every NVFP4 matrix needs an activation divisor: "
            f"{len(NVFP4_TENSOR_SPECS)} matrices, "
            f"{len(INPUT_SCALE_DIVISOR_SPECS)} divisors"
        )
    # The endpoints stay FP8 (the export ships them bf16 and the target reads a
    # byte-wide head), and nothing else may be.
    if {spec.name for spec in FP8_TENSOR_SPECS} != {
        "text/token_embedding",
        "text/output_head",
    }:
        raise ValueError("only the endpoints may be FP8 in the all-NVFP4 contract")
    names = [spec.name for spec in OBJECT_SPECS]
    if len(names) != len(set(names)):
        raise ValueError("duplicate object name in the all-NVFP4 inventory")


validate_inventory()

__all__ = [
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FP8",
    "FP8_MLP_LAYERS",
    "FP8_TENSOR_SPECS",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ALIAS_SPECS",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_MLP_LAYERS",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ROW_SCALE_LAYOUT",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
    "WEIGHTS_ID",
    "tensor_spec",
    "validate_inventory",
]
