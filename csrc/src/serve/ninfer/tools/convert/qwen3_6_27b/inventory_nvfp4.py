"""Persistent-object contract for the additive Qwen3.6-27B NVFP4 artifact."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import TensorSpec

from . import inventory as base


MODEL_ID = base.MODEL_ID
WEIGHTS_ID = "nvfp4"
TARGET_KEY = base.TARGET_KEY

BF16 = base.BF16
FP32 = base.FP32
I32 = base.I32
Q4 = base.Q4
Q5 = base.Q5
Q6 = base.Q6
W8 = base.W8
NVFP4 = "NVFP4"

CONTIGUOUS_LAYOUT = base.CONTIGUOUS_LAYOUT
ROW_SPLIT_LAYOUT = base.ROW_SPLIT_LAYOUT
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"

RESOURCE_SPECS = base.RESOURCE_SPECS
ResourceSpec = base.ResourceSpec
StoredObjectSpec = base.StoredObjectSpec

FULL_ATTENTION_LAYERS = base.FULL_ATTENTION_LAYERS
GDN_LAYERS = base.GDN_LAYERS
EARLY_ATTENTION_INPUT_LAYERS = (3, 7, 11, 15, 19, 23)
NVFP4_ATTENTION_INPUT_LAYERS = tuple(
    layer for layer in FULL_ATTENTION_LAYERS
    if layer not in EARLY_ATTENTION_INPUT_LAYERS
)
BF16_ATTENTION_OUTPUT_LAYERS = (3, 7)
NVFP4_ATTENTION_OUTPUT_LAYERS = tuple(
    layer for layer in FULL_ATTENTION_LAYERS
    if layer not in BF16_ATTENTION_OUTPUT_LAYERS
)
BF16_GDN_OUTPUT_LAYERS = (4,)
NVFP4_GDN_OUTPUT_LAYERS = tuple(
    layer for layer in GDN_LAYERS
    if layer not in BF16_GDN_OUTPUT_LAYERS
)


def _replacement_format(name: str, original: str) -> str:
    if name in ("text/token_embedding", "text/output_head"):
        return W8
    if not name.startswith("text/layers/"):
        return original
    parts = name.split("/")
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    if suffix == "attention/query_key_gate_value":
        return (
            BF16
            if layer in EARLY_ATTENTION_INPUT_LAYERS
            else NVFP4
        )
    if suffix == "attention/output":
        return BF16 if layer in BF16_ATTENTION_OUTPUT_LAYERS else NVFP4
    if suffix == "gdn/query_key_value_z":
        return NVFP4
    if suffix == "gdn/output":
        return BF16 if layer in BF16_GDN_OUTPUT_LAYERS else NVFP4
    if suffix in ("mlp/gate_up", "mlp/down"):
        return NVFP4
    return original


def _tensor(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    else:
        layout = ROW_SPLIT_LAYOUT
    return TensorSpec(name, shape, numeric_format, layout)


def _input_scale_after(spec: TensorSpec) -> str | None:
    if spec.format != NVFP4:
        return None
    prefix, suffix = spec.name.rsplit("/", 1)
    if suffix == "query_key_gate_value" and prefix.endswith("/attention"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/attention"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "query_key_value_z" and prefix.endswith("/gdn"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/gdn"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "gate_up" and prefix.endswith("/mlp"):
        return prefix + "/gate_up_projection/input_scale_divisor"
    if suffix == "down" and prefix.endswith("/mlp"):
        return prefix + "/down_projection/input_scale_divisor"
    return None


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for original in base.TEXT_CORE_TENSOR_SPECS:
        if original.name.endswith("/attention/query_key"):
            name = original.name.removesuffix("query_key") + "query_key_gate_value"
            spec = _tensor(name, (14336, 5120), _replacement_format(name, original.format))
            specs.append(spec)
            scalar_name = _input_scale_after(spec)
            if scalar_name is not None:
                specs.append(_tensor(scalar_name, (), FP32))
            continue
        if original.name.endswith("/attention/gate_value"):
            continue
        if original.name.endswith("/gdn/query_key"):
            name = original.name.removesuffix("query_key") + "query_key_value_z"
            spec = _tensor(name, (16384, 5120), NVFP4)
            specs.append(spec)
            specs.append(
                _tensor(
                    name.removesuffix("query_key_value_z")
                    + "input_projection/input_scale_divisor",
                    (),
                    FP32,
                )
            )
            continue
        if original.name.endswith("/gdn/value_z"):
            continue
        numeric_format = _replacement_format(original.name, original.format)
        spec = _tensor(original.name, original.shape, numeric_format)
        specs.append(spec)
        scalar_name = _input_scale_after(spec)
        if scalar_name is not None:
            specs.append(_tensor(scalar_name, (), FP32))
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

FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT, BLOCK_SCALE_LAYOUT)
FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = (
    base.LogicalRowViewSpec(
        "text/layers/{l}/attention/query",
        "text/layers/{l}/attention/query_key_gate_value",
        0,
        6144,
        (6144, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key_gate_value",
        6144,
        7168,
        (1024, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/query_key_gate_value",
        7168,
        13312,
        (6144, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/query_key_gate_value",
        13312,
        14336,
        (1024, 5120),
        FULL_ATTENTION_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key_value_z",
        0,
        2048,
        (2048, 5120),
        GDN_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key_value_z",
        2048,
        4096,
        (2048, 5120),
        GDN_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/query_key_value_z",
        4096,
        10240,
        (6144, 5120),
        GDN_LAYERS,
    ),
    base.LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/query_key_value_z",
        10240,
        16384,
        (6144, 5120),
        GDN_LAYERS,
    ),
) + base.LOGICAL_ROW_VIEW_SPECS[8:]
ALIAS_SPECS = base.ALIAS_SPECS

NVFP4_TENSOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS if spec.format == NVFP4
)
INPUT_SCALE_DIVISOR_SPECS = tuple(
    spec for spec in TENSOR_SPECS
    if spec.format == FP32 and spec.name.endswith("/input_scale_divisor")
)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("NVFP4 inventory contains duplicate object names")
    if (
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(NVFP4_TENSOR_SPECS),
        len(INPUT_SCALE_DIVISOR_SPECS),
    ) != (954, 2, 12, 333, 1301, 1307, 247, 247):
        raise ValueError("registered NVFP4 inventory is incomplete")
    if FORMAT_COUNTS != {
        BF16: 591,
        FP32: 343,
        I32: 1,
        Q4: 55,
        Q5: 54,
        Q6: 1,
        W8: 9,
        NVFP4: 247,
    }:
        raise ValueError(f"unexpected NVFP4 format allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS != {
        CONTIGUOUS_LAYOUT: 935,
        ROW_SPLIT_LAYOUT: 119,
        BLOCK_SCALE_LAYOUT: 247,
    }:
        raise ValueError(f"unexpected NVFP4 layout allocation: {LAYOUT_COUNTS}")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "BLOCK_SCALE_LAYOUT",
    "CONTIGUOUS_LAYOUT",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "INPUT_SCALE_DIVISOR_SPECS",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "NVFP4",
    "NVFP4_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
    "validate_inventory",
]
