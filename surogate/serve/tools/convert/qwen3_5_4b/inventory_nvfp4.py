"""Persistent-object contract for the Qwen3.5-4B NVFP4 artifact.

The NVFP4 export (AxionML/Qwen3.5-4B-NVFP4) quantises the GDN projections and
every MLP; its self-attention and embeddings stay BF16, so those keep the
groupwise-int artifact's W8 encoding. Source-checkpoint mapping and
materialization live in the sibling conversion recipe.
"""

from __future__ import annotations

from surogate.serve.tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FP32,
    I32,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    VISION_LAYERS,
    W8,
    build_vision_specs,
)


MODEL_ID = "qwen3.5-4b"
WEIGHTS_ID = "nvfp4"
TARGET_KEY = "qwen3_5_4b"

NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"

FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4, FP8)
LAYOUT_NAMES = (
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    BLOCK_SCALE_LAYOUT,
    ROW_SCALE_LAYOUT,
)


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    """Storage layout per format; NVFP4 carries block scales, W8 row splits."""

    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format in (Q4, Q5, Q6, W8):
        layout = ROW_SPLIT_LAYOUT
    elif numeric_format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    elif numeric_format == FP8:
        layout = ROW_SCALE_LAYOUT
    else:
        raise ValueError(f"unsupported Qwen3.5-4B NVFP4 format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)

FULL_ATTENTION_LAYERS = tuple(range(3, 32, 4))
GDN_LAYERS = tuple(layer for layer in range(32) if layer not in FULL_ATTENTION_LAYERS)


_tensor = tensor_spec


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (248320, 2560), W8),
    ]

    for layer in range(32):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (2560,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key_gate_value", (10240, 2560), W8),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (2560, 4096), W8),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (32,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (32,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 8192), BF16),
                    _tensor(prefix + "gdn/a_projection", (32, 2560), BF16),
                    _tensor(prefix + "gdn/b_projection", (32, 2560), BF16),
                    _tensor(prefix + "gdn/query_key_value_z", (12288, 2560), NVFP4),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (2560, 4096), NVFP4),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (2560,), BF16),
                _tensor(prefix + "mlp/gate_up", (18432, 2560), NVFP4),
                _tensor(prefix + "mlp/down", (2560, 9216), NVFP4),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (2560,), BF16),
            _tensor("text/output_head", (248320, 2560), W8),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (131072, 2560), W8),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (2560, 5120), W8),
        _tensor("mtp/embedding_norm", (2560,), BF16),
        _tensor("mtp/hidden_norm", (2560,), BF16),
        _tensor("mtp/layer/input_norm", (2560,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (10240, 2560), W8),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (2560, 4096), W8),
        _tensor("mtp/layer/post_attention_norm", (2560,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (18432, 2560), W8),
        _tensor("mtp/layer/mlp/down", (2560, 9216), W8),
        _tensor("mtp/final_norm", (2560,), BF16),
    )


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    return build_vision_specs(5120)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS: tuple[TensorSpec, ...] = ()  # text-only target

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

# surogate vendor patch (PATCHES.md #15): community GGUF exports frequently
# strip the MTP (nextn) block; such checkpoints convert to an artifact that
# omits the mtp/* objects entirely (the loader binds MTP only when present
# and MTP speculation is refused with a clear error). The draft head stays:
# it derives from the embedding, which every export carries.
TENSOR_SPECS_NO_MTP = TEXT_CORE_TENSOR_SPECS + DRAFT_HEAD_TENSOR_SPECS + VISION_TENSOR_SPECS
OBJECT_SPECS_NO_MTP: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS_NO_MTP


def active_specs(*, mtp: bool) -> tuple[tuple, tuple]:
    """(tensor_specs, object_specs) for the requested artifact variant."""
    if mtp:
        return TENSOR_SPECS, OBJECT_SPECS
    return TENSOR_SPECS_NO_MTP, OBJECT_SPECS_NO_MTP

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


LOGICAL_ROW_VIEW_SPECS = (
    LogicalRowViewSpec(
        "text/layers/{l}/attention/query",
        "text/layers/{l}/attention/query_key_gate_value",
        0,
        4096,
        (4096, 2560),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key_gate_value",
        4096,
        5120,
        (1024, 2560),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/query_key_gate_value",
        5120,
        9216,
        (4096, 2560),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/query_key_gate_value",
        9216,
        10240,
        (1024, 2560),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key_value_z",
        0,
        2048,
        (2048, 2560),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key_value_z",
        2048,
        4096,
        (2048, 2560),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/query_key_value_z",
        4096,
        8192,
        (4096, 2560),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/query_key_value_z",
        8192,
        12288,
        (4096, 2560),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/gate",
        "text/layers/{l}/mlp/gate_up",
        0,
        9216,
        (9216, 2560),
        tuple(range(32)),
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/up",
        "text/layers/{l}/mlp/gate_up",
        9216,
        18432,
        (9216, 2560),
        tuple(range(32)),
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/query",
        "mtp/layer/attention/query_key_gate_value",
        0,
        4096,
        (4096, 2560),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/key",
        "mtp/layer/attention/query_key_gate_value",
        4096,
        5120,
        (1024, 2560),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/output_gate",
        "mtp/layer/attention/query_key_gate_value",
        5120,
        9216,
        (4096, 2560),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/value",
        "mtp/layer/attention/query_key_gate_value",
        9216,
        10240,
        (1024, 2560),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/gate",
        "mtp/layer/mlp/gate_up",
        0,
        9216,
        (9216, 2560),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/up",
        "mtp/layer/mlp/gate_up",
        9216,
        18432,
        (9216, 2560),
        None,
    ),
)


ALIAS_SPECS = (
    LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
    LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
    LogicalAliasSpec(
        "mtp/optimized_proposal_head",
        ("text/draft_head", "text/draft_head_token_ids"),
    ),
    LogicalAliasSpec(
        "text/layers/{l}/gdn/channel_major_convolution",
        ("text/layers/{l}/gdn/convolution",),
        layers=GDN_LAYERS,
        axis_order=(1, 0),
    ),
)
