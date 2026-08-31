"""Persistent-object contract for the complete Qwen3.5-0.8B artifact.

This module contains only target storage roles. Source-checkpoint mapping and
materialization live in the sibling conversion recipe.
"""

from __future__ import annotations

from surogate.serve.tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
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
    tensor_spec,
)


MODEL_ID = "qwen3.5-0.8b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_5_0_8b"

FULL_ATTENTION_LAYERS = tuple(range(3, 24, 4))
GDN_LAYERS = tuple(layer for layer in range(24) if layer not in FULL_ATTENTION_LAYERS)


_tensor = tensor_spec


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (248320, 1024), W8),
    ]

    for layer in range(24):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (1024,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key_gate_value", (5120, 1024), W8),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (1024, 2048), W8),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (16,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (16,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 6144), BF16),
                    _tensor(prefix + "gdn/a_projection", (16, 1024), BF16),
                    _tensor(prefix + "gdn/b_projection", (16, 1024), BF16),
                    _tensor(prefix + "gdn/query_key_value_z", (8192, 1024), W8),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (1024, 2048), W8),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (1024,), BF16),
                _tensor(prefix + "mlp/gate_up", (7168, 1024), W8),
                _tensor(prefix + "mlp/down", (1024, 3584), W8),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (1024,), BF16),
            _tensor("text/output_head", (248320, 1024), W8),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (131072, 1024), W8),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (1024, 2048), W8),
        _tensor("mtp/embedding_norm", (1024,), BF16),
        _tensor("mtp/hidden_norm", (1024,), BF16),
        _tensor("mtp/layer/input_norm", (1024,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (5120, 1024), W8),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (1024, 2048), W8),
        _tensor("mtp/layer/post_attention_norm", (1024,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (7168, 1024), W8),
        _tensor("mtp/layer/mlp/down", (1024, 3584), W8),
        _tensor("mtp/final_norm", (1024,), BF16),
    )


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    # This target's own tower, not the Qwen3.6 default. Serving carries vision
    # on every target; the geometry comes from the model's vision_config, which
    # the DSL declaration is the source of truth for.
    return build_vision_specs(
        1024,
        layers=12,
        hidden=768,
        intermediate=3072,
        qkv_rows=2304,
        merger_hidden=3072,
    )


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = _build_vision_specs()

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
        2048,
        (2048, 1024),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key_gate_value",
        2048,
        2560,
        (512, 1024),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/query_key_gate_value",
        2560,
        4608,
        (2048, 1024),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/query_key_gate_value",
        4608,
        5120,
        (512, 1024),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key_value_z",
        0,
        2048,
        (2048, 1024),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key_value_z",
        2048,
        4096,
        (2048, 1024),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/query_key_value_z",
        4096,
        6144,
        (2048, 1024),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/query_key_value_z",
        6144,
        8192,
        (2048, 1024),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/gate",
        "text/layers/{l}/mlp/gate_up",
        0,
        3584,
        (3584, 1024),
        tuple(range(24)),
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/up",
        "text/layers/{l}/mlp/gate_up",
        3584,
        7168,
        (3584, 1024),
        tuple(range(24)),
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/query",
        "mtp/layer/attention/query_key_gate_value",
        0,
        2048,
        (2048, 1024),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/key",
        "mtp/layer/attention/query_key_gate_value",
        2048,
        2560,
        (512, 1024),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/output_gate",
        "mtp/layer/attention/query_key_gate_value",
        2560,
        4608,
        (2048, 1024),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/value",
        "mtp/layer/attention/query_key_gate_value",
        4608,
        5120,
        (512, 1024),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/gate",
        "mtp/layer/mlp/gate_up",
        0,
        3584,
        (3584, 1024),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/up",
        "mtp/layer/mlp/gate_up",
        3584,
        7168,
        (3584, 1024),
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
