"""Persistent objects built from resolved hyper-connected hybrid checkpoint dimensions."""

from surogate.serve.convert.common.inventory import (
    BF16, CONTIGUOUS_LAYOUT, FP32, GGML_BLOCKS_LAYOUT, I32, RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT, ResourceSpec, StoredObjectSpec, TensorSpec, W8,
    build_vision_specs as _vision_specs,
    VISION_BF16,
    VISION_STORAGE,
)
from surogate.serve.convert.common.qwen3_5 import vision_tower
from surogate.serve.convert.common.qwen4exp import Geometry, geometry_from_config, geometry_from_gguf

MODEL_ID = TARGET_KEY = "qwen4exp"
WEIGHTS_ID = "w8-hc-v1"
PLE_TABLE_RESOURCE = "text/ple/table.iq4nl"
PLE_TABLE_FORMAT = "IQ4_NL"
FORMAT_NAMES = (BF16, FP32, I32, W8)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT)


def tensor_spec(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format == W8:
        layout = ROW_SPLIT_LAYOUT
    else:
        raise ValueError(f"unsupported qwen4exp format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


def _hyper_connection_specs(g: Geometry, prefix: str, with_inject: bool) -> tuple[TensorSpec, ...]:
    specs = [
        tensor_spec(prefix + "norm", (g.residual,), FP32),
        tensor_spec(prefix + "down", (g.hc_low_rank, g.residual), BF16),
        tensor_spec(prefix + "up", (g.residual, g.hc_low_rank), BF16),
    ]
    if with_inject:
        specs.append(tensor_spec(prefix + "inject", (g.hc_streams, g.residual), BF16))
    return tuple(specs)


def build_text_core_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (g.vocab, g.hidden), W8),
    ]
    for layer in range(g.layers):
        prefix = f"text/layers/{layer}/"
        if g.ple_ngram and layer == g.ple_layer:
            specs.extend(
                (
                    tensor_spec(prefix + "ple/key", (g.residual, g.ple_embed), BF16),
                    tensor_spec(prefix + "ple/value", (g.hidden, g.ple_embed), BF16),
                    tensor_spec(prefix + "ple/norm_key", (g.residual,), FP32),
                    tensor_spec(prefix + "ple/norm_query", (g.residual,), FP32),
                    tensor_spec(prefix + "ple/norm_conv", (g.residual,), FP32),
                    tensor_spec(prefix + "ple/convolution", (g.ple_conv_kernel, g.residual), BF16),
                )
            )
        specs.extend(_hyper_connection_specs(g, prefix + "hc_attn/", with_inject=True))
        if layer in g.full_attention_layers:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "attention/query_key_gate_value",
                        (g.attention_fused_rows, g.hidden),
                        W8,
                    ),
                    tensor_spec(prefix + "attention/query_norm", (g.head_dim,), BF16),
                    tensor_spec(prefix + "attention/key_norm", (g.head_dim,), BF16),
                    tensor_spec(prefix + "attention/output", (g.hidden, g.query_size), W8),
                    tensor_spec(
                        prefix + "attention/indexer/query",
                        (g.indexer_heads * g.indexer_head_dim, g.hidden),
                        BF16,
                    ),
                    tensor_spec(prefix + "attention/indexer/key", (g.indexer_head_dim, g.hidden), BF16),
                    tensor_spec(prefix + "attention/indexer/query_norm", (g.indexer_head_dim,), BF16),
                    tensor_spec(prefix + "attention/indexer/key_norm", (g.indexer_head_dim,), BF16),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "gdn/a_log", (g.gdn_value_heads,), FP32),
                    tensor_spec(prefix + "gdn/dt_bias", (g.gdn_value_heads,), FP32),
                    tensor_spec(prefix + "gdn/convolution", (g.gdn_conv_kernel, g.convolution_dim), BF16),
                    tensor_spec(prefix + "gdn/a_b_projection", (2 * g.gdn_value_heads, g.hidden), BF16),
                    tensor_spec(prefix + "gdn/query_key_value_z", (g.gdn_fused_rows, g.hidden), W8),
                    tensor_spec(prefix + "gdn/norm", (g.gdn_value_head_dim,), BF16),
                    tensor_spec(prefix + "gdn/output", (g.hidden, g.value_dim), W8),
                )
            )
        specs.extend(_hyper_connection_specs(g, prefix + "hc_ffn/", with_inject=True))
        specs.extend(
            (
                tensor_spec(prefix + "mlp/router_shared_gate", (g.router_rows, g.hidden), BF16),
                tensor_spec(prefix + "mlp/routed_gate_up", (g.experts * 2 * g.intermediate, g.hidden), W8),
                tensor_spec(prefix + "mlp/routed_down", (g.experts * g.hidden, g.intermediate), W8),
                tensor_spec(prefix + "mlp/shared_gate_up", (2 * g.shared_intermediate, g.hidden), W8),
                tensor_spec(prefix + "mlp/shared_down", (g.hidden, g.shared_intermediate), W8),
            )
        )
    specs.extend(_hyper_connection_specs(g, "text/output_hc/", with_inject=False))
    specs.extend(
        (
            tensor_spec("text/output_head", (g.vocab, g.hidden), W8),
        )
    )
    if g.ple_ngram:
        specs.extend((
            tensor_spec("text/ple/multipliers", (2 * g.ple_ngram,), I32),
            tensor_spec("text/ple/head_offsets", (g.ple_heads,), I32),
            tensor_spec("text/ple/head_vocab_sizes", (g.ple_heads,), I32),
        ))
    return tuple(specs)


def build_mtp_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    prefix = "mtp/"
    layer = prefix + "layer/"
    specs: list[TensorSpec] = [
        tensor_spec(prefix + "embedding_norm", (g.hidden,), BF16),
        tensor_spec(prefix + "hidden_norm", (g.residual,), FP32),
        tensor_spec(prefix + "input_projection", (g.hidden, 2 * g.hidden), W8),
    ]
    specs.extend(_hyper_connection_specs(g, layer + "hc_attn/", with_inject=True))
    specs.extend(
        (
            tensor_spec(layer + "attention/query_key_gate_value", (g.attention_fused_rows, g.hidden), W8),
            tensor_spec(layer + "attention/query_norm", (g.head_dim,), BF16),
            tensor_spec(layer + "attention/key_norm", (g.head_dim,), BF16),
            tensor_spec(layer + "attention/output", (g.hidden, g.query_size), W8),
            tensor_spec(layer + "attention/indexer/query", (g.indexer_heads * g.indexer_head_dim, g.hidden), BF16),
            tensor_spec(layer + "attention/indexer/key", (g.indexer_head_dim, g.hidden), BF16),
            tensor_spec(layer + "attention/indexer/query_norm", (g.indexer_head_dim,), BF16),
            tensor_spec(layer + "attention/indexer/key_norm", (g.indexer_head_dim,), BF16),
        )
    )
    specs.extend(_hyper_connection_specs(g, layer + "hc_ffn/", with_inject=True))
    specs.extend(
        (
            tensor_spec(layer + "mlp/router_shared_gate", (g.router_rows, g.hidden), BF16),
            tensor_spec(layer + "mlp/routed_gate_up", (g.experts * 2 * g.intermediate, g.hidden), W8),
            tensor_spec(layer + "mlp/routed_down", (g.experts * g.hidden, g.intermediate), W8),
            tensor_spec(layer + "mlp/shared_gate_up", (2 * g.shared_intermediate, g.hidden), W8),
            tensor_spec(layer + "mlp/shared_down", (g.hidden, g.shared_intermediate), W8),
        )
    )
    specs.extend(_hyper_connection_specs(g, prefix + "head_hc/", with_inject=False))
    return tuple(specs)



def ple_table_spec(g: Geometry) -> TensorSpec:
    if not g.ple_ngram:
        raise ValueError("this checkpoint does not declare a PLE table")
    return TensorSpec(PLE_TABLE_RESOURCE, (g.ple_table_rows, g.ple_head_dim),
                      PLE_TABLE_FORMAT, GGML_BLOCKS_LAYOUT)


def build_vision_specs(g: Geometry, *, vision_storage: str = VISION_BF16) -> tuple[TensorSpec, ...]:
    """The tower this checkpoint declares, at the width the export profile chooses.

    Nothing this converter writes reaches here: its source is a GGUF, every published
    export of this model drops the tower, and `convert` refuses one that carries it rather
    than emitting a silent text-only artifact -- so both conversion paths ask for
    `vision=False`. What does reach here is the declaration side, which is not nothing: the
    inventory validator, the repack probe and the serve contract all build these specs, and
    the contract compares their widths against what the model declares. `vision_storage` is
    the knob that comparison is about.
    """
    tower = vision_tower(g)
    return _vision_specs(g.hidden, storage=vision_storage, **tower) if tower else ()


def active_specs(*, geometry: Geometry, vision: bool | None = None,
                 mtp: bool | None = None,
                 vision_storage: str = VISION_BF16) -> tuple[tuple, tuple]:
    g = geometry
    tensors = build_text_core_specs(g)
    if vision is not False:
        tensors += build_vision_specs(g, vision_storage=vision_storage)
    if mtp is not False and g.mtp_layers:
        tensors += build_mtp_specs(g)
    leading = (ple_table_spec(g),) if g.ple_ngram else ()
    return tensors, RESOURCE_SPECS + leading + tensors


def validate_inventory(g: Geometry) -> None:
    _, objects = active_specs(geometry=g)
    names = [spec.name for spec in objects]
    if len(names) != len(set(names)):
        raise ValueError("duplicate object name in the qwen4exp inventory")
