"""Persistent-object contract for the Qwen3.8-Flash-Next (`qwen4exp`) artifact.

Geometry (GGUF metadata, HF `Qwen/Qwen3.8-Flash-Next`): 48 layers, hidden 2560,
hyper-connection residual of 4 streams (10,240 wide, low-rank mix 320), 36
gated-delta-net layers (16 k-heads, 48 v-heads, head 128, conv 4) and 12
full-attention layers (24 q-heads, 2 kv-heads, head 256, every 4th layer),
512 routed experts top-10 (FFN 640) plus a shared expert (640), an n-gram
PLE memory feeding one layer, a QSA indexer on the attention layers (4 heads,
dim 128), vocabulary 248,320.

Formats follow the measured trade in design/serve-engine-flash-next.md D7:
routed and shared experts W8G32 (bit-exact from Q8_0, 0.55 % from the
K-quants; the symmetric group-64 formats would be a second quantisation),
attention/GDN projections W8G32 (exact from Q8_0), hyper-connection and PLE
projections BF16, the PLE table kept as the GGUF's IQ4_NL rows verbatim
(28.8 GB, host-resident, decoded by the gather kernel), norms F32 with the
GGUF's `+1` already folded (the converter must not add it again).

Row order convention: every 2-D object is logical [rows = outputs, k = inputs].
GDN value heads are stored in HF *grouped* order (the engine's GDN kernels
expect it); llama.cpp's GGUF converter reorders them to *tiled* order for
ggml's broadcast, and `recipe.py` inverts that.
"""

from __future__ import annotations

from surogate.serve.convert.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    FP32,
    GGML_BLOCKS_LAYOUT,
    I32,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    build_vision_specs,
)

MODEL_ID = "qwen3.8-flash-next"
WEIGHTS_ID = "w8-hc-v1"
TARGET_KEY = "qwen4exp"

# The PLE table is the GGUF's IQ4_NL rows verbatim (18 bytes per 32 values: fp16 scale + 16
# codebook nibbles), 90 bytes per 160-wide row, host-resident and decoded by the gather kernel.
# It is a tensor rather than a resource because only a tensor can carry `runs`, and 28.8 GB
# that already exist byte-identically in the GGUF are not worth copying into the artifact. It
# stays outside `TENSOR_SPECS`: the engine binds it as one flat table, not as part of a layer.
# Its hash parameters are ordinary I32 tensors.
PLE_TABLE_RESOURCE = "text/ple/table.iq4nl"
PLE_TABLE_FORMAT = "IQ4_NL"
PLE_TABLE_ROW_BYTES = 90

FORMAT_NAMES = (BF16, FP32, I32, W8)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT)

LAYERS = 48
HIDDEN = 2560
HC_COUNT = 4
HC_WIDTH = HC_COUNT * HIDDEN  # 10240
HC_LOW_RANK = 320
VOCAB = 248320

FULL_ATTENTION_INTERVAL = 4
FULL_ATTENTION_LAYERS = tuple(
    layer for layer in range(LAYERS) if (layer + 1) % FULL_ATTENTION_INTERVAL == 0
)
GDN_LAYERS = tuple(layer for layer in range(LAYERS) if layer not in FULL_ATTENTION_LAYERS)
PLE_LAYER = 1  # GGUF `qwen4exp.ple.layers = [1]`; HF `ple_layer_ids = [2]` is 1-based

# Attention
QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
QUERY_SIZE = QUERY_HEADS * HEAD_DIM  # 6144
KV_SIZE = KV_HEADS * HEAD_DIM  # 512
ATTENTION_FUSED_ROWS = 2 * QUERY_SIZE + 2 * KV_SIZE  # q | k | gate | v = 13312
INDEXER_HEADS = 4
INDEXER_DIM = 128

# Gated delta net
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_KEY_DIM = GDN_KEY_HEADS * GDN_HEAD_DIM  # 2048
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_DIM  # 6144
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VALUE_DIM  # 10240
GDN_FUSED_ROWS = GDN_CONV_DIM + GDN_VALUE_DIM  # qkv | z = 16384
GDN_CONV_KERNEL = 4

# MoE
EXPERTS = 512
TOP_K = 10
EXPERT_FFN = 640
SHARED_FFN = 640
ROUTER_ROWS = EXPERTS + 1  # 512 router rows + the shared-expert gate row

# PLE
PLE_NGRAM = 3
PLE_HEADS_PER_NGRAM = 8
PLE_HEADS = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM  # 16
PLE_HEAD_DIM = 160
PLE_EMBED = PLE_HEADS * PLE_HEAD_DIM  # 2560
PLE_TABLE_ROWS = 320001536
PLE_CONV_KERNEL = 4


def tensor_spec(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    if numeric_format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format == W8:
        layout = ROW_SPLIT_LAYOUT
    else:
        raise ValueError(f"unsupported qwen4exp format: {numeric_format}")
    return TensorSpec(name, shape, numeric_format, layout)


def _hyper_connection_specs(prefix: str, with_inject: bool) -> tuple[TensorSpec, ...]:
    specs = [
        tensor_spec(prefix + "norm", (HC_WIDTH,), FP32),
        tensor_spec(prefix + "down", (HC_LOW_RANK, HC_WIDTH), BF16),
        tensor_spec(prefix + "up", (HC_WIDTH, HC_LOW_RANK), BF16),
    ]
    if with_inject:
        specs.append(tensor_spec(prefix + "inject", (HC_COUNT, HC_WIDTH), BF16))
    return tuple(specs)


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (VOCAB, HIDDEN), W8),
    ]
    for layer in range(LAYERS):
        prefix = f"text/layers/{layer}/"
        if layer == PLE_LAYER:
            specs.extend(
                (
                    tensor_spec(prefix + "ple/key", (HC_WIDTH, PLE_EMBED), BF16),
                    tensor_spec(prefix + "ple/value", (HIDDEN, PLE_EMBED), BF16),
                    tensor_spec(prefix + "ple/norm_key", (HC_WIDTH,), FP32),
                    tensor_spec(prefix + "ple/norm_query", (HC_WIDTH,), FP32),
                    tensor_spec(prefix + "ple/norm_conv", (HC_WIDTH,), FP32),
                    tensor_spec(prefix + "ple/convolution", (PLE_CONV_KERNEL, HC_WIDTH), BF16),
                )
            )
        specs.extend(_hyper_connection_specs(prefix + "hc_attn/", with_inject=True))
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    tensor_spec(
                        prefix + "attention/query_key_gate_value",
                        (ATTENTION_FUSED_ROWS, HIDDEN),
                        W8,
                    ),
                    tensor_spec(prefix + "attention/query_norm", (HEAD_DIM,), BF16),
                    tensor_spec(prefix + "attention/key_norm", (HEAD_DIM,), BF16),
                    tensor_spec(prefix + "attention/output", (HIDDEN, QUERY_SIZE), W8),
                    # QSA indexer (phase 2): carried now so the artifact is complete.
                    tensor_spec(
                        prefix + "attention/indexer/query",
                        (INDEXER_HEADS * INDEXER_DIM, HIDDEN),
                        BF16,
                    ),
                    tensor_spec(prefix + "attention/indexer/key", (INDEXER_DIM, HIDDEN), BF16),
                    tensor_spec(prefix + "attention/indexer/query_norm", (INDEXER_DIM,), BF16),
                    tensor_spec(prefix + "attention/indexer/key_norm", (INDEXER_DIM,), BF16),
                )
            )
        else:
            specs.extend(
                (
                    tensor_spec(prefix + "gdn/a_log", (GDN_VALUE_HEADS,), FP32),
                    tensor_spec(prefix + "gdn/dt_bias", (GDN_VALUE_HEADS,), FP32),
                    tensor_spec(prefix + "gdn/convolution", (GDN_CONV_KERNEL, GDN_CONV_DIM), BF16),
                    tensor_spec(prefix + "gdn/a_b_projection", (2 * GDN_VALUE_HEADS, HIDDEN), BF16),
                    tensor_spec(prefix + "gdn/query_key_value_z", (GDN_FUSED_ROWS, HIDDEN), W8),
                    tensor_spec(prefix + "gdn/norm", (GDN_HEAD_DIM,), BF16),
                    tensor_spec(prefix + "gdn/output", (HIDDEN, GDN_VALUE_DIM), W8),
                )
            )
        specs.extend(_hyper_connection_specs(prefix + "hc_ffn/", with_inject=True))
        specs.extend(
            (
                tensor_spec(prefix + "mlp/router_shared_gate", (ROUTER_ROWS, HIDDEN), BF16),
                tensor_spec(prefix + "mlp/routed_gate_up", (EXPERTS * 2 * EXPERT_FFN, HIDDEN), W8),
                tensor_spec(prefix + "mlp/routed_down", (EXPERTS * HIDDEN, EXPERT_FFN), W8),
                tensor_spec(prefix + "mlp/shared_gate_up", (2 * SHARED_FFN, HIDDEN), W8),
                tensor_spec(prefix + "mlp/shared_down", (HIDDEN, SHARED_FFN), W8),
            )
        )
    specs.extend(_hyper_connection_specs("text/output_hc/", with_inject=False))
    specs.extend(
        (
            tensor_spec("text/output_head", (VOCAB, HIDDEN), W8),
            # The n-gram memory's hash parameters (the table itself is a resource).
            # Three uint64 multipliers as (lo, hi) int32 pairs.
            tensor_spec("text/ple/multipliers", (2 * PLE_NGRAM,), I32),
            tensor_spec("text/ple/head_offsets", (PLE_HEADS,), I32),
            tensor_spec("text/ple/head_vocab_sizes", (PLE_HEADS,), I32),
        )
    )
    return tuple(specs)


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    """The NextN/MTP draft head, `blk.48` of the separate MTP export.

    Structurally an ordinary trunk full-attention block -- two hyper-connection
    modules, dense attention with the interleaved query/gate, the routed MoE and
    the shared expert -- with six tensors of its own on either side of it:
    `embedding_norm`/`hidden_norm`/`input_projection` fold the next token's
    embedding into the trunk's wide residual on the way in, and `head_hc/`
    collapses the four streams on the way out, standing in for the output norm
    qwen4exp does not have. The embedding table and the LM head are the trunk's:
    the export declares `nextn_shared_target_tensors`, so it carries neither.

    The indexer is bound but unused: the draft attends densely, which is a
    numerical superset of QSA (the trunk only prunes past a 2,048-token budget)
    and the target verifies the draft either way.
    """
    prefix = "mtp/"
    layer = prefix + "layer/"
    specs: list[TensorSpec] = [
        tensor_spec(prefix + "embedding_norm", (HIDDEN,), FP32),
        tensor_spec(prefix + "hidden_norm", (HC_WIDTH,), FP32),
        tensor_spec(prefix + "input_projection", (HIDDEN, 2 * HIDDEN), W8),
    ]
    specs.extend(_hyper_connection_specs(layer + "hc_attn/", with_inject=True))
    specs.extend(
        (
            tensor_spec(layer + "attention/query_key_gate_value", (ATTENTION_FUSED_ROWS, HIDDEN), W8),
            tensor_spec(layer + "attention/query_norm", (HEAD_DIM,), BF16),
            tensor_spec(layer + "attention/key_norm", (HEAD_DIM,), BF16),
            tensor_spec(layer + "attention/output", (HIDDEN, QUERY_SIZE), W8),
            tensor_spec(layer + "attention/indexer/query", (INDEXER_HEADS * INDEXER_DIM, HIDDEN), BF16),
            tensor_spec(layer + "attention/indexer/key", (INDEXER_DIM, HIDDEN), BF16),
            tensor_spec(layer + "attention/indexer/query_norm", (INDEXER_DIM,), BF16),
            tensor_spec(layer + "attention/indexer/key_norm", (INDEXER_DIM,), BF16),
        )
    )
    specs.extend(_hyper_connection_specs(layer + "hc_ffn/", with_inject=True))
    specs.extend(
        (
            tensor_spec(layer + "mlp/router_shared_gate", (ROUTER_ROWS, HIDDEN), BF16),
            tensor_spec(layer + "mlp/routed_gate_up", (EXPERTS * 2 * EXPERT_FFN, HIDDEN), W8),
            tensor_spec(layer + "mlp/routed_down", (EXPERTS * HIDDEN, EXPERT_FFN), W8),
            tensor_spec(layer + "mlp/shared_gate_up", (2 * SHARED_FFN, HIDDEN), W8),
            tensor_spec(layer + "mlp/shared_down", (HIDDEN, SHARED_FFN), W8),
        )
    )
    specs.extend(_hyper_connection_specs(prefix + "head_hc/", with_inject=False))
    return tuple(specs)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()

# Flash-Next's tower is the Qwen3.6 one (27 layers of 1152, head_dim 72), which is
# exactly what the vision kernels implement. Whether an *artifact* carries it is a
# property of the export, not of the model: the community GGUF exports drop the
# tower entirely (the 4-shard Q4_K_XL set has 1,224 tensors and none of them
# vision), while a safetensors checkpoint has it. So the inventory comes in both
# shapes and the converter picks by what the source actually provides -- the same
# way the family already handles MTP-less exports.
VISION_TENSOR_SPECS = build_vision_specs(HIDDEN)

TEXT_ONLY_TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS
TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS + VISION_TENSOR_SPECS
PLE_TABLE_SPEC = TensorSpec(
    PLE_TABLE_RESOURCE, (PLE_TABLE_ROWS, PLE_HEAD_DIM), PLE_TABLE_FORMAT, GGML_BLOCKS_LAYOUT
)
#: The objects that precede the layer stack, in artifact order: the frontend files, then the
#: PLE table.
LEADING_OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + (PLE_TABLE_SPEC,)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = LEADING_OBJECT_SPECS + TENSOR_SPECS
#: For sources that carry no vision tower — the community GGUF exports drop it.
TEXT_ONLY_OBJECT_SPECS: tuple[StoredObjectSpec, ...] = (
    LEADING_OBJECT_SPECS + TEXT_ONLY_TENSOR_SPECS
)


def active_specs(*, vision: bool, mtp: bool = False) -> tuple[tuple, tuple]:
    """The (tensor, object) spec pair matching what the source provides.

    Two independent axes, because the two are carried by different files: the
    vision tower is in the checkpoint or not, and the MTP head arrives as its
    own GGUF alongside the trunk's shards.
    """

    tensors = TENSOR_SPECS if vision else TEXT_ONLY_TENSOR_SPECS
    objects = OBJECT_SPECS if vision else TEXT_ONLY_OBJECT_SPECS
    if mtp:
        tensors = tensors + MTP_TENSOR_SPECS
        objects = objects + MTP_TENSOR_SPECS
    return tensors, objects

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}


def validate_inventory() -> None:
    names = [spec.name for spec in OBJECT_SPECS]
    if len(names) != len(set(names)):
        raise ValueError("duplicate object name in the qwen4exp inventory")
    if len(FULL_ATTENTION_LAYERS) != 12 or len(GDN_LAYERS) != 36:
        raise ValueError("qwen4exp layer split is not 12 attention + 36 GDN")
    if PLE_LAYER not in GDN_LAYERS:
        raise ValueError("the PLE layer is expected to be a GDN layer")
    per_layer_w8 = 4  # routed gate_up, routed down, shared gate_up, shared down
    # The merger's two projections are the tower's only W8 objects; the rest are
    # K-quants. Counted only when the tower is part of this inventory.
    vision_w8 = 2 if VISION_TENSOR_SPECS and VISION_TENSOR_SPECS[0] in TENSOR_SPECS else 0
    expected_w8 = (
        2 + LAYERS * per_layer_w8 + len(FULL_ATTENTION_LAYERS) * 2 + len(GDN_LAYERS) * 2
        + vision_w8
    )
    if FORMAT_COUNTS[W8] != expected_w8:
        raise ValueError(f"expected {expected_w8} W8 objects, found {FORMAT_COUNTS[W8]}")


validate_inventory()

__all__ = [
    "ATTENTION_FUSED_ROWS",
    "BF16",
    "EXPERTS",
    "EXPERT_FFN",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_CONV_DIM",
    "GDN_CONV_KERNEL",
    "GDN_FUSED_ROWS",
    "GDN_HEAD_DIM",
    "GDN_KEY_DIM",
    "GDN_KEY_HEADS",
    "GDN_LAYERS",
    "GDN_VALUE_DIM",
    "GDN_VALUE_HEADS",
    "HC_COUNT",
    "HC_LOW_RANK",
    "HC_WIDTH",
    "HEAD_DIM",
    "HIDDEN",
    "I32",
    "INDEXER_DIM",
    "INDEXER_HEADS",
    "LEADING_OBJECT_SPECS",
    "KV_HEADS",
    "KV_SIZE",
    "LAYERS",
    "LAYOUT_NAMES",
    "MODEL_ID",
    "OBJECT_SPECS",
    "PLE_CONV_KERNEL",
    "PLE_EMBED",
    "PLE_HEADS",
    "PLE_HEAD_DIM",
    "PLE_LAYER",
    "PLE_NGRAM",
    "PLE_TABLE_FORMAT",
    "PLE_TABLE_RESOURCE",
    "PLE_TABLE_ROWS",
    "PLE_TABLE_ROW_BYTES",
    "PLE_TABLE_SPEC",
    "QUERY_HEADS",
    "QUERY_SIZE",
    "RESOURCE_SPECS",
    "ROUTER_ROWS",
    "ResourceSpec",
    "SHARED_FFN",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TOP_K",
    "TensorSpec",
    "VOCAB",
    "W8",
    "WEIGHTS_ID",
    "tensor_spec",
    "validate_inventory",
]
