"""Persistent-object contract for a `Qwen3ForCausalLM` serving artifact.

Qwen3 is the plain dense member of the family this package converts: every layer
is full attention, there is no linear-attention mixer, no MTP block, no vision
tower and no draft head.  Two consequences shape this inventory and are the only
places it departs from its hybrid siblings:

* The fused attention projection carries ``query | key | value`` and nothing
  else.  Every other target in this package fuses ``query | key | gate | value``
  because its attention is output-gated; Qwen3's is not, so the object is named
  ``attention/query_key_value`` and its row count is
  ``query_size + 2 * kv_size``, not ``2 * query_size + 2 * kv_size``.
* Qwen3 applies RMS norm to the query and key head vectors, so each layer stores
  ``attention/query_norm`` and ``attention/key_norm``, each ``head_dim`` wide
  (the hybrid targets store them at their own head width; here it is 128).

Row order convention, shared with the rest of the package: every 2-D object is
logical ``[rows = outputs, k = inputs]``.

Numeric profile: every matrix is ``W8G32_F16S`` in the ``row-split-k128-v1``
layout and every norm is BF16, which is the profile the 0.8B sibling uses and
the one the shared kernels already bind.
"""

from __future__ import annotations

from dataclasses import dataclass

from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    RESOURCE_ENCODING,
    ROW_SPLIT_LAYOUT,
    W8,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    tensor_spec,
)

MODEL_ID = "qwen3-0.6b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3"


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions an artifact's object list depends on.

    These are read off `config.json` at convert time and checked against the
    registered target below, rather than being spelled out a second time: the
    same builder then serves a differently sized `Qwen3ForCausalLM` the day a
    second target header exists for one.
    """

    layers: int
    hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attention_fused_rows(self) -> int:
        """Ungated: query, key and value stacked, with no output-gate block."""
        return self.query_size + 2 * self.kv_size

    @property
    def mlp_gate_up_rows(self) -> int:
        return 2 * self.intermediate


#: `csrc/src/serve/targets/qwen3/impl/config.h`, which this must agree with.
QWEN3_0_6B = Geometry(
    layers=28,
    hidden=1024,
    intermediate=3072,
    vocab=151936,
    query_heads=16,
    kv_heads=8,
    head_dim=128,
)

GEOMETRY = QWEN3_0_6B

def hf_config_for(geometry: Geometry = GEOMETRY) -> dict:
    """The `config.json` a checkpoint of these dimensions would carry.

    The inverse of `recipe.geometry_from_config`, and what lets a caller holding only
    a registered geometry derive the conversion recipes: the declaration those come
    from is compiled against a config either way.
    """
    return declaration.text_config(
        "Qwen3ForCausalLM",
        "qwen3",
        layers=geometry.layers,
        hidden=geometry.hidden,
        intermediate=geometry.intermediate,
        vocab=geometry.vocab,
        query_heads=geometry.query_heads,
        kv_heads=geometry.kv_heads,
        head_dim=geometry.head_dim,
    )


LAYERS = GEOMETRY.layers
HIDDEN = GEOMETRY.hidden
INTERMEDIATE = GEOMETRY.intermediate
VOCAB = GEOMETRY.vocab
QUERY_HEADS = GEOMETRY.query_heads
KV_HEADS = GEOMETRY.kv_heads
HEAD_DIM = GEOMETRY.head_dim
QUERY_SIZE = GEOMETRY.query_size
KV_SIZE = GEOMETRY.kv_size
ATTENTION_FUSED_ROWS = GEOMETRY.attention_fused_rows
FULL_ATTENTION_LAYERS = tuple(range(LAYERS))


#: Qwen3 is text-only, so the artifact carries the four text frontend files and
#: not the two image/video preprocessor configs the vision-capable targets bind.
#: `chat_template.jinja` has no file of its own in the Hugging Face release; the
#: converter lifts it out of `tokenizer_config.json`, which is also where the
#: engine cross-checks it.
RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
    )
)


def build_tensor_specs(geometry: Geometry = GEOMETRY) -> tuple[TensorSpec, ...]:
    """The complete ordered tensor inventory for one Qwen3 text stack."""

    hidden = geometry.hidden
    specs: list[TensorSpec] = [
        tensor_spec("text/token_embedding", (geometry.vocab, hidden), W8),
    ]

    for layer in range(geometry.layers):
        prefix = f"text/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "input_norm", (hidden,), BF16),
                tensor_spec(
                    prefix + "attention/query_key_value",
                    (geometry.attention_fused_rows, hidden),
                    W8,
                ),
                tensor_spec(prefix + "attention/query_norm", (geometry.head_dim,), BF16),
                tensor_spec(prefix + "attention/key_norm", (geometry.head_dim,), BF16),
                tensor_spec(
                    prefix + "attention/output", (hidden, geometry.query_size), W8
                ),
                tensor_spec(prefix + "post_attention_norm", (hidden,), BF16),
                tensor_spec(
                    prefix + "mlp/gate_up", (geometry.mlp_gate_up_rows, hidden), W8
                ),
                tensor_spec(prefix + "mlp/down", (hidden, geometry.intermediate), W8),
            )
        )

    specs.extend(
        (
            tensor_spec("text/final_norm", (hidden,), BF16),
            # Stored, not aliased, even when the checkpoint ties it to the
            # embedding: the head is bound as its own device tensor and the
            # embedding gather and the output matmul want different residency.
            tensor_spec("text/output_head", (geometry.vocab, hidden), W8),
        )
    )
    return tuple(specs)


def build_object_specs(
    geometry: Geometry = GEOMETRY,
) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_tensor_specs(geometry)


TEXT_CORE_TENSOR_SPECS = build_tensor_specs()
TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


def build_logical_row_views(
    geometry: Geometry = GEOMETRY,
) -> tuple[LogicalRowViewSpec, ...]:
    """Row ranges of the fused objects, by the logical projection they carry.

    This is what places a LoRA adapter trained against `q_proj` on the right
    rows of `attention/query_key_value`, and what `verify` slices when it
    compares a converted object against the checkpoint tensor it came from.
    """

    layers = tuple(range(geometry.layers))
    query, kv = geometry.query_size, geometry.kv_size
    fused = "text/layers/{l}/attention/query_key_value"
    gate_up = "text/layers/{l}/mlp/gate_up"
    intermediate = geometry.intermediate
    return (
        LogicalRowViewSpec(
            "text/layers/{l}/attention/query", fused,
            0, query, (query, geometry.hidden), layers,
        ),
        LogicalRowViewSpec(
            "text/layers/{l}/attention/key", fused,
            query, query + kv, (kv, geometry.hidden), layers,
        ),
        LogicalRowViewSpec(
            "text/layers/{l}/attention/value", fused,
            query + kv, query + 2 * kv, (kv, geometry.hidden), layers,
        ),
        LogicalRowViewSpec(
            "text/layers/{l}/mlp/gate", gate_up,
            0, intermediate, (intermediate, geometry.hidden), layers,
        ),
        LogicalRowViewSpec(
            "text/layers/{l}/mlp/up", gate_up,
            intermediate, 2 * intermediate, (intermediate, geometry.hidden), layers,
        ),
    )


LOGICAL_ROW_VIEW_SPECS = build_logical_row_views()

#: No aliases: this target has no MTP head reusing the text embedding, and no
#: object is stored in an order a second consumer needs permuted.
ALIAS_SPECS: tuple[LogicalAliasSpec, ...] = ()


__all__ = [
    "ALIAS_SPECS",
    "ATTENTION_FUSED_ROWS",
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "FORMAT_COUNTS",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GEOMETRY",
    "Geometry",
    "HEAD_DIM",
    "HIDDEN",
    "I32",
    "INTERMEDIATE",
    "KV_HEADS",
    "KV_SIZE",
    "LAYERS",
    "LAYOUT_COUNTS",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "OBJECT_SPECS",
    "QUERY_HEADS",
    "QUERY_SIZE",
    "QWEN3_0_6B",
    "RESOURCE_ENCODING",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VOCAB",
    "W8",
    "WEIGHTS_ID",
    "build_logical_row_views",
    "build_object_specs",
    "build_tensor_specs",
    "tensor_spec",
]
