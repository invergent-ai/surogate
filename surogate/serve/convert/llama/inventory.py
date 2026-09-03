"""Persistent-object contract for a `LlamaForCausalLM` serving artifact.

Llama is the smallest dense stack this package converts, and it reads as the
Qwen3 inventory with two subtractions and one substitution:

* **No per-head query/key norm.**  Qwen3 RMS-norms the query and key head
  vectors and therefore stores ``attention/query_norm`` and
  ``attention/key_norm`` per layer.  Llama normalizes neither, so those objects
  do not exist here at all — a layer holds six objects, not eight — and a
  binder that looked for them would be describing a different architecture.
* **The output head is its own tensor.**  Qwen3-0.6B ties it to the embedding;
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` declares ``tie_word_embeddings: false``
  and ships a genuinely separate ``lm_head.weight``.  The inventory stores the
  head either way (as Qwen3's does), but for this target the tie is a property
  of the checkpoint that the *recipe* reads, not a foregone conclusion.
* Attention is ungated, exactly as in Qwen3: the fused projection carries
  ``query | key | value`` and its row count is ``query_size + 2 * kv_size``.

Row order convention, shared with the rest of the package: every 2-D object is
logical ``[rows = outputs, k = inputs]``.

Numeric profile: every matrix is ``W8G32_F16S`` in the ``row-split-k128-v1``
layout and every norm is BF16 — the same profile Qwen3 uses and the one the
shared kernels already bind.
"""

from __future__ import annotations

from dataclasses import dataclass

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

#: The engine picks a target by comparing an artifact's identity against a
#: package's, string for string: `csrc/src/serve/api/targets/llama/package.h`
#: declares `model_id = "tinyllama-1.1b"` / `target_key = "llama"`, and
#: `csrc/src/serve/targets/llama/impl/package.cpp` admits that pair only with
#: the `groupwise-int` weights profile this inventory encodes. An artifact
#: that spells any of the three differently is refused at load.
MODEL_ID = "tinyllama-1.1b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "llama"

#: Objects a Llama layer stores: input norm, fused QKV, attention output,
#: post-attention norm, fused gate/up, down.  Six, against Qwen3's eight.
LAYER_OBJECT_COUNT = 6


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions an artifact's object list depends on.

    These are read off `config.json` at convert time and checked against the
    registered target below, rather than being spelled out a second time: the
    same builder then serves a differently sized `LlamaForCausalLM` the day a
    second target header exists for one.

    `head_dim` is derived, not read: a Llama `config.json` in the dialect this
    target was exported with carries no `head_dim` key, and the architecture
    fixes it at `hidden_size // num_attention_heads`.
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


#: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, the one registered size of this target.
TINYLLAMA_1_1B = Geometry(
    layers=22,
    hidden=2048,
    intermediate=5632,
    vocab=32000,
    query_heads=32,
    kv_heads=4,
    head_dim=64,
)

GEOMETRY = TINYLLAMA_1_1B

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


#: Llama is text-only, so the artifact carries the four text frontend files and
#: not the two image/video preprocessor configs the vision-capable targets bind.
#: `chat_template.jinja` has no file of its own in the Hugging Face release; the
#: converter lifts it out of `tokenizer_config.json`, which is also where the
#: engine cross-checks it.  The SentencePiece `tokenizer.model` beside it is not
#: bound: the engine reads the `tokenizer.json` the release also ships, and an
#: object no binder consumes is refused at load.
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
    """The complete ordered tensor inventory for one Llama text stack."""

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
                # No query_norm or key_norm: Llama normalizes neither.
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
            # Stored, not aliased. TinyLlama declares tie_word_embeddings=false
            # and ships its own lm_head.weight, so unlike Qwen3 this is not the
            # embedding under a second name; it would still be stored if it were,
            # because the gather and the output matmul want different residency.
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

#: No aliases: this target has no MTP head reusing the text embedding, no tied
#: output head, and no object is stored in an order a second consumer needs
#: permuted.
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
    "LAYER_OBJECT_COUNT",
    "LAYERS",
    "LAYOUT_COUNTS",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "OBJECT_SPECS",
    "QUERY_HEADS",
    "QUERY_SIZE",
    "RESOURCE_ENCODING",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TINYLLAMA_1_1B",
    "TensorSpec",
    "VOCAB",
    "W8",
    "WEIGHTS_ID",
    "build_logical_row_views",
    "build_object_specs",
    "build_tensor_specs",
    "tensor_spec",
]
