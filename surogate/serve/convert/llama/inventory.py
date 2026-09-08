"""Serving objects for llama, shaped by the checkpoint's resolved declaration.

Every matrix uses logical [outputs, inputs] order. The converter's groupwise
profile stores matrices as W8 and norms as BF16; native GGUF objects retain
their actual storage format. The recipe resolves output-head tying from the
checkpoint, and the inventory requires its dimensions explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

MODEL_ID = "llama"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "llama"

#: Objects a Llama layer stores: input norm, fused QKV, attention output,
#: post-attention norm, fused gate/up, down.  Six, against Qwen3's eight.
LAYER_OBJECT_COUNT = 6


@dataclass(frozen=True, slots=True)
class Geometry:
    """Dimensions and declaration resolved for the checkpoint being converted."""

    layers: int
    hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    declared: declaration.Declaration = field(repr=False, compare=False)

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


def build_tensor_specs(geometry: Geometry) -> tuple[TensorSpec, ...]:
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
            # The output head is stored separately even when its source is tied.
            tensor_spec("text/output_head", (geometry.vocab, hidden), W8),
        )
    )
    return tuple(specs)


def build_object_specs(
    geometry: Geometry,
) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_tensor_specs(geometry)


def build_logical_row_views(
    geometry: Geometry,
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



#: No aliases: this target has no MTP head reusing the text embedding, no tied
#: output head, and no object is stored in an order a second consumer needs
#: permuted.
ALIAS_SPECS: tuple[LogicalAliasSpec, ...] = ()
