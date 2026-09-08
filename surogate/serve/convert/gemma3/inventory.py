"""Gemma 3 serving objects, shaped by the resolved checkpoint configuration."""

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

MODEL_ID = "gemma3"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "gemma3"

#: Objects a Gemma 3 layer stores: four sandwich norms, three separate attention
#: projections, two per-head norms, the attention output, and three MLP
#: matrices. Thirteen, against Llama's six and Qwen3's eight.
LAYER_OBJECT_COUNT = 13


@dataclass(frozen=True, slots=True)
class Geometry:
    """Dimensions and attention schedule resolved from the checkpoint."""

    layers: int
    hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    declared: declaration.Declaration = field(repr=False, compare=False)

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple("sliding_attention" if kind == "sliding" else "full_attention"
                     for kind in self.declared.block_types)

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim


RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
    )
)


#: A tied output head is stored as a role on the embedding.
ALIAS_SPECS: tuple[LogicalAliasSpec, ...] = (
    LogicalAliasSpec("text/output_head", ("text/token_embedding",)),
)

#: The roles above, as names: what `build_stored_tensor_specs` leaves out.
ALIASED_OBJECT_NAMES = frozenset(spec.role_pattern for spec in ALIAS_SPECS)


def build_tensor_specs(geometry: Geometry) -> tuple[TensorSpec, ...]:
    """The complete ordered tensor inventory for one Gemma 3 text stack.

    The order is the declaration's: the embedding, then each layer's thirteen
    objects in block-schema order, then the final norm and the output head.
    """

    hidden = geometry.hidden
    specs: list[TensorSpec] = [
        # Stored unscaled. Gemma multiplies the looked-up row by sqrt(hidden)
        # before the first block, and the engine holds that factor as
        # `config.h::embedding_scale`; folding it in here would apply it twice.
        tensor_spec("text/token_embedding", (geometry.vocab, hidden), W8),
    ]

    for layer in range(geometry.layers):
        prefix = f"text/layers/{layer}/"
        specs.extend(
            (
                # The four sandwich norms, all zero-centred: the artifact holds
                # `w` and the runtime applies `1 + w`.
                tensor_spec(prefix + "input_norm", (hidden,), BF16),
                tensor_spec(prefix + "post_attention_norm", (hidden,), BF16),
                tensor_spec(prefix + "pre_feedforward_norm", (hidden,), BF16),
                tensor_spec(prefix + "post_feedforward_norm", (hidden,), BF16),
                # Separate, not fused: see the module docstring.
                tensor_spec(
                    prefix + "attention/query", (geometry.query_size, hidden), W8
                ),
                tensor_spec(prefix + "attention/key", (geometry.kv_size, hidden), W8),
                tensor_spec(prefix + "attention/value", (geometry.kv_size, hidden), W8),
                tensor_spec(prefix + "attention/query_norm", (geometry.head_dim,), BF16),
                tensor_spec(prefix + "attention/key_norm", (geometry.head_dim,), BF16),
                tensor_spec(
                    prefix + "attention/output", (hidden, geometry.query_size), W8
                ),
                tensor_spec(prefix + "mlp/gate", (geometry.intermediate, hidden), W8),
                tensor_spec(prefix + "mlp/up", (geometry.intermediate, hidden), W8),
                tensor_spec(prefix + "mlp/down", (hidden, geometry.intermediate), W8),
            )
        )

    specs.extend(
        (
            tensor_spec("text/final_norm", (hidden,), BF16),
            # Declared, and stored only when the checkpoint unties it; see
            # `ALIAS_SPECS` and `build_stored_tensor_specs` below.
            tensor_spec("text/output_head", (geometry.vocab, hidden), W8),
        )
    )
    return tuple(specs)


def build_stored_tensor_specs(
    geometry: Geometry,
    *,
    tied_output_head: bool = True,
) -> tuple[TensorSpec, ...]:
    """The objects the artifact actually writes, in the same order.

    Aliasing is a storage decision, and the declaration does not express it: the
    model has an `lm_head`, so `emit_inventory` derives a `text/output_head`
    whatever the checkpoint ties. What a *tied* checkpoint stores is one object
    fewer, because the head is the embedding table — see `ALIAS_SPECS`.
    """

    specs = build_tensor_specs(geometry)
    if not tied_output_head:
        return specs
    return tuple(spec for spec in specs if spec.name not in ALIASED_OBJECT_NAMES)


def build_object_specs(
    geometry: Geometry,
    *,
    tied_output_head: bool = True,
) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_stored_tensor_specs(
        geometry, tied_output_head=tied_output_head
    )


def active_specs(
    *,
    tied_output_head: bool,
    geometry: Geometry,
) -> tuple[tuple[TensorSpec, ...], tuple[StoredObjectSpec, ...]]:
    """Both lists for the checkpoint in hand, the way `qwen4exp` selects a variant.

    Which one a run uses is a property of the checkpoint (`tie_word_embeddings`),
    not of the target, so `convert.py` asks for it rather than reading the
    module-level tuples blind.
    """

    tensors = build_stored_tensor_specs(geometry, tied_output_head=tied_output_head)
    return tensors, RESOURCE_SPECS + tensors


LOGICAL_ROW_VIEW_SPECS: tuple[LogicalRowViewSpec, ...] = ()
