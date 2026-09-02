"""Persistent-object contract for a `Gemma3ForCausalLM` serving artifact.

The object list is not invented here.  `surogate/dsl/models/gemma3.py` declares
it — `GEMMA3_MODEL_SERVE_OBJECTS`, `GEMMA3_OUTPUT_HEAD_SERVE_OBJECTS` and the
per-layer `_GEMMA3_SERVE_OBJECTS` on `surogate/dsl/blocks/gemma3.py` — and
`generate/emit_inventory.py` derives exactly the 237 tensors this module spells
out, in exactly this order.  Restating them here follows the Llama and Qwen3
targets, whose converters are also explicit; running

    python emit_inventory.py <checkpoint> --diff \
        surogate.serve.tools.convert.gemma3.inventory

from `serve/tools/generate` is what proves the restatement still agrees with the
declaration.

Read against the Llama inventory beside it, Gemma 3 differs in four ways, and
every one of them comes from the declaration rather than from a family habit:

* **Four norms per layer, not two.**  Gemma sandwiches each sub-block:
  `input_norm -> attention -> post_attention_norm`, then
  `pre_feedforward_norm -> mlp -> post_feedforward_norm`.  Llama and Qwen3 norm
  only the two inputs.
* **Q, K and V stay separate**, where Llama and Qwen3 fuse them into one
  `attention/query_key_value`.  The block schema says so and gives the reason:
  per-head QK norm and rope both want their operand contiguous, and in a fused
  `[q|k|v]` matrix the query rows of successive tokens are not adjacent.  The
  MLP's gate and up stay separate for the same reason the declaration keeps
  them separate — `fuse_gate_up=False` on its `MLPConfig`.
* **Q/K norm is present**, as in Qwen3 and unlike Llama, `head_dim` wide.  There
  is no V norm: that is Gemma 4's, and this checkpoint ships no such tensor.
* **Norms are stored zero-centred.**  Gemma's checkpoint holds `w` and the model
  applies `(1 + w)` (`transformers`' `Gemma3RMSNorm.forward`), which is what the
  declaration's `transform="unfold_unit_offset"` names.  The runtime re-applies
  the one itself, so the artifact must hold the *unfolded* `w` — which is what
  a safetensors checkpoint already stores, so these objects pass through
  untouched.  Only a GGUF source, which stores the folded `1 + w`, has to
  subtract; see `convert/gemma_embedding/sources.py`.

* **The tied head is not stored.**  Gemma 3 ties `lm_head` to the embedding and
  ships no `lm_head.weight`, so `text/output_head` is a logical role served by
  the stored `text/token_embedding` (`ALIAS_SPECS`), the way `mtp/token_embedding`
  is on `qwen3_5_0_8b`.  The declaration still names the head — aliasing is a
  storage decision and `ServeObject` does not express it — which is why
  `TENSOR_SPECS` (what the model has, and what `--diff` compares) and
  `STORED_TENSOR_SPECS` (what the artifact writes) are two lists here.

Row order convention, shared with the rest of the package: every 2-D object is
logical ``[rows = outputs, k = inputs]``.

Numeric profile: every matrix is ``W8G32_F16S`` in the ``row-split-k128-v1``
layout and every norm is BF16 — the profile Llama and Qwen3 use and the one the
shared kernels already bind.
"""

from __future__ import annotations

from dataclasses import dataclass

from surogate.serve.tools.convert.common.inventory import (
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
#: package's, string for string, the way `csrc/src/serve/api/targets/llama/
#: package.h` declares `model_id`/`target_key` for Llama. The Gemma 3 header is
#: `csrc/src/serve/targets/gemma3/impl/config.h`, whose namespace is
#: `sinfer::targets::gemma3_270m`; these are the strings that name it.
MODEL_ID = "gemma3-270m"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "gemma3"

#: Objects a Gemma 3 layer stores: four sandwich norms, three separate attention
#: projections, two per-head norms, the attention output, and three MLP
#: matrices. Thirteen, against Llama's six and Qwen3's eight.
LAYER_OBJECT_COUNT = 13


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions an artifact's object list depends on.

    These are read off `config.json` at convert time and checked against the
    registered target below, rather than being spelled out a second time: the
    same builder then serves a differently sized `Gemma3ForCausalLM` the day a
    second target header exists for one.

    `head_dim` is read, not derived. Gemma 3 decouples it from the hidden size —
    270M is 4 query heads of 256 against a hidden of 640, so
    `hidden // num_attention_heads` would give 160 and be wrong by a factor of
    1.6.
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


#: `google/gemma-3-270m-it`, the one registered size of this target. Matches
#: `csrc/src/serve/targets/gemma3/impl/config.h`, member for member.
GEMMA3_270M = Geometry(
    layers=18,
    hidden=640,
    intermediate=2048,
    vocab=262144,
    query_heads=4,
    kv_heads=1,
    head_dim=256,
)

GEOMETRY = GEMMA3_270M

LAYERS = GEOMETRY.layers
HIDDEN = GEOMETRY.hidden
INTERMEDIATE = GEOMETRY.intermediate
VOCAB = GEOMETRY.vocab
QUERY_HEADS = GEOMETRY.query_heads
KV_HEADS = GEOMETRY.kv_heads
HEAD_DIM = GEOMETRY.head_dim
QUERY_SIZE = GEOMETRY.query_size
KV_SIZE = GEOMETRY.kv_size

#: Every layer is attention — there is no linear-attention mixer anywhere in
#: Gemma 3 — which is the sense `config.h::is_full_attention` uses and the sense
#: the sibling inventories use for this name.
FULL_ATTENTION_LAYERS = tuple(range(LAYERS))

#: The *other* axis, and the one that is easy to confuse with the name above.
#: Gemma 3 alternates local against global attention. Nothing in the artifact
#: depends on the schedule — a windowed layer and a global one store identical
#: objects — but the converter checks the checkpoint's resolved schedule against
#: it, because the engine bakes the schedule in and a checkpoint that disagreed
#: would be served with the wrong masks and the wrong rope bases, in silence.
#:
#: `WINDOWED_ATTENTION` is the converter's copy of what the target header states
#: as data: `config.h::kWindowedAttention`, one bool per layer, true where the
#: layer is windowed, read through `config.h::is_windowed_attention(layer)`.
#: Gemma 3 *derives* that array from a period counted from the end — layer
#: `(layer + 1) % period == 0` is global, so the last layer is — but the period
#: is not the contract: newer `transformers` exports state the schedule as a
#: `layer_types` list and write no period at all, and both spellings have to land
#: on the same array. The period stays because it is how this release's schedule
#: is generated and because the checkpoint may spell it that way.
SLIDING_WINDOW = 512
SLIDING_WINDOW_PERIOD = 6
WINDOWED_ATTENTION: tuple[bool, ...] = tuple(
    (layer + 1) % SLIDING_WINDOW_PERIOD != 0 for layer in range(LAYERS)
)
GLOBAL_ATTENTION_LAYERS = tuple(
    layer for layer, windowed in enumerate(WINDOWED_ATTENTION) if not windowed
)


#: Gemma 3 270M is text-only, so the artifact carries the four text frontend
#: files and not the two image/video preprocessor configs the vision-capable
#: targets bind. Unlike the Qwen releases, this one publishes `chat_template.jinja`
#: as a file of its own; the converter still knows how to lift it out of
#: `tokenizer_config.json`, which is where the engine cross-checks it.
#:
#: The SentencePiece `tokenizer.model` beside them is not bound. The frontend now
#: hands SentencePiece checkpoints to the project tokenizer
#: (`csrc/src/serve/family/impl/frontend/tokenizer.cpp`), but it still reads the
#: scheme, the vocabulary and the added tokens out of `tokenizer.json` — exactly
#: as the Llama target does with its own SentencePiece release — and an object no
#: binder consumes is refused at load.
RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
    )
)


#: One alias, and it is the largest object in the run. `Gemma3ForCausalLM` ties
#: the head to the embedding — `gemma-3-270m-it` ships no `lm_head.weight` at all
#: — so `text/output_head` is a logical *role* served by the stored
#: `text/token_embedding`, exactly as `mtp/token_embedding` is on the
#: `qwen3_5_0_8b` target. Storing it instead would quantise 167.8M elements
#: twice, carry ~170 MB of byte-identical duplicate in the artifact, and hold two
#: ~168 MB tables on a device whose whole model is ~270 MB.
#:
#: Nothing else is aliased: no MTP head reuses the text embedding, and no object
#: is stored in an order a second consumer needs permuted.
#:
#: `csrc/src/serve/targets/gemma3/impl/load/bindings.cpp` is the other half — it
#: binds `text/token_embedding` once and fills both roles from that one handle.
#: An artifact that stored the head separately would be refused there ("artifact
#: object was not consumed by the selected target"), so the two halves have to
#: move together.
ALIAS_SPECS: tuple[LogicalAliasSpec, ...] = (
    LogicalAliasSpec("text/output_head", ("text/token_embedding",)),
)

#: The roles above, as names: what `build_stored_tensor_specs` leaves out.
ALIASED_OBJECT_NAMES = frozenset(spec.role_pattern for spec in ALIAS_SPECS)


def build_tensor_specs(geometry: Geometry = GEOMETRY) -> tuple[TensorSpec, ...]:
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
    geometry: Geometry = GEOMETRY,
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
    geometry: Geometry = GEOMETRY,
    *,
    tied_output_head: bool = True,
) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_stored_tensor_specs(
        geometry, tied_output_head=tied_output_head
    )


def active_specs(
    *,
    tied_output_head: bool,
    geometry: Geometry = GEOMETRY,
) -> tuple[tuple[TensorSpec, ...], tuple[StoredObjectSpec, ...]]:
    """Both lists for the checkpoint in hand, the way `qwen4exp` selects a variant.

    Which one a run uses is a property of the checkpoint (`tie_word_embeddings`),
    not of the target, so `convert.py` asks for it rather than reading the
    module-level tuples blind.
    """

    tensors = build_stored_tensor_specs(geometry, tied_output_head=tied_output_head)
    return tensors, RESOURCE_SPECS + tensors


#: Every object this model *has*, in declaration order — the list
#: `serve/tools/generate/emit_inventory.py --diff` compares against, and the one
#: `tests/test_serve_contract.py` would use. It is not the list the artifact
#: writes: see `STORED_TENSOR_SPECS`.
TEXT_CORE_TENSOR_SPECS = build_tensor_specs()
TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS

#: What a tied checkpoint — which is every published `Gemma3ForCausalLM`, and
#: `gemma-3-270m-it` in particular — actually stores.
STORED_TENSOR_SPECS = build_stored_tensor_specs()
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + STORED_TENSOR_SPECS

# The numeric profile of the artifact, so these count what is written rather than
# what is declared.
FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in STORED_TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in STORED_TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

#: No row views. On the Qwen and Llama targets this is what places a LoRA adapter
#: trained against `q_proj` on the right rows of a fused
#: `attention/query_key_value`, and what a verifier slices to compare one
#: converted object against the checkpoint tensor it came from. Gemma 3 fuses
#: nothing, so every logical projection already *is* an object and a view would
#: name the whole of one.
LOGICAL_ROW_VIEW_SPECS: tuple[LogicalRowViewSpec, ...] = ()

__all__ = [
    "ALIASED_OBJECT_NAMES",
    "ALIAS_SPECS",
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "FORMAT_COUNTS",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GEMMA3_270M",
    "GEOMETRY",
    "GLOBAL_ATTENTION_LAYERS",
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
    "SLIDING_WINDOW",
    "SLIDING_WINDOW_PERIOD",
    "STORED_TENSOR_SPECS",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VOCAB",
    "W8",
    "WEIGHTS_ID",
    "WINDOWED_ATTENTION",
    "active_specs",
    "build_object_specs",
    "build_stored_tensor_specs",
    "build_tensor_specs",
    "tensor_spec",
]
