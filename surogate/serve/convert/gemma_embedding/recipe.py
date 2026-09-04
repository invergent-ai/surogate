"""Where each artifact object comes from in the ``gemma-embedding`` GGUF.

The inventory is not restated here -- :mod:`inventory` derives it from the DSL
declaration. This module answers only the other half: given an object name, which
GGUF tensors build it and what has to happen to them.

Three kinds of work, and only the first is free:

* **Repack.** ``Q8_0`` and the artifact's ``W8G32_F16S`` are the same format
  (int8 codes, one binary16 scale per 32-group), so an object built by row
  algebra over Q8_0 sources moves across bit-exactly -- no dequantize, no
  requantize, no GPU. Every matrix qualifies: the declaration keeps Q, K and V
  separate, so each one is a plain copy of a GGUF tensor. 169 of 315 objects,
  and all but a rounding error of the bytes.

* **Unfold.** Gemma norms are zero-centred: HF stores ``w``, the GGUF stores the
  folded ``1 + w``, and the runtime re-applies the offset itself
  (``rmsnorm(..., unit_offset=true)``). So every norm loses a one on the way in.
  Verified against the HF checkpoint: ``cos(gguf - 1, hf) == 1.000000``.

* **Compose.** The two sentence-transformers Dense modules both declare an
  Identity activation, so ``768 -> 3072 -> 768`` is a single linear map and the
  artifact stores the product. This is the one object that must dequantize: a
  matrix product mixes ``k``, so it is not row algebra.

The GGUF stores ``ne = [k, n]`` with ``k`` fastest, which gguf-py hands back as
``[n, k]`` -- the logical orientation, and the artifact's. No transpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import inventory


@dataclass(frozen=True, slots=True)
class Source:
    """How one artifact object is built from GGUF tensors.

    ``op`` is the identity the declaration named (``ServeObject.transform``) or
    the implicit one for a plain copy. ``tensors`` are GGUF names, in application
    order for a composition.
    """

    tensors: tuple[str, ...]
    op: str = "copy"

    @property
    def repackable(self) -> bool:
        """Whether this object can move from Q8_0 without dequantizing.

        ``unfold`` touches values and ``compose_linear`` mixes columns, so
        neither qualifies; a plain copy does.
        """
        return self.op == "copy"


@dataclass(frozen=True, slots=True)
class TensorRecipe:
    """One artifact object, with its GGUF tensors named in full.

    A `Source` is a template -- one entry per kind of object, with the layer left off
    both ends. This is one entry per object the artifact actually holds, which is what
    a caller holding a checkpoint has to iterate.
    """

    object_name: str
    tensors: tuple[str, ...]
    source: Source

    @property
    def op(self) -> str:
        return self.source.op

    @property
    def repackable(self) -> bool:
        return self.source.repackable


#: Per-layer objects. Keys are the object names the declaration emits, minus the
#: ``text/layers/{layer}/`` prefix; values name GGUF tensors minus ``blk.{i}.``.
LAYER_SOURCES: dict[str, Source] = {
    # The four sandwich norms. GGUF names two of them differently from HF:
    # `ffn_norm` is the *pre*-feedforward norm, and `post_ffw_norm` the post one.
    "input_norm": Source(("attn_norm.weight",), "unfold"),
    "post_attention_norm": Source(("post_attention_norm.weight",), "unfold"),
    "pre_feedforward_norm": Source(("ffn_norm.weight",), "unfold"),
    "post_feedforward_norm": Source(("post_ffw_norm.weight",), "unfold"),
    # Separate, matching the GGUF's own layout. Nothing is interleaved -- Gemma 3
    # has no attention output gate -- so each is a plain copy and repacks exactly.
    "attention/query": Source(("attn_q.weight",)),
    "attention/key": Source(("attn_k.weight",)),
    "attention/value": Source(("attn_v.weight",)),
    "attention/query_norm": Source(("attn_q_norm.weight",), "unfold"),
    "attention/key_norm": Source(("attn_k_norm.weight",), "unfold"),
    "attention/output": Source(("attn_output.weight",)),
    "mlp/gate": Source(("ffn_gate.weight",)),
    "mlp/up": Source(("ffn_up.weight",)),
    "mlp/down": Source(("ffn_down.weight",)),
}

#: Objects outside the stack.
MODEL_SOURCES: dict[str, Source] = {
    "text/token_embedding": Source(("token_embd.weight",)),
    "text/final_norm": Source(("output_norm.weight",), "unfold"),
    # dense_3 @ dense_2: [768,3072] @ [3072,768] -> [768,768], applied in that
    # order because the checkpoint runs 2_Dense first.
    "text/embedding_head": Source(("dense_3.weight", "dense_2.weight"), "compose_linear"),
}

_TOKEN_EMBEDDING = "text/token_embedding"


def source_for(object_name: str) -> Source:
    """The GGUF source for one artifact object name."""

    if object_name.startswith("text/layers/"):
        rest = object_name.split("/", 3)[3]
        return LAYER_SOURCES[rest]
    return MODEL_SOURCES[object_name]


def gguf_names(object_name: str) -> tuple[str, ...]:
    """Fully-qualified GGUF tensor names backing one artifact object."""

    source = source_for(object_name)
    if not object_name.startswith("text/layers/"):
        return source.tensors
    layer = object_name.split("/")[2]
    return tuple(f"blk.{layer}.{name}" for name in source.tensors)


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


# The reader lives beside the recipes because that is where the GGUF repack planner looks
# for it (`serve/ingest.py::_gguf_geometry`). That planner does not reach this target --
# an encoder is ingested through `ENCODER_TARGETS`, which runs the converter directly and
# plans no repack -- but the converter itself needs the same reading, and one spelling of
# it is better than a second that only ingest can see.
def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions off the checkpoint's own config.

    `head_dim` is read, never derived. EmbeddingGemma is 3 query heads of 256 against a
    hidden of 768, so the `hidden // heads` shortcut a Llama converter can take would give
    256 by coincidence here and be wrong for the next size of this encoder.
    """

    head_dim = config.get("head_dim")
    if not head_dim:
        raise ValueError(
            "config declares no head_dim; Gemma 3 does not derive it from "
            "hidden_size // num_attention_heads and the value cannot be guessed"
        )
    return inventory.Geometry(
        layers=int(config["num_hidden_layers"]),
        hidden=int(config["hidden_size"]),
        intermediate=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        query_heads=int(config["num_attention_heads"]),
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(head_dim),
    )


# ---------------------------------------------------------------------------
# source recipe
# ---------------------------------------------------------------------------


def build_recipes(
    geometry: inventory.Geometry = inventory.GEOMETRY,
) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from, for a checkpoint of this geometry.

    Only the layer count varies. The stack is uniform -- EmbeddingGemma alternates
    sliding and full attention but stores the same thirteen tensors either way -- so
    the templates above repeat unchanged, and nothing else about the checkpoint
    reaches the source names.
    """
    # The embedding table before the stack and the rest after it, so this list and the
    # declaration's read side by side.
    names = [_TOKEN_EMBEDDING]
    for layer in range(geometry.layers):
        names.extend(f"text/layers/{layer}/{suffix}" for suffix in LAYER_SOURCES)
    names.extend(name for name in MODEL_SOURCES if name != _TOKEN_EMBEDDING)
    return tuple(
        TensorRecipe(name, gguf_names(name), source_for(name)) for name in names
    )


#: The registered size's recipes, for callers that have no checkpoint in hand.
RECIPE_SPECS = build_recipes()
RECIPES_BY_NAME = {item.object_name: item for item in RECIPE_SPECS}


def validate_recipe_coverage(
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
    geometry: inventory.Geometry = inventory.GEOMETRY,
) -> None:
    """Every declared object has a source, and every source names a declared object.

    Both directions, because the two lists are written by different hands: the declaration
    generates its side, `LAYER_SOURCES` and `MODEL_SOURCES` are typed by hand. An object
    added to the declaration and not here fails at write time with a KeyError; one removed
    from the declaration and left here would be a source silently read for nothing.
    """

    declared = {obj["name"] for obj in inventory.declared_objects(geometry)}
    named = {item.object_name for item in recipes}
    if declared != named:
        raise ValueError(
            "gemma_embedding recipe and declaration disagree: "
            f"declared but unsourced {sorted(declared - named)}, "
            f"sourced but undeclared {sorted(named - declared)}"
        )


validate_recipe_coverage()
