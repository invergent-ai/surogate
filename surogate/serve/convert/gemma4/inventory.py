"""What a dense Gemma 4 serving artifact holds.

Written as a derivation, not a list. `surogate/dsl/blocks/gemma4.py` already says
which objects a layer has, what shape each one is and whether it is quantised, and
`surogate/dsl/models/gemma4.py` says which block runs where; restating any of that
here would create a second place for the geometry to be wrong. Everything below
reads the declaration for one checkpoint's config and maps its storage classes onto
the artifact's.

This target is the **dense k_eq_v pair**: `google/gemma-4-12B` (48 layers, hidden
3840) and `google/gemma-4-31B-it` (60 layers, hidden 5376). The E-series and the
26B-A4B mixture are different architectures with their own targets; the declaration
refuses them by name rather than letting them derive a plausible-looking artifact
from the wrong blocks.

**Two attention geometries in one model, which is what makes Gemma 4 unlike every
other target here.** A windowed layer attends with `head_dim` 256 heads and its own
key *and* value projections. A global layer attends with `global_head_dim` 512
heads, its own key/value head count, and -- under `attention_k_eq_v`, which both
these checkpoints set -- no value projection at all: the value is the key
projection's raw output, RMS-normalised without a weight. The declaration expresses
that last part as a shape that resolves to zero, so the object simply is not in the
inventory, which is exactly how the checkpoint stores it: the 12B ships `v_proj` on
40 of its 48 layers, and the 8 without are the global ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    LogicalAliasSpec,
    ResourceSpec,
    TensorSpec,
    W8,
    tensor_spec,
)

#: The four frontend files a text-only artifact carries.
#:
#: Not the shared `RESOURCE_SPECS`, which names six: the two it adds are the image and video
#: preprocessor configs, and this target binds the text stack only. The engine refuses an
#: artifact carrying an object no binder consumes, so naming them here would make every
#: conversion produce an artifact this target cannot load.
RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
    )
)

#: The architectures this converter serves. All three compile the same declaration;
#: they differ in where the checkpoint keeps the text stack and whether a `text_config`
#: nests the dimensions, which the declaration already handles.
ARCHITECTURES = (
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)

TARGET_KEY = "gemma4"
MODEL_ID = "gemma4"
#: The artifact's weight profile: every quantised object is group-wise int8, the
#: profile the other dense targets carry and the one the shared kernels bind.
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text",)


def architecture_of(config: Mapping[str, Any]) -> str:
    """The architecture a checkpoint declares, refused unless this target serves it.

    Read from the checkpoint rather than fixed here: the 12B is a
    `Gemma4UnifiedForConditionalGeneration` and the 31B a
    `Gemma4ForConditionalGeneration`, and they are the same text architecture.
    """
    declared = config.get("architectures") or ()
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    for name in declared:
        if name in ARCHITECTURES:
            # The 26B-A4B declares exactly these names. Only its config tells it apart, and
            # without this it would derive a plausible-looking artifact from dense blocks --
            # an `mlp/gate` for a layer whose feed-forward is a dense one *beside* 128 routed
            # experts, and no experts at all.
            if text.get("enable_moe_block"):
                raise ValueError(
                    "the gemma4 target serves the dense sizes; this checkpoint sets "
                    "enable_moe_block and belongs to the gemma4_moe target"
                )
            return name
    raise ValueError(
        f"the gemma4 target serves {', '.join(ARCHITECTURES)}; this checkpoint declares "
        f"{list(declared) or 'nothing'}"
    )


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions a dense Gemma 4 checkpoint states about itself.

    Both attention geometries are members, because both are primary: neither is
    derivable from the other and a target that carried only one would size every
    global layer's cache wrong.
    """

    hidden: int
    layers: int
    intermediate: int
    vocab: int
    query_heads: int
    #: The windowed layers' geometry.
    kv_heads: int
    head_dim: int
    #: The global layers' geometry.
    global_kv_heads: int
    global_head_dim: int
    #: True where the layer attends through the window, one entry per layer.
    windowed: tuple[bool, ...]
    sliding_window: int
    rope_theta: float
    sliding_rope_theta: float
    #: The proportion of a global head RoPE actually rotates. The rest of the head is
    #: rotated by a zero frequency, which is the identity -- see `partial_rotary_angles`.
    partial_rotary_factor: float
    #: The value projection is absent from the global layers when this is set.
    k_eq_v: bool
    final_logit_softcapping: float
    rms_epsilon: float

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def global_query_size(self) -> int:
        return self.query_heads * self.global_head_dim

    @property
    def global_kv_size(self) -> int:
        return self.global_kv_heads * self.global_head_dim

    @property
    def embedding_scale(self) -> float:
        """What the embedding lookup is multiplied by before the first block.

        `sqrt(hidden)` **rounded to bf16**, because the reference holds it as a bf16 buffer
        and rounds before multiplying (`Gemma4UnifiedTextScaledWordEmbedding` casts
        `embed_scale` to the weight dtype). Unrounded it is 61.9677 where the model uses
        62.0 -- a systematic 0.05 % scale error on every token at layer 0.
        """
        import struct

        bits = struct.unpack("<I", struct.pack("<f", float(self.hidden) ** 0.5))[0]
        bits += 0x7FFF + ((bits >> 16) & 1)
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]

    @property
    def partial_rotary_angles(self) -> int:
        """How many of a global head's angle pairs carry a non-zero frequency.

        `int(factor * head_dim // 2)`, which is `_compute_proportional_rope_parameters`
        in `transformers/modeling_rope_utils.py`. The remaining `head_dim // 2 - this`
        pairs are zero, so the head rotates over its whole width with an identity tail
        rather than over a contiguous prefix -- the rotated and unrotated channels
        interleave at stride `head_dim // 2`.
        """
        return int(self.partial_rotary_factor * self.global_head_dim // 2)

    def is_windowed(self, layer: int) -> bool:
        return self.windowed[layer]

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(i for i, w in enumerate(self.windowed) if not w)


def windowed_blocks(model: Any, schedule: Sequence[str]) -> tuple[bool, ...]:
    """Which layers attend through the window, by the declaration's own vocabulary.

    Read from `_serve_windowed_blocks_` rather than tested against the string "sliding": the
    E-series names its windowed shared-KV layers `shared_kv_sliding`, and a test on the one
    name called every one of them global -- which sizes their caches for the wrong head, ropes
    them at the wrong base, and masks nothing.
    """
    names = set(getattr(model, "_serve_windowed_blocks_", ()) or ())
    if not names:
        raise ValueError("the declaration names no windowed block types")
    return tuple(str(block) in names for block in schedule)


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    """The geometry a checkpoint's own config states.

    Read through the declaration rather than off the raw config: the layer schedule
    is a `layer_types` list that the declaration normalises (it forces the last layer
    global, as `transformers` does), and the nested `text_config` spellings resolve
    there too. The declaration performs both, exactly as training does.
    """
    architecture = architecture_of(config)
    declared = declaration.declare(architecture, dict(config))
    resolved = declared.config
    schedule = declaration.block_types(resolved, declared.model)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    rope = text.get("rope_parameters") or {}
    full_rope = rope.get("full_attention") or {}
    sliding_rope = rope.get("sliding_attention") or {}
    return Geometry(
        hidden=int(resolved["d_model"]),
        layers=int(resolved["n_layers"]),
        intermediate=int(resolved["d_ff"]),
        vocab=int(resolved["vocab_size"]),
        query_heads=int(resolved["num_query_heads"]),
        kv_heads=int(resolved["num_kv_heads"]),
        head_dim=int(resolved["head_size"]),
        global_kv_heads=int(resolved.get("global_num_kv_heads") or resolved["num_kv_heads"]),
        global_head_dim=int(resolved.get("global_head_dim") or resolved["head_size"]),
        windowed=windowed_blocks(declared.model, schedule),
        sliding_window=int(resolved["sliding_window"]),
        rope_theta=float(full_rope.get("rope_theta", resolved.get("full_rope_theta", 1.0e6))),
        sliding_rope_theta=float(
            sliding_rope.get("rope_theta", resolved.get("sliding_rope_theta", 1.0e4))
        ),
        partial_rotary_factor=float(
            full_rope.get("partial_rotary_factor",
                          resolved.get("full_partial_rotary_factor", 1.0))
        ),
        k_eq_v=bool(resolved.get("k_eq_v")),
        final_logit_softcapping=float(resolved.get("final_logit_softcapping") or 0.0),
        rms_epsilon=float(resolved["eps"]),
    )


#: One alias, and it is the largest object in the run. Every published Gemma 4 ties its
#: head to its embedding, so `text/output_head` is a logical *role* served by the stored
#: `text/token_embedding` rather than an object of its own. Storing it instead would
#: quantise 1.0G elements twice and carry a byte-identical ~1 GB duplicate in a ~12 GB
#: artifact.
#:
#: `csrc/src/serve/targets/gemma4/impl/load/bindings.cpp` is the other half: it binds the
#: embedding once and fills both roles from that one handle. An artifact that stored the
#: head separately is refused there, so the two halves move together.
ALIAS_SPECS: tuple[LogicalAliasSpec, ...] = (
    LogicalAliasSpec("text/output_head", ("text/token_embedding",)),
)

#: The roles above, as names: what a tied checkpoint leaves out of its stored objects.
ALIASED_OBJECT_NAMES = frozenset(spec.role_pattern for spec in ALIAS_SPECS)


def declared_objects(config: Mapping[str, Any]) -> list[declaration.DeclaredObject]:
    """Every object the model *has*, in declaration order.

    Not what the artifact writes: aliasing is a storage decision and `ServeObject` does
    not express one, so the declaration derives a `text/output_head` whichever way the
    checkpoint ties. `stored_objects` is the list that reaches the writer.
    """
    declared = declaration.declare(architecture_of(config), dict(config))
    return list(declared.objects(capabilities=set(CAPABILITIES)))


def stored_objects(
    config: Mapping[str, Any],
    *,
    tied_output_head: bool,
) -> list[declaration.DeclaredObject]:
    """The objects the artifact actually writes, in the same order."""
    objects = declared_objects(config)
    if not tied_output_head:
        return objects
    return [obj for obj in objects if obj.name not in ALIASED_OBJECT_NAMES]


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    """Declared objects as artifact specs.

    The declaration names a storage class, not a width: `quantised` is the target's
    choice, and this target quantises to group-wise int8 throughout. Norms, the
    embedding table's companions and `layer_scalar` stay BF16 -- which is also how
    the checkpoint stores that scalar, a single BF16 element per layer.
    """
    out: list[TensorSpec] = []
    for obj in objects:
        numeric = W8 if obj.format == "quantised" else obj.format.upper()
        if numeric not in DIRECT_FORMATS and numeric != W8:
            numeric = BF16
        out.append(tensor_spec(obj.name, tuple(obj.shape), numeric))
    return tuple(out)


__all__ = [
    "ALIASED_OBJECT_NAMES",
    "ALIAS_SPECS",
    "ARCHITECTURES",
    "BF16",
    "CAPABILITIES",
    "Geometry",
    "MODEL_ID",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "W8",
    "WEIGHTS_ID",
    "architecture_of",
    "declared_objects",
    "geometry_from_config",
    "stored_objects",
    "tensor_specs",
]
