"""What a Gemma 4 *mixture* serving artifact holds.

Written as a derivation, not a list, exactly as the dense target is:
`surogate/dsl/blocks/gemma4.py` says which objects a mixture layer has and what shape
each one is, and `surogate/dsl/models/gemma4.py` says which block runs where.

This target is `google/gemma-4-26B-A4B` and its instruction tune: 30 layers, hidden
2,816, and on **every** layer a 2,112-wide dense feed-forward running *beside* 128
routed experts of 704. It shares the dense target's attention exactly -- windowed
layers at 8 key/value heads of 256, global layers at 2 of 512 with no value
projection under `attention_k_eq_v` -- and differs entirely in the feed-forward:

* the dense branch and the routed branch consume the *same* input, the post-attention
  residual, and their outputs are summed before the block's own
  `post_feedforward_layernorm`. Three norms a dense layer does not have sit in that
  sum: one over the dense branch's output, and one on each side of the routed one.
* the router is unlike any other here. It normalises with **no weight at all**, scales
  by a learned per-channel vector times `hidden ** -0.5`, softmaxes in fp32, takes the
  top 8, renormalises them, and multiplies by a learned **per-expert** scale. Both
  scales are stored; neither is optional, and neither has an analogue in the other
  routed targets.

The E-series and the dense sizes are different architectures with their own targets.
`architecture_of` refuses a checkpoint that does not set `enable_moe_block`, because
the three share `model_type` and a mixture derived from dense blocks would inventory
an `mlp/gate` for a layer that also holds 128 experts -- an artifact that is wrong
rather than absent.
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

#: The four frontend files a text-only artifact carries; see the dense target's note.
RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
    )
)

#: The architectures this converter serves -- the same three names the dense target
#: accepts, because the checkpoints spell the wrapper differently at different sizes.
#: The mixture is told from them by `enable_moe_block`, not by the name.
ARCHITECTURES = (
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)

TARGET_KEY = "gemma4_moe"
MODEL_ID = "gemma4_moe"
#: Group-wise int8 for every quantised object, the profile the other routed targets
#: carry and the one the shared expert-bank kernels bind.
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text",)


def is_mixture(config: Mapping[str, Any]) -> bool:
    """Whether this checkpoint routes experts, as it states."""
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return bool(text.get("enable_moe_block")) and int(text.get("num_experts") or 0) > 0


def architecture_of(config: Mapping[str, Any]) -> str:
    """The architecture a checkpoint declares, refused unless this target serves it.

    Two gates, not one. The name is checked because a Gemma 3 config would otherwise
    reach the declaration, and `enable_moe_block` because the dense 12B, the 31B and the
    whole E-series declare exactly the same names as the 26B-A4B does. They are three
    architectures behind one spelling and the config is the only thing that tells them
    apart.
    """
    declared = config.get("architectures") or ()
    for name in declared:
        if name in ARCHITECTURES:
            if not is_mixture(config):
                raise ValueError(
                    "the gemma4_moe target serves the routed Gemma 4 (26B-A4B); this "
                    "checkpoint sets no enable_moe_block, so it is a dense or E-series "
                    "model and belongs to the gemma4 or gemma4_e target"
                )
            return name
    raise ValueError(
        f"the gemma4_moe target serves {', '.join(ARCHITECTURES)}; this checkpoint declares "
        f"{list(declared) or 'nothing'}"
    )


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions a Gemma 4 mixture checkpoint states about itself.

    The dense target's geometry with the mixture's four numbers added. `intermediate`
    stays the *dense* feed-forward's width and `expert_intermediate` the routed one;
    they are 2,112 and 704 here and neither derives the other, which is why both are
    members.
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
    partial_rotary_factor: float
    k_eq_v: bool
    final_logit_softcapping: float
    rms_epsilon: float
    #: The mixture.
    experts: int
    experts_per_token: int
    expert_intermediate: int

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
        """`sqrt(hidden)` rounded to bf16, as the reference rounds it before multiplying.

        See the dense target's note: unrounded it is a systematic scale error on every
        token at layer 0, and a target serving one size could compile it while this one
        cannot -- the artifact states it.
        """
        import struct

        bits = struct.unpack("<I", struct.pack("<f", float(self.hidden) ** 0.5))[0]
        bits += 0x7FFF + ((bits >> 16) & 1)
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]

    @property
    def partial_rotary_angles(self) -> int:
        """How many of a global head's angle pairs carry a non-zero frequency."""
        return int(self.partial_rotary_factor * self.global_head_dim // 2)

    def is_windowed(self, layer: int) -> bool:
        return self.windowed[layer]

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(i for i, w in enumerate(self.windowed) if not w)


def windowed_blocks(model: Any, schedule: Sequence[str]) -> tuple[bool, ...]:
    """Which layers attend through the window, by the declaration's own vocabulary.

    Read from `_serve_windowed_blocks_` rather than tested against the string "sliding":
    a mixture layer's block is named `moe_sliding`, and a test on the one name would call
    every one of them global -- sizing their caches for the wrong head, roping them at the
    wrong base, and masking nothing.
    """
    names = set(getattr(model, "_serve_windowed_blocks_", ()) or ())
    if not names:
        raise ValueError("the declaration names no windowed block types")
    return tuple(str(block) in names for block in schedule)


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    """The geometry a checkpoint's own config states, read through the declaration."""
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
        experts=int(resolved["num_experts"]),
        experts_per_token=int(resolved["top_k_experts"]),
        expert_intermediate=int(resolved["moe_d_ff"]),
    )


#: One alias, and it is the largest object in the run; see the dense target's note.
ALIAS_SPECS: tuple[LogicalAliasSpec, ...] = (
    LogicalAliasSpec("text/output_head", ("text/token_embedding",)),
)

#: The roles above, as names: what a tied checkpoint leaves out of its stored objects.
ALIASED_OBJECT_NAMES = frozenset(spec.role_pattern for spec in ALIAS_SPECS)


def declared_objects(config: Mapping[str, Any]) -> list[declaration.DeclaredObject]:
    """Every object the model *has*, in declaration order."""
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

    The declaration names a storage class, not a width; this target quantises to
    group-wise int8 throughout, the routed experts included. The router's own three
    tensors stay BF16 -- they are 128 rows, a channel vector and 128 scalars, and the
    routing decision is the one place in the layer where a quantisation error changes
    *which* experts run rather than by how much.
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
    "is_mixture",
    "stored_objects",
    "tensor_specs",
]
