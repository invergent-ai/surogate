"""What an LFM2 serving artifact holds.

Written as a derivation rather than a list. The declaration already says which
objects an LFM2 layer has, what shape each one is and whether it is quantised --
that is what `ServeObject` is for -- so restating it here would create a second
place for the geometry to be wrong, which is the failure the declaration work
exists to remove. Everything below reads the declaration for one checkpoint's
config and maps its two storage classes onto the artifact's.

LFM2 alternates two kinds of layer. Attention layers hold a fused query/key/value
projection with its per-head norms and an output projection; conv layers hold a
short depthwise convolution and the projection either side of it. Both kinds hold
the same two norms and the same SwiGLU MLP, and which kind sits where is the
checkpoint's `full_attn_idxs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    RESOURCE_SPECS,
    TensorSpec,
    W8,
    tensor_spec,
)

ARCHITECTURE = "Lfm2ForCausalLM"
TARGET_KEY = "lfm2"
MODEL_ID = "lfm2"
#: The artifact's weight profile. Every quantised object is group-wise int8, the
#: same profile the dense targets carry; LFM2 declares no other export.
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text",)


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions an LFM2 checkpoint states about itself."""

    hidden: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    conv_kernel: int
    attention_layers: tuple[int, ...]

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkv_rows(self) -> int:
        return self.query_size + 2 * self.kv_size

    @property
    def mlp_gate_up_rows(self) -> int:
        return 2 * self.intermediate

    def is_attention(self, layer: int) -> bool:
        return layer in self.attention_layers


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    """The geometry a checkpoint's own config states.

    Read through the declaration rather than off the raw config: the FFN width is
    an adjustment of `block_ff_dim` rather than the number itself, and the layer
    schedule is a list of attention indices rather than a period. The declaration
    performs both, exactly as training does.
    """
    declared = declaration.declare(ARCHITECTURE, dict(config))
    resolved = declared.config
    schedule = declaration.block_types(resolved, declared.model)
    return Geometry(
        hidden=int(resolved["d_model"]),
        layers=int(resolved["n_layers"]),
        query_heads=int(resolved["num_query_heads"]),
        kv_heads=int(resolved["num_kv_heads"]),
        head_dim=int(resolved["head_size"]),
        intermediate=int(resolved["d_ff"]),
        vocab=int(resolved["vocab_size"]),
        conv_kernel=int(resolved["conv_kernel"]),
        attention_layers=tuple(i for i, kind in enumerate(schedule) if kind == "attention"),
    )


def declared_objects(config: Mapping[str, Any]) -> list[declaration.DeclaredObject]:
    """Every object the artifact stores, in declaration order."""
    declared = declaration.declare(ARCHITECTURE, dict(config))
    return list(declared.objects(capabilities=set(CAPABILITIES)))


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    """Declared objects as artifact specs.

    The declaration names a storage class, not a width: `quantised` is the target's
    choice, and this target quantises to group-wise int8 throughout. Norms, the
    convolution taps and the embedding stay BF16 -- a depthwise tap and a norm
    scale are a handful of numbers each, and quantising them buys nothing.
    """
    out: list[TensorSpec] = []
    for obj in objects:
        numeric = W8 if obj.format == "quantised" else obj.format.upper()
        if numeric not in DIRECT_FORMATS and numeric != W8:
            numeric = BF16
        out.append(tensor_spec(obj.name, tuple(obj.shape), numeric))
    return tuple(out)


__all__ = [
    "ARCHITECTURE",
    "BF16",
    "CAPABILITIES",
    "Geometry",
    "MODEL_ID",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "W8",
    "WEIGHTS_ID",
    "declared_objects",
    "geometry_from_config",
    "tensor_specs",
]
