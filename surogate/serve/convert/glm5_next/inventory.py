"""What a GLM-5.3-Flash serving artifact holds.

Written as a derivation. The declaration already says which objects each of the four block kinds
has, what shape each one is and whether it is quantised, so restating it here would create a
second place for the geometry to be wrong.

Four kinds, because the two axes are independent: a layer runs either Kimi Delta Attention or
multi-head latent attention, and its feed-forward is either dense or the mixture. The released
46-layer checkpoint uses three of the four -- three leading dense KDA layers, then KDA and MLA
layers over the mixture.
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

ARCHITECTURE = "Glm5NextForConditionalGeneration"
TARGET_KEY = "glm5_next"
MODEL_ID = "glm5-next"
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text",)


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions a GLM-5.3 checkpoint states about itself."""

    hidden: int
    layers: int
    query_heads: int
    head_dim: int
    dense_intermediate: int
    expert_intermediate: int
    vocab: int
    experts: int
    experts_per_token: int
    shared_intermediate: int
    routed_scale: float
    hc_streams: int
    kda_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    v_head_dim: int
    attention_layers: tuple[int, ...]
    dense_layers: tuple[int, ...]

    @property
    def residual(self) -> int:
        """Hyper-connections carry `hc_streams` copies of the hidden state."""
        return self.hc_streams * self.hidden

    @property
    def kda_dim(self) -> int:
        return self.kda_heads * self.kda_head_dim

    @property
    def query_dim(self) -> int:
        return self.query_heads * self.qk_head_dim

    @property
    def value_dim(self) -> int:
        return self.query_heads * self.v_head_dim


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    """The geometry the checkpoint states, read through the declaration so the resolution rules
    are training's rather than a second copy of them."""
    declared = declaration.declare(ARCHITECTURE, dict(config))
    resolved = declared.config
    schedule = declaration.block_types(resolved, declared.model)
    if resolved.get("qk_rope_head_dim", 0):
        raise ValueError(
            "this checkpoint declares a rotary split on its latent attention; every released "
            "GLM-5.3 is the NoPE variant and the served attention applies no rotary at all"
        )
    return Geometry(
        hidden=int(resolved["d_model"]),
        layers=int(resolved["n_layers"]),
        query_heads=int(resolved["num_attention_heads"]),
        head_dim=int(resolved["qk_nope_head_dim"]),
        dense_intermediate=int(resolved["d_ff"]),
        expert_intermediate=int(resolved["moe_d_ff"]),
        vocab=int(resolved["vocab_size"]),
        experts=int(resolved["num_experts"]),
        experts_per_token=int(resolved["num_experts_per_tok"]),
        shared_intermediate=int(resolved.get("shared_expert_intermediate", 0) or 0),
        routed_scale=float(resolved.get("routed_scaling_factor", 1.0) or 1.0),
        hc_streams=int(resolved.get("hc_mult") or resolved.get("hc_count") or 0),
        kda_heads=int(resolved["linear_num_heads"]),
        kda_head_dim=int(resolved["linear_head_dim"]),
        kda_conv_kernel=int(resolved["linear_conv_kernel_dim"]),
        q_lora_rank=int(resolved["q_lora_rank"]),
        kv_lora_rank=int(resolved["kv_lora_rank"]),
        qk_head_dim=int(resolved["qk_nope_head_dim"]),
        v_head_dim=int(resolved["v_head_dim"]),
        attention_layers=tuple(i for i, kind in enumerate(schedule) if kind.startswith("mla")),
        dense_layers=tuple(i for i, kind in enumerate(schedule) if not kind.endswith("_moe")),
    )


def declared_objects(config: Mapping[str, Any]) -> list[declaration.DeclaredObject]:
    declared = declaration.declare(ARCHITECTURE, dict(config))
    return list(declared.objects(capabilities=set(CAPABILITIES)))


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    """Declared objects as artifact specs.

    `quantised` is a storage class, not a width, and this target's choice is group-wise int8.
    The norms, the router and its bias, the convolution taps and the two scalar-per-head gates
    stay in the precision the declaration names: a router that picks eight of 288 experts and a
    decay that is exponentiated are both places where a coarse width costs more than it saves.
    """
    out: list[TensorSpec] = []
    for obj in objects:
        numeric = W8 if obj.format == "quantised" else obj.format.upper()
        if numeric not in DIRECT_FORMATS and numeric != W8:
            numeric = BF16
        out.append(tensor_spec(obj.name, tuple(obj.shape), numeric))
    return tuple(out)


def build_tensor_specs(config: Mapping[str, Any]) -> tuple[TensorSpec, ...]:
    """The planner's spelling of `tensor_specs(declared_objects(config))`."""
    return tensor_specs(declared_objects(config))


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
    "build_tensor_specs",
    "declared_objects",
    "geometry_from_config",
    "tensor_specs",
]
