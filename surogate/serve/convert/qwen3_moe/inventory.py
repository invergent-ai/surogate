"""What a Qwen3-MoE serving artifact holds.

Written as a derivation, not a list. The declaration already says which objects a layer has,
what shape each one is and whether it is quantised, so restating it here would create a second
place for the geometry to be wrong.

Every layer is the same: classic Qwen3 attention -- one fused query/key/value parent with no
output-gate rows, per-head q and k norms, an output projection -- over a routed mixture. The
mixture has no always-on expert, so the router is the experts and nothing else; the hybrid MoE
family fuses a shared expert's gate onto the router as an extra row, and this one has no such
row to fuse.
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

ARCHITECTURE = "Qwen3MoeForCausalLM"
TARGET_KEY = "qwen3_moe"
MODEL_ID = "qwen3-moe"
#: Every quantised object is group-wise int8, the profile the dense targets carry. A GGUF
#: source keeps its own K-quants instead; that is the repack path, not a second profile.
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text",)


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions a Qwen3-MoE checkpoint states about itself."""

    hidden: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    experts: int
    experts_per_token: int

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
    def routed_gate_up_rows(self) -> int:
        return self.experts * 2 * self.intermediate

    @property
    def routed_down_rows(self) -> int:
        return self.experts * self.hidden


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    """The geometry the checkpoint's own config states, read through the declaration so the
    resolution rules are training's rather than a second copy of them."""
    declared = declaration.declare(ARCHITECTURE, dict(config))
    resolved = declared.config
    shared = int(resolved.get("shared_expert_intermediate", 0) or 0)
    if shared:
        raise ValueError(
            "this checkpoint declares a shared expert of width "
            f"{shared}; the qwen3_moe target serves the routed-only mixture "
            "(Qwen3-30B-A3B and its siblings), whose router has one row per expert and no "
            "shared gate. Serving it here would drop that expert's contribution silently."
        )
    return Geometry(
        hidden=int(resolved["d_model"]),
        layers=int(resolved["n_layers"]),
        query_heads=int(resolved["num_query_heads"]),
        kv_heads=int(resolved["num_kv_heads"]),
        head_dim=int(resolved["head_size"]),
        intermediate=int(resolved["d_ff"]),
        vocab=int(resolved["vocab_size"]),
        experts=int(resolved["num_experts"]),
        experts_per_token=int(resolved["num_experts_per_tok"]),
    )


def declared_objects(config: Mapping[str, Any]) -> list[declaration.DeclaredObject]:
    """Every object the artifact stores, in declaration order."""
    declared = declaration.declare(ARCHITECTURE, dict(config))
    return list(declared.objects(capabilities=set(CAPABILITIES)))


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    """Declared objects as artifact specs.

    The declaration names a storage class, not a width: `quantised` is the target's choice, and
    this target quantises to group-wise int8. Norms, the router and the embedding stay BF16 --
    a router is one row per expert and a norm a handful of numbers, and quantising either buys
    nothing while costing the router's selection its precision.
    """
    out: list[TensorSpec] = []
    for obj in objects:
        numeric = W8 if obj.format == "quantised" else obj.format.upper()
        if numeric not in DIRECT_FORMATS and numeric != W8:
            numeric = BF16
        out.append(tensor_spec(obj.name, tuple(obj.shape), numeric))
    return tuple(out)


def build_tensor_specs(config: Mapping[str, Any]) -> tuple[TensorSpec, ...]:
    """The artifact's tensor specs for one checkpoint's config.

    The GGUF repack planner's spelling of `tensor_specs(declared_objects(config))`: it holds a
    config and wants the specs, and every converter it plans against offers this name.
    """
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
