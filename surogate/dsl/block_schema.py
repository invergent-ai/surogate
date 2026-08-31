"""Block schema declarations for Phase 4 storage and distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DistributionKind = Literal["replicated", "sharded_dim", "expert_parallel", "router_replicated"]
Residency = Literal["auto", "gpu", "cpu_pinned_stream", "cpu_pageable", "nvme_offload"]
SlotKind = Literal["activation", "param", "scratch", "param_grad", "activation_grad"]
Lifetime = Literal["op", "layer", "block", "model", "persistent"]
RoutingKind = Literal["none", "topk_softmax", "topk_sigmoid", "expert_choice"]
#: Numeric format a serving artifact stores a weight in. The declaration already
#: carries format policy — `quantizable=False` on router and shared-expert weights
#: is exactly this decision — so naming the serving format here keeps one source of
#: truth rather than restating the geometry in a converter.
#: `quantised` means the export profile chooses the width — the 35B stores routed
#: experts Q4, their down projections Q5 and the output head Q6 where the 0.8B
#: stores all three W8. The declaration fixes only what the *model* fixes: a norm
#: is never quantised, a weight may be. This is the same distinction `quantizable`
#: already draws on parameters.
ServeFormat = Literal["quantised", "w8", "bf16", "fp32", "i32", "raw"]


@dataclass(frozen=True)
class ServeObject:
    """One object as a serving artifact stores it.

    A serving artifact does not store a model the way training holds it: weights
    are quantised, laid out for the kernels that read them, and fused — a serve
    target binds one `attention/query_key_gate_value` where the declaration has
    four projections. Those are real decisions, but they are decisions *about this
    model*, which is why they belong beside the parameters they describe rather
    than in a converter that restates the geometry to express them.

    `components` names the declared parameters this object is built from, in row
    order, which is what lets an adapter trained on one logical projection be
    placed on the right rows of a fused tensor. `transform` names the repacking
    the converter applies when the composition is not a plain concatenation — the
    algorithm stays in the converter; only its identity is declared here.
    """

    name: str
    format: ServeFormat = "bf16"
    shape: tuple[str | int, ...] = ()
    components: tuple[str, ...] = ()
    transform: str | None = None
    residency: Residency = "gpu"
    #: Layers this object exists on: every layer, or only the ones running this
    #: mixer. `None` means "wherever the block it belongs to runs".
    scope: Literal["block", "model"] = "block"
    #: What a target must implement to consume this object; see `ServeSection`.
    capability: str = "text"


@dataclass(frozen=True)
class ServeSection:
    """A sub-stack a serving artifact stores under its own prefix.

    A served checkpoint is not only the text stack: it carries a speculative
    draft head, and on some targets a vision tower and a DFlash stack. Those are
    model components, not policy, and the declaration is the single source of
    truth for what an artifact contains — so they are declared here rather than
    left for a converter to know about.

    `repeat` names the config key holding the sub-stack's layer count; `1` means
    the section is not indexed and its objects sit directly under the prefix.
    """

    prefix: str
    objects: tuple["ServeObject", ...] = ()
    repeat: str | int = 1
    #: What a target must implement to consume this section. A target that does
    #: not is not merely uninterested — the engine refuses to load an artifact
    #: holding objects no binder consumes — so the section is declared and simply
    #: not exported for it.
    capability: str = "text"


@dataclass(frozen=True)
class StreamingHint:
    prefetch_distance: int = 0
    eviction_policy: str = "after_use"
    sticky: bool = False


@dataclass(frozen=True)
class DistributionDecl:
    kind: DistributionKind = "replicated"
    shard_dim: int | None = None
    mode: str | None = None
    num_shards: int | str | None = None
    experts_per_rank: int | str | None = None
    global_experts: int | str | None = None

    @classmethod
    def replicated(cls) -> "DistributionDecl":
        return cls(kind="replicated")

    @classmethod
    def router_replicated(cls) -> "DistributionDecl":
        return cls(kind="router_replicated")

    @classmethod
    def sharded_dim(cls, *, dim: int, mode: str, num_shards: int | str | None = None) -> "DistributionDecl":
        return cls(kind="sharded_dim", shard_dim=dim, mode=mode, num_shards=num_shards)

    @classmethod
    def expert_parallel(
        cls, *, experts_per_rank: int | str | None = "auto", global_experts: int | str | None = None
    ) -> "DistributionDecl":
        return cls(kind="expert_parallel", experts_per_rank=experts_per_rank, global_experts=global_experts)


@dataclass(frozen=True)
class SlotDecl:
    name: str
    kind: SlotKind = "activation"
    shape: tuple[str | int, ...] = ()
    lifetime: Lifetime = "layer"
    dtype: str | None = None
    residency: Residency = "gpu"
    distribution: DistributionDecl = field(default_factory=DistributionDecl.replicated)
    save_for_backward: bool = False
    grouped: bool = False
    streaming_hint: StreamingHint | None = None


@dataclass(frozen=True)
class RoutingSchema:
    kind: RoutingKind = "none"
    topk: int | str | None = None
    norm_topk_prob: bool | str | None = None
    scoring_bias: bool = False
    shared_experts: int | str = 0


@dataclass(frozen=True)
class EPTopology:
    ep_size_param: str = "ep_size"
    weight_transfer_eligible: bool = False


@dataclass(frozen=True)
class BlockSchema:
    slots: tuple[SlotDecl, ...] = ()
    routing: RoutingSchema | None = None
    ep_topology: EPTopology | None = None
    #: How a serving artifact stores this block's weights. Empty for blocks no
    #: serve target covers; a serve target's object inventory is emitted from
    #: these rather than restating the geometry in a converter.
    serve_objects: tuple[ServeObject, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    def get_slot(self, name: str) -> SlotDecl | None:
        for slot in self.slots:
            if slot.name == name:
                return slot
        return None

    def contract_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        block_family = self.attrs.get("block_family")
        if not isinstance(block_family, str) or not block_family:
            errors.append("schema attrs.block_family must be a non-empty string")

        seen_slots: set[str] = set()
        for slot in self.slots:
            if not slot.name:
                errors.append("schema slots must have non-empty names")
            elif slot.name in seen_slots:
                errors.append(f"duplicate schema slot '{slot.name}'")
            seen_slots.add(slot.name)
            if not slot.shape:
                errors.append(f"schema slot '{slot.name}' must declare a non-empty shape")

        family = block_family.lower() if isinstance(block_family, str) else ""
        is_moe = "moe" in family
        if is_moe:
            if self.routing is None or self.routing.kind == "none":
                errors.append(f"MoE schema '{block_family}' must declare routing metadata")
            if self.ep_topology is None:
                errors.append(f"MoE schema '{block_family}' must declare EP topology metadata")
            grouped_expert_params = [
                slot
                for slot in self.slots
                if slot.kind == "param" and slot.grouped and slot.distribution.kind == "expert_parallel"
            ]
            if not grouped_expert_params:
                errors.append(f"MoE schema '{block_family}' must declare grouped expert-parallel param slots")
            router = self.get_slot("router_weight")
            if router is not None and router.distribution.kind != "router_replicated":
                errors.append(f"MoE schema '{block_family}' router_weight must be router_replicated")
        elif self.routing is not None and self.routing.kind != "none":
            errors.append(f"non-MoE schema '{block_family}' must not declare routing metadata")

        return tuple(errors)

    def validate_contract(self) -> None:
        errors = self.contract_errors()
        if errors:
            raise ValueError("; ".join(errors))
