"""Block schema declarations for Phase 4 storage and distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DistributionKind = Literal["replicated", "sharded_dim", "expert_parallel", "router_replicated"]
Residency = Literal["auto", "gpu", "cpu_pinned_stream", "cpu_pageable", "nvme_offload"]
SlotKind = Literal["activation", "param", "scratch", "param_grad", "activation_grad"]
Lifetime = Literal["op", "layer", "block", "model", "persistent"]
RoutingKind = Literal["none", "topk_softmax", "topk_sigmoid", "expert_choice"]


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
