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
