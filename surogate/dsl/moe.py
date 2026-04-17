"""MoE registry for the Python DSL.

Router/EP tags are single-value ``StrEnum``s (no per-entry metadata);
shape differences live on ``MoEConfig``. ``MoE.*`` constants preset the
common expert variants (standard top-k, GPT-OSS sigmoid+interleaved,
Qwen3-MoE shared-expert).

See ``MIGRATION.md`` Phase 8.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Mapping

from .activations import Activation, ActivationSpec


class RouterKind(StrEnum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


class EPStrategyKind(StrEnum):
    STATIC = "static"
    LLEP = "llep"
    DEEP_EP = "deep_ep"


@dataclass(frozen=True, kw_only=True)
class MoEConfig:
    """Declarative MoE config.

    ``gate_up_interleaved`` is the GPT-OSS packing: a single expert
    weight with [gate0, up0, gate1, up1, ...] rows. ``shared_expert``
    is the Qwen3-MoE pattern where a dense expert adds to the routed
    outputs.
    """

    num_experts: int = 8
    top_k: int = 2
    router: RouterKind = RouterKind.SOFTMAX
    normalize_weights: bool = True
    gate_up_interleaved: bool = False
    shared_expert: bool = False
    expert_activation: ActivationSpec = Activation.SILU
    ep_strategy: EPStrategyKind = EPStrategyKind.STATIC


@dataclass(frozen=True, kw_only=True)
class MoESpec:
    """MoE kind = factory class + preset config."""

    name: str
    factory: type
    config: MoEConfig = field(default_factory=MoEConfig)


class MoE:
    """Known MoE kinds.

    Preset configurations for the common MoE shapes. The ``factory``
    slot is ``object`` until the generic MoE experts class lands;
    callers construct concrete MoE modules directly today.
    """

    STANDARD: ClassVar[MoESpec] = MoESpec(
        name="standard",
        factory=object,
        config=MoEConfig(),
    )
    GPT_OSS: ClassVar[MoESpec] = MoESpec(
        name="gpt_oss",
        factory=object,
        config=MoEConfig(
            router=RouterKind.SIGMOID,
            gate_up_interleaved=True,
            expert_activation=Activation.GPT_OSS,
        ),
    )
    QWEN3_MOE: ClassVar[MoESpec] = MoESpec(
        name="qwen3_moe",
        factory=object,
        config=MoEConfig(shared_expert=True, top_k=4),
    )


_BY_NAME: Mapping[str, MoESpec] = MappingProxyType({m.name: m for m in vars(MoE).values() if isinstance(m, MoESpec)})


def moe_from_name(name: str) -> MoESpec:
    """Resolve an MoE-kind string to a spec."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown MoE kind '{name}'. Known: {sorted(_BY_NAME)}")
    return spec


__all__ = [
    "EPStrategyKind",
    "MoE",
    "MoEConfig",
    "MoESpec",
    "RouterKind",
    "moe_from_name",
]
