"""MLP registry for the Python DSL.

Declarative ``MLPConfig`` collapses the fused-SwiGLU / gated / simple MLP
shapes into a single generic block; ``MLP.*`` constants are populated by
``surogate.dsl.modules.mlp`` on import and preset the common variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .activations import Activation, ActivationSpec


@dataclass(frozen=True, kw_only=True)
class MLPConfig:
    """Declarative MLP config.

    ``gated`` selects gate*up gating (SwiGLU / GatedMLP) vs a plain
    two-matmul MLP. ``fuse_gate_up`` controls whether the gate and up
    projections share a concatenated weight (the SwiGLU case) or remain
    separate matrices.
    """

    activation: ActivationSpec = Activation.SILU
    gated: bool = True
    fuse_gate_up: bool = True
    bias: bool = False


@dataclass(frozen=True, kw_only=True)
class MLPSpec:
    """MLP kind = factory class + preset config."""

    name: str
    factory: type
    config: MLPConfig = field(default_factory=MLPConfig)


class MLP:
    """Known MLP kinds. Populated by ``surogate.dsl.modules.mlp`` on import."""


_BY_NAME: dict[str, MLPSpec] = {}


def _register(spec: MLPSpec) -> MLPSpec:
    """Register an ``MLPSpec`` by name. Idempotent — re-registering the
    same name (e.g. after ``importlib.reload``) replaces the previous
    entry rather than crashing."""
    _BY_NAME[spec.name] = spec
    return spec


def mlp_from_name(name: str) -> MLPSpec:
    """Resolve an MLP-kind string to a spec."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown MLP kind '{name}'. Known: {sorted(_BY_NAME)}")
    return spec


def known_mlp_kinds() -> Mapping[str, MLPSpec]:
    """Return an immutable view of registered MLP kinds."""
    return MappingProxyType(_BY_NAME)


__all__ = [
    "MLP",
    "MLPConfig",
    "MLPSpec",
    "known_mlp_kinds",
    "mlp_from_name",
]
