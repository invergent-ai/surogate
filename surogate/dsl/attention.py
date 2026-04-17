"""Attention registry for the Python DSL.

Declarative ``AttentionConfig`` replaces per-model attention classes;
``Attention.*`` constants tag the concrete mixer kinds (GQA today,
Mamba2 / gated-delta in the future).

See ``MIGRATION.md`` Phase 8.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Mapping

from .rope import RoPE, RoPESpec


@dataclass(frozen=True, kw_only=True)
class AttentionConfig:
    """Declarative attention config.

    Variants that differ only in configuration (QK-norm, sandwich norm,
    sliding window, fused vs separate QKV) collapse into a single
    generic attention class driven by this object.
    """

    # QKV layout
    fuse_qkv: bool = True
    k_eq_v: bool = False

    # Norms
    qk_norm: bool = False
    sandwich_norm: bool = False

    # Position encoding
    rope: RoPESpec = RoPE.STANDARD

    # Attention masking
    sliding_window: int = 0
    window_per_layer: bool = False

    # Biases
    qkv_bias: bool = False
    out_bias: bool = False

    # Attention-specific extras
    has_sinks: bool = False  # GPT-OSS: per-head learnable sink scalars
    softmax_scale: float | None = None  # None = default (1/sqrt(D)); Gemma4 uses 1.0
    partial_rotary_factor: float = 1.0  # fraction of head dim rotated by RoPE

    # QK-norm numerics
    eps: float = 1e-6


@dataclass(frozen=True, kw_only=True)
class AttentionSpec:
    """Attention kind = factory class + a preset config.

    ``factory`` is a ``Module`` subclass. It is typed loosely as ``type``
    to avoid an import cycle with ``surogate.dsl.nn``.
    """

    name: str
    factory: type
    config: AttentionConfig = field(default_factory=AttentionConfig)


class Attention:
    """Known attention kinds.

    Today this is populated lazily by ``surogate.dsl.nn`` / ``modules``
    at import time (see Phase 8 step 2 in ``MIGRATION.md``). Until the
    generic attention class lands, callers construct concrete attention
    modules directly; the spec tags are defined here so the registry
    surface exists for downstream code.
    """


_BY_NAME: dict[str, AttentionSpec] = {}


def _register(spec: AttentionSpec) -> AttentionSpec:
    """Register an ``AttentionSpec`` by name. Idempotent — re-registering
    the same name (e.g. after ``importlib.reload``) replaces the previous
    entry rather than crashing. Use distinct names for genuinely
    different kinds."""
    _BY_NAME[spec.name] = spec
    return spec


def attention_from_name(name: str) -> AttentionSpec:
    """Resolve an attention-kind string to a spec."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown attention kind '{name}'. Known: {sorted(_BY_NAME)}")
    return spec


def known_attention_kinds() -> Mapping[str, AttentionSpec]:
    """Return an immutable view of registered attention kinds."""
    return MappingProxyType(_BY_NAME)


__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionSpec",
    "attention_from_name",
    "known_attention_kinds",
]
