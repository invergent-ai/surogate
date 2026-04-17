"""RoPE registry for the Python DSL.

Rotary Position Embedding variants. Names are the HF-config keys
(``rope_type`` / ``rope_scaling.type``); ``cpp_op`` names match the ops
registered in ``csrc/src/runtime/executor/op_registrations.cpp``.

See ``MIGRATION.md`` Phase 8.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping


@dataclass(frozen=True, kw_only=True)
class RoPESpec:
    """Describes a RoPE variant for the DSL -> C++ pipeline."""

    name: str
    cpp_op: str
    requires_sections: bool = False
    requires_scaling: bool = False
    scaling_fn: Callable[..., Any] | None = None


class RoPE:
    """Known RoPE variants. Immutable, IDE-discoverable."""

    STANDARD: ClassVar[RoPESpec] = RoPESpec(name="standard", cpp_op="rope")
    MROPE: ClassVar[RoPESpec] = RoPESpec(name="mrope", cpp_op="mrope", requires_sections=True)


_BY_NAME: Mapping[str, RoPESpec] = MappingProxyType({r.name: r for r in vars(RoPE).values() if isinstance(r, RoPESpec)})


def rope_from_name(name: str) -> RoPESpec:
    """Resolve an HF-config RoPE name to a spec."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown RoPE '{name}'. Known: {sorted(_BY_NAME)}")
    return spec


__all__ = [
    "RoPE",
    "RoPESpec",
    "rope_from_name",
]
