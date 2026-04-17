"""Norm registry for the Python DSL.

Norms are pure factory classes — no config dataclass. ``Norm.*`` tags
are string constants for grep-ability; ``make_norm(...)`` resolves them
to the concrete ``nn.*`` class at call time so this module avoids
importing ``nn.py`` at module load.

See ``MIGRATION.md`` Phase 8.6.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping


class Norm:
    """Known norm tags. Strings so the registry avoids importing ``nn``
    at module load, but still IDE-discoverable as ``Norm.RMSNORM``."""

    RMSNORM: str = "rmsnorm"
    RMSNORM_PLUS1: str = "rmsnorm_plus1"


def _rmsnorm() -> type:
    from .nn import RMSNorm  # lazy: nn.py is heavy

    return RMSNorm


def _rmsnorm_plus1() -> type:
    from .nn import RMSNormPlus1  # lazy

    return RMSNormPlus1


_RESOLVERS: Mapping[str, Callable[[], type]] = MappingProxyType(
    {
        Norm.RMSNORM: _rmsnorm,
        Norm.RMSNORM_PLUS1: _rmsnorm_plus1,
    }
)


def norm_from_name(name: str) -> type:
    """Resolve a norm tag string to its concrete ``nn.*`` class."""
    resolver = _RESOLVERS.get(name)
    if resolver is None:
        raise ValueError(f"Unknown norm '{name}'. Known: {sorted(_RESOLVERS)}")
    return resolver()


def make_norm(kind: str, d_model: int, eps: float = 1e-6, **kwargs: Any):
    """Construct a norm module of the given kind.

    ``kind`` is a ``Norm.*`` tag (``Norm.RMSNORM``, ``Norm.RMSNORM_PLUS1``).
    Extra ``kwargs`` are forwarded to the underlying class.
    """
    cls = norm_from_name(kind)
    return cls(d_model, eps=eps, **kwargs)


__all__ = [
    "Norm",
    "make_norm",
    "norm_from_name",
]
