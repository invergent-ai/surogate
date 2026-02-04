"""Initialization primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive


@primitive(impl="kernels.zeros")
def zeros(*, shape: list[int | str], dtype: str = "bf16") -> Tensor["*"]:
    """Create zero-filled tensor."""
    ...


@primitive(impl="kernels.ones")
def ones(*, shape: list[int | str], dtype: str = "bf16") -> Tensor["*"]:
    """Create one-filled tensor."""
    ...


@primitive(impl="kernels.fill_normal")
def fill_normal(
    *,
    shape: list[int | str],
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 0,
    dtype: str = "bf16",
) -> Tensor["*"]:
    """Create tensor filled with normal random values."""
    ...
