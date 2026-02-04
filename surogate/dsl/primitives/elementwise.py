"""Elementwise operation primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive


@primitive(impl="kernels.add")
def add(a: Tensor["*"], b: Tensor["*"]) -> Tensor["*"]:
    """Element-wise addition."""
    ...


@primitive(impl="kernels.mul")
def mul(a: Tensor["*"], b: Tensor["*"]) -> Tensor["*"]:
    """Element-wise multiplication."""
    ...


@primitive(impl="kernels.scale")
def scale(x: Tensor["*"], *, factor: float) -> Tensor["*"]:
    """Scale tensor by constant factor."""
    ...


@primitive(impl="kernels.bias_add")
def bias_add(x: Tensor["*", "C"], bias: Tensor["C"]) -> Tensor["*", "C"]:
    """Add bias along last dimension."""
    ...
