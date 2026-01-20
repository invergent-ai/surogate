"""Tensor manipulation primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive


@primitive(impl="kernels.view")
def view(x: Tensor["*"], *, shape: list[int | str]) -> Tensor["*"]:
    """Reshape tensor (no data movement if contiguous)."""
    ...


@primitive(impl="kernels.transpose")
def transpose(
    x: Tensor["D0", "D1"],
    *,
    dim0: int = 0,
    dim1: int = 1,
) -> Tensor["D1", "D0"]:
    """Transpose two dimensions."""
    ...


@transpose.backward
def transpose_backward(d_out: Tensor["D1", "D0"]) -> Tensor["D0", "D1"]:
    """Backward pass for transpose (just transpose back)."""
    ...


@primitive(impl="kernels.concat")
def concat(*tensors: Tensor["*"], dim: int = 0) -> Tensor["*"]:
    """Concatenate tensors along dimension."""
    ...


@primitive(impl="kernels.split")
def split(
    x: Tensor["*"],
    *,
    split_size: int | list[int],
    dim: int = 0,
) -> tuple[Tensor["*"], ...]:
    """Split tensor along dimension."""
    ...
