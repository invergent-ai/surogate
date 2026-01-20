"""Normalization primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save


@primitive(impl="kernels.rmsnorm")
def rmsnorm(
    x: Tensor["*", "C"],
    weight: Tensor["C"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["*", "C"], Tensor["*"]]:
    """RMS normalization. Returns (y, rstd)."""
    ...


@rmsnorm.backward
@save("x", "rstd")
def rmsnorm_backward(
    d_y: Tensor["*", "C"],
    x: Tensor["*", "C"],
    weight: Tensor["C"],
    rstd: Tensor["*"],
) -> tuple[Tensor["*", "C"], Tensor["C"]]:
    """Backward pass for RMS norm. Returns (d_x, d_weight)."""
    ...


@primitive(impl="kernels.fused_residual_rmsnorm")
def fused_residual_rmsnorm(
    residual: Tensor["*", "C"],
    x: Tensor["*", "C"],
    weight: Tensor["C"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["*", "C"], Tensor["*", "C"], Tensor["*"]]:
    """Fused residual add + RMS norm. Returns (residual_out, y, rstd)."""
    ...


@fused_residual_rmsnorm.backward
@save("residual_out", "rstd")
def fused_residual_rmsnorm_backward(
    d_y: Tensor["*", "C"],
    d_residual_next: Tensor["*", "C"],
    residual_out: Tensor["*", "C"],
    weight: Tensor["C"],
    rstd: Tensor["*"],
) -> tuple[Tensor["*", "C"], Tensor["*", "C"], Tensor["C"]]:
    """Backward pass. Returns (d_residual, d_input, d_weight)."""
    ...
