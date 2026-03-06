"""Qwen3.5-specific primitives."""

from __future__ import annotations

from ..decorators import primitive
from ..tensor_type import Tensor


@primitive(impl="kernels.qwen3_5_decay")
def qwen3_5_decay(
    a: Tensor["B", "T", "H"],
    a_log: Tensor["H"],
    dt_bias: Tensor["H"],
) -> Tensor["B", "T", "H"]:
    """Compute Qwen3.5 decay term: -exp(A_log) * softplus(a + dt_bias)."""
    ...

