"""Activation primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save


@primitive(impl="kernels.swiglu")
def swiglu(x: Tensor["*", "2M"]) -> Tensor["*", "M"]:
    """SwiGLU activation: silu(gate) * up where x = [gate, up]."""
    ...


@swiglu.backward
@save("x")
def swiglu_backward(
    d_out: Tensor["*", "M"],
    x: Tensor["*", "2M"],
) -> Tensor["*", "2M"]:
    """Backward pass for SwiGLU."""
    ...


@primitive(impl="kernels.silu")
def silu(x: Tensor["*"]) -> Tensor["*"]:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    ...


@silu.backward
@save("x")
def silu_backward(d_out: Tensor["*"], x: Tensor["*"]) -> Tensor["*"]:
    """Backward pass for SiLU."""
    ...


@primitive(impl="kernels.relu2")
def relu2(x: Tensor["*"]) -> Tensor["*"]:
    """ReLU squared activation: max(0, x)^2."""
    ...


@primitive(impl="kernels.silu_mul")
def silu_mul(gate: Tensor["*"], up: Tensor["*"]) -> Tensor["*"]:
    """SiLU(gate) * up activation."""
    ...
