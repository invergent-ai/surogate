"""Linear Module."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, param, forward, save
from ..graph_builder import graph


@module
class Linear:
    """Linear projection: y = x @ W^T (+ bias)."""

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        # Derived constants (like DSL let: section)
        self.C = in_dim
        self.O = out_dim

    @param
    def weight(self) -> Tensor["O", "C"]:
        """Weight matrix [out_dim, in_dim]."""
        ...

    @param(condition=lambda self: self.use_bias)
    def bias(self) -> Tensor["O"]:
        """Optional bias vector [out_dim]."""
        ...

    @forward
    @save("x")
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "O"]:
        with graph() as g:
            # Flatten batch dimensions
            x_flat = g.view(x, shape=["B * T", "C"])

            # Matrix multiply
            y_flat = g.matmul(x_flat, "weight", transpose="NT")

            # Reshape back
            y_tmp = g.view(y_flat, shape=["B", "T", "O"])

            # Optional bias
            if self.use_bias:
                y = g.bias_add(y_tmp, "bias")
            else:
                y = y_tmp

            return y
