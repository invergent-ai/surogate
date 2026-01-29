"""Linear Module."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, param, forward, save
from ..graph_builder import graph
from ..dim import Dim, B, T


@module
class Linear:
    """Linear projection: y = x @ W^T (+ bias)."""

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        # Typed dimensions bound to config parameters
        self.C = Dim("in_dim")
        self.O = Dim("out_dim")

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
            x_flat = g.view(x, shape=[B * T, self.C])

            # Matrix multiply (optionally fused with bias)
            if self.use_bias:
                y_flat = g.matmul_bias(x_flat, "weight", "bias", transpose="NT")
            else:
                y_flat = g.matmul(x_flat, "weight", transpose="NT")

            # Reshape back
            y = g.view(y_flat, shape=[B, T, self.O])

            return y
