"""RMSNorm Modules."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, param, forward
from ..graph_builder import graph
from ..dim import Dim, B, T


@module
class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Typed dimension - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")

    @param
    def weight(self) -> Tensor["C"]:
        """Normalization weight [d_model]."""
        ...

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            y, _ = g.rmsnorm(x, "weight", eps=self.eps)
            return y


@module
class FusedResidualRMSNorm:
    """Fused residual addition + RMS normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Typed dimension - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")

    @param
    def weight(self) -> Tensor["C"]:
        """Normalization weight [d_model]."""
        ...

    @forward
    def forward(
        self,
        residual: Tensor["B", "T", "C"],
        x: Tensor["B", "T", "C"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Returns (residual_out, normalized)."""
        with graph() as g:
            residual_out, y, _ = g.fused_residual_rmsnorm(
                residual, x, "weight", eps=self.eps
            )
            return residual_out, y
