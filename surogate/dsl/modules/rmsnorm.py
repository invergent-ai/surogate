"""RMSNorm Modules."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, forward, Param
from ..graph_builder import graph
from ..dim import Dim, B, T


@module
class RMSNorm:
    """Root Mean Square Layer Normalization."""

    # Default HF weight path template.
    # Use {prefix} for the norm submodule path
    # (e.g., "model.layers.{layer}.input_layernorm").
    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Typed dimension - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")

    # Normalization weight
    weight = Param(Tensor["C"])

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            y, _ = g.rmsnorm(x, "weight", eps=self.eps)
            return y


@module
class FusedResidualRMSNorm:
    """Fused residual addition + RMS normalization."""

    # Same weight as RMSNorm.
    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Typed dimension - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")

    # Normalization weight
    weight = Param(Tensor["C"])

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
