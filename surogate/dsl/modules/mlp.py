"""MLP Modules."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, forward, save, Param
from ..graph_builder import graph
from ..dim import Dim, B, T
from ..hf import fuse


@module
class SwiGLUMLP:
    """SwiGLU MLP: down(swiglu(up(x)))."""

    # Default HF weight path templates.
    # Use {prefix} for the MLP submodule path
    # (e.g., "model.layers.{layer}.mlp").
    # SwiGLU fuses up_proj and gate_proj into a single up_weight.
    _hf_mapping_defaults_ = {
        "up_weight": fuse(
            "{prefix}.up_proj.weight",
            "{prefix}.gate_proj.weight",
            dim=0,
        ),
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")
        self.M = Dim("M")

        # Derived dimensions (DimExpr)
        self.MUp = 2 * self.M  # gate + up concatenated

    # MLP weights
    up_weight = Param(Tensor["MUp", "C"])
    down_weight = Param(Tensor["C", "M"])

    @forward
    @save("x", "up")
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            # Flatten
            x_flat = g.view(x, shape=[B * T, self.C])

            # Up projection (gate + up combined)
            up_flat = g.matmul(x_flat, "up_weight", transpose="NT")
            up = g.view(up_flat, shape=[B, T, self.MUp])

            # SwiGLU activation
            act = g.swiglu(up)

            # Down projection
            act_flat = g.view(act, shape=[B * T, self.M])
            y_flat = g.matmul(act_flat, "down_weight", transpose="NT")
            y = g.view(y_flat, shape=[B, T, self.C])

            return y


@module
class GatedMLP:
    """Gated MLP with configurable activation."""

    # Default HF weight path templates.
    # GatedMLP uses separate gate and up projections (not fused).
    _hf_mapping_defaults_ = {
        "gate_weight": "{prefix}.gate_proj.weight",
        "up_weight": "{prefix}.up_proj.weight",
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",  # silu, relu, relu2, gelu
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")
        self.M = Dim("M")

    # MLP weights
    gate_weight = Param(Tensor["M", "C"])
    up_weight = Param(Tensor["M", "C"])
    down_weight = Param(Tensor["C", "M"])

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=[B * T, self.C])

            # Gate and up projections
            gate_flat = g.matmul(x_flat, "gate_weight", transpose="NT")
            up_flat = g.matmul(x_flat, "up_weight", transpose="NT")

            # Apply activation to gate
            if self.activation == "silu":
                gate_act = g.silu(gate_flat)
            elif self.activation == "relu":
                gate_act = g.relu(gate_flat)
            elif self.activation == "relu2":
                gate_act = g.relu2(gate_flat)
            elif self.activation == "gelu":
                gate_act = g.gelu(gate_flat)
            else:
                gate_act = g.silu(gate_flat)  # default

            # Gating
            hidden = g.mul(gate_act, up_flat)

            # Down projection
            y_flat = g.matmul(hidden, "down_weight", transpose="NT")
            y = g.view(y_flat, shape=[B, T, self.C])

            return y
