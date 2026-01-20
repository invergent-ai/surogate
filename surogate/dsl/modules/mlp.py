"""MLP Modules."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, param, forward, save
from ..graph_builder import graph


@module
class SwiGLUMLP:
    """SwiGLU MLP: down(swiglu(up(x)))."""

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff  # gate + up concatenated

    @param
    def up_weight(self) -> Tensor["MUp", "C"]:
        """Up projection weight [2*d_ff, d_model] (gate+up fused)."""
        ...

    @param
    def down_weight(self) -> Tensor["C", "M"]:
        """Down projection weight [d_model, d_ff]."""
        ...

    @forward
    @save("x", "up")
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            # Flatten
            x_flat = g.view(x, shape=["B * T", "C"])

            # Up projection (gate + up combined)
            up_flat = g.matmul(x_flat, "up_weight", transpose="NT")
            up = g.view(up_flat, shape=["B", "T", "MUp"])

            # SwiGLU activation
            act = g.swiglu(up)

            # Down projection
            act_flat = g.view(act, shape=["B * T", "M"])
            y_flat = g.matmul(act_flat, "down_weight", transpose="NT")
            y = g.view(y_flat, shape=["B", "T", "C"])

            return y


@module
class GatedMLP:
    """Gated MLP with configurable activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",  # silu, relu, relu2, gelu
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.C = d_model
        self.M = d_ff

    @param
    def gate_weight(self) -> Tensor["M", "C"]:
        """Gate projection weight [d_ff, d_model]."""
        ...

    @param
    def up_weight(self) -> Tensor["M", "C"]:
        """Up projection weight [d_ff, d_model]."""
        ...

    @param
    def down_weight(self) -> Tensor["C", "M"]:
        """Down projection weight [d_model, d_ff]."""
        ...

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=["B * T", "C"])

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
            y = g.view(y_flat, shape=["B", "T", "C"])

            return y
