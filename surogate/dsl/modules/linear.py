"""Linear projection module."""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer


class Linear(Module):
    """Linear projection: y = x @ W^T (+ bias)."""

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
        "bias": "{prefix}.bias",
    }

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.C = Dim("in_dim")
        self.O = Dim("out_dim")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        w = tracer.register_param("weight", ("O", "C"))
        if self.use_bias:
            b = tracer.register_param("bias", ("O",), when="use_bias")

        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "O"),
            share_policy="when_recomputed",
        )

        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))
        if self.use_bias:
            y_flat = g.matmul_bias(x_flat, w, b, transpose="NT", out_name=tracer.prefixed("y_flat"))
        else:
            y_flat = g.matmul(x_flat, w, transpose="NT", out_name=tracer.prefixed("y_flat"))
        out = g.view(y_flat, shape=[B, T, self.O], out_name=out_slot)

        return Proxy(out_slot, out)
