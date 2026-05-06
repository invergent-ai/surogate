"""Short causal convolution mixer modules."""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer
from ..specs import LoRATarget


class Lfm2ShortConv(Module):
    """LFM2 short depthwise causal convolution mixer.

    Matches ``transformers.models.lfm2.modeling_lfm2.Lfm2ShortConv`` for
    training/full-sequence execution:

    ``B, C, x = in_proj(normed_x).transpose(1, 2).chunk(3, dim=1)``
    ``out = out_proj((C * causal_conv1d(B * x)).transpose(1, 2))``
    """

    _hf_mapping_defaults_ = {
        "in_proj_weight": "{prefix}.in_proj.weight",
        "in_proj_bias": "{prefix}.in_proj.bias",
        "conv_weight": "{prefix}.conv.weight",
        "conv_bias": "{prefix}.conv.bias",
        "out_proj_weight": "{prefix}.out_proj.weight",
        "out_proj_bias": "{prefix}.out_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        conv_kernel: int = 3,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.conv_kernel = conv_kernel
        self.use_bias = use_bias

        self.C = Dim("C")
        self.ThreeC = 3 * self.C
        self.K = conv_kernel

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        hidden = self.d_model

        in_w = tracer.register_param(
            "in_proj_weight",
            ("3 * C", "C"),
            lora_targets=[
                LoRATarget(name="conv_b", offset=0, size=hidden),
                LoRATarget(name="conv_c", offset=hidden, size=hidden),
                LoRATarget(name="conv_x", offset=2 * hidden, size=hidden),
            ],
        )
        in_b = tracer.register_param("in_proj_bias", ("3 * C",), when="use_bias")
        conv_w = tracer.register_param("conv_weight", ("C", 1, self.K), quantizable=False)
        conv_b = tracer.register_param("conv_bias", ("C",), when="use_bias", quantizable=False)
        out_w = tracer.register_param(
            "out_proj_weight",
            ("C", "C"),
            lora_targets=[LoRATarget(name="out", size=hidden)],
        )
        out_b = tracer.register_param("out_proj_bias", ("C",), when="use_bias")

        in_proj_slot = tracer.register_activation(
            "in_proj",
            ("B", "T", "3 * C"),
            aliases=["in_proj_flat"],
            save=True,
            share_policy="when_recomputed",
        )
        bx_slot = tracer.register_activation(
            "bx",
            ("B", "C", "T"),
            save=True,
            share_policy="when_recomputed",
        )
        conv_slot = tracer.register_activation(
            "conv_out",
            ("B", "C", "T"),
            save=True,
            share_policy="when_recomputed",
        )
        gated_slot = tracer.register_activation(
            "gated_conv",
            ("B", "C", "T"),
            save=True,
            share_policy="when_recomputed",
        )
        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "C"),
            aliases=["out_proj_flat"],
            share_policy="when_recomputed",
        )

        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )
        if self.use_bias:
            in_proj_flat = g.matmul_bias(
                x_flat,
                in_w,
                in_b,
                transpose="NT",
                out_name=tracer.prefixed("in_proj_flat"),
            )
        else:
            in_proj_flat = g.matmul(
                x_flat,
                in_w,
                transpose="NT",
                out_name=tracer.prefixed("in_proj_flat"),
            )
        in_proj = g.view(
            in_proj_flat,
            shape=[B, T, self.ThreeC],
            out_name=in_proj_slot,
        )
        in_proj_cf = g.transpose(in_proj, dim0=1, dim1=2)
        b_gate, c_gate, conv_x = g.split(
            in_proj_cf,
            split_size=[self.d_model, self.d_model, self.d_model],
            dim=1,
            out_names=[
                tracer.prefixed("b_gate"),
                tracer.prefixed("c_gate"),
                tracer.prefixed("conv_x"),
            ],
        )
        bx = g.mul(b_gate, conv_x, out_name=bx_slot)
        conv_w_2d = g.view(
            conv_w,
            shape=[self.C, self.K],
            out_name=tracer.prefixed("conv_w2d"),
        )
        conv_out = g.mamba_conv1d(
            bx,
            conv_w_2d,
            conv_b if self.use_bias else None,
            activation="",
            out_name=conv_slot,
        )
        gated = g.mul(c_gate, conv_out, out_name=gated_slot)
        gated_bt = g.transpose(gated, dim0=1, dim1=2)
        gated_flat = g.view(
            gated_bt,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("gated_flat"),
        )
        if self.use_bias:
            out_flat = g.matmul_bias(
                gated_flat,
                out_w,
                out_b,
                transpose="NT",
                out_name=tracer.prefixed("out_proj_flat"),
            )
        else:
            out_flat = g.matmul(
                gated_flat,
                out_w,
                transpose="NT",
                out_name=tracer.prefixed("out_proj_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=out_slot,
        )

        return Proxy(out_slot, out)
