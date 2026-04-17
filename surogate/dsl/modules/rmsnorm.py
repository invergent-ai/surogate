"""Normalization modules (RMSNorm family)."""

from __future__ import annotations

from typing import Any

from ..dim import Dim
from ..nn import Module, Proxy, Tracer


class RMSNorm(Module):
    """RMS Layer Normalization.

    When called with one argument ``(x,)`` performs standard rmsnorm.
    When called with two arguments ``(residual, x)`` performs fused
    residual-add + rmsnorm.
    """

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.C = Dim("C")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy | tuple[Proxy, ...]:
        g = tracer.graph
        weight = tracer.register_param("weight", ("C",), quantizable=False)

        if len(args) == 2:
            # Fused residual + rmsnorm
            residual, x = args
            full_y_name = tracer.prefixed("y")
            save_y = full_y_name in {"ln1", "ln"} or full_y_name.endswith("attn_norm_y")

            res_slot = tracer.register_activation(
                "res",
                ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            y_slot = tracer.register_activation(
                "y",
                ("B", "T", "C"),
                save=save_y,
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd",
                ("B", "T"),
                dtype="fp32",
                save=True,
                share_policy="per_layer",
            )

            res_ref, y_ref, rstd_ref = g.fused_residual_rmsnorm(
                residual.ref,
                x.ref,
                weight,
                eps=self.eps,
                res_out_name=res_slot,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return (
                Proxy(res_slot, res_ref),
                Proxy(y_slot, y_ref),
            )

        elif len(args) == 1:
            x = args[0]

            y_slot = tracer.register_activation(
                "y",
                ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd",
                ("B", "T"),
                dtype="fp32",
                save=True,
                share_policy="per_layer",
            )

            y_ref, rstd_ref = g.rmsnorm(
                x.ref,
                weight,
                eps=self.eps,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return Proxy(y_slot, y_ref)

        raise ValueError(f"RMSNorm expects 1 or 2 inputs, got {len(args)}")


class RMSNormPlus1(Module):
    """RMS Layer Normalization with weight + 1 bias (Qwen3.5 style).

    Like RMSNorm, but adds 1.0 to the weight before applying normalization.
    When called with one argument ``(x,)`` performs standard rmsnorm(x, weight+1).
    When called with two arguments ``(residual, x)`` performs fused
    residual-add + rmsnorm(x, weight+1).
    """

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.C = Dim("C")

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy | tuple[Proxy, ...]:
        g = tracer.graph
        weight = tracer.register_param("weight", ("C",), quantizable=False)

        # Create ones and add to weight for the +1 bias
        ones_ref = g.ones(shape=[self.C], dtype="bf16")
        weight_eff = g.add(weight, ones_ref, out_name=tracer.prefixed("weight_eff"))

        if len(args) == 2:
            # Fused residual + rmsnorm
            residual, x = args

            res_slot = tracer.register_activation(
                "res",
                ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            y_slot = tracer.register_activation(
                "y",
                ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd",
                ("B", "T"),
                dtype="fp32",
                save=True,
                share_policy="per_layer",
            )

            res_ref, y_ref, rstd_ref = g.fused_residual_rmsnorm(
                residual.ref,
                x.ref,
                weight_eff,
                eps=self.eps,
                res_out_name=res_slot,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return (
                Proxy(res_slot, res_ref),
                Proxy(y_slot, y_ref),
            )

        elif len(args) == 1:
            x = args[0]

            y_slot = tracer.register_activation(
                "y",
                ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd",
                ("B", "T"),
                dtype="fp32",
                save=True,
                share_policy="per_layer",
            )

            y_ref, rstd_ref = g.rmsnorm(
                x.ref,
                weight_eff,
                eps=self.eps,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return Proxy(y_slot, y_ref)

        raise ValueError(f"RMSNormPlus1 expects 1 or 2 inputs, got {len(args)}")


# Backwards-compat alias (schema name used by ``surogate.dsl.hf``).
FusedResidualRMSNorm = RMSNorm
