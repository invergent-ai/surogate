"""Mamba2 SSM mixer."""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer


class Mamba2Mixer(Module):
    """Mamba2 SSM module for hybrid architectures like Nemotron-H."""

    _hf_mapping_defaults_ = {
        "in_proj_weight": "{prefix}.in_proj.weight",
        "in_proj_bias": "{prefix}.in_proj.bias",
        "conv_weight": "{prefix}.conv1d.weight",
        "conv_bias": "{prefix}.conv1d.bias",
        "A_log": "{prefix}.A_log",
        "D_param": "{prefix}.D",
        "dt_bias": "{prefix}.dt_bias",
        "gated_norm_weight": "{prefix}.norm.weight",
        "out_proj_weight": "{prefix}.out_proj.weight",
        "out_proj_bias": "{prefix}.out_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        ssm_state_size: int = 128,
        n_groups: int = 8,
        conv_kernel: int = 4,
        chunk_size: int = 256,
        eps: float = 1e-5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        time_step_limit: tuple[float, float] | None = None,
        use_conv_bias: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.eps = eps
        self.dt_min = dt_min
        self.dt_max = dt_max
        import math as _math

        dt_max_default = 1e9
        if time_step_limit is None:
            time_step_limit = (0.0, dt_max_default)
        elif isinstance(time_step_limit, (list, tuple)) and len(time_step_limit) == 2:
            lo = float(time_step_limit[0])
            hi = float(time_step_limit[1])
            if not _math.isfinite(lo):
                lo = 0.0
            if not _math.isfinite(hi):
                hi = dt_max_default
            time_step_limit = (lo, hi)
        self.time_step_limit = time_step_limit
        self.use_conv_bias = use_conv_bias
        self.use_bias = use_bias

        # Derived dimensions
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.conv_dim = self.intermediate_size + 2 * n_groups * ssm_state_size
        self.projection_size = self.intermediate_size + self.conv_dim + mamba_num_heads

        # Use concrete integers for Mamba-specific dims to avoid conflicts
        # with attention dims (D, K) in hybrid models.
        self.C = Dim("C")  # d_model — shared with attention

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        P = self.projection_size
        I = self.intermediate_size
        H = self.mamba_num_heads
        D_conv = self.conv_dim
        CK = self.conv_kernel

        # -- params ----------------------------------------------------------
        in_proj_w = tracer.register_param("in_proj_weight", (P, "C"))
        in_proj_b = tracer.register_param("in_proj_bias", (P,), when="use_bias")
        tracer.register_param("conv_weight", (D_conv, CK), quantizable=False)
        tracer.register_param("conv_bias", (D_conv,), when="use_conv_bias", quantizable=False)
        tracer.register_param("A_log", (H,), dtype="fp32", quantizable=False)
        tracer.register_param("D_param", (H,), dtype="fp32", quantizable=False)
        tracer.register_param("dt_bias", (H,), dtype="fp32", quantizable=False)
        tracer.register_param("gated_norm_weight", (I,), quantizable=False)
        out_proj_w = tracer.register_param("out_proj_weight", ("C", I))
        out_proj_b = tracer.register_param("out_proj_bias", ("C",), when="use_bias")

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "projected",
            ("B", "T", P),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "gate",
            ("B", "T", I),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "conv_out",
            ("B", D_conv, "T"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "hidden_states",
            ("B", I, "T"),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "ssm_out",
            ("B", "T", I),
            save=True,
            share_policy="fft_share",
        )
        tracer.register_activation(
            "ssm_state",
            ("B", H, self.mamba_head_dim, self.ssm_state_size),
            save=True,
            share_policy="fft_share",
            description="Final SSM state for caching",
        )
        tracer.register_activation(
            "gated_out",
            ("B", "T", I),
            save=True,
            share_policy="fft_share",
        )
        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="Mamba2 mixer output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )
        if self.use_bias:
            projected_flat = g.matmul_bias(
                x_flat,
                in_proj_w,
                in_proj_b,
                transpose="NT",
                out_name=tracer.prefixed("projected_flat"),
            )
        else:
            projected_flat = g.matmul(
                x_flat,
                in_proj_w,
                transpose="NT",
                out_name=tracer.prefixed("projected_flat"),
            )
        projected = g.view(
            projected_flat,
            shape=[B, T, P],
            out_name=tracer.prefixed("projected"),
        )

        # Split projection
        gate, conv_input, dt = g.mamba_split_proj(
            projected,
            intermediate_size=self.intermediate_size,
            conv_dim=self.conv_dim,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            gate_name=tracer.prefixed("gate"),
            conv_input_name=tracer.prefixed("conv_input"),
            dt_name=tracer.prefixed("dt"),
        )

        # Causal 1D convolution
        if self.use_conv_bias:
            conv_out = g.mamba_conv1d(
                conv_input,
                tracer.prefixed("conv_weight"),
                tracer.prefixed("conv_bias"),
                activation="silu",
                out_name=tracer.prefixed("conv_out"),
            )
        else:
            conv_out = g.mamba_conv1d(
                conv_input,
                tracer.prefixed("conv_weight"),
                None,
                activation="silu",
                out_name=tracer.prefixed("conv_out"),
            )

        # Split conv output
        hidden_states, ssm_B, ssm_C = g.mamba_split_conv_out(
            conv_out,
            intermediate_size=self.intermediate_size,
            groups_state_size=self.n_groups * self.ssm_state_size,
            n_groups=self.n_groups,
            ssm_state_size=self.ssm_state_size,
            hidden_name=tracer.prefixed("hidden_states"),
            B_name=tracer.prefixed("ssm_B"),
            C_name=tracer.prefixed("ssm_C"),
        )

        # SSM scan
        ssm_out, ssm_state = g.mamba_ssm_scan(
            hidden_states,
            dt,
            tracer.prefixed("A_log"),
            ssm_B,
            ssm_C,
            tracer.prefixed("D_param"),
            dt_bias=tracer.prefixed("dt_bias"),
            dt_softplus=True,
            dt_min=self.time_step_limit[0],
            dt_max=self.time_step_limit[1],
            chunk_size=self.chunk_size,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            ssm_state_size=self.ssm_state_size,
            n_groups=self.n_groups,
            out_name=tracer.prefixed("ssm_out"),
            state_name=tracer.prefixed("ssm_state"),
        )

        # Reshape SSM output
        ssm_out_flat = g.view(
            ssm_out,
            shape=[B, T, I],
            out_name=tracer.prefixed("ssm_out_flat"),
        )

        # Gated RMSNorm
        gated_out = g.mamba_gated_rmsnorm(
            ssm_out_flat,
            gate,
            tracer.prefixed("gated_norm_weight"),
            eps=self.eps,
            n_groups=self.n_groups,
            out_name=tracer.prefixed("gated_out"),
        )

        # Output projection
        gated_flat = g.view(
            gated_out,
            shape=[B * T, I],
            out_name=tracer.prefixed("gated_flat"),
        )
        if self.use_bias:
            out_flat = g.matmul_bias(
                gated_flat,
                out_proj_w,
                out_proj_b,
                transpose="NT",
                out_name=tracer.prefixed("out_flat"),
            )
        else:
            out_flat = g.matmul(
                gated_flat,
                out_proj_w,
                transpose="NT",
                out_name=tracer.prefixed("out_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=out_slot,
        )

        return Proxy(out_slot, out)


# Self-register as an ``AttentionSpec`` kind for discovery.
from ..attention import AttentionSpec, _register as _register_attention  # noqa: E402

_register_attention(AttentionSpec(name="mamba2", factory=Mamba2Mixer))
