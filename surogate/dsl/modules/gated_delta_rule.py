"""Gated DeltaNet linear-attention mixer."""

from __future__ import annotations

from typing import Any

from ..dim import B, Dim, T
from ..nn import Module, Proxy, Tracer


class GatedDeltaNetMixer(Module):
    """Gated DeltaNet linear attention mixer (Qwen3.5 style).

    Implements the linear attention path:
    - in_proj_qkv: project to QKV space (then conv1d + silu)
    - in_proj_z: gate projection
    - in_proj_b: beta projection (sigmoid)
    - in_proj_a: decay projection
    - A_log, dt_bias: decay parameters (FP32)
    - chunk_gated_delta_rule: fused chunked attention
    - gated_rmsnorm: gated normalization with Z gate
    - out_proj: output projection
    """

    _hf_mapping_defaults_ = {
        "in_proj_qkv_weight": "{prefix}.in_proj_qkv.weight",
        "in_proj_z_weight": "{prefix}.in_proj_z.weight",
        "in_proj_b_weight": "{prefix}.in_proj_b.weight",
        "in_proj_a_weight": "{prefix}.in_proj_a.weight",
        "conv_weight": "{prefix}.conv1d.weight",
        "A_log": "{prefix}.A_log",
        "dt_bias": "{prefix}.dt_bias",
        "norm_weight": "{prefix}.norm.weight",
        "out_weight": "{prefix}.out_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.chunk_size = chunk_size
        self.eps = eps

        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(
                "GatedDeltaNetMixer requires linear_num_value_heads to be divisible by linear_num_key_heads"
            )

        self.C = Dim("C")
        # Use concrete integers for linear-attention dims to avoid conflicts
        self.Hk = linear_num_key_heads
        self.Hv = linear_num_value_heads
        self.Kd = linear_key_head_dim
        self.Vd = linear_value_head_dim
        self.KeyDim = self.Hk * self.Kd
        self.ValueDim = self.Hv * self.Vd
        self.ConvK = linear_conv_kernel_dim
        self.ConvDim = self.KeyDim * 2 + self.ValueDim
        self.HeadRepeat = self.Hv // self.Hk

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (x,) = args

        # -- params ----------------------------------------------------------
        tracer.register_param("in_proj_qkv_weight", (self.ConvDim, "C"))
        tracer.register_param("in_proj_z_weight", (self.ValueDim, "C"))
        tracer.register_param("in_proj_b_weight", (self.Hv, "C"))
        tracer.register_param("in_proj_a_weight", (self.Hv, "C"))
        tracer.register_param(
            "conv_weight",
            (self.ConvDim, 1, self.ConvK),
            quantizable=False,
        )
        tracer.register_param("A_log", (self.Hv,), dtype="fp32", quantizable=False)
        tracer.register_param("dt_bias", (self.Hv,), dtype="fp32", quantizable=False)
        tracer.register_param("norm_weight", (self.Vd,), quantizable=False)
        out_proj_w = tracer.register_param("out_weight", ("C", self.ValueDim))

        # -- activation slots ------------------------------------------------
        out_slot = tracer.register_activation(
            "out",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="GatedDeltaNet mixer output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # QKV projection -> conv1d
        mixed_qkv_flat = g.matmul(
            x_flat,
            tracer.prefixed("in_proj_qkv_weight"),
            transpose="NT",
            out_name=tracer.prefixed("mixed_qkv_flat"),
        )
        mixed_qkv = g.view(
            mixed_qkv_flat,
            shape=[B, T, self.ConvDim],
            out_name=tracer.prefixed("mixed_qkv"),
        )
        mixed_qkv_cf = g.transpose(mixed_qkv, dim0=1, dim1=2)

        conv_weight_2d = g.view(
            tracer.prefixed("conv_weight"),
            shape=[self.ConvDim, self.ConvK],
            out_name=tracer.prefixed("conv_w2d"),
        )
        conv_out_cf = g.mamba_conv1d(
            mixed_qkv_cf,
            conv_weight_2d,
            None,
            activation="silu",
            out_name=tracer.prefixed("conv_out_cf"),
        )
        conv_out = g.transpose(conv_out_cf, dim0=1, dim1=2)

        # Split Q/K/V
        q_flat, k_flat, v_flat = g.split(
            conv_out,
            split_size=[self.KeyDim, self.KeyDim, self.ValueDim],
            dim=2,
        )
        query = g.view(
            q_flat,
            shape=[B, T, self.Hk, self.Kd],
            out_name=tracer.prefixed("query"),
        )
        key = g.view(
            k_flat,
            shape=[B, T, self.Hk, self.Kd],
            out_name=tracer.prefixed("key"),
        )
        value = g.view(
            v_flat,
            shape=[B, T, self.Hv, self.Vd],
            out_name=tracer.prefixed("value"),
        )

        # Z gate
        z_flat = g.matmul(
            x_flat,
            tracer.prefixed("in_proj_z_weight"),
            transpose="NT",
            out_name=tracer.prefixed("z_flat"),
        )
        z = g.view(z_flat, shape=[B, T, self.Hv, self.Vd], out_name=tracer.prefixed("z"))

        # Beta (sigmoid)
        b_flat = g.matmul(
            x_flat,
            tracer.prefixed("in_proj_b_weight"),
            transpose="NT",
            out_name=tracer.prefixed("b_flat"),
        )
        b = g.view(b_flat, shape=[B, T, self.Hv], out_name=tracer.prefixed("b"))
        beta = g.sigmoid(b)

        # Decay
        a_flat = g.matmul(
            x_flat,
            tracer.prefixed("in_proj_a_weight"),
            transpose="NT",
            out_name=tracer.prefixed("a_flat"),
        )
        a = g.view(a_flat, shape=[B, T, self.Hv], out_name=tracer.prefixed("a"))
        g_decay = g.qwen3_5_decay(
            a,
            tracer.prefixed("A_log"),
            tracer.prefixed("dt_bias"),
            out_name=tracer.prefixed("decay"),
        )

        # Repeat heads if needed
        if self.HeadRepeat > 1:
            query = g.repeat_interleave_heads(
                query,
                repeats=self.HeadRepeat,
                out_name=tracer.prefixed("query_rep"),
            )
            key = g.repeat_interleave_heads(
                key,
                repeats=self.HeadRepeat,
                out_name=tracer.prefixed("key_rep"),
            )

        # Chunked gated delta rule
        core_attn_out, _ = g.custom(
            "chunk_gated_delta_rule",
            query,
            key,
            value,
            g_decay,
            beta,
            num_outputs=2,
            scale=0.0,
            chunk_size=self.chunk_size,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Gated RMSNorm
        core_flat = g.view(
            core_attn_out,
            shape=[B * T * self.Hv, self.Vd],
            out_name=tracer.prefixed("core_flat"),
        )
        z_norm_flat = g.view(
            z,
            shape=[B * T * self.Hv, self.Vd],
            out_name=tracer.prefixed("z_norm_flat"),
        )
        gated_flat = g.mamba_gated_rmsnorm(
            core_flat,
            z_norm_flat,
            tracer.prefixed("norm_weight"),
            eps=self.eps,
            n_groups=1,
            norm_before_gate=True,
            out_name=tracer.prefixed("gated_flat"),
        )
        gated = g.view(
            gated_flat,
            shape=[B, T, self.ValueDim],
            out_name=tracer.prefixed("gated"),
        )

        # Output projection
        gated_bt_flat = g.view(
            gated,
            shape=[B * T, self.ValueDim],
            out_name=tracer.prefixed("gated_bt_flat"),
        )
        out_flat = g.matmul(
            gated_bt_flat,
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


# Backwards-compat alias.
ChunkGatedDeltaRule = GatedDeltaNetMixer


# Self-register as an ``AttentionSpec`` kind for discovery.
from ..attention import AttentionSpec, _register as _register_attention  # noqa: E402

_register_attention(AttentionSpec(name="gated_delta_net", factory=GatedDeltaNetMixer))
