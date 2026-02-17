"""NemotronH Hybrid Architecture Blocks.

Nemotron-H uses a hybrid architecture with interleaved block types:
- M = Mamba2 block (State Space Model)
- * = Attention block (GQA)
- - = MLP block (dense feed-forward)

Each block has the structure:
    residual, x = fused_residual_rmsnorm(residual, x)
    x = mixer(x)  # mixer depends on block type
    # residual connection handled in next block's norm
"""

from __future__ import annotations

import math

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import B, T


@block
class NemotronHMamba2Block:
    """Mamba2 block for Nemotron-H hybrid architecture.

    Implements the Mamba2 SSM mixer with:
    - Input projection (gate, hidden_states, B, C, dt)
    - Causal 1D convolution
    - State Space Model scan
    - Gated RMSNorm
    - Output projection
    """

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
    ):
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
        dt_max_default = 1e9
        if time_step_limit is None:
            time_step_limit = (0.0, dt_max_default)
        elif isinstance(time_step_limit, (list, tuple)) and len(time_step_limit) == 2:
            lo = float(time_step_limit[0])
            hi = float(time_step_limit[1])
            if not math.isfinite(lo):
                lo = 0.0
            if not math.isfinite(hi):
                hi = dt_max_default
            time_step_limit = (lo, hi)
        self.time_step_limit = time_step_limit
        self.use_conv_bias = use_conv_bias
        self.use_bias = use_bias

        # Derived dimensions - compute actual values for shape resolution
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.conv_dim = self.intermediate_size + 2 * n_groups * ssm_state_size
        self.projection_size = self.intermediate_size + self.conv_dim + mamba_num_heads

        # Dimension aliases for Param shape annotations
        # These must be set to actual integer values for DSL shape resolution
        self.C = d_model
        self.I = self.intermediate_size
        self.H = mamba_num_heads
        self.D = mamba_head_dim
        self.N = ssm_state_size
        self.G = n_groups
        self.K = conv_kernel
        self.P = self.projection_size
        self.D_conv = self.conv_dim

    # Pre-block normalization (never quantized)
    norm_weight = Param(Tensor["C"], quantizable=False)

    # Input projection
    in_proj_weight = Param(Tensor["P", "C"])
    in_proj_bias = Param(Tensor["P"], when="use_bias")

    # Convolution
    conv_weight = Param(Tensor["D_conv", "K"], quantizable=False)
    conv_bias = Param(Tensor["D_conv"], when="use_conv_bias", quantizable=False)

    # SSM parameters (must be FP32 — C++ kernels use .get<float>())
    # Never quantized: tiny 1D tensors where quantization error is fatal
    A_log = Param(Tensor["H", "fp32"], frozen=False, quantizable=False)
    D_param = Param(Tensor["H", "fp32"], quantizable=False)
    dt_bias = Param(Tensor["H", "fp32"], quantizable=False)

    # Gated RMSNorm (never quantized — 1D norm weight)
    gated_norm_weight = Param(Tensor["I"], quantizable=False)

    # Output projection
    out_proj_weight = Param(Tensor["C", "I"])
    out_proj_bias = Param(Tensor["C"], when="use_bias")

    # =========================================================================
    # Activation slots
    # =========================================================================

    # Pre-norm
    ln = Activation(
        Tensor["B", "T", "C"],
        recompute=True,
        recompute_from=["res_in", "ln_rstd", "@param:norm_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="always",
        share_policy="when_recomputed",
    )
    ln_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
    )

    # Input projection output
    projected = Activation(
        Tensor["B", "T", "P"],
        save=True,
        recompute=True,
        recompute_from=["ln", "@param:in_proj_weight"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        share_policy="fft_share",
    )

    # Split projection: gate, conv_input, dt
    gate = Activation(
        Tensor["B", "T", "I"],
        save=True,
        share_policy="fft_share",
    )
    conv_input = Activation(
        Tensor["B", "D_conv", "T"],
        save=True,
        share_policy="fft_share",
    )
    dt = Activation(
        Tensor["B", "I", "T"],
        save=True,
        share_policy="fft_share",
    )

    # Conv output
    conv_out = Activation(
        Tensor["B", "D_conv", "T"],
        save=True,
        recompute=True,
        recompute_from=["conv_input", "@param:conv_weight", "?@param:conv_bias"],
        recompute_op="mamba_conv1d",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )

    # Split conv output: hidden_states, B, C
    hidden_states = Activation(
        Tensor["B", "I", "T"],
        save=True,
        share_policy="fft_share",
    )
    ssm_B = Activation(
        Tensor["B", "G", "N", "T"],
        save=True,
        share_policy="fft_share",
    )
    ssm_C = Activation(
        Tensor["B", "G", "N", "T"],
        save=True,
        share_policy="fft_share",
    )

    # SSM scan output
    ssm_out = Activation(
        Tensor["B", "T", "I"],
        save=True,
        recompute=True,
        recompute_group="ssm_scan",
        recompute_outputs=["ssm_out", "ssm_state"],
        recompute_from=["hidden_states", "dt", "ssm_B", "ssm_C",
                        "@param:A_log", "@param:D_param", "@param:dt_bias"],
        recompute_op="mamba_ssm_scan",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    ssm_state = Activation(
        Tensor["B", "H", "D", "N"],
        save=True,
        recompute=True,
        recompute_group="ssm_scan",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Final SSM state for caching",
    )

    # Gated norm output
    gated_out = Activation(
        Tensor["B", "T", "I"],
        save=True,
        recompute=True,
        recompute_from=["ssm_out", "gate", "@param:gated_norm_weight"],
        recompute_op="mamba_gated_rmsnorm",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )

    # Output projection
    out = Activation(
        Tensor["B", "T", "C"],
        share_policy="per_layer",
        description="Block output",
    )

    # Residual (input to this block)
    res_in = Activation(
        Tensor["B", "T", "C"],
        share_policy="per_layer",
    )

    # =========================================================================
    # Gradient slots
    # =========================================================================

    d_ln = Gradient(Tensor["B", "T", "C"], gradient_of="ln")
    d_projected = Gradient(Tensor["B", "T", "P"], gradient_of="projected")
    d_gate = Gradient(Tensor["B", "T", "I"], gradient_of="gate")
    d_conv_out = Gradient(Tensor["B", "D_conv", "T"], gradient_of="conv_out")
    d_hidden_states = Gradient(Tensor["B", "I", "T"], gradient_of="hidden_states")
    d_ssm_B = Gradient(Tensor["B", "G", "N", "T"], gradient_of="ssm_B")
    d_ssm_C = Gradient(Tensor["B", "G", "N", "T"], gradient_of="ssm_C")
    d_ssm_out = Gradient(Tensor["B", "T", "I"], gradient_of="ssm_out")
    d_gated_out = Gradient(Tensor["B", "T", "I"], gradient_of="gated_out")
    d_out = Gradient(Tensor["B", "T", "C"], gradient_of="out")
    d_res_in = Gradient(Tensor["B", "T", "C"], gradient_of="res_in")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Forward pass. Returns (output, residual_out)."""
        with graph() as g:
            # Fused residual + RMSNorm
            res_in, ln, ln_rstd = g.fused_residual_rmsnorm(
                residual, x, "norm_weight", eps=self.eps,
                res_out_name="res_in",
                y_name="ln",
                rstd_name="ln_rstd",
            )

            # Input projection
            ln_flat = g.view(ln, shape=[B * T, self.C])
            if self.use_bias:
                projected_flat = g.matmul_bias(ln_flat, "in_proj_weight", "in_proj_bias", transpose="NT")
            else:
                projected_flat = g.matmul(ln_flat, "in_proj_weight", transpose="NT")
            projected = g.view(projected_flat, shape=[B, T, self.P], out_name="projected")

            # Split projection into gate, conv_input (hidden_states_B_C), dt
            gate, conv_input, dt = g.mamba_split_proj(
                projected,
                intermediate_size=self.intermediate_size,
                conv_dim=self.conv_dim,
                num_heads=self.mamba_num_heads,
                head_dim=self.mamba_head_dim,
                gate_name="gate",
                conv_input_name="conv_input",
                dt_name="dt",
            )

            # Causal 1D convolution
            if self.use_conv_bias:
                conv_out = g.mamba_conv1d(conv_input, "conv_weight", "conv_bias",
                                          activation="silu", out_name="conv_out")
            else:
                conv_out = g.mamba_conv1d(conv_input, "conv_weight", None,
                                          activation="silu", out_name="conv_out")

            # Split conv output into hidden_states, B, C
            hidden_states, ssm_B, ssm_C = g.mamba_split_conv_out(
                conv_out,
                intermediate_size=self.intermediate_size,
                groups_state_size=self.n_groups * self.ssm_state_size,
                n_groups=self.n_groups,
                ssm_state_size=self.ssm_state_size,
                hidden_name="hidden_states",
                B_name="ssm_B",
                C_name="ssm_C",
            )

            # SSM scan
            ssm_out, ssm_state = g.mamba_ssm_scan(
                hidden_states, dt, "A_log", ssm_B, ssm_C, "D_param",
                dt_bias="dt_bias",
                dt_softplus=True,
                dt_min=self.time_step_limit[0],
                dt_max=self.time_step_limit[1],
                chunk_size=self.chunk_size,
                num_heads=self.mamba_num_heads,
                head_dim=self.mamba_head_dim,
                ssm_state_size=self.ssm_state_size,
                n_groups=self.n_groups,
                out_name="ssm_out",
                state_name="ssm_state",
            )

            # Reshape SSM output for gated norm
            ssm_out_flat = g.view(ssm_out, shape=[B, T, self.I])

            # Gated RMSNorm
            gated_out = g.mamba_gated_rmsnorm(
                ssm_out_flat, gate, "gated_norm_weight",
                eps=self.eps,
                n_groups=self.n_groups,
                out_name="gated_out",
            )

            # Output projection
            gated_flat = g.view(gated_out, shape=[B * T, self.I])
            if self.use_bias:
                out_flat = g.matmul_bias(gated_flat, "out_proj_weight", "out_proj_bias", transpose="NT")
            else:
                out_flat = g.matmul(gated_flat, "out_proj_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="out")

            return out, res_in


@block
class NemotronHAttentionBlock:
    """Attention block for Nemotron-H hybrid architecture.

    Standard GQA attention with pre-norm:
    - Pre-norm (RMSNorm)
    - Q, K, V projections
    - FlashAttention
    - Output projection
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq: int = 4096,
        eps: float = 1e-5,
        attention_bias: bool = False,
        use_rope: bool = False,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.eps = eps
        self.attention_bias = attention_bias
        self.use_rope = use_rope

        # Dimension aliases for Param shape annotations
        # Set to actual integer values for DSL shape resolution
        self.C = d_model
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.D = head_dim
        self.MaxSeq = max_seq

        # Derived dimensions - computed integer values
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_dim
        self.AttnDim = num_query_heads * head_dim

    # Pre-block normalization (never quantized)
    norm_weight = Param(Tensor["C"], quantizable=False)

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="attention_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    out_bias = Param(Tensor["C"], when="attention_bias")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True, when="use_rope")

    # =========================================================================
    # Activation slots
    # =========================================================================
    # NOTE: Use standard names (ln1, ln1_rstd, res_att) so the LoRA backward
    # hooks can find them via simplified_acts/simplified_grads fields.

    ln1 = Activation(
        Tensor["B", "T", "C"],
        recompute=True,
        recompute_from=["res_att", "ln1_rstd", "@param:norm_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="always",
        share_policy="when_recomputed",
    )
    ln1_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
    )

    qkv = Activation(
        Tensor["B", "T", "QKV"],
        aliases=["qkv_flat"],
        save=True,
        recompute=False,
        recompute_from=["ln1", "@param:qkv_weight", "?@param:qkv_bias"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["q", "k", "v"],
        share_policy="when_recomputed",
    )
    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        recompute=True,
        recompute_from=["qkv", "@global:freq_cis", "@input:position_ids"],
        recompute_op="rope",
        recompute_policy="fft_only",
        share_policy="when_recomputed",
        when="use_rope",
    )

    att = Activation(
        Tensor["B", "T", "AttnDim"],
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_outputs=["att", "lse"],
        recompute_from=["qkv_rope"],
        recompute_op="flash_attention",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        recompute=True,
        recompute_from=["att", "@param:out_weight"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["o"],
        share_policy="fft_share",
    )

    res_att = Activation(
        Tensor["B", "T", "C"],
        share_policy="per_layer",
    )

    # =========================================================================
    # Gradient slots
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_rope = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_rope", when="use_rope")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_att_out = Gradient(Tensor["B", "T", "C"], gradient_of="att_out")
    d_res_att = Gradient(Tensor["B", "T", "C"], gradient_of="res_att")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Forward pass. Returns (output, residual_out)."""
        with graph() as g:
            # Fused residual + RMSNorm
            res_att, ln1, ln1_rstd = g.fused_residual_rmsnorm(
                residual, x, "norm_weight", eps=self.eps,
                res_out_name="res_att",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            # QKV projection
            ln1_flat = g.view(ln1, shape=[B * T, self.C])
            if self.attention_bias:
                qkv_flat = g.matmul_bias(ln1_flat, "qkv_weight", "qkv_bias",
                                         transpose="NT", out_name="qkv_flat")
            else:
                qkv_flat = g.matmul(ln1_flat, "qkv_weight",
                                    transpose="NT", out_name="qkv_flat")
            # Keep packed QKV as [B, T, QKV] to align LoRA hooks and flash_attention expectations.
            qkv = g.view(qkv_flat, shape=[B, T, self.QKV], out_name="qkv")

            # RoPE (optional — Nemotron-H attention does not use positional encoding)
            if self.use_rope:
                attn_input = g.rope(qkv, "rope_freqs", position_ids, rotary_dim=self.head_dim, out_name="qkv_rope")
            else:
                attn_input = qkv

            # FlashAttention
            att, lse = g.flash_attention(attn_input, causal=True, out_name="att", lse_name="lse")

            # Output projection
            att_flat = g.view(att, shape=[B * T, self.AttnDim])
            if self.attention_bias:
                att_out_flat = g.matmul_bias(att_flat, "out_weight", "out_bias",
                                             transpose="NT", out_name="att_out_flat")
            else:
                att_out_flat = g.matmul(att_flat, "out_weight",
                                        transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=[B, T, self.C], out_name="att_out")

            return att_out, res_att


@block
class NemotronHMLPBlock:
    """MLP block for Nemotron-H hybrid architecture.

    Simple feed-forward block with pre-norm:
    - Pre-norm (RMSNorm)
    - Up projection
    - Activation (relu2 by default for Nemotron)
    - Down projection

    Note: Unlike standard transformers, this block has NO attention.
    It's used for "-" pattern in the hybrid_override_pattern.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        eps: float = 1e-5,
        activation: str = "relu2",
        mlp_bias: bool = False,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.eps = eps
        self.activation = activation
        self.mlp_bias = mlp_bias

        # Dimension aliases for Param shape annotations
        # Set to actual integer values for DSL shape resolution
        self.C = d_model
        self.M = d_ff

    # Pre-block normalization (never quantized)
    norm_weight = Param(Tensor["C"], quantizable=False)

    # MLP weights
    up_weight = Param(Tensor["M", "C"])
    up_bias = Param(Tensor["M"], when="mlp_bias")
    down_weight = Param(Tensor["C", "M"])
    down_bias = Param(Tensor["C"], when="mlp_bias")

    # =========================================================================
    # Activation slots
    # =========================================================================

    ln = Activation(
        Tensor["B", "T", "C"],
        recompute=True,
        recompute_from=["res_in", "ln_rstd", "@param:norm_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="always",
        share_policy="when_recomputed",
    )
    ln_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
    )

    mlp_up = Activation(
        Tensor["B", "T", "M"],
        save=True,
        recompute=True,
        recompute_from=["ln", "@param:up_weight", "?@param:up_bias"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="always",
        lora_targets=["up"],
        share_policy="when_recomputed",
    )
    swiglu = Activation(
        Tensor["B", "T", "M"],
        save=True,
        recompute=True,
        recompute_from=["mlp_up"],
        recompute_op="relu2",  # Default for Nemotron
        recompute_policy="always",
        share_policy="when_recomputed",
    )
    mlp_down = Activation(
        Tensor["B", "T", "C"],
        recompute=True,
        recompute_from=["swiglu", "@param:down_weight"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="always",
        lora_targets=["down"],
        share_policy="when_recomputed",
    )

    out = Activation(
        Tensor["B", "T", "C"],
        share_policy="per_layer",
    )
    res_in = Activation(
        Tensor["B", "T", "C"],
        share_policy="per_layer",
    )

    # =========================================================================
    # Gradient slots
    # =========================================================================

    d_ln = Gradient(Tensor["B", "T", "C"], gradient_of="ln")
    d_mlp_up = Gradient(Tensor["B", "T", "M"], gradient_of="mlp_up")
    d_swiglu = Gradient(Tensor["B", "T", "M"], gradient_of="swiglu")
    d_mlp_down = Gradient(Tensor["B", "T", "C"], gradient_of="mlp_down")
    d_out = Gradient(Tensor["B", "T", "C"], gradient_of="out")
    d_res_in = Gradient(Tensor["B", "T", "C"], gradient_of="res_in")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Forward pass. Returns (output, residual_out)."""
        with graph() as g:
            # Fused residual + RMSNorm
            res_in, ln, ln_rstd = g.fused_residual_rmsnorm(
                residual, x, "norm_weight", eps=self.eps,
                res_out_name="res_in",
                y_name="ln",
                rstd_name="ln_rstd",
            )

            # MLP
            ln_flat = g.view(ln, shape=[B * T, self.C])
            if self.mlp_bias:
                up_flat = g.matmul_bias(ln_flat, "up_weight", "up_bias", transpose="NT")
            else:
                up_flat = g.matmul(ln_flat, "up_weight", transpose="NT")
            mlp_up = g.view(up_flat, shape=[B, T, self.M], out_name="mlp_up")

            # Activation (relu2 for Nemotron, configurable)
            if self.activation == "relu2":
                swiglu = g.relu2(mlp_up, out_name="swiglu")
            elif self.activation == "silu":
                swiglu = g.silu(mlp_up, out_name="swiglu")
            elif self.activation == "gelu":
                swiglu = g.gelu(mlp_up, out_name="swiglu")
            else:
                swiglu = g.relu2(mlp_up, out_name="swiglu")  # Default

            # Down projection
            swiglu_flat = g.view(swiglu, shape=[B * T, self.M])
            if self.mlp_bias:
                out_flat = g.matmul_bias(swiglu_flat, "down_weight", "down_bias", transpose="NT")
            else:
                out_flat = g.matmul(swiglu_flat, "down_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="out")

            return out, res_in


@block
class NemotronHMoEBlock:
    """MoE block for Nemotron-H hybrid architecture (optional).

    Mixture of Experts with:
    - Pre-norm (RMSNorm)
    - Router (sigmoid-based for Nemotron-H)
    - Routed experts
    - Shared expert
    """

    def __init__(
        self,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        shared_expert_intermediate_size: int = 0,
        eps: float = 1e-5,
        mlp_bias: bool = False,
        activation: str = "relu2",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        ep_size: int = 1,
    ):
        self.d_model = d_model
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.use_shared_expert = shared_expert_intermediate_size > 0
        self.eps = eps
        self.mlp_bias = mlp_bias
        self.activation = activation
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size
        self.num_local_experts = num_experts // ep_size if ep_size > 1 else num_experts

        # Dimension aliases for Param shape annotations
        # Set to actual integer values for DSL shape resolution
        self.C = d_model
        self.M = moe_intermediate_size
        self.E = num_experts
        self.K = num_experts_per_tok
        self.SharedM = shared_expert_intermediate_size if shared_expert_intermediate_size > 0 else moe_intermediate_size

    # Pre-block normalization (never quantized)
    norm_weight = Param(Tensor["C"], quantizable=False)

    # Router
    router_weight = Param(Tensor["E", "C"], quantizable=False)
    e_score_correction_bias = Param(Tensor["E", "fp32"], quantizable=False)

    # Experts (batched format)
    # offload_group="moe_experts" signals the runtime to store these on CPU
    # and stream to GPU on demand when expert offloading is enabled.
    experts_up = Param(Tensor["E", "M", "C"], offload_group="moe_experts")
    experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")

    # Shared expert (optional)
    shared_expert_up = Param(Tensor["SharedM", "C"], when="use_shared_expert")
    shared_expert_down = Param(Tensor["C", "SharedM"], when="use_shared_expert")

    # =========================================================================
    # Activation slots (simplified - full MoE would have more)
    # =========================================================================

    ln = Activation(Tensor["B", "T", "C"])
    ln_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)

    router_logits = Activation(Tensor["B * T", "E"], dtype="fp32", save=True)
    router_probs = Activation(Tensor["B * T", "E"], dtype="fp32", save=True)
    routing_weights = Activation(Tensor["B * T", "K"], dtype="fp32", save=True)
    routing_indices = Activation(Tensor["B * T", "K"], dtype="int32", save=True)

    permuted_input = Activation(Tensor["B * T * K", "C"], save=True)
    scatter_indices = Activation(Tensor["B * T * K"], dtype="int32", save=True)

    # EP dispatch outputs (only present when ep_size > 1)
    ep_recv_input = Activation(
        Tensor["B * T * K", "C"], save=True,
        when=lambda self: self.ep_size > 1,
        description="EP-dispatched input tokens (post-A2A)",
    )
    ep_recv_scatter = Activation(
        Tensor["B * T * K"], dtype="int32", save=True,
        when=lambda self: self.ep_size > 1,
        description="EP-dispatched scatter indices (post-A2A)",
    )

    expert_up = Activation(Tensor["B * T * K", "M"], save=True)
    expert_act = Activation(Tensor["B * T * K", "M"], save=True)
    expert_down = Activation(Tensor["B * T * K", "C"], save=True)
    ep_combined = Activation(
        Tensor["B * T * K", "C"], save=True,
        when=lambda self: self.ep_size > 1,
        description="EP-combined expert output (post reverse-A2A)",
    )
    moe_out = Activation(Tensor["B * T", "C"], save=True)

    # Shared expert activations
    shared_up_out = Activation(Tensor["B * T", "SharedM"], when="use_shared_expert")
    shared_act = Activation(Tensor["B * T", "SharedM"], when="use_shared_expert")
    shared_out = Activation(Tensor["B * T", "C"], when="use_shared_expert")

    out = Activation(Tensor["B", "T", "C"])
    res_in = Activation(Tensor["B", "T", "C"])

    # =========================================================================
    # Gradient slots
    # =========================================================================

    d_ln = Gradient(Tensor["B", "T", "C"], gradient_of="ln")
    d_router_logits = Gradient(Tensor["B * T", "E"], dtype="fp32", gradient_of="router_logits")
    d_routing_weights = Gradient(Tensor["B * T", "K"], dtype="fp32", gradient_of="routing_weights")
    d_ep_recv_input = Gradient(Tensor["B * T * K", "C"], gradient_of="ep_recv_input",
                               when=lambda self: self.ep_size > 1)
    d_permuted_input = Gradient(Tensor["B * T * K", "C"], gradient_of="permuted_input")
    d_expert_up = Gradient(Tensor["B * T * K", "M"], gradient_of="expert_up")
    d_expert_act = Gradient(Tensor["B * T * K", "M"], gradient_of="expert_act")
    d_expert_down = Gradient(Tensor["B * T * K", "C"], gradient_of="expert_down")
    d_moe_out = Gradient(Tensor["B * T", "C"], gradient_of="moe_out")
    d_out = Gradient(Tensor["B", "T", "C"], gradient_of="out")
    d_res_in = Gradient(Tensor["B", "T", "C"], gradient_of="res_in")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Forward pass. Returns (output, residual_out)."""
        with graph() as g:
            # Fused residual + RMSNorm
            res_in, ln, ln_rstd = g.fused_residual_rmsnorm(
                residual, x, "norm_weight", eps=self.eps,
                res_out_name="res_in",
                y_name="ln",
                rstd_name="ln_rstd",
            )

            # Router
            ln_flat = g.view(ln, shape=[B * T, self.C], out_name="ln_flat")
            router_logits = g.matmul(ln_flat, "router_weight", transpose="NT", out_name="router_logits")

            # Nemotron-H uses sigmoid routing
            router_probs = g.moe_sigmoid(router_logits, out_name="router_probs")

            # Top-k selection with optional scaling factor and correction bias
            routing_weights, routing_indices = g.moe_topk(
                router_probs, top_k=self.num_experts_per_tok, normalize=self.norm_topk_prob,
                scaling_factor=self.routed_scaling_factor,
                correction_bias="e_score_correction_bias",
                weights_name="routing_weights", indices_name="routing_indices",
            )

            # Permute for grouped expert computation
            permuted_input, scatter_indices = g.moe_permute(
                ln_flat, routing_indices, top_k=self.num_experts_per_tok,
                out_name="permuted_input", scatter_name="scatter_indices",
            )

            # EP dispatch: route tokens to expert-owning GPUs (no-op when ep_size=1)
            if self.ep_size > 1:
                ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                    permuted_input, routing_indices, scatter_indices,
                    num_experts=self.num_experts, ep_size=self.ep_size,
                    top_k=self.num_experts_per_tok,
                    out_name="ep_recv_input", recv_scatter_name="ep_recv_scatter",
                )
                gemm_input = ep_recv_input
                gemm_scatter = ep_recv_scatter
            else:
                gemm_input = permuted_input
                gemm_scatter = scatter_indices

            # Expert computation (simple up + activation + down)
            expert_up = g.moe_grouped_gemm(
                gemm_input, "experts_up", gemm_scatter,
            )

            # Activation
            if self.activation == "relu2":
                expert_act = g.relu2(expert_up)
            else:
                expert_act = g.silu(expert_up)

            expert_down = g.moe_grouped_gemm_down(
                expert_act, "experts_down", gemm_scatter,
            )

            # EP combine: route expert outputs back (no-op when ep_size=1)
            if self.ep_size > 1:
                expert_down = g.ep_combine(
                    expert_down,
                    num_experts=self.num_experts, ep_size=self.ep_size,
                    top_k=self.num_experts_per_tok,
                    out_name="ep_combined",
                )

            # Unpermute and combine
            moe_out = g.moe_unpermute(
                expert_down, routing_weights, scatter_indices, top_k=self.num_experts_per_tok,
                out_name="moe_out",
            )

            # Shared expert (if enabled)
            if self.use_shared_expert:
                shared_up_out = g.matmul(ln_flat, "shared_expert_up", transpose="NT", out_name="shared_up_out")
                if self.activation == "relu2":
                    shared_act = g.relu2(shared_up_out, out_name="shared_act")
                else:
                    shared_act = g.silu(shared_up_out, out_name="shared_act")
                shared_out = g.matmul(shared_act, "shared_expert_down", transpose="NT", out_name="shared_out")
                moe_out = g.add(moe_out, shared_out, out_name="moe_combined")

            # Reshape back
            out = g.view(moe_out, shape=[B, T, self.C], out_name="out")

            return out, res_in
