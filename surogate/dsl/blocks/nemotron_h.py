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

    # Pre-block normalization
    norm_weight = Param(Tensor["C"])

    # Input projection
    in_proj_weight = Param(Tensor["P", "C"])
    in_proj_bias = Param(Tensor["P"], when="use_bias")

    # Convolution
    conv_weight = Param(Tensor["D_conv", "K"])
    conv_bias = Param(Tensor["D_conv"], when="use_conv_bias")

    # SSM parameters
    A_log = Param(Tensor["H"], frozen=False)  # Log of state decay
    D_param = Param(Tensor["H"])              # Skip connection
    dt_bias = Param(Tensor["H"])              # Time step bias

    # Gated RMSNorm
    gated_norm_weight = Param(Tensor["I"])

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
        Tensor["B", "T", "D_conv"],
        save=True,
        share_policy="fft_share",
    )
    dt = Activation(
        Tensor["B", "T", "H"],
        save=True,
        share_policy="fft_share",
    )

    # Conv output
    conv_out = Activation(
        Tensor["B", "T", "D_conv"],
        save=True,
        recompute=True,
        recompute_from=["conv_input", "@param:conv_weight", "?@param:conv_bias"],
        recompute_op="mamba_conv1d",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )

    # Split conv output: hidden_states, B, C
    hidden_states = Activation(
        Tensor["B", "T", "I"],
        save=True,
        share_policy="fft_share",
    )
    ssm_B = Activation(
        Tensor["B", "T", "G", "N"],
        save=True,
        share_policy="fft_share",
    )
    ssm_C = Activation(
        Tensor["B", "T", "G", "N"],
        save=True,
        share_policy="fft_share",
    )

    # SSM scan output
    ssm_out = Activation(
        Tensor["B", "T", "H", "D"],
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
    d_conv_out = Gradient(Tensor["B", "T", "D_conv"], gradient_of="conv_out")
    d_hidden_states = Gradient(Tensor["B", "T", "I"], gradient_of="hidden_states")
    d_ssm_B = Gradient(Tensor["B", "T", "G", "N"], gradient_of="ssm_B")
    d_ssm_C = Gradient(Tensor["B", "T", "G", "N"], gradient_of="ssm_C")
    d_ssm_out = Gradient(Tensor["B", "T", "H", "D"], gradient_of="ssm_out")
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
                hidden_name="hidden_states",
                B_name="ssm_B",
                C_name="ssm_C",
            )

            # Reshape for SSM scan
            hidden_states_4d = g.view(hidden_states, shape=[B, T, self.H, self.D])
            ssm_B_4d = g.view(ssm_B, shape=[B, T, self.G, self.N])
            ssm_C_4d = g.view(ssm_C, shape=[B, T, self.G, self.N])

            # SSM scan
            ssm_out, ssm_state = g.mamba_ssm_scan(
                hidden_states_4d, dt, "A_log", ssm_B_4d, ssm_C_4d, "D_param",
                dt_bias="dt_bias",
                dt_softplus=True,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                chunk_size=self.chunk_size,
                out_name="ssm_out",
                state_name="ssm_state",
            )

            # Reshape SSM output for gated norm
            ssm_out_flat = g.view(ssm_out, shape=[B, T, self.I])

            # Gated RMSNorm
            gated_out = g.mamba_gated_rmsnorm(
                ssm_out_flat, gate, "gated_norm_weight",
                eps=self.eps,
                group_size=self.intermediate_size // self.n_groups,
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
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.eps = eps
        self.attention_bias = attention_bias

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

    # Pre-block normalization
    norm_weight = Param(Tensor["C"])

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="attention_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    out_bias = Param(Tensor["C"], when="attention_bias")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

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

    qkv = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        recompute=True,
        recompute_from=["ln", "@param:qkv_weight", "?@param:qkv_bias"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["q", "k", "v"],
        share_policy="fft_share",
    )
    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        recompute=True,
        recompute_from=["qkv", "@global:freq_cis", "@input:position_ids"],
        recompute_op="rope",
        recompute_policy="fft_only",
        share_policy="fft_share",
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
        recompute=True,
        recompute_from=["att", "@param:out_weight"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["o"],
        share_policy="fft_share",
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
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_rope = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_rope")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_att_out = Gradient(Tensor["B", "T", "C"], gradient_of="att_out")
    d_out = Gradient(Tensor["B", "T", "C"], gradient_of="out")
    d_res_in = Gradient(Tensor["B", "T", "C"], gradient_of="res_in")

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
            res_in, ln, ln_rstd = g.fused_residual_rmsnorm(
                residual, x, "norm_weight", eps=self.eps,
                res_out_name="res_in",
                y_name="ln",
                rstd_name="ln_rstd",
            )

            # QKV projection
            ln_flat = g.view(ln, shape=[B * T, self.C])
            if self.attention_bias:
                qkv_flat = g.matmul_bias(ln_flat, "qkv_weight", "qkv_bias", transpose="NT")
            else:
                qkv_flat = g.matmul(ln_flat, "qkv_weight", transpose="NT")
            qkv = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D], out_name="qkv")

            # RoPE
            qkv_rope = g.rope(qkv, "rope_freqs", position_ids, rotary_dim=self.head_dim, out_name="qkv_rope")

            # FlashAttention
            att, lse = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            # Output projection
            att_flat = g.view(att, shape=[B * T, self.AttnDim])
            if self.attention_bias:
                out_flat = g.matmul_bias(att_flat, "out_weight", "out_bias", transpose="NT")
            else:
                out_flat = g.matmul(att_flat, "out_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="out")

            return out, res_in


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

    # Pre-block normalization
    norm_weight = Param(Tensor["C"])

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
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    mlp_act = Activation(
        Tensor["B", "T", "M"],
        save=True,
        recompute=True,
        recompute_from=["mlp_up"],
        recompute_op="relu2",  # Default for Nemotron
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    mlp_down = Activation(
        Tensor["B", "T", "C"],
        recompute=True,
        recompute_from=["mlp_act", "@param:down_weight"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        share_policy="fft_share",
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
    d_mlp_act = Gradient(Tensor["B", "T", "M"], gradient_of="mlp_act")
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
                mlp_act = g.relu2(mlp_up, out_name="mlp_act")
            elif self.activation == "silu":
                mlp_act = g.silu(mlp_up, out_name="mlp_act")
            elif self.activation == "gelu":
                mlp_act = g.gelu(mlp_up, out_name="mlp_act")
            else:
                mlp_act = g.relu2(mlp_up, out_name="mlp_act")  # Default

            # Down projection
            mlp_act_flat = g.view(mlp_act, shape=[B * T, self.M])
            if self.mlp_bias:
                out_flat = g.matmul_bias(mlp_act_flat, "down_weight", "down_bias", transpose="NT")
            else:
                out_flat = g.matmul(mlp_act_flat, "down_weight", transpose="NT")
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

        # Dimension aliases for Param shape annotations
        # Set to actual integer values for DSL shape resolution
        self.C = d_model
        self.M = moe_intermediate_size
        self.E = num_experts
        self.K = num_experts_per_tok
        self.SharedM = shared_expert_intermediate_size if shared_expert_intermediate_size > 0 else moe_intermediate_size

    # Pre-block normalization
    norm_weight = Param(Tensor["C"])

    # Router
    router_weight = Param(Tensor["E", "C"])

    # Experts (batched format)
    experts_up = Param(Tensor["E", "M", "C"])
    experts_down = Param(Tensor["E", "C", "M"])

    # Shared expert (optional)
    shared_expert_up = Param(Tensor["SharedM", "C"], when="use_shared_expert")
    shared_expert_down = Param(Tensor["C", "SharedM"], when="use_shared_expert")

    # =========================================================================
    # Activation slots (simplified - full MoE would have more)
    # =========================================================================

    ln = Activation(Tensor["B", "T", "C"])
    ln_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)

    router_logits = Activation(Tensor["B * T", "E"], save=True)
    router_probs = Activation(Tensor["B * T", "E"], save=True)
    routing_weights = Activation(Tensor["B * T", "K"], save=True)
    routing_indices = Activation(Tensor["B * T", "K"], dtype="int32", save=True)

    permuted_input = Activation(Tensor["B * T * K", "C"], save=True)
    scatter_indices = Activation(Tensor["B * T * K"], dtype="int32", save=True)

    expert_up = Activation(Tensor["B * T * K", "M"], save=True)
    expert_act = Activation(Tensor["B * T * K", "M"], save=True)
    expert_down = Activation(Tensor["B * T * K", "C"], save=True)
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
    d_router_logits = Gradient(Tensor["B * T", "E"], gradient_of="router_logits")
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

            # Top-k selection
            routing_weights, routing_indices = g.moe_topk(
                router_probs, top_k=self.num_experts_per_tok, normalize=self.norm_topk_prob,
                weights_name="routing_weights", indices_name="routing_indices",
            )

            # Permute for grouped expert computation
            permuted_input, scatter_indices = g.moe_permute(
                ln_flat, routing_indices, top_k=self.num_experts_per_tok,
                out_name="permuted_input", scatter_name="scatter_indices",
            )

            # Expert computation (simple up + activation + down)
            expert_up = g.moe_grouped_gemm(
                permuted_input, "experts_up", scatter_indices,
            )

            # Activation
            if self.activation == "relu2":
                expert_act = g.relu2(expert_up)
            else:
                expert_act = g.silu(expert_up)

            expert_down = g.moe_grouped_gemm_down(
                expert_act, "experts_down", scatter_indices,
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
                moe_out = g.add(moe_out, shared_out, out_name="moe_out")

            # Reshape back
            out = g.view(moe_out, shape=[B, T, self.C], out_name="out")

            return out, res_in
