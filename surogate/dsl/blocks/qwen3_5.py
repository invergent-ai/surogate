"""Qwen3.5 dense transformer blocks."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T


def _resolve_rotary_dim(head_size: int, partial_rotary_factor: float) -> int:
    rotary = int(round(float(head_size) * float(partial_rotary_factor)))
    rotary = max(2, min(rotary, head_size))
    if rotary % 2 != 0:
        rotary -= 1
    return max(2, rotary)


@block
class Qwen3_5AttentionBlock:
    """Qwen3.5 full-attention decoder block (token mixer + MLP)."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
    ):
        self.d_model = d_model  # Hidden dimension of the model
        self.num_query_heads = num_query_heads  # Number of query attention heads
        self.num_kv_heads = num_kv_heads  # Number of key/value heads (GQA); may be < num_query_heads
        self.head_size = head_size  # Feature dimension per attention head (D)
        self.d_ff = d_ff  # Feed-forward intermediate dimension (before SwiGLU split)
        self.max_seq = max_seq  # Maximum sequence length for RoPE frequency cache
        self.eps = eps  # Epsilon for RMSNorm numerical stability
        self.use_qkv_bias = use_qkv_bias  # Whether Q/K/V/out projections include bias terms
        self.partial_rotary_factor = partial_rotary_factor  # Fraction of head_size to apply RoPE to
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)  # MRoPE split sizes for (temporal, height, width) axes

        self.C = Dim("C")  # Hidden / channel dimension (= d_model)
        self.Hq = Dim("Hq")  # Number of query heads (= num_query_heads)
        self.Hkv = Dim("Hkv")  # Number of KV heads (= num_kv_heads)
        self.D = Dim("D")  # Per-head feature dimension (= head_size)
        self.M = Dim("M")  # MLP intermediate dimension after SwiGLU split (= d_ff // 2)
        self.MaxSeq = Dim("MaxSeq")  # Max sequence length (= max_seq)
        self.AttnDim = self.Hq * self.D  # Total attention output dim: Hq * D
        self.QProjDim = 2 * self.AttnDim  # Q projection includes a gating channel: 2 * Hq * D
        self.KVDim = self.Hkv * self.D  # K or V projection dimension: Hkv * D
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D  # Packed QKV concat dimension (Q gate excluded)
        self.MUp = 2 * self.M  # Up-projection output size: gate + up channels for SwiGLU
        self.RotaryDim = _resolve_rotary_dim(head_size, partial_rotary_factor)  # Rotary embedding dimension (even, <= head_size)

    # Norm / MLP
    ln1_weight = Param(Tensor["C"], quantizable=False)
    ln2_weight = Param(Tensor["C"], quantizable=False)
    mlp_up_weight = Param(Tensor["MUp", "C"])
    mlp_down_weight = Param(Tensor["C", "M"])

    # Full-attention path
    full_q_proj_weight = Param(Tensor["QProjDim", "C"])
    full_q_proj_bias = Param(Tensor["QProjDim"], when="use_qkv_bias")
    full_k_proj_weight = Param(Tensor["KVDim", "C"])
    full_k_proj_bias = Param(Tensor["KVDim"], when="use_qkv_bias")
    full_v_proj_weight = Param(Tensor["KVDim", "C"])
    full_v_proj_bias = Param(Tensor["KVDim"], when="use_qkv_bias")
    full_out_weight = Param(Tensor["C", "AttnDim"])
    full_out_bias = Param(Tensor["C"], when="use_qkv_bias")
    q_norm_weight = Param(Tensor["D"], quantizable=False)
    k_norm_weight = Param(Tensor["D"], quantizable=False)
    rope_freqs = Param(Tensor["MaxSeq", "RotaryDim // 2", 2, "fp32"], frozen=True, quantizable=False)

    # =========================================================================
    # Activation slots (forward pass intermediate tensors)
    # =========================================================================

    # Pre-attention normalization
    ln1 = Activation(Tensor["B", "T", "C"], aliases=["ln1_flat"], share_policy="when_recomputed")
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True, share_policy="per_layer")

    # QKV (concat of separate Q, K, V projections — declared for cross-layer sharing)
    qkv = Activation(Tensor["B", "T", "QKV"], save=True, share_policy="when_recomputed")
    qkv_rope = Activation(Tensor["B", "T", "QKV"], save=True, share_policy="when_recomputed")

    # Attention output (flash_attention)
    att = Activation(Tensor["B", "T", "AttnDim"], aliases=["att_flat"], save=True, share_policy="always_recompute")
    lse = Activation(Tensor["B", "Hq", "T"], dtype="fp32", save=True, share_policy="always_recompute")
    att_out = Activation(Tensor["B", "T", "C"], aliases=["att_out_flat"], share_policy="when_recomputed")

    # First residual
    res_att = Activation(Tensor["B", "T", "C"], aliases=["residual_att"], share_policy="when_recomputed")

    # Pre-MLP normalization
    ln2 = Activation(Tensor["B", "T", "C"], aliases=["ln2_flat"], share_policy="when_recomputed")
    ln2_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True, share_policy="per_layer")

    # MLP
    mlp_up = Activation(Tensor["B", "T", "MUp"], aliases=["mlp_up_flat"], share_policy="when_recomputed")
    swiglu = Activation(Tensor["B", "T", "M"], aliases=["swiglu_flat"], share_policy="when_recomputed")
    mlp_down = Activation(Tensor["B", "T", "C"], aliases=["mlp_down_flat"], share_policy="when_recomputed")

    # Second residual (block output)
    res_ffn = Activation(Tensor["B", "T", "C"], aliases=["residual_ffn"], share_policy="per_layer")

    # =========================================================================
    # Gradient slots
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_rope = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_rope")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_ln2 = Gradient(Tensor["B", "T", "C"], gradient_of="ln2")
    d_mlp_up = Gradient(Tensor["B", "T", "MUp"], gradient_of="mlp_up")
    d_swiglu = Gradient(Tensor["B", "T", "M"], gradient_of="swiglu")
    d_mlp_down = Gradient(Tensor["B", "T", "C"], gradient_of="mlp_down")
    d_res_att = Gradient(Tensor["B", "T", "C"], gradient_of="res_att")
    d_res_ffn = Gradient(Tensor["B", "T", "C"], gradient_of="res_ffn")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor[3, "B", "T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        with graph() as g:
            ones_c = g.ones(shape=[self.C], dtype="bf16")
            ones_d = g.ones(shape=[self.D], dtype="bf16")

            ln1_weight_eff = g.add("ln1_weight", ones_c, out_name="ln1_weight_eff")
            res_ffn, ln1_out, _ = g.fused_residual_rmsnorm(
                residual,
                x,
                ln1_weight_eff,
                eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            ln1_flat = g.view(ln1_out, shape=[B * T, self.C], out_name="ln1_flat")
            if self.use_qkv_bias:
                q_proj = g.matmul_bias(
                    ln1_flat,
                    "full_q_proj_weight",
                    "full_q_proj_bias",
                    transpose="NT",
                    out_name="full_q_proj",
                )
                k_proj = g.matmul_bias(
                    ln1_flat,
                    "full_k_proj_weight",
                    "full_k_proj_bias",
                    transpose="NT",
                    out_name="full_k_proj",
                )
                v_proj = g.matmul_bias(
                    ln1_flat,
                    "full_v_proj_weight",
                    "full_v_proj_bias",
                    transpose="NT",
                    out_name="full_v_proj",
                )
            else:
                q_proj = g.matmul(ln1_flat, "full_q_proj_weight", transpose="NT", out_name="full_q_proj")
                k_proj = g.matmul(ln1_flat, "full_k_proj_weight", transpose="NT", out_name="full_k_proj")
                v_proj = g.matmul(ln1_flat, "full_v_proj_weight", transpose="NT", out_name="full_v_proj")

            q_proj_4d = g.view(q_proj, shape=[B, T, self.Hq, 2 * self.D], out_name="full_q_proj_4d")
            q, gate_4d = g.split(q_proj_4d, split_size=[self.head_size, self.head_size], dim=3)
            q = g.view(q, shape=[B, T, self.Hq, self.D], out_name="full_q")
            gate_4d = g.view(gate_4d, shape=[B, T, self.Hq, self.D], out_name="full_gate")
            k = g.view(k_proj, shape=[B, T, self.Hkv, self.D], out_name="full_k")
            v = g.view(v_proj, shape=[B, T, self.Hkv, self.D], out_name="full_v")
            qkv = g.concat(q, k, v, dim=2)

            q_norm_weight_eff = g.add("q_norm_weight", ones_d, out_name="q_norm_weight_eff")
            k_norm_weight_eff = g.add("k_norm_weight", ones_d, out_name="k_norm_weight_eff")
            qkv_norm, _, _ = g.qkv_qk_norm(
                qkv,
                q_norm_weight_eff,
                k_norm_weight_eff,
                eps=self.eps,
            )
            qkv_rope = g.mrope(
                qkv_norm,
                "rope_freqs",
                position_ids,
                rotary_dim=self.RotaryDim,
                mrope_section=self.mrope_section,
                out_name="qkv_rope",
            )
            attn_out, _ = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            attn_4d = g.view(attn_out, shape=[B, T, self.Hq, self.D], out_name="att_4d")
            gate_sigmoid = g.sigmoid(gate_4d)
            gated_attn_4d = g.mul(attn_4d, gate_sigmoid)
            gated_attn_flat = g.view(gated_attn_4d, shape=[B * T, self.AttnDim], out_name="att_flat")

            if self.use_qkv_bias:
                attn_out_flat = g.matmul_bias(
                    gated_attn_flat,
                    "full_out_weight",
                    "full_out_bias",
                    transpose="NT",
                    out_name="att_out_flat",
                )
            else:
                attn_out_flat = g.matmul(
                    gated_attn_flat,
                    "full_out_weight",
                    transpose="NT",
                    out_name="att_out_flat",
                )
            attn_out_proj = g.view(attn_out_flat, shape=[B, T, self.C], out_name="att_out")

            ln2_weight_eff = g.add("ln2_weight", ones_c, out_name="ln2_weight_eff")
            res_att, ln2_out, _ = g.fused_residual_rmsnorm(
                res_ffn,
                attn_out_proj,
                ln2_weight_eff,
                eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            ln2_flat = g.view(ln2_out, shape=[B * T, self.C], out_name="ln2_flat")
            mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", role="mlp_up_proj", out_name="mlp_up_flat")
            mlp_up = g.view(mlp_up_flat, shape=[B, T, self.MUp], out_name="mlp_up")
            mlp_act = g.swiglu(mlp_up, role="mlp_activation", out_name="swiglu")
            mlp_act_flat = g.view(mlp_act, shape=[B * T, self.M], out_name="swiglu_flat")
            out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", role="mlp_down_proj", out_name="mlp_down_flat")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="mlp_down")

            return out, res_att


@block
class Qwen3_5LinearBlock:
    """Qwen3.5 linear-attention (Gated DeltaNet) decoder block (token mixer + MLP)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        linear_conv_kernel_dim: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ):
        self.d_model = d_model  # Hidden dimension of the model
        self.d_ff = d_ff  # Feed-forward intermediate dimension (before SwiGLU split)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim  # Causal conv1d kernel width applied to QKV features
        self.linear_key_head_dim = linear_key_head_dim  # Feature dimension per key/query head
        self.linear_value_head_dim = linear_value_head_dim  # Feature dimension per value head
        self.linear_num_key_heads = linear_num_key_heads  # Number of key (and query) heads
        self.linear_num_value_heads = linear_num_value_heads  # Number of value heads; must be a multiple of key heads
        self.chunk_size = chunk_size  # Chunk size for chunk-wise GDeltaNet recurrence
        self.eps = eps  # Epsilon for RMSNorm numerical stability

        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                "Qwen3_5LinearBlock requires linear_num_value_heads to be divisible by linear_num_key_heads"
            )

        self.C = Dim("C")  # Hidden / channel dimension (= d_model)
        self.M = Dim("M")  # MLP intermediate dimension after SwiGLU split (= d_ff // 2)
        self.MUp = 2 * self.M  # Up-projection output size: gate + up channels for SwiGLU
        self.Hk = self.linear_num_key_heads  # Number of key heads
        self.Hv = self.linear_num_value_heads  # Number of value heads
        self.Kd = self.linear_key_head_dim  # Per-head key dimension
        self.Vd = self.linear_value_head_dim  # Per-head value dimension
        self.KeyDim = self.Hk * self.Kd  # Total key/query projection dimension: Hk * Kd
        self.ValueDim = self.Hv * self.Vd  # Total value projection dimension: Hv * Vd
        self.ConvK = self.linear_conv_kernel_dim  # Conv1d kernel width (alias for Dim use)
        self.ConvDim = self.KeyDim * 2 + self.ValueDim  # Packed Q+K+V conv input dimension
        self.HeadRepeat = self.Hv // self.Hk  # GQA-like repeat factor to expand K heads to match V heads

    # Norm / MLP
    ln1_weight = Param(Tensor["C"], quantizable=False)
    ln2_weight = Param(Tensor["C"], quantizable=False)
    mlp_up_weight = Param(Tensor["MUp", "C"])
    mlp_down_weight = Param(Tensor["C", "M"])

    # Linear-attention path
    lin_in_proj_qkv_weight = Param(Tensor["ConvDim", "C"])
    lin_in_proj_z_weight = Param(Tensor["ValueDim", "C"])
    lin_in_proj_b_weight = Param(Tensor["Hv", "C"])
    lin_in_proj_a_weight = Param(Tensor["Hv", "C"])
    lin_conv_weight = Param(Tensor["ConvDim", 1, "ConvK"], quantizable=False)
    lin_A_log = Param(Tensor["Hv", "fp32"], quantizable=False)
    lin_dt_bias = Param(Tensor["Hv", "fp32"], quantizable=False)
    lin_norm_weight = Param(Tensor["Vd"], quantizable=False)
    lin_out_weight = Param(Tensor["C", "ValueDim"])

    # =========================================================================
    # Activation slots (shared MLP part — same names as AttentionBlock)
    # =========================================================================

    ln1 = Activation(Tensor["B", "T", "C"], aliases=["ln1_flat"], share_policy="when_recomputed")
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True, share_policy="per_layer")

    # First residual
    res_att = Activation(Tensor["B", "T", "C"], aliases=["residual_att"], share_policy="when_recomputed")

    # Pre-MLP normalization
    ln2 = Activation(Tensor["B", "T", "C"], aliases=["ln2_flat"], share_policy="when_recomputed")
    ln2_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True, share_policy="per_layer")

    # MLP
    mlp_up = Activation(Tensor["B", "T", "MUp"], aliases=["mlp_up_flat"], share_policy="when_recomputed")
    swiglu = Activation(Tensor["B", "T", "M"], aliases=["swiglu_flat"], share_policy="when_recomputed")
    mlp_down = Activation(Tensor["B", "T", "C"], aliases=["mlp_down_flat"], share_policy="when_recomputed")

    # Second residual (block output)
    res_ffn = Activation(Tensor["B", "T", "C"], aliases=["residual_ffn"], share_policy="per_layer")

    # =========================================================================
    # Gradient slots
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_ln2 = Gradient(Tensor["B", "T", "C"], gradient_of="ln2")
    d_mlp_up = Gradient(Tensor["B", "T", "MUp"], gradient_of="mlp_up")
    d_swiglu = Gradient(Tensor["B", "T", "M"], gradient_of="swiglu")
    d_mlp_down = Gradient(Tensor["B", "T", "C"], gradient_of="mlp_down")
    d_res_att = Gradient(Tensor["B", "T", "C"], gradient_of="res_att")
    d_res_ffn = Gradient(Tensor["B", "T", "C"], gradient_of="res_ffn")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor[3, "B", "T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        del position_ids  # Unused in linear-attention layers.

        with graph() as g:
            ones_c = g.ones(shape=[self.C], dtype="bf16")
            ln1_weight_eff = g.add("ln1_weight", ones_c, out_name="ln1_weight_eff")
            res_ffn, ln1_out, _ = g.fused_residual_rmsnorm(
                residual,
                x,
                ln1_weight_eff,
                eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            ln1_flat = g.view(ln1_out, shape=[B * T, self.C], out_name="ln1_flat")
            mixed_qkv_flat = g.matmul(
                ln1_flat,
                "lin_in_proj_qkv_weight",
                transpose="NT",
                role="gdn_in_proj_qkv",
                out_name="lin_mixed_qkv_flat",
            )
            mixed_qkv = g.view(mixed_qkv_flat, shape=[B, T, self.ConvDim], out_name="lin_mixed_qkv")
            mixed_qkv_cf = g.transpose(mixed_qkv, dim0=1, dim1=2)

            conv_weight_2d = g.view("lin_conv_weight", shape=[self.ConvDim, self.ConvK], out_name="lin_conv_w2d")
            conv_out_cf = g.mamba_conv1d(
                mixed_qkv_cf,
                conv_weight_2d,
                None,
                activation="silu",
                out_name="lin_conv_out_cf",
            )
            conv_out = g.transpose(conv_out_cf, dim0=1, dim1=2)

            q_flat, k_flat, v_flat = g.split(
                conv_out,
                split_size=[self.KeyDim, self.KeyDim, self.ValueDim],
                dim=2,
            )
            query = g.view(q_flat, shape=[B, T, self.Hk, self.Kd], out_name="lin_query")
            key = g.view(k_flat, shape=[B, T, self.Hk, self.Kd], out_name="lin_key")
            value = g.view(v_flat, shape=[B, T, self.Hv, self.Vd], out_name="lin_value")

            z_flat = g.matmul(ln1_flat, "lin_in_proj_z_weight", transpose="NT", role="gdn_in_proj_z", out_name="lin_z_flat")
            z = g.view(z_flat, shape=[B, T, self.Hv, self.Vd], out_name="lin_z")

            b_flat = g.matmul(ln1_flat, "lin_in_proj_b_weight", transpose="NT", role="gdn_in_proj_b", out_name="lin_b_flat")
            b = g.view(b_flat, shape=[B, T, self.Hv], out_name="lin_b")
            beta = g.sigmoid(b)

            a_flat = g.matmul(ln1_flat, "lin_in_proj_a_weight", transpose="NT", role="gdn_in_proj_a", out_name="lin_a_flat")
            a = g.view(a_flat, shape=[B, T, self.Hv], out_name="lin_a")
            g_decay = g.qwen3_5_decay(a, "lin_A_log", "lin_dt_bias", out_name="lin_decay")

            if self.HeadRepeat > 1:
                query = g.repeat_interleave_heads(query, repeats=self.HeadRepeat, out_name="lin_query_rep")
                key = g.repeat_interleave_heads(key, repeats=self.HeadRepeat, out_name="lin_key_rep")

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

            core_flat = g.view(core_attn_out, shape=[B * T * self.Hv, self.Vd], out_name="lin_core_flat")
            z_norm_flat = g.view(z, shape=[B * T * self.Hv, self.Vd], out_name="lin_z_norm_flat")
            gated_flat = g.mamba_gated_rmsnorm(
                core_flat,
                z_norm_flat,
                "lin_norm_weight",
                eps=self.eps,
                n_groups=1,
                norm_before_gate=True,
                out_name="lin_gated_flat",
            )
            gated = g.view(gated_flat, shape=[B, T, self.ValueDim], out_name="lin_gated")

            gated_bt_flat = g.view(gated, shape=[B * T, self.ValueDim], out_name="lin_gated_bt_flat")
            attn_out_flat = g.matmul(gated_bt_flat, "lin_out_weight", transpose="NT", out_name="lin_att_out_flat")
            attn_out = g.view(attn_out_flat, shape=[B, T, self.C], out_name="lin_att_out")

            ln2_weight_eff = g.add("ln2_weight", ones_c, out_name="ln2_weight_eff")
            res_att, ln2_out, _ = g.fused_residual_rmsnorm(
                res_ffn,
                attn_out,
                ln2_weight_eff,
                eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            ln2_flat = g.view(ln2_out, shape=[B * T, self.C], out_name="ln2_flat")
            mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", role="mlp_up_proj", out_name="mlp_up_flat")
            mlp_up = g.view(mlp_up_flat, shape=[B, T, self.MUp], out_name="mlp_up")
            mlp_act = g.swiglu(mlp_up, role="mlp_activation", out_name="swiglu")
            mlp_act_flat = g.view(mlp_act, shape=[B * T, self.M], out_name="swiglu_flat")
            out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", role="mlp_down_proj", out_name="mlp_down_flat")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="mlp_down")

            return out, res_att
