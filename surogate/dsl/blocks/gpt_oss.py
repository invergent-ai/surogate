"""GPT-OSS Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T


@block
class GptOssBlock:
    """GPT-OSS transformer block with sinks and MoE experts.

    Structure:
        residual, x = fused_residual_rmsnorm(residual, x)
        x = attention(x, sinks) + residual
        residual, x = fused_residual_rmsnorm(residual, x)
        x = moe(x) + residual
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,  # Per-expert intermediate size
        max_seq: int,
        num_experts: int,
        num_experts_per_tok: int,
        eps: float = 1e-5,
        use_qkv_bias: bool = True,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.M = Dim("M")
        self.MaxSeq = Dim("MaxSeq")
        self.E = Dim("E")
        self.K = Dim("K")

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D
        self.MUp = 2 * self.M  # interleaved gate + up

    # LayerNorm weights
    ln1_weight = Param(Tensor["C"])
    ln2_weight = Param(Tensor["C"])

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    out_bias = Param(Tensor["C"], when="use_qkv_bias")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)
    sinks = Param(Tensor["Hq"])

    # Router weights
    router_weight = Param(Tensor["E", "C"])
    router_bias = Param(Tensor["E"])

    # Expert weights (batched format: [num_experts, ...])
    # offload_group="moe_experts" signals the runtime to store these on CPU
    experts_gate_up = Param(Tensor["E", "MUp", "C"], offload_group="moe_experts")
    experts_gate_up_bias = Param(Tensor["E", "MUp"], offload_group="moe_experts")
    experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")
    experts_down_bias = Param(Tensor["E", "C"], offload_group="moe_experts")

    # =========================================================================
    # Activation slots (forward pass intermediate tensors)
    # =========================================================================

    # Pre-attention normalization
    ln1 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln1_flat"],
        recompute=True,
        recompute_from=["res_ffn", "ln1_rstd", "@param:ln1_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    ln1_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
        description="RMSNorm reciprocal std for LN1",
    )

    # QKV projection and RoPE
    qkv = Activation(
        Tensor["B", "T", "QKV"],
        aliases=["qkv_flat", "qkv_biased"],
        save=True,
        recompute=True,
        recompute_from=["ln1", "@param:qkv_weight", "?@param:qkv_bias"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "qkv", "transpose": "NT"},
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
        recompute_attrs={"rotary_dim": "D"},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="QKV after RoPE",
    )

    # Attention
    att = Activation(
        Tensor["B", "T", "AttnDim"],
        aliases=["att_flat", "attn"],
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_outputs=["att", "lse"],
        recompute_from=["qkv_rope", "@param:sinks"],
        recompute_op="flash_attention",
        recompute_attrs={"attn_impl": "cudnn"},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Attention output (pre out-proj)",
    )
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Log-sum-exp from flash attention",
    )
    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        recompute=True,
        recompute_from=["att", "@param:out_weight", "?@param:out_bias"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "attn_out", "transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["o"],
        share_policy="fft_share",
        description="After output projection",
    )

    # First residual (residual + input before attention)
    res_ffn = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_ffn"],
        share_policy="per_layer",
        description="Residual + input (pre-attention)",
    )

    # Mid residual after attention + LN2
    res_att = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_att"],
        recompute=True,
        recompute_group="ln2_fused",
        recompute_outputs=["res_att", "ln2"],
        recompute_from=["res_ffn", "att_out", "ln2_rstd", "@param:ln2_weight"],
        recompute_op="fused_residual_rmsnorm_apply_saved",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Residual after attention (pre-MoE)",
    )

    # Pre-MoE normalization
    ln2 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln2_flat"],
        save=True,
        recompute=True,
        recompute_group="ln2_fused",
        recompute_policy="fft_only",
        share_policy="fft_share",
    )
    ln2_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
        description="RMSNorm reciprocal std for LN2",
    )

    # Router
    router_logits = Activation(
        Tensor["B * T", "E"],
        save=True,
        recompute=True,
        recompute_from=["ln2", "@param:router_weight", "@param:router_bias"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Router logits before top-k",
    )

    # Routing (top-k selection with softmax on top-k logits)
    routing_weights = Activation(
        Tensor["B * T", "K"],
        save=True,
        recompute=True,
        recompute_group="moe_topk",
        recompute_outputs=["routing_weights", "routing_indices"],
        recompute_from=["router_logits"],
        recompute_op="moe_topk",
        recompute_attrs={"top_k": "K", "normalize": False, "softmax": True},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Routing weights for selected experts",
    )
    routing_indices = Activation(
        Tensor["B * T", "K"],
        dtype="int32",
        save=True,
        recompute=True,
        recompute_group="moe_topk",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Expert indices for each token",
    )

    # MoE permutation
    permuted_input = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        recompute=True,
        recompute_group="moe_permute",
        recompute_outputs=["permuted_input", "scatter_indices"],
        recompute_from=["ln2", "routing_indices"],
        recompute_op="moe_permute",
        recompute_attrs={"top_k": "K"},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Permuted input for grouped GEMM",
    )
    scatter_indices = Activation(
        Tensor["B * T * K"],
        dtype="int32",
        save=True,
        recompute=True,
        recompute_group="moe_permute",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Indices for scattering back to original order",
    )

    # Expert computations
    expert_gate_up = Activation(
        Tensor["B * T * K", "MUp"],
        save=True,
        recompute=True,
        recompute_from=["permuted_input", "@param:experts_gate_up", "scatter_indices"],
        recompute_op="moe_grouped_gemm_gate_up",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Expert gate+up projection output",
    )
    expert_gate_up_bias = Activation(
        Tensor["B * T * K", "MUp"],
        save=True,
        recompute=True,
        recompute_from=["expert_gate_up", "@param:experts_gate_up_bias"],
        recompute_op="moe_expert_bias_add",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Expert gate+up projection after bias",
    )
    expert_act = Activation(
        Tensor["B * T * K", "M"],
        save=True,
        recompute=True,
        recompute_from=["expert_gate_up_bias"],
        recompute_op="gpt_oss_moe_act",
        recompute_attrs={"alpha": 1.702, "limit": 7.0},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Expert GPT-OSS activation output",
    )
    expert_down = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        recompute=True,
        recompute_from=["expert_act", "@param:experts_down", "scatter_indices"],
        recompute_op="moe_grouped_gemm_down",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Expert down projection output",
    )
    expert_down_bias = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        recompute=True,
        recompute_from=["expert_down", "@param:experts_down_bias"],
        recompute_op="moe_expert_bias_add",
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Expert down projection after bias",
    )

    # MoE output
    moe_out = Activation(
        Tensor["B * T", "C"],
        aliases=["moe_out_flat"],
        save=True,
        recompute=True,
        recompute_from=["expert_down_bias", "routing_weights", "scatter_indices"],
        recompute_op="moe_unpermute",
        recompute_attrs={"top_k": "K"},
        recompute_policy="fft_only",
        share_policy="fft_share",
        description="Combined MoE output",
    )

    # Final output (MoE output in block output slot)
    mlp_down = Activation(
        Tensor["B", "T", "C"], aliases=["mlp_down_flat"],
        share_policy="per_layer",
        description="MoE output (block output)",
    )

    # =========================================================================
    # Gradient slots (backward pass)
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_rope = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_rope")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_att_out = Gradient(Tensor["B", "T", "C"], gradient_of="att_out")
    d_ln2 = Gradient(Tensor["B", "T", "C"], gradient_of="ln2")
    d_res_att = Gradient(Tensor["B", "T", "C"], gradient_of="res_att")
    d_router_logits = Gradient(Tensor["B * T", "E"], gradient_of="router_logits")
    d_permuted_input = Gradient(Tensor["B * T * K", "C"], gradient_of="permuted_input",
                                alias_of="permuted_input")
    d_expert_gate_up = Gradient(Tensor["B * T * K", "MUp"], gradient_of="expert_gate_up")
    d_expert_gate_up_bias = Gradient(Tensor["B * T * K", "MUp"], gradient_of="expert_gate_up_bias",
                                     alias_of="expert_gate_up_bias")
    d_expert_act = Gradient(Tensor["B * T * K", "M"], gradient_of="expert_act",
                            alias_of="expert_act")
    d_expert_down = Gradient(Tensor["B * T * K", "C"], gradient_of="expert_down")
    d_expert_down_bias = Gradient(Tensor["B * T * K", "C"], gradient_of="expert_down_bias",
                                  alias_of="expert_down_bias")
    d_moe_out = Gradient(Tensor["B * T", "C"], gradient_of="moe_out")
    d_mlp_down = Gradient(Tensor["B", "T", "C"], gradient_of="mlp_down")
    d_res_ffn = Gradient(Tensor["B", "T", "C"], gradient_of="res_ffn")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Returns (out, residual_out)."""
        with graph() as g:
            # ================================================================
            # Pre-attention norm
            # ================================================================
            res_ffn, ln1_out, ln1_rstd = g.fused_residual_rmsnorm(
                residual, x, "ln1_weight", eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            # ================================================================
            # Attention
            # ================================================================
            ln1_flat = g.view(ln1_out, shape=[B * T, self.C], out_name="ln1_flat")
            if self.use_qkv_bias:
                qkv_flat = g.matmul_bias(ln1_flat, "qkv_weight", "qkv_bias", transpose="NT", out_name="qkv_flat")
            else:
                qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT", out_name="qkv_flat")
            qkv_packed = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D], out_name="qkv")

            # RoPE
            qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, out_name="qkv_rope")

            # FlashAttention with sinks
            attn_out, _ = g.flash_attention(qkv_rope, causal=True, sinks="sinks",
                                            out_name="att", lse_name="lse")

            # Output projection
            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim], out_name="att_flat")
            if self.use_qkv_bias:
                att_out_flat = g.matmul_bias(attn_flat, "out_weight", "out_bias",
                                             transpose="NT", out_name="att_out_flat")
            else:
                att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=[B, T, self.C], out_name="att_out")

            # ================================================================
            # Pre-MoE norm
            # ================================================================
            res_att, ln2_out, ln2_rstd = g.fused_residual_rmsnorm(
                res_ffn, att_out, "ln2_weight", eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            # ================================================================
            # MoE: Router -> Experts -> Combine
            # ================================================================
            ln2_flat = g.view(ln2_out, shape=[B * T, self.C], out_name="ln2_flat")

            # Router: compute routing logits and select top-k experts
            router_logits = g.matmul_bias(ln2_flat, "router_weight", "router_bias",
                                          transpose="NT", out_name="router_logits")

            # Top-k selection (softmax over selected logits)
            routing_weights, routing_indices = g.moe_topk(
                router_logits,
                top_k=self.num_experts_per_tok,
                normalize=False,
                softmax=True,
                sort_by_index=True,
                weights_name="routing_weights", indices_name="routing_indices",
            )

            # Permute inputs for grouped expert computation
            permuted_input, scatter_indices = g.moe_permute(
                ln2_flat, routing_indices, top_k=self.num_experts_per_tok,
                out_name="permuted_input", scatter_name="scatter_indices",
            )

            # Grouped GEMM for gate+up projection
            expert_gate_up = g.moe_grouped_gemm_gate_up(
                permuted_input,
                "experts_gate_up",
                scatter_indices,
                gate_up_interleaved=True,
                out_name="expert_gate_up",
            )
            expert_gate_up_bias = g.moe_expert_bias_add(
                expert_gate_up, "experts_gate_up_bias", out_name="expert_gate_up_bias",
            )

            # GPT-OSS MoE activation
            expert_act = g.gpt_oss_moe_act(
                expert_gate_up_bias, alpha=1.702, limit=7.0,
                out_name="expert_act",
            )

            # Grouped GEMM for down projection
            expert_down = g.moe_grouped_gemm_down(
                expert_act, "experts_down", scatter_indices, out_name="expert_down",
            )
            expert_down_bias = g.moe_expert_bias_add(
                expert_down, "experts_down_bias", out_name="expert_down_bias",
            )

            # Unpermute and combine with routing weights
            moe_out = g.moe_unpermute(
                expert_down_bias, routing_weights, scatter_indices, top_k=self.num_experts_per_tok,
                out_name="moe_out",
            )

            # Reshape back to (B, T, C)
            out = g.view(moe_out, shape=[B, T, self.C], out_name="mlp_down")

            return out, res_att
