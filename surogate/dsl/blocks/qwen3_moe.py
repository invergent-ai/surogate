"""Qwen3 MoE Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T


@block
class Qwen3MoEBlock:
    """Qwen3 MoE transformer block with QK-Norm and Mixture of Experts.

    Structure:
        residual, x = fused_residual_rmsnorm(residual, x)
        x = attention(x) + residual
        residual, x = fused_residual_rmsnorm(residual, x)
        x = moe(x) + residual  (router -> experts -> combine)

    Features:
        - QK normalization in attention
        - Top-k expert routing with norm_topk_prob
        - Optional shared expert
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
        num_experts_per_tok: int,  # top_k
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        norm_topk_prob: bool = True,
        use_shared_expert: bool = False,
        shared_expert_intermediate: int = 0,
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
        self.use_qk_norm = use_qk_norm
        self.norm_topk_prob = norm_topk_prob
        self.use_shared_expert = use_shared_expert
        self.shared_expert_intermediate = shared_expert_intermediate if shared_expert_intermediate > 0 else d_ff

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
        self.MUp = 2 * self.M  # gate + up fused

        # Shared expert dimensions
        self.SharedM = Dim("SharedM")
        self.SharedMUp = 2 * self.SharedM

    # LayerNorm weights
    ln1_weight = Param(Tensor["C"])
    ln2_weight = Param(Tensor["C"])

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    q_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    k_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

    # Router weight
    router_weight = Param(Tensor["E", "C"])

    # Expert weights (batched format: [num_experts, ...])
    experts_gate_up = Param(Tensor["E", "MUp", "C"])
    experts_down = Param(Tensor["E", "C", "M"])

    # Shared expert weights (optional - present when shared_expert_intermediate > 0)
    shared_expert_gate = Param(Tensor["SharedM", "C"], when=lambda self: getattr(self, 'use_shared_expert', False) or getattr(self, 'shared_expert_intermediate', 0) > 0)
    shared_expert_up = Param(Tensor["SharedM", "C"], when=lambda self: getattr(self, 'use_shared_expert', False) or getattr(self, 'shared_expert_intermediate', 0) > 0)
    shared_expert_down = Param(Tensor["C", "SharedM"], when=lambda self: getattr(self, 'use_shared_expert', False) or getattr(self, 'shared_expert_intermediate', 0) > 0)

    # =========================================================================
    # Activation slots (forward pass intermediate tensors)
    # =========================================================================

    # Pre-attention normalization
    ln1 = Activation(Tensor["B", "T", "C"], aliases=["ln1_flat"])
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                          description="RMSNorm reciprocal std for LN1")

    # QKV projection and RoPE
    qkv = Activation(Tensor["B", "T", "QKV"], aliases=["qkv_flat", "qkv_biased"])
    qkv_rope = Activation(Tensor["B", "T", "QKV"],
                          description="QKV after QK-Norm + RoPE")

    # QK-norm RSTDs (conditional on use_qk_norm)
    q_rstd = Activation(Tensor["B", "T", "Hq"], dtype="fp32", save=True,
                        when="use_qk_norm", description="Q head RMSNorm rstd")
    k_rstd = Activation(Tensor["B", "T", "Hkv"], dtype="fp32", save=True,
                        when="use_qk_norm", description="K head RMSNorm rstd")

    # Attention
    att = Activation(Tensor["B", "T", "AttnDim"], aliases=["att_flat", "attn"],
                     description="Attention output (pre out-proj)")
    lse = Activation(Tensor["B", "Hq", "T"], dtype="fp32", save=True,
                     description="Log-sum-exp from flash attention")
    att_out = Activation(Tensor["B", "T", "C"], aliases=["att_out_flat"],
                         description="After output projection")

    # Mid residual (attention output added to residual)
    residual_mid = Activation(Tensor["B", "T", "C"],
                              description="Residual after attention")

    # Pre-MoE normalization
    ln2 = Activation(Tensor["B", "T", "C"], aliases=["ln2_flat"])
    ln2_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                          description="RMSNorm reciprocal std for LN2")

    # Router
    router_logits = Activation(Tensor["B * T", "E"],
                               description="Router logits before softmax/sigmoid")
    router_probs = Activation(Tensor["B * T", "E"],
                              description="Router probabilities after normalization")

    # Routing (top-k selection)
    routing_weights = Activation(Tensor["B * T", "K"], save=True,
                                 description="Routing weights for selected experts")
    routing_indices = Activation(Tensor["B * T", "K"], dtype="int32", save=True,
                                 description="Expert indices for each token")

    # MoE permutation
    permuted_input = Activation(Tensor["B * T * K", "C"],
                                description="Permuted input for grouped GEMM")
    scatter_indices = Activation(Tensor["B * T * K"], dtype="int32", save=True,
                                 description="Indices for scattering back to original order")

    # Expert computations
    expert_gate_up = Activation(Tensor["B * T * K", "MUp"],
                                description="Expert gate+up projection output")
    expert_act = Activation(Tensor["B * T * K", "M"],
                            description="Expert SwiGLU activation output")
    expert_down = Activation(Tensor["B * T * K", "C"],
                             description="Expert down projection output")

    # MoE output
    moe_out = Activation(Tensor["B * T", "C"], aliases=["moe_out_flat"],
                         description="Combined MoE output")

    # Shared expert (conditional)
    shared_gate = Activation(Tensor["B * T", "SharedM"], when="use_shared_expert",
                             description="Shared expert gate projection")
    shared_up = Activation(Tensor["B * T", "SharedM"], when="use_shared_expert",
                           description="Shared expert up projection")
    shared_gate_act = Activation(Tensor["B * T", "SharedM"], when="use_shared_expert",
                                 description="Shared expert gate activation (SiLU)")
    shared_hidden = Activation(Tensor["B * T", "SharedM"], when="use_shared_expert",
                               description="Shared expert hidden state (gate * up)")
    shared_out = Activation(Tensor["B * T", "C"], when="use_shared_expert",
                            description="Shared expert output")

    # Final residual
    res_ffn = Activation(Tensor["B", "T", "C"], aliases=["residual_ffn"],
                         description="Residual + MoE (block output)")

    # =========================================================================
    # Gradient slots (backward pass)
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_rope = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_rope")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_att_out = Gradient(Tensor["B", "T", "C"], gradient_of="att_out")
    d_ln2 = Gradient(Tensor["B", "T", "C"], gradient_of="ln2")
    d_router_logits = Gradient(Tensor["B * T", "E"], gradient_of="router_logits")
    d_permuted_input = Gradient(Tensor["B * T * K", "C"], gradient_of="permuted_input")
    d_expert_gate_up = Gradient(Tensor["B * T * K", "MUp"], gradient_of="expert_gate_up")
    d_expert_act = Gradient(Tensor["B * T * K", "M"], gradient_of="expert_act")
    d_expert_down = Gradient(Tensor["B * T * K", "C"], gradient_of="expert_down")
    d_moe_out = Gradient(Tensor["B * T", "C"], gradient_of="moe_out")
    d_shared_hidden = Gradient(Tensor["B * T", "SharedM"], gradient_of="shared_hidden",
                               when="use_shared_expert")
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
            residual_mid, ln1_out, _ = g.fused_residual_rmsnorm(
                residual, x, "ln1_weight", eps=self.eps
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

            # QK-Norm + RoPE (fused)
            if self.use_qk_norm:
                qkv_rope, _, _ = g.qkv_qk_norm_rope(
                    qkv_packed,
                    "q_norm_weight",
                    "k_norm_weight",
                    "rope_freqs",
                    position_ids,
                    eps=self.eps,
                    out_name="qkv_rope",
                    q_rstd_name="q_rstd",
                    k_rstd_name="k_rstd",
                )
            else:
                qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, out_name="qkv_rope")

            # FlashAttention
            attn_out, _ = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            # Output projection
            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim], out_name="att_flat")
            att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=[B, T, self.C], out_name="att_out")

            # ================================================================
            # Pre-MoE norm
            # ================================================================
            residual_out, ln2_out, _ = g.fused_residual_rmsnorm(
                residual_mid, att_out, "ln2_weight", eps=self.eps
            )

            # ================================================================
            # MoE: Router -> Experts -> Combine
            # ================================================================
            ln2_flat = g.view(ln2_out, shape=[B * T, self.C], out_name="ln2_flat")

            # Router: compute routing logits and select top-k experts
            router_logits = g.matmul(ln2_flat, "router_weight", transpose="NT")

            # Softmax or sigmoid for routing probabilities
            if self.norm_topk_prob:
                router_probs = g.moe_sigmoid(router_logits)
            else:
                router_probs = g.moe_softmax(router_logits)

            # Top-k selection
            routing_weights, routing_indices = g.moe_topk(
                router_probs, top_k=self.num_experts_per_tok, normalize=self.norm_topk_prob
            )

            # Permute inputs for grouped expert computation
            permuted_input, scatter_indices = g.moe_permute(
                ln2_flat, routing_indices, top_k=self.num_experts_per_tok
            )

            # Grouped GEMM for gate+up projection
            expert_gate_up = g.moe_grouped_gemm_gate_up(
                permuted_input, "experts_gate_up", scatter_indices
            )

            # SwiGLU activation
            expert_act = g.swiglu(expert_gate_up)

            # Grouped GEMM for down projection
            expert_down = g.moe_grouped_gemm_down(
                expert_act, "experts_down", scatter_indices
            )

            # Unpermute and combine with routing weights
            moe_out = g.moe_unpermute(
                expert_down, routing_weights, scatter_indices, top_k=self.num_experts_per_tok
            )

            # ================================================================
            # Shared expert (optional)
            # ================================================================
            if self.use_shared_expert:
                # Shared expert forward pass
                shared_gate = g.matmul(ln2_flat, "shared_expert_gate", transpose="NT")
                shared_up = g.matmul(ln2_flat, "shared_expert_up", transpose="NT")
                shared_gate_act = g.silu(shared_gate)
                shared_hidden = g.mul(shared_gate_act, shared_up)
                shared_out = g.matmul(shared_hidden, "shared_expert_down", transpose="NT")
                # Add shared expert output to MoE output
                moe_out = g.add(moe_out, shared_out)

            # Reshape back to (B, T, C)
            out = g.view(moe_out, shape=[B, T, self.C], out_name="res_ffn")

            return out, residual_out
