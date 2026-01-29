"""Qwen3 MoE Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param
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
            ln1_flat = g.view(ln1_out, shape=[B * T, self.C])
            if self.use_qkv_bias:
                qkv_flat = g.matmul_bias(ln1_flat, "qkv_weight", "qkv_bias", transpose="NT")
            else:
                qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT")
            qkv_packed = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D])

            # QK-Norm + RoPE (fused)
            if self.use_qk_norm:
                qkv_rope, _, _ = g.qkv_qk_norm_rope(
                    qkv_packed,
                    "q_norm_weight",
                    "k_norm_weight",
                    "rope_freqs",
                    position_ids,
                    eps=self.eps,
                )
            else:
                qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids)

            # FlashAttention
            attn_out, _ = g.flash_attention(qkv_rope, causal=True)

            # Output projection
            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim])
            att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT")
            att_out = g.view(att_out_flat, shape=[B, T, self.C])

            # ================================================================
            # Pre-MoE norm
            # ================================================================
            residual_out, ln2_out, _ = g.fused_residual_rmsnorm(
                residual_mid, att_out, "ln2_weight", eps=self.eps
            )

            # ================================================================
            # MoE: Router -> Experts -> Combine
            # ================================================================
            ln2_flat = g.view(ln2_out, shape=[B * T, self.C])

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
            out = g.view(moe_out, shape=[B, T, self.C])

            return out, residual_out
