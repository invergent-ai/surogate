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
        ep_size: int = 1,
    ):
        self.d_model = d_model  # Hidden dimension of the model
        self.num_query_heads = num_query_heads  # Number of query attention heads
        self.num_kv_heads = num_kv_heads  # Number of key/value heads (GQA); may be < num_query_heads
        self.head_size = head_size  # Feature dimension per attention head (D)
        self.d_ff = d_ff  # Per-expert feed-forward intermediate dimension (before SwiGLU split)
        self.max_seq = max_seq  # Maximum sequence length for RoPE frequency cache
        self.num_experts = num_experts  # Total number of experts across all EP ranks
        self.num_experts_per_tok = num_experts_per_tok  # Top-k experts selected per token
        self.eps = eps  # Epsilon for RMSNorm numerical stability
        self.use_qkv_bias = use_qkv_bias  # Whether Q/K/V/out projections include bias terms
        self.use_qk_norm = use_qk_norm  # Whether to apply per-head RMSNorm to Q and K before RoPE
        self.norm_topk_prob = norm_topk_prob  # Renormalize routing weights to sum to 1 after top-k selection
        self.use_shared_expert = use_shared_expert  # Whether a dense shared expert is added to every token
        self.shared_expert_intermediate = shared_expert_intermediate if shared_expert_intermediate > 0 else d_ff  # Shared expert intermediate dim; falls back to d_ff if not specified
        self.ep_size = ep_size  # Expert parallelism degree (1 = no EP, all experts local)
        self.num_local_experts = num_experts // ep_size if ep_size > 1 else num_experts  # Number of experts owned by this EP rank

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")  # Hidden / channel dimension (= d_model)
        self.Hq = Dim("Hq")  # Number of query heads (= num_query_heads)
        self.Hkv = Dim("Hkv")  # Number of KV heads (= num_kv_heads)
        self.D = Dim("D")  # Per-head feature dimension (= head_size)
        self.M = Dim("M")  # Per-expert MLP intermediate dimension after SwiGLU split (= d_ff // 2)
        self.MaxSeq = Dim("MaxSeq")  # Max sequence length (= max_seq)
        self.E = Dim("E")  # Total number of experts (= num_experts)
        self.K = Dim("K")  # Top-k experts per token (= num_experts_per_tok)

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D  # Packed QKV projection dimension: (Hq + 2*Hkv) * D
        self.AttnDim = self.Hq * self.D  # Total attention output dimension: Hq * D
        self.MUp = 2 * self.M  # Expert gate+up fused projection output size: 2 * M

        # Shared expert dimensions
        self.SharedM = Dim("SharedM")  # Shared expert intermediate dimension (= shared_expert_intermediate // 2)
        self.SharedMUp = 2 * self.SharedM  # Shared expert gate+up fused projection output size: 2 * SharedM

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
    # offload_group="moe_experts" signals the runtime to store these on CPU
    # and stream to GPU on demand when expert offloading is enabled.
    experts_gate_up = Param(Tensor["E", "MUp", "C"], offload_group="moe_experts")
    experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")

    # Shared expert weights (optional - present when shared_expert_intermediate > 0)
    shared_expert_gate = Param(Tensor["SharedM", "C"], when=lambda self: getattr(self, 'use_shared_expert', False) or getattr(self, 'shared_expert_intermediate', 0) > 0)
    shared_expert_up = Param(Tensor["SharedM", "C"], when=lambda self: getattr(self, 'use_shared_expert', False) or getattr(self, 'shared_expert_intermediate', 0) > 0)
    shared_expert_down = Param(Tensor["C", "SharedM"], when=lambda self: getattr(self, 'use_shared_expert', False) or getattr(self, 'shared_expert_intermediate', 0) > 0)

    # =========================================================================
    # Activation slots (forward pass intermediate tensors)
    # =========================================================================

    # Pre-attention normalization
    ln1 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln1_flat"],
        share_policy="fft_share",  # Share only in FFT mode
    )
    ln1_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",  # Always save per-layer (needed for recompute)
        description="RMSNorm reciprocal std for LN1",
    )

    # QKV projection and RoPE
    qkv = Activation(
        Tensor["B", "T", "QKV"],
        aliases=["qkv_flat", "qkv_biased"],
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
    )
    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        description="QKV after QK-Norm + RoPE",
    )

    # QK-norm RSTDs (conditional on use_qk_norm)
    q_rstd = Activation(
        Tensor["B", "T", "Hq"],
        dtype="fp32",
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        when="use_qk_norm",
        description="Q head RMSNorm rstd",
    )
    k_rstd = Activation(
        Tensor["B", "T", "Hkv"],
        dtype="fp32",
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        when="use_qk_norm",
        description="K head RMSNorm rstd",
    )

    # Attention
    att = Activation(
        Tensor["B", "T", "AttnDim"],
        aliases=["att_flat", "attn"],
        save=True,
        share_policy="always_recompute",
        description="Attention output (pre out-proj)",
    )
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        share_policy="always_recompute",
        description="Log-sum-exp from flash attention",
    )
    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        share_policy="fft_share",  # Share only in FFT mode
        description="After output projection",
    )

    # First residual (residual + input before attention)
    res_ffn = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_ffn"],
        # res_ffn is stored via residual manager; do not mark as recompute.
        share_policy="per_layer",  # Managed by residual manager, not shared
        description="Residual + input (pre-attention)",
    )

    # Mid residual after attention + LN2
    res_att = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_att"],
        share_policy="fft_share",  # Share only in FFT mode
        description="Residual after attention (pre-MoE)",
    )

    # Pre-MoE normalization
    ln2 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln2_flat"],
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
    )
    ln2_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",  # Always save per-layer (needed for recompute)
        description="RMSNorm reciprocal std for LN2",
    )

    # Router
    router_logits = Activation(
        Tensor["B * T", "E"],
        save=True,
        # Router matmul uses quantized weights under QLoRA; skip recompute in LoRA mode.
        share_policy="fft_share",  # Share only in FFT mode
        description="Router logits before softmax/sigmoid",
    )
    router_probs = Activation(
        Tensor["B * T", "E"],
        save=True,
        # Depends on router_logits; skip recompute in LoRA mode.
        share_policy="fft_share",  # Share only in FFT mode
        description="Router probabilities after normalization",
    )

    # Routing (top-k selection)
    routing_weights = Activation(
        Tensor["B * T", "K"],
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        description="Routing weights for selected experts",
    )
    routing_indices = Activation(
        Tensor["B * T", "K"],
        dtype="int32",
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert indices for each token",
    )

    # MoE permutation
    permuted_input = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        # Recompute only in FFT mode; in LoRA/QLoRA use saved tensors to avoid
        # recompute drift and quantized-weight path mismatches.
        share_policy="fft_share",  # Share only in FFT mode
        description="Permuted input for grouped GEMM",
    )
    scatter_indices = Activation(
        Tensor["B * T * K"],
        dtype="int32",
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        description="Indices for scattering back to original order",
    )

    # EP dispatch outputs (only present when ep_size > 1)
    ep_recv_input = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        when=lambda self: self.ep_size > 1,
        share_policy="per_layer",
        description="EP-dispatched input tokens (post-A2A, local experts only)",
    )
    ep_recv_scatter = Activation(
        Tensor["B * T * K"],
        dtype="int32",
        save=True,
        when=lambda self: self.ep_size > 1,
        share_policy="per_layer",
        description="EP-dispatched scatter indices (post-A2A)",
    )

    # Expert computations
    expert_gate_up = Activation(
        Tensor["B * T * K", "MUp"],
        save=True,
        # Expert GEMM uses quantized weights under QLoRA; skip recompute in LoRA mode.
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert gate+up projection output",
    )
    expert_act = Activation(
        Tensor["B * T * K", "M"],
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert SwiGLU activation output",
    )
    expert_down = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        # Expert GEMM uses quantized weights under QLoRA; skip recompute in LoRA mode.
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert down projection output",
    )

    # EP combine output (only present when ep_size > 1)
    ep_combined = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        when=lambda self: self.ep_size > 1,
        share_policy="per_layer",
        description="EP-combined expert output (post reverse-A2A, pre-EP order)",
    )

    # MoE output
    moe_out = Activation(
        Tensor["B * T", "C"],
        aliases=["moe_out_flat"],
        save=True,
        share_policy="fft_share",  # Share only in FFT mode
        description="Combined MoE output",
    )

    # Shared expert (conditional)
    shared_gate = Activation(
        Tensor["B * T", "SharedM"], when="use_shared_expert",
        share_policy="fft_share",  # Share only in FFT mode
        description="Shared expert gate projection",
    )
    shared_up = Activation(
        Tensor["B * T", "SharedM"], when="use_shared_expert",
        share_policy="fft_share",  # Share only in FFT mode
        description="Shared expert up projection",
    )
    shared_gate_act = Activation(
        Tensor["B * T", "SharedM"], when="use_shared_expert",
        share_policy="fft_share",  # Share only in FFT mode
        description="Shared expert gate activation (SiLU)",
    )
    shared_hidden = Activation(
        Tensor["B * T", "SharedM"], when="use_shared_expert",
        share_policy="fft_share",  # Share only in FFT mode
        description="Shared expert hidden state (gate * up)",
    )
    shared_out = Activation(
        Tensor["B * T", "C"], when="use_shared_expert",
        share_policy="fft_share",  # Share only in FFT mode
        description="Shared expert output",
    )

    # Final output (MoE output in block output slot)
    mlp_down = Activation(
        Tensor["B", "T", "C"], aliases=["mlp_down_flat"],
        share_policy="per_layer",  # Block output, not shared
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
    d_ep_recv_input = Gradient(Tensor["B * T * K", "C"], gradient_of="ep_recv_input",
                               when=lambda self: self.ep_size > 1)
    d_permuted_input = Gradient(Tensor["B * T * K", "C"], gradient_of="permuted_input",
                                alias_of="permuted_input")
    d_expert_gate_up = Gradient(Tensor["B * T * K", "MUp"], gradient_of="expert_gate_up")
    d_expert_act = Gradient(Tensor["B * T * K", "M"], gradient_of="expert_act",
                            alias_of="expert_act")
    d_expert_down = Gradient(Tensor["B * T * K", "C"], gradient_of="expert_down")
    d_moe_out = Gradient(Tensor["B * T", "C"], gradient_of="moe_out")
    d_mlp_down = Gradient(Tensor["B", "T", "C"], gradient_of="mlp_down")
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
            router_logits = g.matmul(ln2_flat, "router_weight", transpose="NT", out_name="router_logits")

            # Qwen3 MoE uses softmax routing; norm_topk_prob controls renormalization after top-k.
            router_probs = g.moe_softmax(router_logits, out_name="router_probs")

            # Top-k selection
            routing_weights, routing_indices = g.moe_topk(
                router_probs, top_k=self.num_experts_per_tok, normalize=self.norm_topk_prob,
                weights_name="routing_weights", indices_name="routing_indices",
            )

            # Permute inputs for grouped expert computation
            permuted_input, scatter_indices = g.moe_permute(
                ln2_flat, routing_indices, top_k=self.num_experts_per_tok,
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

            # Grouped GEMM for gate+up projection
            expert_gate_up = g.moe_grouped_gemm_gate_up(
                gemm_input, "experts_gate_up", gemm_scatter, role="moe_gate_up", out_name="expert_gate_up",
            )

            # SwiGLU activation
            expert_act = g.swiglu(expert_gate_up, role="mlp_activation", out_name="expert_act")

            # Grouped GEMM for down projection
            expert_down = g.moe_grouped_gemm_down(
                expert_act, "experts_down", gemm_scatter, role="moe_down", out_name="expert_down",
            )

            # EP combine: route expert outputs back (no-op when ep_size=1)
            if self.ep_size > 1:
                expert_down = g.ep_combine(
                    expert_down,
                    num_experts=self.num_experts, ep_size=self.ep_size,
                    top_k=self.num_experts_per_tok,
                    out_name="ep_combined",
                )

            # Unpermute and combine with routing weights
            moe_out = g.moe_unpermute(
                expert_down, routing_weights, scatter_indices, top_k=self.num_experts_per_tok,
                out_name="moe_out",
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
                # Write combined output into the persistent MoE output slot
                moe_out = g.add(moe_out, shared_out, out_name="moe_out")

            # Reshape back to (B, T, C)
            out = g.view(moe_out, shape=[B, T, self.C], out_name="mlp_down")

            return out, res_att
