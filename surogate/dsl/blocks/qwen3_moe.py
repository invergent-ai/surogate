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
        recompute=True,
        # Recompute LN1 directly from saved res_ffn + rstd.
        # This matches implementation (res_ffn is retrieved from residual storage, not re-derived).
        recompute_from=["res_ffn", "ln1_rstd", "@param:ln1_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="fft_only",
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
        recompute=True,
        recompute_from=["ln1", "@param:qkv_weight", "?@param:qkv_bias"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "qkv", "transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["q", "k", "v"],
        share_policy="fft_share",  # Share only in FFT mode
    )
    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_outputs=["qkv_rope", "q_rstd", "k_rstd"],
        recompute_from=[
            "qkv",
            "@global:freq_cis",
            "@input:position_ids",
            "?@param:q_norm_weight",
            "?@param:k_norm_weight",
        ],
        recompute_op="qkv_qk_norm_rope",
        recompute_attrs={"rotary_dim": "D"},
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="QKV after QK-Norm + RoPE",
    )

    # QK-norm RSTDs (conditional on use_qk_norm)
    q_rstd = Activation(
        Tensor["B", "T", "Hq"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        when="use_qk_norm",
        description="Q head RMSNorm rstd",
    )
    k_rstd = Activation(
        Tensor["B", "T", "Hkv"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        when="use_qk_norm",
        description="K head RMSNorm rstd",
    )

    # Attention
    att = Activation(
        Tensor["B", "T", "AttnDim"],
        aliases=["att_flat", "attn"],
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_outputs=["att", "lse"],
        recompute_from=["qkv_rope"],
        recompute_op="flash_attention",
        recompute_attrs={"attn_impl": "cudnn"},
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Attention output (pre out-proj)",
    )
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Log-sum-exp from flash attention",
    )
    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        recompute=True,
        recompute_from=["att", "@param:out_weight"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "attn_out", "transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["o"],
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
        recompute=True,
        recompute_group="ln2_fused",
        recompute_outputs=["res_att", "ln2"],
        recompute_from=["res_ffn", "att_out", "ln2_rstd", "@param:ln2_weight"],
        recompute_op="fused_residual_rmsnorm_apply_saved",
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
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
        recompute=True,
        recompute_from=["ln2", "@param:router_weight"],
        recompute_op="matmul",
        recompute_attrs={"transpose": "NT"},
        # Router matmul uses quantized weights under QLoRA; skip recompute in LoRA mode.
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Router logits before softmax/sigmoid",
    )
    router_probs = Activation(
        Tensor["B * T", "E"],
        save=True,
        recompute=True,
        recompute_from=["router_logits"],
        recompute_op="moe_router_probs",
        # Depends on router_logits; skip recompute in LoRA mode.
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Router probabilities after normalization",
    )

    # Routing (top-k selection)
    routing_weights = Activation(
        Tensor["B * T", "K"],
        save=True,
        recompute=True,
        recompute_group="moe_topk",
        recompute_outputs=["routing_weights", "routing_indices"],
        recompute_from=["router_probs"],
        recompute_op="moe_topk",
        recompute_attrs={"top_k": "K", "normalize": "norm_topk_prob"},
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Routing weights for selected experts",
    )
    routing_indices = Activation(
        Tensor["B * T", "K"],
        dtype="int32",
        save=True,
        recompute=True,
        recompute_group="moe_topk",
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
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
        # Recompute only in FFT mode; in LoRA/QLoRA use saved tensors to avoid
        # recompute drift and quantized-weight path mismatches.
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Permuted input for grouped GEMM",
    )
    scatter_indices = Activation(
        Tensor["B * T * K"],
        dtype="int32",
        save=True,
        recompute=True,
        recompute_group="moe_permute",
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Indices for scattering back to original order",
    )

    # Expert computations
    expert_gate_up = Activation(
        Tensor["B * T * K", "MUp"],
        save=True,
        recompute=True,
        recompute_from=["permuted_input", "@param:experts_gate_up", "scatter_indices"],
        recompute_op="moe_grouped_gemm_gate_up",
        # Expert GEMM uses quantized weights under QLoRA; skip recompute in LoRA mode.
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert gate+up projection output",
    )
    expert_act = Activation(
        Tensor["B * T * K", "M"],
        save=True,
        recompute=True,
        recompute_from=["expert_gate_up"],
        recompute_op="swiglu",
        recompute_attrs={"activation": "swiglu"},
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert SwiGLU activation output",
    )
    expert_down = Activation(
        Tensor["B * T * K", "C"],
        save=True,
        recompute=True,
        recompute_from=["expert_act", "@param:experts_down", "scatter_indices"],
        recompute_op="moe_grouped_gemm_down",
        # Expert GEMM uses quantized weights under QLoRA; skip recompute in LoRA mode.
        recompute_policy="fft_only",
        share_policy="fft_share",  # Share only in FFT mode
        description="Expert down projection output",
    )

    # MoE output
    moe_out = Activation(
        Tensor["B * T", "C"],
        aliases=["moe_out_flat"],
        save=True,
        recompute=True,
        recompute_from=["expert_down", "routing_weights", "scatter_indices"],
        recompute_op="moe_unpermute",
        recompute_attrs={"top_k": "K"},
        recompute_policy="fft_only",
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

            # Grouped GEMM for gate+up projection
            expert_gate_up = g.moe_grouped_gemm_gate_up(
                permuted_input, "experts_gate_up", scatter_indices, out_name="expert_gate_up",
            )

            # SwiGLU activation
            expert_act = g.swiglu(expert_gate_up, out_name="expert_act")

            # Grouped GEMM for down projection
            expert_down = g.moe_grouped_gemm_down(
                expert_act, "experts_down", scatter_indices, out_name="expert_down",
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
