"""Qwen3 MoE Model."""

from __future__ import annotations

from ..tensor_type import Tensor, Array
from ..decorators import model, forward, hf_config, Param, Activation, Gradient
from ..graph_builder import graph
from ..hf import fuse, stack_experts


@model
@hf_config(
    architecture="Qwen3MoeForCausalLM",
    model_type="qwen3_moe",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="moe_intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    use_qkv_bias="attention_bias",
    num_experts="num_experts",
    num_experts_per_tok="num_experts_per_tok",
    # Shared expert config (optional in HF config)
    shared_expert_intermediate="shared_expert_intermediate_size",
)
class Qwen3MoEModel:
    """Qwen3 Mixture of Experts model.

    All layers use MoE (decoder_sparse_step=1, mlp_only_layers=[]).

    Features:
        - QK normalization in attention
        - Qwen3-style routing with norm_topk_prob=True
        - Optional shared expert
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 2048,
        n_layers: int = 28,
        num_query_heads: int = 16,
        num_kv_heads: int = 4,
        d_ff: int = 1408,  # Per-expert intermediate size (moe_intermediate_size)
        max_seq: int = 40960,
        head_size: int = 128,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        use_shared_expert: bool = False,
        shared_expert_intermediate: int = 0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.head_size = head_size
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.use_shared_expert = use_shared_expert
        self.shared_expert_intermediate = shared_expert_intermediate

        # Derived
        self.D = head_size if head_size > 0 else d_model // num_query_heads

    # Model weights
    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="model.embed_tokens.weight")
    blocks = Param(Array["n_layers", "Qwen3MoEBlock"])
    final_norm = Param(Tensor["d_model"], hf_mapping="model.norm.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")

    # =========================================================================
    # IO slots (runtime-provided inputs/outputs)
    # =========================================================================

    token_ids = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                           description="Input token IDs")
    position_ids = Activation(Tensor["T"], dtype="int32", scope="global",
                              description="Position IDs for RoPE")
    targets = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                         aliases=["labels"], description="Target labels for loss")

    # Precomputed constants
    freq_cis = Activation(Tensor["max_seq", "D", 2], dtype="fp32", scope="global",
                          aliases=["rope_freqs"], description="Precomputed RoPE frequencies")

    # =========================================================================
    # Global activation slots (model-level, not per-block)
    # =========================================================================

    # Embedding output
    x0 = Activation(Tensor["B", "T", "d_model"], aliases=["encoded"], scope="global",
                    description="Embedded input (after embedding lookup)")
    residual0 = Activation(Tensor["B", "T", "d_model"], scope="global",
                           description="Initial residual stream (zeros)")

    # After stacked blocks
    xN = Activation(Tensor["B", "T", "d_model"], scope="global",
                    description="Output from stacked blocks")
    residualN = Activation(Tensor["B", "T", "d_model"], scope="global",
                           description="Residual after stacked blocks")

    # Final normalization
    residual_final = Activation(Tensor["B", "T", "d_model"], scope="global",
                                description="Final residual (before norm)")
    xF = Activation(Tensor["B", "T", "d_model"], aliases=["ln_final"],
                    scope="global", description="After final layer norm")
    xF_flat = Activation(Tensor["B * T", "d_model"], scope="global",
                         description="Flattened for LM head")
    ln_final_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                               scope="global", description="Final LN rstd")

    # Loss
    loss = Activation(Tensor["B * T"], dtype="fp32", scope="global",
                      aliases=["losses"], description="Cross-entropy loss per token")

    # =========================================================================
    # Global gradient slots
    # =========================================================================

    d_loss = Gradient(Tensor["B * T"], dtype="fp32", gradient_of="loss", scope="global",
                      description="Gradient seed for backward pass")
    d_xF = Gradient(Tensor["B", "T", "d_model"], gradient_of="xF", scope="global")
    d_xN = Gradient(Tensor["B", "T", "d_model"], gradient_of="xN", scope="global")
    d_residualN = Gradient(Tensor["B", "T", "d_model"], gradient_of="residualN", scope="global")
    d_x0 = Gradient(Tensor["B", "T", "d_model"], gradient_of="x0", scope="global")

    # HF mappings for block parameters
    _hf_block_mappings_ = {
        # Layer norms
        "ln1_weight": "model.layers.{layer}.input_layernorm.weight",
        "ln2_weight": "model.layers.{layer}.post_attention_layernorm.weight",

        # Attention (same as Qwen3)
        "qkv_weight": fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0,
        ),
        "qkv_bias": fuse(
            "model.layers.{layer}.self_attn.q_proj.bias",
            "model.layers.{layer}.self_attn.k_proj.bias",
            "model.layers.{layer}.self_attn.v_proj.bias",
            dim=0,
        ),
        "out_weight": "model.layers.{layer}.self_attn.o_proj.weight",
        "q_norm_weight": "model.layers.{layer}.self_attn.q_norm.weight",
        "k_norm_weight": "model.layers.{layer}.self_attn.k_norm.weight",

        # Router
        "router_weight": "model.layers.{layer}.mlp.gate.weight",

        # Experts (batched format) - stack individual expert weights from HF format
        # NOTE: Param names must match C++ TensorTarget expectations in weight_mapping.cpp
        "experts_gate_up": stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
            fuse_gate_up=True,  # Also load up_proj and fuse into gate_up
        ),
        "experts_down": stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        ),

        # Shared expert (optional)
        # NOTE: Param names must match C++ TensorTarget expectations in weight_mapping.cpp
        "shared_expert_gate": "model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
        "shared_expert_up": "model.layers.{layer}.mlp.shared_expert.up_proj.weight",
        "shared_expert_down": "model.layers.{layer}.mlp.shared_expert.down_proj.weight",
    }

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor["T", "int32"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            # Embedding lookup
            x0 = g.embedding(token_ids, "embedding")

            # Initialize residual stream
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            # Stacked MoE blocks (handled by runtime)
            xN, residualN = g.call(
                "StackedBlocks",
                x0,
                residual0,
                position_ids,
                num_outputs=2,
                blocks="blocks",
                n_layers=self.n_layers,
            )

            # Final norm - use explicit names for outputs to map to C++ activation slots
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN, xN, "final_norm", eps=self.eps,
                y_name="xF"  # Map to LNFinal slot in C++
            )

            # Fused LM head + loss
            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            loss = g.fused_lm_head_loss(xF_flat, "lm_head", targets,
                                        compute_accuracy=True, out_name="loss")

            return loss
