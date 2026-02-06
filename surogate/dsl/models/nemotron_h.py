"""Nemotron-H Hybrid Model.

Nemotron-H is a hybrid architecture that interleaves different block types:
- M = Mamba2 (State Space Model)
- * = Attention (GQA)
- - = MLP (dense feed-forward)
- E = MoE (Mixture of Experts)

The hybrid_override_pattern string defines the sequence, e.g.:
"M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
or with MoE blocks:
"MEMEMEME*EMEMEME*EMEMEME*EMEMEME*EMEMEME*EMEME"

This allows mixing the efficiency of SSMs with the expressiveness of attention.
"""

from __future__ import annotations

from ..tensor_type import Tensor, Array
from ..decorators import model, forward, hf_config, Param, Activation, Gradient
from ..graph_builder import graph
from ..hf import fuse, stack_experts


def parse_hybrid_pattern(pattern: str) -> list[str]:
    """Parse hybrid_override_pattern into list of block types.

    Args:
        pattern: String like "M-M-M-M*-M-M-M" or "MEMEMEME*EMEMEME" where:
            M = Mamba2 block
            * = Attention block
            - = MLP block
            E = MoE block

    Returns:
        List of block types: ["mamba", "mlp", "mamba", ..., "attention", "moe", ...]
    """
    block_types = []
    for char in pattern:
        if char == "M":
            block_types.append("mamba")
        elif char == "*":
            block_types.append("attention")
        elif char == "-":
            block_types.append("mlp")
        elif char == "E":
            block_types.append("moe")
        else:
            raise ValueError(f"Invalid character '{char}' in hybrid_override_pattern. "
                           f"Expected 'M', '*', '-', or 'E'")
    return block_types


@model
@hf_config(
    architecture="NemotronHForCausalLM",
    model_type="nemotron_h",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    # Attention config
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    head_dim="head_dim",
    # MLP config
    d_ff="intermediate_size",
    # Mamba config
    mamba_num_heads="mamba_num_heads",
    mamba_head_dim="mamba_head_dim",
    ssm_state_size="ssm_state_size",
    n_groups="n_groups",
    conv_kernel="conv_kernel",
    chunk_size="chunk_size",
    # Common config
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    eps="layer_norm_epsilon",
    # Hybrid pattern
    hybrid_pattern="hybrid_override_pattern",
    # MoE config (optional)
    num_experts="n_routed_experts",
    num_experts_per_tok="num_experts_per_tok",
    moe_intermediate_size="moe_intermediate_size",
    shared_expert_intermediate="moe_shared_expert_intermediate_size",
    # Activation (for mlp_up_factor determination)
    mlp_activation="mlp_hidden_act",
)
class NemotronHModel:
    """Nemotron-H hybrid model with interleaved Mamba2, Attention, MLP, and MoE blocks.

    Architecture:
        - Embedding layer
        - N hybrid blocks (type determined by hybrid_override_pattern)
        - Final layer norm
        - LM head

    Each block type:
        - M (Mamba2): SSM-based sequence mixing
        - * (Attention): GQA attention
        - - (MLP): Dense feed-forward only
        - E (MoE): Mixture of Experts
    """

    def __init__(
        self,
        vocab_size: int = 131072,
        d_model: int = 4096,
        n_layers: int = 52,
        # Attention params
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        # MLP params
        d_ff: int = 21504,
        # Mamba params
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        ssm_state_size: int = 128,
        n_groups: int = 8,
        conv_kernel: int = 4,
        chunk_size: int = 128,
        # Common params
        max_seq: int = 4096,
        eps: float = 1e-5,
        # Hybrid pattern
        hybrid_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
        # Bias options
        attention_bias: bool = False,
        mlp_bias: bool = False,
        use_conv_bias: bool = True,
        use_mamba_bias: bool = False,
        # MoE options (for hybrid patterns with MoE)
        num_experts: int = 0,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 7688,
        shared_expert_intermediate: int = 0,
        # Activation
        mlp_activation: str = "relu2",
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.d_ff = d_ff
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.max_seq = max_seq
        self.eps = eps
        self.hybrid_pattern = hybrid_pattern
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_conv_bias = use_conv_bias
        self.use_mamba_bias = use_mamba_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate = shared_expert_intermediate
        self.mlp_activation = mlp_activation

        # Parse hybrid pattern to get block types
        self.block_types = parse_hybrid_pattern(hybrid_pattern)
        if len(self.block_types) != n_layers:
            raise ValueError(
                f"hybrid_override_pattern length ({len(self.block_types)}) "
                f"must match n_layers ({n_layers})"
            )

        # Count block types for array parameters
        self.n_mamba_blocks = sum(1 for t in self.block_types if t == "mamba")
        self.n_attn_blocks = sum(1 for t in self.block_types if t == "attention")
        self.n_mlp_blocks = sum(1 for t in self.block_types if t == "mlp")
        self.n_moe_blocks = sum(1 for t in self.block_types if t == "moe")

        # Boolean flags for conditional params (some block types may not be present)
        self.has_mamba_blocks = self.n_mamba_blocks > 0
        self.has_attn_blocks = self.n_attn_blocks > 0
        self.has_mlp_blocks = self.n_mlp_blocks > 0
        self.has_moe_blocks = self.n_moe_blocks > 0

        # Derived dimensions
        self.D = head_dim if head_dim > 0 else d_model // num_query_heads
        self.mamba_intermediate = mamba_num_heads * mamba_head_dim
        self.mamba_conv_dim = self.mamba_intermediate + 2 * n_groups * ssm_state_size
        self.mamba_proj_size = self.mamba_intermediate + self.mamba_conv_dim + mamba_num_heads

    # Model weights
    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="backbone.embeddings.weight")
    final_norm = Param(Tensor["d_model"], hf_mapping="backbone.norm_f.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")

    # Block arrays - using specialized block types based on pattern
    # The runtime will handle dispatching to correct block based on layer index
    # Each array is conditional on having at least one block of that type
    mamba_blocks = Param(Array["n_mamba_blocks", "NemotronHMamba2Block"], when="has_mamba_blocks")
    attn_blocks = Param(Array["n_attn_blocks", "NemotronHAttentionBlock"], when="has_attn_blocks")
    mlp_blocks = Param(Array["n_mlp_blocks", "NemotronHMLPBlock"], when="has_mlp_blocks")
    moe_blocks = Param(Array["n_moe_blocks", "NemotronHMoEBlock"], when="has_moe_blocks")

    # =========================================================================
    # IO slots
    # =========================================================================

    token_ids = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                           description="Input token IDs")
    position_ids = Activation(Tensor["T"], dtype="int32", scope="global",
                              description="Position IDs for RoPE")
    targets = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                         aliases=["labels"], description="Target labels for loss")

    # Precomputed constants (for attention blocks)
    freq_cis = Activation(Tensor["max_seq", "D", 2], dtype="fp32", scope="global",
                          aliases=["rope_freqs"], description="Precomputed RoPE frequencies")

    # =========================================================================
    # Global activation slots
    # =========================================================================

    x0 = Activation(Tensor["B", "T", "d_model"], aliases=["encoded"], scope="global",
                    description="Embedded input")
    residual0 = Activation(Tensor["B", "T", "d_model"], scope="global",
                           description="Initial residual stream (zeros)")

    xN = Activation(Tensor["B", "T", "d_model"], scope="global",
                    description="Output from stacked blocks")
    residualN = Activation(Tensor["B", "T", "d_model"], scope="global",
                           description="Residual after stacked blocks")

    residual_final = Activation(Tensor["B", "T", "d_model"], scope="global",
                                description="Final residual (before norm)")
    xF = Activation(Tensor["B", "T", "d_model"], aliases=["ln_final"],
                    scope="global", description="After final layer norm")
    xF_flat = Activation(Tensor["B * T", "d_model"], scope="global",
                         description="Flattened for LM head")
    ln_final_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                               scope="global", description="Final LN rstd")

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

    # HuggingFace weight mappings for each block type
    # These use layer-specific patterns for the hybrid architecture
    # Note: Nemotron uses 'backbone.layers' prefix (not 'model.layers')
    _hf_block_mappings_ = {
        # Common to all blocks - Nemotron uses 'norm.weight' directly
        "norm_weight": "backbone.layers.{layer}.norm.weight",

        # Mamba2 block weights
        "in_proj_weight": "backbone.layers.{layer}.mixer.in_proj.weight",
        "in_proj_bias": "backbone.layers.{layer}.mixer.in_proj.bias",
        "conv_weight": "backbone.layers.{layer}.mixer.conv1d.weight",
        "conv_bias": "backbone.layers.{layer}.mixer.conv1d.bias",
        "A_log": "backbone.layers.{layer}.mixer.A_log",
        "D_param": "backbone.layers.{layer}.mixer.D",
        "dt_bias": "backbone.layers.{layer}.mixer.dt_bias",
        "gated_norm_weight": "backbone.layers.{layer}.mixer.norm.weight",
        "out_proj_weight": "backbone.layers.{layer}.mixer.out_proj.weight",
        "out_proj_bias": "backbone.layers.{layer}.mixer.out_proj.bias",

        # Attention block weights
        "qkv_weight": fuse(
            "backbone.layers.{layer}.mixer.q_proj.weight",
            "backbone.layers.{layer}.mixer.k_proj.weight",
            "backbone.layers.{layer}.mixer.v_proj.weight",
            dim=0,
        ),
        "qkv_bias": fuse(
            "backbone.layers.{layer}.mixer.q_proj.bias",
            "backbone.layers.{layer}.mixer.k_proj.bias",
            "backbone.layers.{layer}.mixer.v_proj.bias",
            dim=0,
        ),
        "out_weight": "backbone.layers.{layer}.mixer.o_proj.weight",
        "out_bias": "backbone.layers.{layer}.mixer.o_proj.bias",

        # MLP block weights (for dense MLP blocks)
        "up_weight": "backbone.layers.{layer}.mixer.up_proj.weight",
        "up_bias": "backbone.layers.{layer}.mixer.up_proj.bias",
        "down_weight": "backbone.layers.{layer}.mixer.down_proj.weight",
        "down_bias": "backbone.layers.{layer}.mixer.down_proj.bias",

        # MoE block weights
        # Nemotron MoE uses relu2 activation (no gate), so only up_proj and down_proj per expert
        # Use standard names (experts_gate_up, experts_down) for BnB loader compatibility
        "router_weight": "backbone.layers.{layer}.mixer.gate.weight",
        "experts_gate_up": stack_experts(
            "backbone.layers.{layer}.mixer.experts.{expert}.up_proj.weight",
            fuse_gate_up=False,  # No gate_proj in Nemotron MoE, just up_proj
        ),
        "experts_down": stack_experts(
            "backbone.layers.{layer}.mixer.experts.{expert}.down_proj.weight",
        ),
    }

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor["T", "int32"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            # Embedding
            x0 = g.embedding(token_ids, "embedding")
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            # Stacked hybrid blocks
            # The runtime handles dispatching to correct block type based on layer
            xN, residualN = g.call(
                "HybridStackedBlocks",
                x0,
                residual0,
                position_ids,
                num_outputs=2,
                mamba_blocks="mamba_blocks",
                attn_blocks="attn_blocks",
                mlp_blocks="mlp_blocks",
                moe_blocks="moe_blocks",
                block_types=self.block_types,
                n_layers=self.n_layers,
            )

            # Final norm
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN, xN, "final_norm", eps=self.eps,
                res_out_name="residual_final",
                y_name="xF",
                rstd_name="ln_final_rstd",
            )

            # Fused LM head + loss
            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            loss = g.fused_lm_head_loss(xF_flat, "lm_head", targets,
                                        compute_accuracy=True, out_name="loss")

            return loss


# Convenience function to create model from HuggingFace config
def from_hf_config(config: dict) -> NemotronHModel:
    """Create NemotronHModel from HuggingFace config dict.

    Args:
        config: HuggingFace config dictionary

    Returns:
        NemotronHModel instance
    """
    return NemotronHModel(
        vocab_size=config.get("vocab_size", 131072),
        d_model=config.get("hidden_size", 4096),
        n_layers=config.get("num_hidden_layers", 52),
        num_query_heads=config.get("num_attention_heads", 32),
        num_kv_heads=config.get("num_key_value_heads", 8),
        head_dim=config.get("head_dim", 128),
        d_ff=config.get("intermediate_size", 21504),
        mamba_num_heads=config.get("mamba_num_heads", 128),
        mamba_head_dim=config.get("mamba_head_dim", 64),
        ssm_state_size=config.get("ssm_state_size", 128),
        n_groups=config.get("n_groups", 8),
        conv_kernel=config.get("conv_kernel", 4),
        chunk_size=config.get("chunk_size", 128),
        max_seq=config.get("max_position_embeddings", 4096),
        eps=config.get("layer_norm_epsilon", 1e-5),
        hybrid_pattern=config.get("hybrid_override_pattern",
                                  "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"),
        attention_bias=config.get("attention_bias", False),
        mlp_bias=config.get("mlp_bias", False),
        use_conv_bias=config.get("use_conv_bias", True),
        use_mamba_bias=config.get("use_bias", False),
        mlp_activation=config.get("mlp_hidden_act", "relu2"),
    )
