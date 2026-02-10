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

import math

from ..tensor_type import Tensor, Array
from ..decorators import model, forward, hf_config, Param, Activation, Gradient
from ..graph_builder import graph
from ..hf import (
    build_mamba_mappings, build_simple_mlp_mappings, build_attn_mappings,
    stack_experts,
)


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


# Standard hybrid pattern alphabet used across the DSL/C++ boundary:
#   M = Mamba, A = Attention, P = MLP (Plain), E = MoE
_NEMOTRON_TO_STANDARD = str.maketrans({"*": "A", "-": "P"})


def to_standard_hybrid_pattern(nemotron_pattern: str) -> str:
    """Translate Nemotron's hybrid_override_pattern to the standard alphabet.

    Nemotron uses ``*`` for Attention and ``-`` for MLP.  The standard
    pattern recognised by the C++ runtime uses ``A`` and ``P`` instead.
    ``M`` (Mamba) and ``E`` (MoE) are the same in both formats.
    """
    return nemotron_pattern.translate(_NEMOTRON_TO_STANDARD)


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
    time_step_limit="time_step_limit",
    time_step_min="time_step_min",
    time_step_max="time_step_max",
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
    shared_expert_intermediate_size="moe_shared_expert_intermediate_size",
    # Router scaling
    routed_scaling_factor="routed_scaling_factor",
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
        time_step_limit: tuple[float, float] | None = None,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        # Common params
        max_seq: int = 4096,
        eps: float = 1e-5,
        use_rope: bool = False,
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
        shared_expert_intermediate_size: int = 0,
        # Router scaling
        routed_scaling_factor: float = 1.0,
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
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.max_seq = max_seq
        self.eps = eps
        # Store in standard alphabet (M/A/P/E) for export to C++ runtime
        self.hybrid_pattern = to_standard_hybrid_pattern(hybrid_pattern)
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_conv_bias = use_conv_bias
        self.use_mamba_bias = use_mamba_bias
        # Alias for mamba block's when="use_bias" condition evaluation
        self.use_bias = use_mamba_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.mlp_activation = mlp_activation

        # Nemotron-H attention does not use RoPE (Mamba provides positional info)
        self.use_rope = use_rope

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
    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="backbone.embeddings.weight", quantizable=False)
    final_norm = Param(Tensor["d_model"], hf_mapping="backbone.norm_f.weight", quantizable=False)
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight", quantizable=False)

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

    # Precomputed constants (for attention blocks that use RoPE)
    freq_cis = Activation(Tensor["max_seq", "D", 2], dtype="fp32", scope="global",
                          aliases=["rope_freqs"], description="Precomputed RoPE frequencies",
                          when="use_rope")

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

    # HuggingFace weight mappings for each block type.
    # Composed from module-level _hf_mapping_defaults_ where possible.
    # Note: Nemotron uses 'backbone.layers' prefix and 'mixer' submodule
    # (not 'model.layers' / 'self_attn' / 'mlp').
    _hf_block_mappings_ = {
        # Common to all blocks - Nemotron uses 'norm.weight' directly
        "norm_weight": "backbone.layers.{layer}.norm.weight",

        # Mamba2 block weights (from Mamba2Mixer._hf_mapping_defaults_)
        **build_mamba_mappings(
            layer_prefix="backbone.layers.{layer}",
            mamba_suffix="mixer",
        ),

        # Attention block weights (from GQAAttention._hf_mapping_defaults_)
        **build_attn_mappings(
            layer_prefix="backbone.layers.{layer}",
            attn_suffix="mixer",
        ),
        # Nemotron attention also has output bias (beyond GQAAttention defaults)
        "out_bias": "backbone.layers.{layer}.mixer.o_proj.bias",

        # MLP block weights (from SimpleMLP._hf_mapping_defaults_)
        **build_simple_mlp_mappings(
            layer_prefix="backbone.layers.{layer}",
            mlp_suffix="mixer",
        ),

        # MoE block weights (NemotronMoEBlock uses experts_up, not experts_gate_up)
        # Nemotron MoE uses relu2 activation (no gate), so only up_proj and down_proj
        "router_weight": "backbone.layers.{layer}.mixer.gate.weight",
        "e_score_correction_bias": "backbone.layers.{layer}.mixer.gate.e_score_correction_bias",
        "experts_up": stack_experts(
            "backbone.layers.{layer}.mixer.experts.{expert}.up_proj.weight",
        ),
        "experts_down": stack_experts(
            "backbone.layers.{layer}.mixer.experts.{expert}.down_proj.weight",
        ),
        # Shared expert (optional, present when use_shared_expert=True)
        "shared_expert_up": "backbone.layers.{layer}.mixer.shared_experts.up_proj.weight",
        "shared_expert_down": "backbone.layers.{layer}.mixer.shared_experts.down_proj.weight",
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
    dt_max_default = 1e9
    time_step_min = config.get("time_step_min", 0.001)
    time_step_max = config.get("time_step_max", 0.1)
    time_step_limit = config.get("time_step_limit")
    if not time_step_limit:
        time_step_limit = (0.0, dt_max_default)
    elif isinstance(time_step_limit, (list, tuple)) and len(time_step_limit) == 2:
        lo = float(time_step_limit[0])
        hi = float(time_step_limit[1])
        if not math.isfinite(lo):
            lo = 0.0
        if not math.isfinite(hi):
            hi = dt_max_default
        time_step_limit = (lo, hi)
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
        time_step_limit=time_step_limit,
        time_step_min=time_step_min,
        time_step_max=time_step_max,
        max_seq=config.get("max_position_embeddings", 4096),
        eps=config.get("layer_norm_epsilon", 1e-5),
        hybrid_pattern=config.get("hybrid_override_pattern",
                                  "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"),
        attention_bias=config.get("attention_bias", False),
        mlp_bias=config.get("mlp_bias", False),
        use_conv_bias=config.get("use_conv_bias", True),
        use_mamba_bias=config.get("use_bias", False),
        num_experts=config.get("n_routed_experts", 0),
        num_experts_per_tok=config.get("num_experts_per_tok", 2),
        moe_intermediate_size=config.get("moe_intermediate_size", 7688),
        shared_expert_intermediate_size=config.get("moe_shared_expert_intermediate_size", 0),
        routed_scaling_factor=config.get("routed_scaling_factor", 1.0),
        mlp_activation=config.get("mlp_hidden_act", "relu2"),
    )
