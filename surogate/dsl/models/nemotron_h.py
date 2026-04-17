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

from .. import nn
from ..blocks.nemotron_h import (
    NemotronHAttentionBlock,
    NemotronHMamba2Block,
    NemotronHMLPBlock,
    NemotronHMoEBlock,
)
from ..hf import (
    build_attn_mappings,
    build_mamba_mappings,
    build_simple_mlp_mappings,
    stack_experts,
)
from ..nn import NEMOTRON_MODEL_NAME_REMAP
from ..specs import ActivationScope


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
            raise ValueError(f"Invalid character '{char}' in hybrid_override_pattern. Expected 'M', '*', '-', or 'E'")
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


@nn.hf_config(
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
class NemotronHModel(nn.Model):
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

    _name_remap_ = NEMOTRON_MODEL_NAME_REMAP

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
        # Model-level weight mappings
        "embedding": "backbone.embeddings.weight",
        "final_norm": "backbone.norm_f.weight",
        "lm_head": "lm_head.weight",
    }

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
        # Expert parallel
        ep_size: int = 1,
        # Activation
        mlp_activation: str = "relu2",
    ):
        super().__init__()
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
        self.ep_size = ep_size
        self.mlp_activation = mlp_activation

        # Nemotron-H attention does not use RoPE (Mamba provides positional info)
        self.use_rope = use_rope

        # Parse hybrid pattern to get block types
        self.block_types = parse_hybrid_pattern(hybrid_pattern)
        if len(self.block_types) != n_layers:
            raise ValueError(
                f"hybrid_override_pattern length ({len(self.block_types)}) must match n_layers ({n_layers})"
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

        # Build block configs for HybridBlockStack
        block_configs = []
        if self.has_mamba_blocks:
            block_configs.append(
                (
                    "mamba_blocks",
                    NemotronHMamba2Block,
                    self.n_mamba_blocks,
                    dict(
                        d_model=d_model,
                        mamba_num_heads=mamba_num_heads,
                        mamba_head_dim=mamba_head_dim,
                        ssm_state_size=ssm_state_size,
                        n_groups=n_groups,
                        conv_kernel=conv_kernel,
                        chunk_size=chunk_size,
                        eps=eps,
                        dt_min=time_step_min,
                        dt_max=time_step_max,
                        time_step_limit=time_step_limit,
                        use_conv_bias=use_conv_bias,
                        use_bias=use_mamba_bias,
                    ),
                )
            )
        if self.has_attn_blocks:
            block_configs.append(
                (
                    "attn_blocks",
                    NemotronHAttentionBlock,
                    self.n_attn_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        max_seq=max_seq,
                        eps=eps,
                        attention_bias=attention_bias,
                        use_rope=use_rope,
                    ),
                )
            )
        if self.has_mlp_blocks:
            block_configs.append(
                (
                    "mlp_blocks",
                    NemotronHMLPBlock,
                    self.n_mlp_blocks,
                    dict(
                        d_model=d_model,
                        d_ff=d_ff,
                        eps=eps,
                        activation=mlp_activation,
                        mlp_bias=mlp_bias,
                    ),
                )
            )
        if self.has_moe_blocks:
            block_configs.append(
                (
                    "moe_blocks",
                    NemotronHMoEBlock,
                    self.n_moe_blocks,
                    dict(
                        d_model=d_model,
                        moe_intermediate_size=moe_intermediate_size,
                        num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok,
                        shared_expert_intermediate_size=shared_expert_intermediate_size,
                        eps=eps,
                        mlp_bias=mlp_bias,
                        activation=mlp_activation,
                        routed_scaling_factor=routed_scaling_factor,
                        ep_size=ep_size,
                    ),
                )
            )

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=self.block_types,
            n_layers=n_layers,
        )
        self.final_norm = nn.RMSNorm(d_model, eps=eps)
        self.lm_head = nn.LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        if self.use_rope:
            self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

        # Global intermediate slots
        _h = ("B", "T", "d_model")
        self._register_activation("residual0", _h, scope=G)
        self._register_activation("x0", _h, aliases=["encoded"], scope=G)
        self._register_activation("xN", _h, scope=G)
        self._register_activation("residualN", _h, scope=G)
        self._register_activation("residual_final", _h, scope=G)
        self._register_activation("xF", _h, aliases=["ln_final"], scope=G)
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

        x = self.embedding(token_ids)
        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
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
        hybrid_pattern=config.get("hybrid_override_pattern", "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"),
        attention_bias=config.get("attention_bias", False),
        mlp_bias=config.get("mlp_bias", False),
        use_conv_bias=config.get("use_conv_bias", True),
        use_mamba_bias=config.get("use_bias", False),
        num_experts=config.get("n_routed_experts", 0),
        num_experts_per_tok=config.get("num_experts_per_tok", 2),
        moe_intermediate_size=config.get("moe_intermediate_size", 7688),
        shared_expert_intermediate_size=config.get("moe_shared_expert_intermediate_size", 0),
        routed_scaling_factor=config.get("routed_scaling_factor", 1.0),
        ep_size=config.get("ep_size", 1),
        mlp_activation=config.get("mlp_hidden_act", "relu2"),
    )
