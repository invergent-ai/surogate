"""NemotronH Hybrid Architecture Blocks.

Nemotron-H uses a hybrid architecture with interleaved block types:
- M = Mamba2 block (State Space Model)
- * = Attention block (GQA)
- - = MLP block (dense feed-forward)
- E = MoE block (Mixture of Experts)

Each block has the structure:
    residual, x = fused_residual_rmsnorm(residual, x)
    x = mixer(x)  # mixer depends on block type
    # residual connection handled in next block's norm
"""

from __future__ import annotations

from .. import nn
from ..dim import B, Dim, T
from ..nn import (
    NEMOTRON_ATTN_BLOCK_REMAP,
    NEMOTRON_MAMBA_BLOCK_REMAP,
    NEMOTRON_MLP_BLOCK_REMAP,
    NEMOTRON_MOE_BLOCK_REMAP,
)


class NemotronHMamba2Block(nn.Block):
    """Mamba2 block for Nemotron-H hybrid architecture."""

    _name_remap_ = NEMOTRON_MAMBA_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        ssm_state_size: int = 128,
        n_groups: int = 8,
        conv_kernel: int = 4,
        chunk_size: int = 256,
        eps: float = 1e-5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        time_step_limit: tuple[float, float] | None = None,
        use_conv_bias: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.eps = eps
        self.use_conv_bias = use_conv_bias
        self.use_bias = use_bias

        # Derived dimensions
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.conv_dim = self.intermediate_size + 2 * n_groups * ssm_state_size
        self.projection_size = self.intermediate_size + self.conv_dim + mamba_num_heads
        self.C = d_model
        self.I = self.intermediate_size
        self.H = mamba_num_heads
        self.D = mamba_head_dim
        self.N = ssm_state_size
        self.G = n_groups
        self.K = conv_kernel
        self.P = self.projection_size
        self.D_conv = self.conv_dim

        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.mixer = nn.Mamba2Mixer(
            d_model,
            mamba_num_heads=mamba_num_heads,
            mamba_head_dim=mamba_head_dim,
            ssm_state_size=ssm_state_size,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            eps=eps,
            dt_min=dt_min,
            dt_max=dt_max,
            time_step_limit=time_step_limit,
            use_conv_bias=use_conv_bias,
            use_bias=use_bias,
        )

    def forward(self, x, residual):
        residual, h = self.norm(residual, x)
        h = self.mixer(h)
        return h, residual


class NemotronHAttentionBlock(nn.Block):
    """Attention block for Nemotron-H hybrid architecture."""

    _name_remap_ = NEMOTRON_ATTN_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq: int = 4096,
        eps: float = 1e-5,
        attention_bias: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.eps = eps
        self.attention_bias = attention_bias
        self.use_rope = use_rope

        # Dimension aliases for shape resolution
        self.C = d_model
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.D = head_dim
        self.MaxSeq = max_seq
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_dim
        self.AttnDim = num_query_heads * head_dim

        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.mixer = nn.NemotronAttention(
            d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            max_seq=max_seq,
            attention_bias=attention_bias,
            use_rope=use_rope,
        )

    def forward(self, x, residual, position_ids):
        residual, h = self.norm(residual, x)
        h = self.mixer(h, position_ids)
        return h, residual


class NemotronHMLPBlock(nn.Block):
    """MLP block for Nemotron-H hybrid architecture."""

    _name_remap_ = NEMOTRON_MLP_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        eps: float = 1e-5,
        activation: str = "relu2",
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.eps = eps
        self.mlp_bias = mlp_bias
        self.activation = activation

        # Dimension aliases for shape resolution
        self.C = d_model
        self.M = d_ff

        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.mixer = nn.SimpleMLP(
            d_model,
            d_ff,
            activation=activation,
            use_bias=mlp_bias,
        )

    def forward(self, x, residual):
        residual, h = self.norm(residual, x)
        h = self.mixer(h)
        return h, residual


class NemotronHMoEBlock(nn.Block):
    """MoE block for Nemotron-H hybrid architecture."""

    _name_remap_ = NEMOTRON_MOE_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        shared_expert_intermediate_size: int = 0,
        eps: float = 1e-5,
        mlp_bias: bool = False,
        activation: str = "relu2",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.use_shared_expert = shared_expert_intermediate_size > 0
        self.eps = eps
        self.mlp_bias = mlp_bias
        self.activation = activation
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size

        # Dimension aliases for shape resolution
        self.C = d_model
        self.M = moe_intermediate_size
        self.E = num_experts
        self.K = num_experts_per_tok
        self.SharedM = shared_expert_intermediate_size if shared_expert_intermediate_size > 0 else moe_intermediate_size

        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.mixer = nn.NemotronMoEExperts(
            d_model,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            activation=activation,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )
        if self.use_shared_expert:
            self.shared_expert = nn.NemotronSharedExpert(
                d_model,
                shared_expert_intermediate_size,
                activation=activation,
            )

    def forward(self, x, residual):
        C = Dim("C")
        residual, h = self.norm(residual, x)
        h_flat = self._view(h, [B * T, C], name="ln_flat")
        moe_out = self.mixer(h_flat)
        if self.use_shared_expert:
            shared_out = self.shared_expert(h_flat)
            moe_out = self._add(moe_out, shared_out, name="moe_combined")
        self._register_activation(
            "out",
            ("B", "T", "C"),
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        out = self._view(moe_out, [B, T, C], name="out")
        return out, residual
