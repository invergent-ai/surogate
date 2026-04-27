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
from ..block_schema import BlockSchema, DistributionDecl, EPTopology, RoutingSchema, SlotDecl, StreamingHint
from ..dim import B, Dim, T
from ..modules import Mamba2Mixer, NemotronAttention, NemotronMoEExperts, NemotronSharedExpert, RMSNorm, SimpleMLP
from .common import STANDARD_MODEL_NAME_REMAP

# Nemotron uses the standard model scope; alias for call-site clarity.
NEMOTRON_MODEL_NAME_REMAP = STANDARD_MODEL_NAME_REMAP

# Shared "norm → ln" prefixes used by every Nemotron block remap.
_NEMOTRON_NORM_IN = {
    "norm_weight": "norm_weight",
    "norm_res": "res_in",
    "norm_y": "ln",
    "norm_rstd": "ln_rstd",
}
_NEMOTRON_NORM_ATTN = {
    "norm_weight": "norm_weight",
    "norm_res": "res_att",
    "norm_y": "ln1",
    "norm_rstd": "ln1_rstd",
}


NEMOTRON_MAMBA_BLOCK_REMAP: dict[str, str] = {
    **_NEMOTRON_NORM_IN,
    # Strip mixer_ prefix from Mamba2Mixer params
    "mixer_in_proj_weight": "in_proj_weight",
    "mixer_in_proj_bias": "in_proj_bias",
    "mixer_conv_weight": "conv_weight",
    "mixer_conv_bias": "conv_bias",
    "mixer_A_log": "A_log",
    "mixer_D_param": "D_param",
    "mixer_dt_bias": "dt_bias",
    "mixer_gated_norm_weight": "gated_norm_weight",
    "mixer_out_proj_weight": "out_proj_weight",
    "mixer_out_proj_bias": "out_proj_bias",
    # Strip mixer_ prefix from Mamba2Mixer activations
    "mixer_x_flat": "x_flat",
    "mixer_projected_flat": "projected_flat",
    "mixer_projected": "projected",
    "mixer_gate": "gate",
    "mixer_conv_input": "conv_input",
    "mixer_dt": "dt",
    "mixer_conv_out": "conv_out",
    "mixer_hidden_states": "hidden_states",
    "mixer_ssm_B": "ssm_B",
    "mixer_ssm_C": "ssm_C",
    "mixer_ssm_out": "ssm_out",
    "mixer_ssm_out_flat": "ssm_out_flat",
    "mixer_ssm_state": "ssm_state",
    "mixer_gated_out": "gated_out",
    "mixer_gated_flat": "gated_flat",
    "mixer_out_flat": "out_flat",
    "mixer_out": "out",
}

NEMOTRON_ATTN_BLOCK_REMAP: dict[str, str] = {
    **_NEMOTRON_NORM_ATTN,
    # Strip mixer_ prefix from NemotronAttention params
    "mixer_qkv_weight": "qkv_weight",
    "mixer_qkv_bias": "qkv_bias",
    "mixer_out_weight": "out_weight",
    "mixer_out_bias": "out_bias",
    "mixer_rope_freqs": "rope_freqs",
    # Strip mixer_ prefix from NemotronAttention activations
    "mixer_x_flat": "x_flat",
    "mixer_qkv_flat": "qkv_flat",
    "mixer_qkv": "qkv",
    "mixer_qkv_rope": "qkv_rope",
    "mixer_att": "att",
    "mixer_att_flat": "att_flat",
    "mixer_attn": "attn",
    "mixer_lse": "lse",
    "mixer_att_out_flat": "att_out_flat",
    "mixer_att_out": "att_out",
}

NEMOTRON_MLP_BLOCK_REMAP: dict[str, str] = {
    **_NEMOTRON_NORM_IN,
    # Strip mixer_ prefix from SimpleMLP params
    "mixer_up_weight": "up_weight",
    "mixer_up_bias": "up_bias",
    "mixer_down_weight": "down_weight",
    "mixer_down_bias": "down_bias",
    # Strip mixer_ prefix from SimpleMLP activations
    "mixer_x_flat": "mlp_x_flat",
    "mixer_up_flat": "mlp_up_flat",
    "mixer_up": "mlp_up",
    "mixer_act": "swiglu",
    "mixer_act_flat": "swiglu_flat",
    "mixer_down_flat": "mlp_down_flat",
    "mixer_down": "mlp_down",
}

NEMOTRON_MOE_BLOCK_REMAP: dict[str, str] = {
    **_NEMOTRON_NORM_IN,
    # Strip mixer_ prefix from NemotronMoEExperts params
    "mixer_router_weight": "router_weight",
    "mixer_e_score_correction_bias": "e_score_correction_bias",
    "mixer_experts_up": "experts_up",
    "mixer_experts_down": "experts_down",
    # Strip mixer_ prefix from NemotronMoEExperts activations
    "mixer_router_logits": "router_logits",
    "mixer_router_probs": "router_probs",
    "mixer_routing_weights": "routing_weights",
    "mixer_routing_indices": "routing_indices",
    "mixer_permuted_input": "permuted_input",
    "mixer_scatter_indices": "scatter_indices",
    "mixer_ep_recv_input": "ep_recv_input",
    "mixer_ep_recv_scatter": "ep_recv_scatter",
    "mixer_expert_up": "expert_up",
    "mixer_expert_act": "expert_act",
    "mixer_expert_down": "expert_down",
    "mixer_ep_combined": "ep_combined",
    "mixer_out": "moe_out",
    "mixer_out_flat": "moe_out_flat",
    # Strip shared_expert_ prefix from NemotronSharedExpert params
    "shared_expert_up": "shared_expert_up",
    "shared_expert_down": "shared_expert_down",
    # Strip shared_expert_ prefix from activations
    "shared_expert_up_out": "shared_up_out",
    "shared_expert_act": "shared_act",
    "shared_expert_out": "shared_out",
}


class NemotronHMamba2Block(nn.Block):
    """Mamba2 block for Nemotron-H hybrid architecture."""

    _name_remap_ = NEMOTRON_MAMBA_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("projected", shape=("B", "T", "P"), save_for_backward=True),
            SlotDecl("gate", shape=("B", "T", "I"), save_for_backward=True),
            SlotDecl("conv_out", shape=("B", "D_conv", "T"), save_for_backward=True),
            SlotDecl("hidden_states", shape=("B", "I", "T"), save_for_backward=True),
            SlotDecl("ssm_out", shape=("B", "T", "I"), save_for_backward=True),
            SlotDecl("ssm_state", shape=("B", "H", "D", "N"), save_for_backward=True),
            SlotDecl("gated_out", shape=("B", "T", "I"), save_for_backward=True),
            SlotDecl("out", shape=("B", "T", "C")),
            SlotDecl(
                "in_proj_weight",
                kind="param",
                shape=("P", "C"),
                residency="auto",
                distribution=DistributionDecl.sharded_dim(dim=0, mode="zero2"),
                streaming_hint=StreamingHint(prefetch_distance=2),
            ),
            SlotDecl(
                "out_proj_weight",
                kind="param",
                shape=("C", "I"),
                residency="auto",
                distribution=DistributionDecl.sharded_dim(dim=0, mode="zero2"),
                streaming_hint=StreamingHint(prefetch_distance=2),
            ),
        ),
        attrs={"block_family": "nemotron_mamba2"},
    )

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

        self.norm = RMSNorm(d_model, eps=eps)
        self.mixer = Mamba2Mixer(
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
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("res_att", shape=("B", "T", "C")),
            SlotDecl("ln1", shape=("B", "T", "C"), save_for_backward=True),
            SlotDecl("ln1_rstd", shape=("B", "T"), dtype="fp32", save_for_backward=True),
            SlotDecl("qkv", shape=("B", "T", "QKV"), save_for_backward=True),
            SlotDecl("att", shape=("B", "T", "AttnDim"), save_for_backward=True),
            SlotDecl("lse", shape=("B", "Hq", "T"), dtype="fp32", save_for_backward=True),
            SlotDecl("att_out", shape=("B", "T", "C")),
        ),
        attrs={"block_family": "nemotron_attention"},
    )

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

        self.norm = RMSNorm(d_model, eps=eps)
        self.mixer = NemotronAttention(
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
    schema = BlockSchema(
        slots=(
            SlotDecl("up_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("mlp_up", shape=("B", "T", "M"), save_for_backward=True),
            SlotDecl("swiglu", shape=("B", "T", "M"), save_for_backward=True),
            SlotDecl("mlp_down", shape=("B", "T", "C")),
        ),
        attrs={"block_family": "nemotron_mlp"},
    )

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

        self.norm = RMSNorm(d_model, eps=eps)
        self.mixer = SimpleMLP(
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
    schema = BlockSchema(
        slots=(
            SlotDecl(
                "router_weight", kind="param", shape=("E", "C"), distribution=DistributionDecl.router_replicated()
            ),
            SlotDecl(
                "experts_up",
                kind="param",
                shape=("E", "M", "C"),
                residency="auto",
                distribution=DistributionDecl.expert_parallel(global_experts="num_experts"),
                grouped=True,
                streaming_hint=StreamingHint(prefetch_distance=1),
            ),
            SlotDecl(
                "experts_down",
                kind="param",
                shape=("E", "C", "M"),
                residency="auto",
                distribution=DistributionDecl.expert_parallel(global_experts="num_experts"),
                grouped=True,
                streaming_hint=StreamingHint(prefetch_distance=1),
            ),
            SlotDecl(
                "permuted_input", shape=("dispatched_tokens", "C"), distribution=DistributionDecl.expert_parallel()
            ),
        ),
        routing=RoutingSchema(
            kind="topk_sigmoid",
            topk="num_experts_per_tok",
            norm_topk_prob="norm_topk_prob",
            scoring_bias=True,
            shared_experts="use_shared_expert",
        ),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        attrs={"block_family": "nemotron_moe"},
    )

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

        self.norm = RMSNorm(d_model, eps=eps)
        self.mixer = NemotronMoEExperts(
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
            self.shared_expert = NemotronSharedExpert(
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
