"""Qwen3.5 MoE transformer blocks (hybrid full-attention + linear-attention with MoE)."""

from __future__ import annotations

from .. import nn
from ..dim import B, Dim, T
from ..nn import (
    QWEN3_5_MOE_ATTN_BLOCK_REMAP,
    QWEN3_5_MOE_LINEAR_BLOCK_REMAP,
)


class Qwen3_5MoEAttentionBlock(nn.Block):
    """Qwen3.5 MoE full-attention decoder block.

    Uses Qwen3_5Attention (gated output, partial MRoPE, QK-Norm with weight+1),
    RMSNormPlus1 norms, and MoE FFN with shared expert + sigmoid gate.
    """

    _name_remap_ = QWEN3_5_MOE_ATTN_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.partial_rotary_factor = partial_rotary_factor
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")

        # Derived dimensions for shape resolution
        self.D = head_size
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.M = d_ff
        self.MaxSeq = max_seq
        self.AttnDim = num_query_heads * head_size
        self.QProjDim = 2 * self.AttnDim
        self.KVDim = num_kv_heads * head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.RotaryDim = nn._resolve_rotary_dim(head_size, partial_rotary_factor)

        self.attn_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.self_attn = nn.Qwen3_5Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
            eps=eps,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
        )
        self.mlp_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.moe = nn.MoEExpertsGated(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob=False,
            ep_size=ep_size,
        )
        self.shared_expert = nn.MoESharedExpert(d_model, shared_expert_intermediate)

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.attn_norm(residual, x)
        # Attention
        h = self.self_attn(h, position_ids)
        # Pre-MoE normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.mlp_norm(residual, h)
        # Flatten for MoE
        h_flat = self._view(h, [B * T, self.C], name="ln2_flat")
        # MoE experts
        moe_out = self.moe(h_flat)
        # Shared expert with sigmoid gate
        shared_out = self.shared_expert(h_flat)
        self._register_param("shared_expert_gate_proj_weight", (1, "C"))
        shared_gate = self._matmul(h_flat, "shared_expert_gate_proj_weight", name="shared_expert_gate_proj")
        shared_gate = self._sigmoid(shared_gate, name="shared_expert_gate_sigmoid")
        shared_out = self._mul(shared_out, shared_gate, name="shared_expert_gated")
        moe_out = self._add(moe_out, shared_out, name="moe_combined")
        # Register output slot and reshape back
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        out = self._view(moe_out, [B, T, self.C], name="mlp_down")
        return out, residual


class Qwen3_5MoELinearBlock(nn.Block):
    """Qwen3.5 MoE linear-attention (Gated DeltaNet) decoder block.

    Uses GatedDeltaNet linear attention, RMSNormPlus1 norms,
    and MoE FFN with shared expert + sigmoid gate.
    """

    _name_remap_ = QWEN3_5_MOE_LINEAR_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        chunk_size: int = 64,
        eps: float = 1e-6,
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.chunk_size = chunk_size
        self.eps = eps
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")

        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(
                "Qwen3_5MoELinearBlock requires linear_num_value_heads to be divisible by linear_num_key_heads"
            )

        # Derived dimensions for shape resolution
        self.M = d_ff
        self.Hk = linear_num_key_heads
        self.Hv = linear_num_value_heads
        self.Kd = linear_key_head_dim
        self.Vd = linear_value_head_dim
        self.KeyDim = self.Hk * self.Kd
        self.ValueDim = self.Hv * self.Vd
        self.ConvK = linear_conv_kernel_dim
        self.ConvDim = self.KeyDim * 2 + self.ValueDim
        self.HeadRepeat = self.Hv // self.Hk

        self.attn_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.mixer = nn.GatedDeltaNetMixer(
            d_model,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            chunk_size=chunk_size,
            eps=eps,
        )
        self.mlp_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.moe = nn.MoEExpertsGated(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob=False,
            ep_size=ep_size,
        )
        self.shared_expert = nn.MoESharedExpert(d_model, shared_expert_intermediate)

    def forward(self, x, residual, position_ids):
        del position_ids  # Unused in linear-attention layers.

        # Pre-attention normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.attn_norm(residual, x)
        # Linear attention (Gated DeltaNet)
        h = self.mixer(h)
        # Pre-MoE normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.mlp_norm(residual, h)
        # Flatten for MoE
        h_flat = self._view(h, [B * T, self.C], name="ln2_flat")
        # MoE experts
        moe_out = self.moe(h_flat)
        # Shared expert with sigmoid gate
        shared_out = self.shared_expert(h_flat)
        self._register_param("shared_expert_gate_proj_weight", (1, "C"))
        shared_gate = self._matmul(h_flat, "shared_expert_gate_proj_weight", name="shared_expert_gate_proj")
        shared_gate = self._sigmoid(shared_gate, name="shared_expert_gate_sigmoid")
        shared_out = self._mul(shared_out, shared_gate, name="shared_expert_gated")
        moe_out = self._add(moe_out, shared_out, name="moe_combined")
        # Register output slot and reshape back
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        out = self._view(moe_out, [B, T, self.C], name="mlp_down")
        return out, residual
