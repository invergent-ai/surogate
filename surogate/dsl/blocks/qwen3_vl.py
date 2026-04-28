"""Qwen3-VL Transformer Block (text)."""

from __future__ import annotations

from .. import nn
from ..block_schema import BlockSchema, SlotDecl
from ..modules import GenericMLP, Qwen3VLAttention, RMSNorm
from .common import VL_DENSE_BLOCK_NAME_REMAP


class Qwen3VLBlock(nn.Block):
    """Qwen3-VL text transformer block with QK-Norm + MRoPE."""

    _name_remap_ = VL_DENSE_BLOCK_NAME_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("ln1", shape=("B", "T", "C"), save_for_backward=True),
            SlotDecl("ln1_rstd", shape=("B", "T"), dtype="fp32", save_for_backward=True),
            SlotDecl("qkv", shape=("B", "T", "QKV"), save_for_backward=True),
            SlotDecl("qkv_norm", shape=("B", "T", "QKV"), save_for_backward=True),
            SlotDecl("q_rstd", shape=("B", "T", "Hq"), dtype="fp32", save_for_backward=True),
            SlotDecl("k_rstd", shape=("B", "T", "Hkv"), dtype="fp32", save_for_backward=True),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
            SlotDecl("att", shape=("B", "T", "AttnDim"), save_for_backward=True),
            SlotDecl("lse", shape=("B", "Hq", "T"), dtype="fp32", save_for_backward=True),
            SlotDecl("att_out", shape=("B", "T", "C")),
            SlotDecl("res_att", shape=("B", "T", "C")),
            SlotDecl("ln2_rstd", shape=("B", "T"), dtype="fp32", save_for_backward=True),
            SlotDecl("mlp_up", shape=("B", "T", "2M")),
            SlotDecl("swiglu", shape=("B", "T", "M")),
            SlotDecl("mlp_down", shape=("B", "T", "C")),
        ),
        attrs={"block_family": "qwen3_vl_dense"},
    )

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        mrope_section: tuple[int, int, int] | list[int] = (24, 20, 20),
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = Qwen3VLAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
            eps=eps,
            mrope_section=mrope_section,
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization (fused residual + rmsnorm)
        residual, h = self.attn_norm(residual, x)
        # Attention (QK-Norm + MRoPE)
        h = self.self_attn(h, position_ids)
        # Pre-MLP normalization (fused residual + rmsnorm)
        residual, h = self.mlp_norm(residual, h)
        # MLP (SwiGLU)
        h = self.mlp(h)
        return h, residual
