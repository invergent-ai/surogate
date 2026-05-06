"""LFM2 hybrid decoder blocks."""

from __future__ import annotations

from .. import nn
from ..attention import AttentionConfig
from ..block_schema import BlockSchema, SlotDecl
from ..mlp import MLPConfig
from ..modules import GenericGQAttention, GenericMLP, Lfm2ShortConv, RMSNorm
from .common import STANDARD_MODEL_NAME_REMAP

LFM2_MODEL_NAME_REMAP = STANDARD_MODEL_NAME_REMAP

LFM2_ATTENTION_BLOCK_REMAP: dict[str, str] = {
    "operator_norm_weight": "operator_norm_weight",
    "operator_norm_res": "res_operator",
    "operator_norm_y": "operator_ln",
    "operator_norm_rstd": "operator_ln_rstd",
    "self_attn_qkv_weight": "qkv_weight",
    "self_attn_qkv_bias": "qkv_bias",
    "self_attn_out_weight": "out_weight",
    "self_attn_out_bias": "out_bias",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_x_flat": "attn_x_flat",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_flat": "qkv_flat",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_attn": "attn",
    "self_attn_lse": "lse",
    "self_attn_att_out": "operator_out",
    "self_attn_att_out_flat": "operator_out_flat",
    "ffn_norm_weight": "ffn_norm_weight",
    "ffn_norm_res": "res_ffn",
    "ffn_norm_y": "ffn_ln",
    "ffn_norm_rstd": "ffn_ln_rstd",
    "mlp_up_weight": "mlp_up_weight",
    "mlp_down_weight": "mlp_down_weight",
    "mlp_x_flat": "mlp_x_flat",
    "mlp_up": "mlp_up",
    "mlp_up_flat": "mlp_up_flat",
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_down": "mlp_down",
    "mlp_down_flat": "mlp_down_flat",
}

LFM2_CONV_BLOCK_REMAP: dict[str, str] = {
    "operator_norm_weight": "operator_norm_weight",
    "operator_norm_res": "res_operator",
    "operator_norm_y": "operator_ln",
    "operator_norm_rstd": "operator_ln_rstd",
    "short_conv_in_proj_weight": "conv_in_proj_weight",
    "short_conv_in_proj_bias": "conv_in_proj_bias",
    "short_conv_conv_weight": "conv_weight",
    "short_conv_conv_bias": "conv_bias",
    "short_conv_out_proj_weight": "conv_out_proj_weight",
    "short_conv_out_proj_bias": "conv_out_proj_bias",
    "short_conv_x_flat": "conv_x_flat",
    "short_conv_in_proj": "conv_in_proj",
    "short_conv_in_proj_flat": "conv_in_proj_flat",
    "short_conv_bx": "conv_bx",
    "short_conv_conv_out": "operator_out_cf",
    "short_conv_gated_conv": "gated_conv",
    "short_conv_out": "operator_out",
    "short_conv_out_proj_flat": "operator_out_flat",
    "ffn_norm_weight": "ffn_norm_weight",
    "ffn_norm_res": "res_ffn",
    "ffn_norm_y": "ffn_ln",
    "ffn_norm_rstd": "ffn_ln_rstd",
    "mlp_up_weight": "mlp_up_weight",
    "mlp_down_weight": "mlp_down_weight",
    "mlp_x_flat": "mlp_x_flat",
    "mlp_up": "mlp_up",
    "mlp_up_flat": "mlp_up_flat",
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_down": "mlp_down",
    "mlp_down_flat": "mlp_down_flat",
}


class Lfm2AttentionBlock(nn.Block):
    """LFM2 full-attention decoder layer."""

    _name_remap_ = LFM2_ATTENTION_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("q_norm_weight", kind="param", shape=("D",)),
            SlotDecl("k_norm_weight", kind="param", shape=("D",)),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
            SlotDecl("operator_out", shape=("B", "T", "C")),
            SlotDecl("res_ffn", shape=("B", "T", "C")),
            SlotDecl("mlp_down", shape=("B", "T", "C")),
        ),
        attrs={"block_family": "lfm2_attention"},
    )

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.use_qk_norm = True
        self.use_qkv_bias = False
        self.use_out_bias = False
        self.operator_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(qk_norm=True, eps=eps),
        )
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff, config=MLPConfig())

    def forward(self, x, residual, position_ids):
        residual, h = self.operator_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.ffn_norm(residual, h)
        h = self.mlp(h)
        return h, residual


class Lfm2ConvBlock(nn.Block):
    """LFM2 short-convolution decoder layer."""

    _name_remap_ = LFM2_CONV_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("conv_in_proj_weight", kind="param", shape=("3C", "C"), residency="auto"),
            SlotDecl("conv_weight", kind="param", shape=("C", 1, "K")),
            SlotDecl("conv_out_proj_weight", kind="param", shape=("C", "C"), residency="auto"),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("conv_in_proj", shape=("B", "T", "3C"), save_for_backward=True),
            SlotDecl("conv_bx", shape=("B", "C", "T"), save_for_backward=True),
            SlotDecl("operator_out_cf", shape=("B", "C", "T"), save_for_backward=True),
            SlotDecl("operator_out", shape=("B", "T", "C")),
            SlotDecl("res_ffn", shape=("B", "T", "C")),
            SlotDecl("mlp_down", shape=("B", "T", "C")),
        ),
        attrs={"block_family": "lfm2_conv"},
    )

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        conv_kernel: int = 3,
        eps: float = 1e-5,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.use_bias = conv_bias
        self.operator_norm = RMSNorm(d_model, eps=eps)
        self.short_conv = Lfm2ShortConv(d_model, conv_kernel=conv_kernel, use_bias=conv_bias)
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff, config=MLPConfig())

    def forward(self, x, residual, position_ids):
        residual, h = self.operator_norm(residual, x)
        h = self.short_conv(h)
        residual, h = self.ffn_norm(residual, h)
        h = self.mlp(h)
        return h, residual
