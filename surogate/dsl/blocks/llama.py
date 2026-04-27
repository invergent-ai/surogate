"""LLaMA Transformer Block."""

from __future__ import annotations

from .. import nn
from ..block_schema import BlockSchema, SlotDecl
from ..modules import GenericGQAttention, GenericMLP, RMSNorm
from ..attention import AttentionConfig
from ..mlp import MLPConfig
from .common import DENSE_BLOCK_NAME_REMAP


class LlamaBlock(nn.Block):
    """LLaMA transformer block: GQA attention + SwiGLU MLP."""

    _name_remap_ = DENSE_BLOCK_NAME_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("ln1", shape=("B", "T", "C"), save_for_backward=True),
            SlotDecl("ln1_rstd", shape=("B", "T"), dtype="fp32", save_for_backward=True),
            SlotDecl("qkv", shape=("B", "T", "QKV"), save_for_backward=True),
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
        attrs={"block_family": "llama_dense"},
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
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(eps=eps),
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff, config=MLPConfig())

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)
        h = self.mlp(h)
        return h, residual
