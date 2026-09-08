"""Spark-X2.5 attention and GELU decoder blocks."""

from .. import nn
from ..activations import Activation
from ..attention import AttentionConfig
from ..block_schema import BlockSchema, ServeObject, SlotDecl
from ..mlp import MLPConfig
from ..modules import GenericGQAttention, GenericMLP, RMSNorm
from .common import DENSE_BLOCK_NAME_REMAP

SPARK_BLOCK_OBJECTS = (
    ServeObject("input_norm", "bf16", ("C",), ("ln1_weight",)),
    ServeObject("attention/query_key_value", "w8", ("QKV", "C"), ("qkv_weight",)),
    ServeObject("attention/output_gate", "bf16", ("QueryHeads", "C"), ("attn_gate_weight",)),
    ServeObject("attention/output", "w8", ("C", "AttnDim"), ("out_weight",)),
    ServeObject("post_attention_norm", "bf16", ("C",), ("ln2_weight",)),
    ServeObject("mlp/gate_up", "w8", ("MUp", "C"), ("mlp_gate_weight", "mlp_up_weight")),
    ServeObject("mlp/down", "w8", ("C", "M"), ("mlp_down_weight",)),
)


class Spark2_5Block(nn.Block):
    _name_remap_ = {
        **DENSE_BLOCK_NAME_REMAP,
        "self_attn_output_gate_weight": "attn_gate_weight",
    }
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("attn_gate_weight", kind="param", shape=("Hq", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_gate_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_up_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("res_ffn", shape=("B", "T", "C"), dtype="fp32"),
            SlotDecl("res_att", shape=("B", "T", "C"), dtype="fp32"),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        serve_objects=SPARK_BLOCK_OBJECTS,
        attrs={"block_family": "spark2_5"},
    )

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        eps,
        sliding_window,
        partial_rotary_factor,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps, residual_dtype="fp32")
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(
                headwise_output_gate=True,
                fused_qkv_lora=True,
                sliding_window=sliding_window,
                partial_rotary_factor=partial_rotary_factor,
                eps=eps,
            ),
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps, residual_dtype="fp32")
        self.mlp = GenericMLP(
            d_model,
            d_ff,
            config=MLPConfig(activation=Activation.GELU_EXACT, gated=True, fuse_gate_up=False),
        )

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)
        return self.mlp(h), residual


class Spark2_5FullBlock(Spark2_5Block):
    """Full causal attention with the checkpoint's global rotary prefix."""


class Spark2_5SlidingBlock(Spark2_5Block):
    """Windowed causal attention with the checkpoint's local rotary prefix."""
