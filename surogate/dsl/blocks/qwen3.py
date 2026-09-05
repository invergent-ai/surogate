"""Qwen3 Transformer Block."""

from __future__ import annotations

from .. import nn
from ..block_schema import BlockSchema, SlotDecl
from ..modules import GenericGQAttention, GenericMLP, RMSNorm
from ..attention import AttentionConfig
from ..mlp import MLPConfig
from ..block_schema import ServeObject
from .common import DENSE_BLOCK_NAME_REMAP


#: How a serving artifact stores this block. Shapes are written against the symbols
#: `serve/convert/common/declaration.symbols_for` resolves from the runtime config.
#:
#: The declaration fuses q, k and v into one parameter and the artifact stores the
#: same matrix, so `attention/query_key_value` is a pass-through; ungated attention
#: means there is no gate block between k and v. The SwiGLU parameter fuses `up |
#: gate` and the artifact stores `gate | up`, so that object names the halves in its
#: own row order rather than the parameter as a whole.
_QWEN3_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("input_norm", "bf16", ("C",), ("ln1_weight",)),
    ServeObject("attention/query_key_value", "quantised", ("QKV", "C"), ("qkv_weight",)),
    ServeObject("attention/query_norm", "bf16", ("HeadDim",), ("q_norm_weight",)),
    ServeObject("attention/key_norm", "bf16", ("HeadDim",), ("k_norm_weight",)),
    ServeObject("attention/output", "quantised", ("C", "AttnDim"), ("out_weight",)),
    ServeObject("post_attention_norm", "bf16", ("C",), ("ln2_weight",)),
    ServeObject("mlp/gate_up", "quantised", ("MUp", "C"),
                ("mlp_up_weight.gate", "mlp_up_weight.up")),
    ServeObject("mlp/down", "quantised", ("C", "M"), ("mlp_down_weight",)),
)


class Qwen3Block(nn.Block):
    """Qwen3 transformer block: QK-norm GQA + SwiGLU MLP."""

    _name_remap_ = DENSE_BLOCK_NAME_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("res_att", shape=("B", "T", "C")),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        serve_objects=_QWEN3_SERVE_OBJECTS,
        attrs={"block_family": "qwen3_dense"},
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
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(
                qk_norm=use_qk_norm,
                qkv_bias=use_qkv_bias,
                eps=eps,
            ),
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff, config=MLPConfig())

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)
        h = self.mlp(h)
        return h, residual
