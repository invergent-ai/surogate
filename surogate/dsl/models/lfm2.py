"""LFM2 Model."""

from __future__ import annotations

from .. import nn
from ..blocks.lfm2 import LFM2_MODEL_NAME_REMAP, Lfm2AttentionBlock, Lfm2ConvBlock
from ..hf import fuse, tied_to
from ..modules import Embedding, LMHead, RMSNorm
from ..specs import ActivationScope


def _compute_lfm2_intermediate_size(
    intermediate_size: int,
    *,
    block_auto_adjust_ff_dim: bool,
    block_ffn_dim_multiplier: float | int | None,
    block_multiple_of: int,
) -> int:
    """Mirror HF ``Lfm2MLP`` intermediate-size adjustment."""
    if not block_auto_adjust_ff_dim:
        return intermediate_size

    adjusted = int(2 * intermediate_size / 3)
    if block_ffn_dim_multiplier is not None:
        adjusted = int(float(block_ffn_dim_multiplier) * adjusted)
    return block_multiple_of * ((adjusted + block_multiple_of - 1) // block_multiple_of)


def _resolve_lfm2_layer_types(
    *,
    n_layers: int,
    layer_types: list[str] | None,
    full_attn_idxs: list[int] | None,
) -> list[str]:
    if layer_types is None:
        attn = set(full_attn_idxs if full_attn_idxs is not None else list(range(n_layers)))
        layer_types = ["full_attention" if i in attn else "conv" for i in range(n_layers)]

    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) must match n_layers ({n_layers})")

    block_types = []
    for layer_type in layer_types:
        if layer_type == "full_attention":
            block_types.append("attention")
        elif layer_type == "conv":
            block_types.append("conv")
        else:
            raise ValueError(f"Unsupported LFM2 layer type '{layer_type}'. Expected 'full_attention' or 'conv'")
    return block_types


@nn.hf_config(
    architecture="Lfm2ForCausalLM",
    model_type="lfm2",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    eps="norm_eps",
    conv_kernel="conv_L_cache",
    conv_bias="conv_bias",
    block_multiple_of="block_multiple_of",
    block_ffn_dim_multiplier="block_ffn_dim_multiplier",
    block_auto_adjust_ff_dim="block_auto_adjust_ff_dim",
    full_attn_idxs="full_attn_idxs",
    layer_types="layer_types",
    tie_word_embeddings="tie_word_embeddings",
)
class Lfm2Model(nn.Model):
    """LFM2 hybrid model with full-attention and short-conv decoder layers."""

    _name_remap_ = LFM2_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        # Attention operator
        "operator_norm_weight": "model.layers.{layer}.operator_norm.weight",
        "qkv_weight": fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0,
        ),
        "out_weight": "model.layers.{layer}.self_attn.out_proj.weight",
        "q_norm_weight": "model.layers.{layer}.self_attn.q_layernorm.weight",
        "k_norm_weight": "model.layers.{layer}.self_attn.k_layernorm.weight",
        # Short-conv operator
        "conv_in_proj_weight": "model.layers.{layer}.conv.in_proj.weight",
        "conv_in_proj_bias": "model.layers.{layer}.conv.in_proj.bias",
        "conv_weight": "model.layers.{layer}.conv.conv.weight",
        "conv_bias": "model.layers.{layer}.conv.conv.bias",
        "conv_out_proj_weight": "model.layers.{layer}.conv.out_proj.weight",
        "conv_out_proj_bias": "model.layers.{layer}.conv.out_proj.bias",
        # Shared FFN
        "ffn_norm_weight": "model.layers.{layer}.ffn_norm.weight",
        "mlp_up_weight": fuse(
            "model.layers.{layer}.feed_forward.w3.weight",
            "model.layers.{layer}.feed_forward.w1.weight",
            dim=0,
        ),
        "mlp_down_weight": "model.layers.{layer}.feed_forward.w2.weight",
        # Model-level weights
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.embedding_norm.weight",
        "lm_head": tied_to("embedding"),
    }

    def __init__(
        self,
        vocab_size: int = 65536,
        d_model: int = 2560,
        n_layers: int = 32,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        d_ff: int = 12288,
        max_seq: int = 128000,
        head_size: int = 0,
        eps: float = 1e-5,
        conv_kernel: int = 3,
        conv_bias: bool = False,
        block_multiple_of: int = 256,
        block_ffn_dim_multiplier: float | int | None = 1.0,
        block_auto_adjust_ff_dim: bool = True,
        full_attn_idxs: list[int] | None = None,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq = max_seq
        self.eps = eps
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.use_bias = conv_bias
        self.use_qk_norm = True
        self.use_qkv_bias = False
        self.use_out_bias = False
        self.block_multiple_of = block_multiple_of
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim
        self.tie_word_embeddings = tie_word_embeddings

        self.head_size = head_size if head_size > 0 else d_model // num_query_heads
        self.D = self.head_size
        self.d_ff = _compute_lfm2_intermediate_size(
            d_ff,
            block_auto_adjust_ff_dim=block_auto_adjust_ff_dim,
            block_ffn_dim_multiplier=block_ffn_dim_multiplier,
            block_multiple_of=block_multiple_of,
        )
        self.M = self.d_ff
        self.K = conv_kernel

        self.block_types = _resolve_lfm2_layer_types(
            n_layers=n_layers,
            layer_types=layer_types,
            full_attn_idxs=full_attn_idxs,
        )
        self.layer_types = layer_types
        self.full_attn_idxs = full_attn_idxs
        self.hybrid_pattern = "".join("A" if t == "attention" else "C" for t in self.block_types)
        self.n_attn_blocks = sum(1 for t in self.block_types if t == "attention")
        self.n_conv_blocks = sum(1 for t in self.block_types if t == "conv")
        self.has_attn_blocks = self.n_attn_blocks > 0
        self.has_conv_blocks = self.n_conv_blocks > 0

        block_configs = []
        if self.has_attn_blocks:
            block_configs.append(
                (
                    "attn_blocks",
                    Lfm2AttentionBlock,
                    self.n_attn_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=self.head_size,
                        d_ff=self.d_ff,
                        max_seq=max_seq,
                        eps=eps,
                    ),
                )
            )
        if self.has_conv_blocks:
            block_configs.append(
                (
                    "conv_blocks",
                    Lfm2ConvBlock,
                    self.n_conv_blocks,
                    dict(
                        d_model=d_model,
                        d_ff=self.d_ff,
                        conv_kernel=conv_kernel,
                        eps=eps,
                        conv_bias=conv_bias,
                    ),
                )
            )

        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=self.block_types,
            n_layers=n_layers,
        )
        self.final_norm = RMSNorm(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        if self.has_attn_blocks:
            self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

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
