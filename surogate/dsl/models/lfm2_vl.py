"""LFM2-VL Model (text backbone).

LFM2-VL is an LFM2 hybrid text stack with a SigLIP2-NaFlex vision tower and a
pixel-unshuffle MLP connector in front of it. As with the other VL models here,
the DSL declares the *text* backbone: the tower and the projector run outside
and hand in already-projected image features, which are scattered into the
embedding stream at the image-token positions.

The only text-side differences from ``Lfm2ForCausalLM`` are that the weights sit
under ``model.language_model.`` and that the LM head is a real tensor at the root
rather than being tied to the embedding.
"""

from __future__ import annotations

from .. import nn
from ..blocks.common import VL_MODEL_NAME_REMAP
from ..blocks.lfm2 import Lfm2AttentionBlock, Lfm2ConvBlock
from ..hf import fuse
from ..models.lfm2 import _compute_lfm2_intermediate_size, _resolve_lfm2_layer_types
from ..modules import Embedding, LMHead, RMSNorm
from ..specs import ActivationScope

_LAYER_PREFIX = "model.language_model.layers.{layer}"


@nn.hf_config(
    architecture="Lfm2VlForConditionalGeneration",
    model_type="lfm2_vl",
    d_model="text_config.hidden_size",
    n_layers="text_config.num_hidden_layers",
    num_query_heads="text_config.num_attention_heads",
    num_kv_heads="text_config.num_key_value_heads",
    d_ff="text_config.intermediate_size",
    vocab_size="text_config.vocab_size",
    max_seq="text_config.max_position_embeddings",
    eps="text_config.norm_eps",
    conv_kernel="text_config.conv_L_cache",
    conv_bias="text_config.conv_bias",
    block_multiple_of="text_config.block_multiple_of",
    block_ffn_dim_multiplier="text_config.block_ffn_dim_multiplier",
    block_auto_adjust_ff_dim="text_config.block_auto_adjust_ff_dim",
    full_attn_idxs="text_config.full_attn_idxs",
    layer_types="text_config.layer_types",
    tie_word_embeddings="tie_word_embeddings",
    image_token_id="image_token_id",
    downsample_factor="downsample_factor",
    projector_hidden_size="projector_hidden_size",
    vision_hidden_size="vision_config.hidden_size",
    vision_layers="vision_config.num_hidden_layers",
)
class Lfm2VlModel(nn.Model):
    """LFM2-VL text backbone: LFM2 hybrid blocks fed image features by scatter."""

    _name_remap_ = VL_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        # Attention operator
        "operator_norm_weight": f"{_LAYER_PREFIX}.operator_norm.weight",
        "qkv_weight": fuse(
            f"{_LAYER_PREFIX}.self_attn.q_proj.weight",
            f"{_LAYER_PREFIX}.self_attn.k_proj.weight",
            f"{_LAYER_PREFIX}.self_attn.v_proj.weight",
            dim=0,
        ),
        "out_weight": f"{_LAYER_PREFIX}.self_attn.out_proj.weight",
        "q_norm_weight": f"{_LAYER_PREFIX}.self_attn.q_layernorm.weight",
        "k_norm_weight": f"{_LAYER_PREFIX}.self_attn.k_layernorm.weight",
        # Short-conv operator
        "conv_in_proj_weight": f"{_LAYER_PREFIX}.conv.in_proj.weight",
        "conv_in_proj_bias": f"{_LAYER_PREFIX}.conv.in_proj.bias",
        "conv_weight": f"{_LAYER_PREFIX}.conv.conv.weight",
        "conv_bias": f"{_LAYER_PREFIX}.conv.conv.bias",
        "conv_out_proj_weight": f"{_LAYER_PREFIX}.conv.out_proj.weight",
        "conv_out_proj_bias": f"{_LAYER_PREFIX}.conv.out_proj.bias",
        # Shared FFN
        "ffn_norm_weight": f"{_LAYER_PREFIX}.ffn_norm.weight",
        "mlp_up_weight": fuse(
            f"{_LAYER_PREFIX}.feed_forward.w3.weight",
            f"{_LAYER_PREFIX}.feed_forward.w1.weight",
            dim=0,
        ),
        "mlp_down_weight": f"{_LAYER_PREFIX}.feed_forward.w2.weight",
        # Model-level weights
        "embedding": "model.language_model.embed_tokens.weight",
        "final_norm": "model.language_model.embedding_norm.weight",
        "lm_head": "lm_head.weight",
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
        image_token_id: int = 396,
        downsample_factor: int = 2,
        projector_hidden_size: int = 2560,
        vision_hidden_size: int = 0,
        vision_layers: int = 0,
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
        # Vision-side geometry: carried for the converter and the projector that
        # runs outside this graph, not used by the text forward.
        self.image_token_id = image_token_id
        self.downsample_factor = downsample_factor
        self.projector_hidden_size = projector_hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_layers = vision_layers
        self.projector_in_features = vision_hidden_size * downsample_factor * downsample_factor

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

    def forward(self, token_ids, position_ids, visual_pos_masks, visual_embeds, targets):
        G = ActivationScope.GLOBAL

        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        # Image features arrive already projected to d_model; the mask marks the
        # image-token positions the processor expanded in the prompt.
        self._register_activation("visual_pos_masks", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("visual_embeds", ("B * T", "d_model"), scope=G)
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
        x = self._mask_scatter(x, visual_pos_masks, visual_embeds, name="x0")
        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss
