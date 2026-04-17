"""Qwen3-VL Model (text backbone)."""

from __future__ import annotations

from .. import nn
from ..blocks.qwen3_vl import Qwen3VLBlock
from ..hf import build_dense_block_mappings
from ..modules.attention import Qwen3Attention
from ..nn import VL_MODEL_NAME_REMAP
from ..specs import ActivationScope


@nn.hf_config(
    architecture="Qwen3VLForConditionalGeneration",
    model_type="qwen3_vl",
    d_model="text_config.hidden_size",
    n_layers="text_config.num_hidden_layers",
    num_query_heads="text_config.num_attention_heads",
    num_kv_heads="text_config.num_key_value_heads",
    d_ff="text_config.intermediate_size",
    vocab_size="text_config.vocab_size",
    max_seq="text_config.max_position_embeddings",
    head_size="text_config.head_dim",
    eps="text_config.rms_norm_eps",
    use_qkv_bias="text_config.attention_bias",
    mrope_section="text_config.rope_scaling.mrope_section",
    deepstack_visual_indexes="vision_config.deepstack_visual_indexes",
)
class Qwen3VLModel(nn.Model):
    """Qwen3-VL text model using Qwen3VLBlock (MRoPE)."""

    _name_remap_ = VL_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        **build_dense_block_mappings(
            attn_module=Qwen3Attention,
            layer_prefix="model.language_model.layers.{layer}",
        ),
        # Model-level weight mappings
        "embedding": "model.language_model.embed_tokens.weight",
        "final_norm": "model.language_model.norm.weight",
        "lm_head": "lm_head.weight",
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 4096,
        n_layers: int = 32,
        num_query_heads: int = 32,
        num_kv_heads: int = 32,
        d_ff: int = 22016,
        max_seq: int = 128000,
        head_size: int = 128,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        mrope_section: tuple[int, int, int] | list[int] | None = None,
        deepstack_visual_indexes: list[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.head_size = head_size
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias

        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (24, 20, 20)
        self.mrope_section = list(mrope_section)

        if isinstance(deepstack_visual_indexes, (list, tuple)):
            self.deepstack_layers = len(deepstack_visual_indexes)
        else:
            self.deepstack_layers = 3

        self.D = head_size if head_size > 0 else d_model // num_query_heads

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.BlockStack(
            n_layers,
            Qwen3VLBlock,
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            d_ff=d_ff,
            max_seq=max_seq,
            eps=eps,
            use_qkv_bias=use_qkv_bias,
            mrope_section=mrope_section,
        )
        self.final_norm = nn.RMSNorm(d_model, eps=eps)
        self.lm_head = nn.LMHead(vocab_size, d_model)

    def forward(
        self,
        token_ids,
        position_ids,
        visual_pos_masks,
        visual_embeds,
        deepstack_visual_embeds_0,
        deepstack_visual_embeds_1,
        deepstack_visual_embeds_2,
        targets,
    ):
        G = ActivationScope.GLOBAL

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", (3, "B", "T"), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        self._register_activation("visual_pos_masks", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("visual_embeds", ("B * T", "d_model"), scope=G)
        self._register_activation("deepstack_visual_embeds_0", ("B * T", "d_model"), scope=G)
        self._register_activation("deepstack_visual_embeds_1", ("B * T", "d_model"), scope=G)
        self._register_activation("deepstack_visual_embeds_2", ("B * T", "d_model"), scope=G)
        self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

        # Global intermediate slots
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

        # Embedding + visual injection
        x = self.embedding(token_ids)
        x = self._mask_scatter(x, visual_pos_masks, visual_embeds, name="x0")

        residual = self._zeros(["B", "T", "d_model"])

        # Stacked blocks with deepstack visual args
        x, residual = self.blocks(
            x,
            residual,
            position_ids,
            visual_pos_masks,
            deepstack_visual_embeds_0,
            deepstack_visual_embeds_1,
            deepstack_visual_embeds_2,
            deepstack_layers=self.deepstack_layers,
        )

        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss
