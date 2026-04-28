"""Qwen3.5 dense models (hybrid full-attention + linear-attention)."""

from __future__ import annotations

from .. import nn
from ..modules import Embedding, LMHead, RMSNormPlus1
from ..modules.attention import _resolve_rotary_dim
from ..blocks.qwen3_5 import Qwen3_5AttentionBlock, Qwen3_5LinearBlock
from ..hf import build_mlp_mappings, build_norm_mappings
from ..blocks.qwen3_5 import QWEN3_5_MODEL_NAME_REMAP, QWEN3_5_VL_MODEL_NAME_REMAP
from ..specs import ActivationScope


def _parse_qwen3_5_layer_types(
    layer_types: list[str] | None,
    n_layers: int,
    full_attention_interval: int,
) -> list[str]:
    """Convert HF layer_types to DSL HybridStackedBlocks types."""
    if layer_types is None:
        interval = max(1, int(full_attention_interval))
        layer_types = ["linear_attention" if ((i + 1) % interval) != 0 else "full_attention" for i in range(n_layers)]
    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) must match n_layers ({n_layers})")

    out: list[str] = []
    for t in layer_types:
        if t == "linear_attention":
            out.append("mamba")
        elif t == "full_attention":
            out.append("attention")
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type '{t}'. Expected 'linear_attention' or 'full_attention'.")
    return out


def _build_qwen3_5_block_mappings(layer_prefix: str) -> dict[str, object]:
    """HF mappings shared by Qwen3.5 dense model variants."""
    return {
        **build_norm_mappings(layer_prefix),
        **build_mlp_mappings(layer_prefix),
        # Full-attention params
        "full_q_proj_weight": f"{layer_prefix}.self_attn.q_proj.weight",
        "full_q_proj_bias": f"{layer_prefix}.self_attn.q_proj.bias",
        "full_k_proj_weight": f"{layer_prefix}.self_attn.k_proj.weight",
        "full_k_proj_bias": f"{layer_prefix}.self_attn.k_proj.bias",
        "full_v_proj_weight": f"{layer_prefix}.self_attn.v_proj.weight",
        "full_v_proj_bias": f"{layer_prefix}.self_attn.v_proj.bias",
        "full_out_weight": f"{layer_prefix}.self_attn.o_proj.weight",
        "full_out_bias": f"{layer_prefix}.self_attn.o_proj.bias",
        "q_norm_weight": f"{layer_prefix}.self_attn.q_norm.weight",
        "k_norm_weight": f"{layer_prefix}.self_attn.k_norm.weight",
        # Linear-attention params
        "lin_in_proj_qkv_weight": f"{layer_prefix}.linear_attn.in_proj_qkv.weight",
        "lin_in_proj_z_weight": f"{layer_prefix}.linear_attn.in_proj_z.weight",
        "lin_in_proj_b_weight": f"{layer_prefix}.linear_attn.in_proj_b.weight",
        "lin_in_proj_a_weight": f"{layer_prefix}.linear_attn.in_proj_a.weight",
        "lin_conv_weight": f"{layer_prefix}.linear_attn.conv1d.weight",
        "lin_A_log": f"{layer_prefix}.linear_attn.A_log",
        "lin_dt_bias": f"{layer_prefix}.linear_attn.dt_bias",
        "lin_norm_weight": f"{layer_prefix}.linear_attn.norm.weight",
        "lin_out_weight": f"{layer_prefix}.linear_attn.out_proj.weight",
        # Model-level weight mappings
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }


def _build_qwen3_5_conditional_block_mappings(layer_prefix: str) -> dict[str, object]:
    """HF mappings for Qwen3.5 conditional generation model."""
    return {
        **build_norm_mappings(layer_prefix),
        **build_mlp_mappings(layer_prefix),
        # Full-attention params
        "full_q_proj_weight": f"{layer_prefix}.self_attn.q_proj.weight",
        "full_q_proj_bias": f"{layer_prefix}.self_attn.q_proj.bias",
        "full_k_proj_weight": f"{layer_prefix}.self_attn.k_proj.weight",
        "full_k_proj_bias": f"{layer_prefix}.self_attn.k_proj.bias",
        "full_v_proj_weight": f"{layer_prefix}.self_attn.v_proj.weight",
        "full_v_proj_bias": f"{layer_prefix}.self_attn.v_proj.bias",
        "full_out_weight": f"{layer_prefix}.self_attn.o_proj.weight",
        "full_out_bias": f"{layer_prefix}.self_attn.o_proj.bias",
        "q_norm_weight": f"{layer_prefix}.self_attn.q_norm.weight",
        "k_norm_weight": f"{layer_prefix}.self_attn.k_norm.weight",
        # Linear-attention params
        "lin_in_proj_qkv_weight": f"{layer_prefix}.linear_attn.in_proj_qkv.weight",
        "lin_in_proj_z_weight": f"{layer_prefix}.linear_attn.in_proj_z.weight",
        "lin_in_proj_b_weight": f"{layer_prefix}.linear_attn.in_proj_b.weight",
        "lin_in_proj_a_weight": f"{layer_prefix}.linear_attn.in_proj_a.weight",
        "lin_conv_weight": f"{layer_prefix}.linear_attn.conv1d.weight",
        "lin_A_log": f"{layer_prefix}.linear_attn.A_log",
        "lin_dt_bias": f"{layer_prefix}.linear_attn.dt_bias",
        "lin_norm_weight": f"{layer_prefix}.linear_attn.norm.weight",
        "lin_out_weight": f"{layer_prefix}.linear_attn.out_proj.weight",
        # Model-level weight mappings
        "embedding": "model.language_model.embed_tokens.weight",
        "final_norm": "model.language_model.norm.weight",
        "lm_head": "lm_head.weight",
    }


@nn.hf_config(
    architecture="Qwen3_5ForCausalLM",
    model_type="qwen3_5_text",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    use_qkv_bias="attention_bias",
    partial_rotary_factor="rope_parameters.partial_rotary_factor",
    mrope_section="rope_parameters.mrope_section",
    linear_conv_kernel_dim="linear_conv_kernel_dim",
    linear_key_head_dim="linear_key_head_dim",
    linear_value_head_dim="linear_value_head_dim",
    linear_num_key_heads="linear_num_key_heads",
    linear_num_value_heads="linear_num_value_heads",
    layer_types="layer_types",
    full_attention_interval="full_attention_interval",
)
class Qwen3_5CausalModel(nn.Model):
    """Qwen3.5 dense text model for ``Qwen3_5ForCausalLM``."""

    _name_remap_ = QWEN3_5_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_qwen3_5_block_mappings("model.layers.{layer}")

    def __init__(
        self,
        vocab_size: int = 248320,
        d_model: int = 4096,
        n_layers: int = 32,
        num_query_heads: int = 16,
        num_kv_heads: int = 4,
        d_ff: int = 12288,
        max_seq: int = 32768,
        head_size: int = 256,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        chunk_size: int = 64,
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

        self.partial_rotary_factor = partial_rotary_factor
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.full_attention_interval = full_attention_interval
        self.chunk_size = chunk_size

        # Derived
        self.D = head_size if head_size > 0 else d_model // num_query_heads
        self.rotary_dim = _resolve_rotary_dim(self.D, self.partial_rotary_factor)

        self.block_types = _parse_qwen3_5_layer_types(
            layer_types=layer_types,
            n_layers=n_layers,
            full_attention_interval=full_attention_interval,
        )
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["linear_attention" if t == "mamba" else "full_attention" for t in self.block_types]
        )
        self.n_linear_blocks = sum(1 for t in self.block_types if t == "mamba")
        self.n_attn_blocks = sum(1 for t in self.block_types if t == "attention")
        self.has_linear_blocks = self.n_linear_blocks > 0
        self.has_attn_blocks = self.n_attn_blocks > 0

        # Use mamba_blocks / attn_blocks naming for HybridBlockStack
        # (mamba = linear_attention, attention = full_attention)
        self.n_mamba_blocks = self.n_linear_blocks
        self.n_attention_blocks = self.n_attn_blocks

        # Build block configs for HybridBlockStack
        block_configs = []
        if self.n_linear_blocks > 0:
            block_configs.append(
                (
                    "mamba_blocks",
                    Qwen3_5LinearBlock,
                    self.n_linear_blocks,
                    dict(
                        d_model=d_model,
                        d_ff=d_ff,
                        linear_conv_kernel_dim=linear_conv_kernel_dim,
                        linear_key_head_dim=linear_key_head_dim,
                        linear_value_head_dim=linear_value_head_dim,
                        linear_num_key_heads=linear_num_key_heads,
                        linear_num_value_heads=linear_num_value_heads,
                        chunk_size=chunk_size,
                        eps=eps,
                    ),
                )
            )
        if self.n_attn_blocks > 0:
            block_configs.append(
                (
                    "attn_blocks",
                    Qwen3_5AttentionBlock,
                    self.n_attn_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        eps=eps,
                        use_qkv_bias=use_qkv_bias,
                        partial_rotary_factor=partial_rotary_factor,
                        mrope_section=mrope_section,
                    ),
                )
            )

        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=self.block_types,
            n_layers=n_layers,
        )
        self.final_norm = RMSNormPlus1(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", (3, "B", "T"), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        self._register_activation(
            "freq_cis", ("max_seq", "rotary_dim // 2", 2), dtype="fp32", scope=G, aliases=["rope_freqs"]
        )

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

        x = self.embedding(token_ids)
        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss


@nn.hf_config(
    architecture="Qwen3_5ForConditionalGeneration",
    model_type="qwen3_5",
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
    partial_rotary_factor="text_config.rope_parameters.partial_rotary_factor",
    mrope_section="text_config.rope_parameters.mrope_section",
    linear_conv_kernel_dim="text_config.linear_conv_kernel_dim",
    linear_key_head_dim="text_config.linear_key_head_dim",
    linear_value_head_dim="text_config.linear_value_head_dim",
    linear_num_key_heads="text_config.linear_num_key_heads",
    linear_num_value_heads="text_config.linear_num_value_heads",
    layer_types="text_config.layer_types",
    full_attention_interval="text_config.full_attention_interval",
    use_visual_inputs="vision_config",
)
class Qwen3_5ConditionalModel(nn.Model):
    """Qwen3.5 dense text model for ``Qwen3_5ForConditionalGeneration``."""

    _name_remap_ = QWEN3_5_VL_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_qwen3_5_conditional_block_mappings(
        "model.language_model.layers.{layer}",
    )

    def __init__(
        self,
        vocab_size: int = 248320,
        d_model: int = 4096,
        n_layers: int = 32,
        num_query_heads: int = 16,
        num_kv_heads: int = 4,
        d_ff: int = 12288,
        max_seq: int = 32768,
        head_size: int = 256,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        chunk_size: int = 64,
        use_visual_inputs: bool | dict | None = False,
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

        self.partial_rotary_factor = partial_rotary_factor
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.full_attention_interval = full_attention_interval
        self.chunk_size = chunk_size
        self.use_visual_inputs = bool(use_visual_inputs)

        # Derived
        self.D = head_size if head_size > 0 else d_model // num_query_heads
        self.rotary_dim = _resolve_rotary_dim(self.D, self.partial_rotary_factor)

        self.block_types = _parse_qwen3_5_layer_types(
            layer_types=layer_types,
            n_layers=n_layers,
            full_attention_interval=full_attention_interval,
        )
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["linear_attention" if t == "mamba" else "full_attention" for t in self.block_types]
        )
        self.n_linear_blocks = sum(1 for t in self.block_types if t == "mamba")
        self.n_attn_blocks = sum(1 for t in self.block_types if t == "attention")
        self.has_linear_blocks = self.n_linear_blocks > 0
        self.has_attn_blocks = self.n_attn_blocks > 0

        # Use mamba_blocks / attn_blocks naming for HybridBlockStack
        self.n_mamba_blocks = self.n_linear_blocks
        self.n_attention_blocks = self.n_attn_blocks

        # Build block configs for HybridBlockStack
        block_configs = []
        if self.n_linear_blocks > 0:
            block_configs.append(
                (
                    "mamba_blocks",
                    Qwen3_5LinearBlock,
                    self.n_linear_blocks,
                    dict(
                        d_model=d_model,
                        d_ff=d_ff,
                        linear_conv_kernel_dim=linear_conv_kernel_dim,
                        linear_key_head_dim=linear_key_head_dim,
                        linear_value_head_dim=linear_value_head_dim,
                        linear_num_key_heads=linear_num_key_heads,
                        linear_num_value_heads=linear_num_value_heads,
                        chunk_size=chunk_size,
                        eps=eps,
                    ),
                )
            )
        if self.n_attn_blocks > 0:
            block_configs.append(
                (
                    "attn_blocks",
                    Qwen3_5AttentionBlock,
                    self.n_attn_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        eps=eps,
                        use_qkv_bias=use_qkv_bias,
                        partial_rotary_factor=partial_rotary_factor,
                        mrope_section=mrope_section,
                    ),
                )
            )

        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=self.block_types,
            n_layers=n_layers,
        )
        self.final_norm = RMSNormPlus1(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(
        self,
        token_ids,
        position_ids,
        visual_pos_masks,
        visual_embeds,
        targets,
    ):
        G = ActivationScope.GLOBAL

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", (3, "B", "T"), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        if self.use_visual_inputs:
            self._register_activation(
                "visual_pos_masks", ("B", "T"), dtype="int32", scope=G, description="Mask for visual token positions"
            )
            self._register_activation(
                "visual_embeds", ("B * T", "d_model"), scope=G, description="Visual embeddings (packed by mask)"
            )
        self._register_activation(
            "freq_cis", ("max_seq", "rotary_dim // 2", 2), dtype="fp32", scope=G, aliases=["rope_freqs"]
        )

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
        if self.use_visual_inputs:
            x = self._mask_scatter(x, visual_pos_masks, visual_embeds, name="x0")

        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss
