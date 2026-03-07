"""Qwen3.5 dense models (hybrid full-attention + linear-attention)."""

from __future__ import annotations

from ..tensor_type import Tensor, Array
from ..decorators import model, forward, hf_config, Param, Activation, Gradient
from ..graph_builder import graph
from ..hf import build_norm_mappings, build_mlp_mappings


def _parse_qwen3_5_layer_types(
    layer_types: list[str] | None,
    n_layers: int,
    full_attention_interval: int,
) -> list[str]:
    """Convert HF layer_types to DSL HybridStackedBlocks types."""
    if layer_types is None:
        interval = max(1, int(full_attention_interval))
        layer_types = [
            "linear_attention" if ((i + 1) % interval) != 0 else "full_attention"
            for i in range(n_layers)
        ]
    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) must match n_layers ({n_layers})")

    out: list[str] = []
    for t in layer_types:
        if t == "linear_attention":
            out.append("mamba")
        elif t == "full_attention":
            out.append("attention")
        else:
            raise ValueError(
                f"Unsupported Qwen3.5 layer type '{t}'. Expected 'linear_attention' or 'full_attention'."
            )
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
    }


class _Qwen3_5DenseBase:
    """Shared implementation for Qwen3.5 dense text backbones."""

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
        self.rotary_dim = max(2, int(round(self.D * self.partial_rotary_factor)))
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1
        self.rotary_dim = max(2, min(self.rotary_dim, self.D))

        self.block_types = _parse_qwen3_5_layer_types(
            layer_types=layer_types,
            n_layers=n_layers,
            full_attention_interval=full_attention_interval,
        )
        self.layer_types = layer_types if layer_types is not None else [
            "linear_attention" if t == "mamba" else "full_attention" for t in self.block_types
        ]
        self.n_linear_blocks = sum(1 for t in self.block_types if t == "mamba")
        self.n_attn_blocks = sum(1 for t in self.block_types if t == "attention")
        self.has_linear_blocks = self.n_linear_blocks > 0
        self.has_attn_blocks = self.n_attn_blocks > 0

    # Block arrays
    linear_blocks = Param(Array["n_linear_blocks", "Qwen3_5LinearBlock"], when="has_linear_blocks")
    attn_blocks = Param(Array["n_attn_blocks", "Qwen3_5AttentionBlock"], when="has_attn_blocks")

    # IO
    token_ids = Activation(Tensor["B", "T"], dtype="int32", scope="global", description="Input token IDs")
    position_ids = Activation(Tensor[3, "B", "T"], dtype="int32", scope="global", description="Position IDs")
    targets = Activation(Tensor["B", "T"], dtype="int32", scope="global", aliases=["labels"])

    # Precomputed constants
    freq_cis = Activation(
        Tensor["max_seq", "rotary_dim // 2", 2],
        dtype="fp32",
        scope="global",
        aliases=["rope_freqs"],
        description="Precomputed RoPE frequencies",
    )

    # Global activations
    x0 = Activation(Tensor["B", "T", "d_model"], aliases=["encoded"], scope="global")
    residual0 = Activation(Tensor["B", "T", "d_model"], scope="global")
    xN = Activation(Tensor["B", "T", "d_model"], scope="global")
    residualN = Activation(Tensor["B", "T", "d_model"], scope="global")
    residual_final = Activation(Tensor["B", "T", "d_model"], scope="global")
    xF = Activation(Tensor["B", "T", "d_model"], aliases=["ln_final"], scope="global")
    xF_flat = Activation(Tensor["B * T", "d_model"], scope="global")
    ln_final_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True, scope="global")
    loss = Activation(Tensor["B * T"], dtype="fp32", scope="global", aliases=["losses"])

    # Global gradients
    d_loss = Gradient(Tensor["B * T"], dtype="fp32", gradient_of="loss", scope="global")
    d_xF = Gradient(Tensor["B", "T", "d_model"], gradient_of="xF", scope="global")
    d_xN = Gradient(Tensor["B", "T", "d_model"], gradient_of="xN", scope="global")
    d_residualN = Gradient(Tensor["B", "T", "d_model"], gradient_of="residualN", scope="global")
    d_x0 = Gradient(Tensor["B", "T", "d_model"], gradient_of="x0", scope="global")

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor[3, "B", "T", "int32"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            x0 = g.embedding(token_ids, "embedding")
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            xN, residualN = g.call(
                "HybridStackedBlocks",
                x0,
                residual0,
                position_ids,
                num_outputs=2,
                mamba_blocks="linear_blocks",
                attn_blocks="attn_blocks",
                block_types=self.block_types,
                n_layers=self.n_layers,
            )

            final_ones = g.ones(shape=["d_model"], dtype="bf16")
            final_norm_eff = g.add("final_norm", final_ones, out_name="final_norm_eff")
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN,
                xN,
                final_norm_eff,
                eps=self.eps,
                res_out_name="residual_final",
                y_name="xF",
                rstd_name="ln_final_rstd",
            )

            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            loss = g.fused_lm_head_loss(
                xF_flat,
                "lm_head",
                targets,
                compute_accuracy=True,
                out_name="loss",
            )
            return loss


@model
@hf_config(
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
class Qwen3_5CausalModel(_Qwen3_5DenseBase):
    """Qwen3.5 dense text model for `Qwen3_5ForCausalLM`."""

    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="model.embed_tokens.weight")
    final_norm = Param(Tensor["d_model"], hf_mapping="model.norm.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")

    _hf_block_mappings_ = _build_qwen3_5_block_mappings("model.layers.{layer}")


@model
@hf_config(
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
)
class Qwen3_5ConditionalModel(_Qwen3_5DenseBase):
    """Qwen3.5 dense text model for `Qwen3_5ForConditionalGeneration`."""

    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="model.language_model.embed_tokens.weight")
    final_norm = Param(Tensor["d_model"], hf_mapping="model.language_model.norm.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")

    _hf_block_mappings_ = _build_qwen3_5_block_mappings("model.language_model.layers.{layer}")

    # Multimodal external inputs (for vision-enabled Qwen3.5 conditional models).
    visual_pos_masks = Activation(
        Tensor["B", "T"],
        dtype="int32",
        scope="global",
        description="Mask for visual token positions",
    )
    visual_embeds = Activation(
        Tensor["B * T", "d_model"],
        scope="global",
        description="Visual embeddings (packed by mask)",
    )

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor[3, "B", "T", "int32"],
        visual_pos_masks: Tensor["B", "T", "int32"],
        visual_embeds: Tensor["B * T", "d_model"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            x0 = g.embedding(token_ids, "embedding")
            x0 = g.mask_scatter(x0, visual_pos_masks, visual_embeds, out_name="x0")
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            xN, residualN = g.call(
                "HybridStackedBlocks",
                x0,
                residual0,
                position_ids,
                num_outputs=2,
                mamba_blocks="linear_blocks",
                attn_blocks="attn_blocks",
                block_types=self.block_types,
                n_layers=self.n_layers,
            )

            final_ones = g.ones(shape=["d_model"], dtype="bf16")
            final_norm_eff = g.add("final_norm", final_ones, out_name="final_norm_eff")
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN,
                xN,
                final_norm_eff,
                eps=self.eps,
                res_out_name="residual_final",
                y_name="xF",
                rstd_name="ln_final_rstd",
            )

            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            loss = g.fused_lm_head_loss(
                xF_flat,
                "lm_head",
                targets,
                compute_accuracy=True,
                out_name="loss",
            )
            return loss
