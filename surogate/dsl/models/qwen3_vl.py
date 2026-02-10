"""Qwen3-VL Model (text backbone)."""

from __future__ import annotations

from ..tensor_type import Tensor, Array
from ..decorators import model, forward, hf_config, Param, Activation, Gradient
from ..graph_builder import graph
from ..hf import build_dense_block_mappings
from ..modules.attention import Qwen3Attention


@model
@hf_config(
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
class Qwen3VLModel:
    """Qwen3-VL text model using Qwen3VLBlock (MRoPE)."""

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

        # Derived
        self.D = head_size if head_size > 0 else d_model // num_query_heads

    # Model weights
    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="model.language_model.embed_tokens.weight")
    blocks = Param(Array["n_layers", "Qwen3VLBlock"])
    final_norm = Param(Tensor["d_model"], hf_mapping="model.language_model.norm.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")

    # =========================================================================
    # IO slots (runtime-provided inputs/outputs)
    # =========================================================================

    token_ids = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                           description="Input token IDs")
    position_ids = Activation(Tensor[3, "B", "T"], dtype="int32", scope="global",
                              description="3D position IDs for MRoPE")
    targets = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                         aliases=["labels"], description="Target labels for loss")
    visual_pos_masks = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                                  description="Mask for visual token positions")
    visual_embeds = Activation(Tensor["B * T", "d_model"], scope="global",
                               description="Visual embeddings (packed by mask)")
    deepstack_visual_embeds_0 = Activation(Tensor["B * T", "d_model"], scope="global",
                                           description="Deepstack visual embeddings layer 0")
    deepstack_visual_embeds_1 = Activation(Tensor["B * T", "d_model"], scope="global",
                                           description="Deepstack visual embeddings layer 1")
    deepstack_visual_embeds_2 = Activation(Tensor["B * T", "d_model"], scope="global",
                                           description="Deepstack visual embeddings layer 2")

    # Precomputed constants
    freq_cis = Activation(Tensor["max_seq", "D", 2], dtype="fp32", scope="global",
                          aliases=["rope_freqs"], description="Precomputed RoPE frequencies")

    # =========================================================================
    # Global activation slots (model-level, not per-block)
    # =========================================================================

    x0 = Activation(Tensor["B", "T", "d_model"], aliases=["encoded"], scope="global",
                    description="Embedded input (after embedding lookup)")
    residual0 = Activation(Tensor["B", "T", "d_model"], scope="global",
                           description="Initial residual stream (zeros)")

    xN = Activation(Tensor["B", "T", "d_model"], scope="global",
                    description="Output from stacked blocks")
    residualN = Activation(Tensor["B", "T", "d_model"], scope="global",
                           description="Residual after stacked blocks")

    residual_final = Activation(Tensor["B", "T", "d_model"], scope="global",
                                description="Final residual (before norm)")
    xF = Activation(Tensor["B", "T", "d_model"], aliases=["ln_final"],
                    scope="global", description="After final layer norm")
    xF_flat = Activation(Tensor["B * T", "d_model"], scope="global",
                         description="Flattened for LM head")
    ln_final_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                               scope="global", description="Final LN rstd")

    loss = Activation(Tensor["B * T"], dtype="fp32", scope="global",
                      aliases=["losses"], description="Cross-entropy loss per token")

    # =========================================================================
    # Global gradient slots
    # =========================================================================

    d_loss = Gradient(Tensor["B * T"], dtype="fp32", gradient_of="loss", scope="global",
                      description="Gradient seed for backward pass")
    d_xF = Gradient(Tensor["B", "T", "d_model"], gradient_of="xF", scope="global")
    d_xN = Gradient(Tensor["B", "T", "d_model"], gradient_of="xN", scope="global")
    d_residualN = Gradient(Tensor["B", "T", "d_model"], gradient_of="residualN", scope="global")
    d_x0 = Gradient(Tensor["B", "T", "d_model"], gradient_of="x0", scope="global")

    # Block mappings composed from module-level defaults (Qwen3Attention + SwiGLUMLP + RMSNorm).
    _hf_block_mappings_ = build_dense_block_mappings(attn_module=Qwen3Attention,
                                                     layer_prefix="model.language_model.layers.{layer}")

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor[3, "B", "T", "int32"],
        visual_pos_masks: Tensor["B", "T", "int32"],
        visual_embeds: Tensor["B * T", "d_model"],
        deepstack_visual_embeds_0: Tensor["B * T", "d_model"],
        deepstack_visual_embeds_1: Tensor["B * T", "d_model"],
        deepstack_visual_embeds_2: Tensor["B * T", "d_model"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            # Embedding lookup
            x0 = g.embedding(token_ids, "embedding")
            x0 = g.mask_scatter(x0, visual_pos_masks, visual_embeds, out_name="x0")

            # Initialize residual stream
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            # Stacked blocks (handled by runtime)
            xN, residualN = g.call(
                "StackedBlocks",
                x0,
                residual0,
                position_ids,
                visual_pos_masks,
                deepstack_visual_embeds_0,
                deepstack_visual_embeds_1,
                deepstack_visual_embeds_2,
                num_outputs=2,
                blocks="blocks",
                n_layers=self.n_layers,
                deepstack_layers=self.deepstack_layers,
            )

            # Final norm
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN, xN, "final_norm", eps=self.eps,
                res_out_name="residual_final",
                y_name="xF",
                rstd_name="ln_final_rstd",
            )

            # Fused LM head + loss
            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            loss = g.fused_lm_head_loss(xF_flat, "lm_head", targets,
                                        compute_accuracy=True, out_name="loss")

            return loss
