"""LFM2-MoE Model."""

from __future__ import annotations

from .. import nn
from ..blocks.lfm2 import LFM2_MODEL_NAME_REMAP, Lfm2AttentionBlock, Lfm2ConvBlock
from ..blocks.lfm2_moe import Lfm2MoeAttentionBlock, Lfm2MoeConvBlock
from ..hf import fuse, stack_experts, tied_to
from ..modules import Embedding, LMHead, RMSNorm
from ..specs import ActivationScope

_LAYER_PREFIX = "model.layers.{layer}"
_FFN = f"{_LAYER_PREFIX}.feed_forward"


def _resolve_lfm2_moe_layer_types(
    *,
    n_layers: int,
    layer_types: list[str] | None,
    full_attn_idxs: list[int] | None,
    num_dense_layers: int,
) -> list[str]:
    """One block type per layer, over both axes.

    The operator axis comes from ``layer_types`` (HF requires it for LFM2-MoE;
    ``full_attn_idxs`` is accepted too, so a checkpoint carrying only the older
    key still loads). The feed-forward axis is positional: the first
    ``num_dense_layers`` layers keep a dense MLP and the rest are sparse.
    """
    if layer_types is None:
        attn = set(full_attn_idxs if full_attn_idxs is not None else range(n_layers))
        layer_types = ["full_attention" if i in attn else "conv" for i in range(n_layers)]

    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) must match n_layers ({n_layers})")

    block_types = []
    for index, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            operator = "attention"
        elif layer_type == "conv":
            operator = "conv"
        else:
            raise ValueError(f"Unsupported LFM2-MoE layer type '{layer_type}'. Expected 'full_attention' or 'conv'")
        block_types.append(operator if index < num_dense_layers else f"{operator}_moe")
    return block_types


@nn.hf_config(
    architecture="Lfm2MoeForCausalLM",
    model_type="lfm2_moe",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    moe_d_ff="moe_intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    eps="norm_eps",
    conv_kernel="conv_L_cache",
    conv_bias="conv_bias",
    num_dense_layers="num_dense_layers",
    num_experts="num_experts",
    num_experts_per_tok="num_experts_per_tok",
    norm_topk_prob="norm_topk_prob",
    use_expert_bias="use_expert_bias",
    routed_scaling_factor="routed_scaling_factor",
    full_attn_idxs="full_attn_idxs",
    layer_types="layer_types",
    tie_word_embeddings="tie_word_embeddings",
)
class Lfm2MoeModel(nn.Model):
    """LFM2-MoE hybrid model: attention/short-conv operators, dense then sparse FFNs."""

    _name_remap_ = LFM2_MODEL_NAME_REMAP
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
        "ffn_norm_weight": f"{_LAYER_PREFIX}.ffn_norm.weight",
        # Dense feed-forward (the leading layers). w1 gates, w3 lifts, w2 projects
        # back; the fused row order is [up; gate], which is what the loader packs.
        "mlp_up_weight": fuse(f"{_FFN}.w3.weight", f"{_FFN}.w1.weight", dim=0),
        "mlp_down_weight": f"{_FFN}.w2.weight",
        # Sparse feed-forward. The checkpoint stores one tensor per expert under
        # the same w1/w3/w2 names, so the gate/up pair must be named explicitly --
        # the default derivation only knows gate_proj/up_proj.
        "router_weight": f"{_FFN}.gate.weight",
        # The aux-loss-free selection bias. LFM2-MoE parks it on the feed-forward
        # rather than under experts/ the way Laguna does.
        "e_score_correction_bias": f"{_FFN}.expert_bias",
        "experts_gate_up": stack_experts(
            f"{_FFN}.experts.{{expert}}.w1.weight",
            fuse_gate_up=True,
            up_pattern=f"{_FFN}.experts.{{expert}}.w3.weight",
        ),
        "experts_down": stack_experts(f"{_FFN}.experts.{{expert}}.w2.weight"),
        # Model-level weights
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.embedding_norm.weight",
        "lm_head": tied_to("embedding"),
    }

    def __init__(
        self,
        vocab_size: int = 65536,
        d_model: int = 2048,
        n_layers: int = 32,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        d_ff: int = 7168,
        moe_d_ff: int = 1792,
        max_seq: int = 128000,
        head_size: int = 0,
        eps: float = 1e-5,
        conv_kernel: int = 3,
        conv_bias: bool = False,
        num_dense_layers: int = 2,
        num_experts: int = 32,
        num_experts_per_tok: int = 4,
        norm_topk_prob: bool = True,
        use_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        full_attn_idxs: list[int] | None = None,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool = True,
        ep_size: int = 1,
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
        self.tie_word_embeddings = tie_word_embeddings
        self.num_dense_layers = num_dense_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.use_expert_bias = use_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size

        self.head_size = head_size if head_size > 0 else d_model // num_query_heads
        self.D = self.head_size
        # Unlike LFM2, the MoE variant carries no ff-dim adjustment: the dense
        # layers use intermediate_size as written and the experts their own width.
        self.d_ff = d_ff
        self.M = moe_d_ff
        self.moe_d_ff = moe_d_ff
        self.K = conv_kernel

        self.block_types = _resolve_lfm2_moe_layer_types(
            n_layers=n_layers,
            layer_types=layer_types,
            full_attn_idxs=full_attn_idxs,
            num_dense_layers=num_dense_layers,
        )
        self.layer_types = layer_types
        self.full_attn_idxs = full_attn_idxs
        self.hybrid_pattern = "".join(
            {"attention": "A", "conv": "C", "attention_moe": "a", "conv_moe": "c"}[t] for t in self.block_types
        )
        self.n_attn_blocks = sum(1 for t in self.block_types if t == "attention")
        self.n_conv_blocks = sum(1 for t in self.block_types if t == "conv")
        self.n_attention_moe_blocks = sum(1 for t in self.block_types if t == "attention_moe")
        self.n_conv_moe_blocks = sum(1 for t in self.block_types if t == "conv_moe")
        self.has_attn_blocks = (self.n_attn_blocks + self.n_attention_moe_blocks) > 0
        self.has_conv_blocks = (self.n_conv_blocks + self.n_conv_moe_blocks) > 0

        attention_kwargs = dict(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=self.head_size,
            max_seq=max_seq,
            eps=eps,
        )
        moe_kwargs = dict(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

        block_configs = []
        if self.n_attn_blocks:
            block_configs.append(
                ("attn_blocks", Lfm2AttentionBlock, self.n_attn_blocks, dict(**attention_kwargs, d_ff=self.d_ff))
            )
        if self.n_conv_blocks:
            block_configs.append(
                (
                    "conv_blocks",
                    Lfm2ConvBlock,
                    self.n_conv_blocks,
                    dict(d_model=d_model, d_ff=self.d_ff, conv_kernel=conv_kernel, eps=eps, conv_bias=conv_bias),
                )
            )
        if self.n_attention_moe_blocks:
            block_configs.append(
                (
                    "attention_moe_blocks",
                    Lfm2MoeAttentionBlock,
                    self.n_attention_moe_blocks,
                    dict(**attention_kwargs, d_ff=moe_d_ff, **moe_kwargs),
                )
            )
        if self.n_conv_moe_blocks:
            block_configs.append(
                (
                    "conv_moe_blocks",
                    Lfm2MoeConvBlock,
                    self.n_conv_moe_blocks,
                    dict(
                        d_model=d_model,
                        d_ff=moe_d_ff,
                        conv_kernel=conv_kernel,
                        eps=eps,
                        conv_bias=conv_bias,
                        **moe_kwargs,
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
