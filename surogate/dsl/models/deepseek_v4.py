"""DeepSeek-V4 (``DeepseekV4ForCausalLM``).

DeepSeek-V4 drops V3's MLA. Every layer is shared-KV MQA — ``num_key_value_heads == 1``
and K and V are *the same tensor* — over a 128-token sliding window, with a long-range
compressor branch (CSA at m=4 plus a Lightning Indexer, or HCA at m'=128) concatenated
onto the KV axis inside the attention block, a gpt-oss-style per-head attention sink, and
a grouped low-rank output projection. The feed-forward is a sqrt-softplus MoE on *every*
layer (no dense prefix), with a shared expert and a frozen ``tid2eid`` hash router on the
first few layers. The residual is ``hc_mult`` parallel streams tied together by
manifold-constrained hyper-connections (mHC).

Declared here, faithfully: the two-axis layer schedule, every attention and MoE tensor
with its checkpoint name and shape, the sliding-window attention skeleton (shared KV
head, per-head sink, grouped output projection), the aux-loss-free routing bias, and the
batched expert layout.

Deferred, with the missing primitive named in each case (the block module docstring has
the full argument, and each site carries a ``DEFERRED`` comment):

* **mHC** — needs Sinkhorn-Knopp (sum-over-dim + divide) and a per-token stream mix.
  The blocks run the ordinary single-stream pre-norm residual. Note V4 *also* keeps
  ``model.norm``, unlike ``qwen4_exp`` where the output mix replaced the final norm, so
  ``final_norm`` is a real tensor here and the standard global slot vocabulary applies.
* **CSA / HCA compressors and the Lightning Indexer** — running-window state, per-window
  softmax pooling, the Ca/Cb overlap scheme, top-k over compressed blocks. Attention runs
  pure sliding-window; the tensors are declared as serve objects.
* **V4's RoPE** — trailing-channel interleaved rotation, and the conjugate rotation that
  strips RoPE off the (shared-with-K) values after attention.
* **Hash routing** — ``tid2eid[input_ids]`` expert selection.
* **``sqrt`` of the router score** and the ``swiglu_limit`` clamps.
* **MTP** (``num_nextn_predict_layers``) — captured in config, not declared.

Because of those holes this declaration is **not numerically faithful**, and the IR says
so: ``numerically_faithful == False`` plus a ``*_deferred`` boolean per mechanism in the
runtime config, so nothing downstream can mistake it for a trainable V4.
"""

from __future__ import annotations

from .. import nn
from ..blocks.common import STANDARD_MODEL_NAME_REMAP
from ..blocks.deepseek_v4 import (
    DEEPSEEK_V4_ATTENTION_KINDS,
    DEEPSEEK_V4_BLOCK_CLASSES,
    DEEPSEEK_V4_MLP_KINDS,
    DeepseekV4CsaMoEBlock,
    DeepseekV4HcaMoEBlock,
    DeepseekV4SlidingMoEBlock,
)
from ..hf import transform
from ..modules import Embedding, LMHead, RMSNorm
from ..specs import ActivationScope

_LAYER = "model.layers.{layer}"
_ATTN = f"{_LAYER}.self_attn"
_MLP = f"{_LAYER}.mlp"

#: Checkpoint tensors this declaration does NOT load, with the mechanism that would
#: consume them. Kept as data (rather than in ``_hf_block_mappings_``) so the names are
#: recorded without handing the weight loader parameters the graph never allocates.
#: Every one of these is a deferral listed in the module docstring.
DEEPSEEK_V4_DEFERRED_HF_TENSORS: dict[str, str] = {
    # Manifold-constrained hyper-connections, two sites per layer plus the head.
    "attn_hc.fn": f"{_LAYER}.attn_hc.fn",
    "attn_hc.base": f"{_LAYER}.attn_hc.base",
    "attn_hc.scale": f"{_LAYER}.attn_hc.scale",
    "ffn_hc.fn": f"{_LAYER}.ffn_hc.fn",
    "ffn_hc.base": f"{_LAYER}.ffn_hc.base",
    "ffn_hc.scale": f"{_LAYER}.ffn_hc.scale",
    "hc_head.fn": "model.hc_head.hc_fn",
    "hc_head.base": "model.hc_head.hc_base",
    "hc_head.scale": "model.hc_head.hc_scale",
    # Long-range compressor (CSA and HCA layers).
    "compressor.kv_proj": f"{_ATTN}.compressor.kv_proj.weight",
    "compressor.gate_proj": f"{_ATTN}.compressor.gate_proj.weight",
    "compressor.position_bias": f"{_ATTN}.compressor.position_bias",
    "compressor.kv_norm": f"{_ATTN}.compressor.kv_norm.weight",
    # Lightning Indexer (CSA layers only).
    "indexer.kv_proj": f"{_ATTN}.compressor.indexer.kv_proj.weight",
    "indexer.gate_proj": f"{_ATTN}.compressor.indexer.gate_proj.weight",
    "indexer.position_bias": f"{_ATTN}.compressor.indexer.position_bias",
    "indexer.kv_norm": f"{_ATTN}.compressor.indexer.kv_norm.weight",
    "indexer.q_b_proj": f"{_ATTN}.compressor.indexer.q_b_proj.weight",
    "indexer.scorer": f"{_ATTN}.compressor.indexer.scorer.weights_proj.weight",
}

#: What this declaration does not compute. The runtime config carries the same list as
#: one ``<mechanism>_deferred`` boolean each (booleans reach the IR, lists do not).
DEEPSEEK_V4_DEFERRED_MECHANISMS: tuple[str, ...] = (
    "mhc_hyper_connections",
    "csa_hca_compressor",
    "lightning_indexer",
    "interleaved_trailing_rope",
    "inverse_rope_on_attention_output",
    "hash_routing",
    "sqrt_softplus_router_score",
    "swiglu_limit_clamp",
    "mtp_draft_layers",
)


def _resolve_deepseek_v4_layer_types(
    *,
    n_layers: int,
    layer_types: list[str] | None,
    mlp_layer_types: list[str] | None,
    num_hash_layers: int,
) -> tuple[list[str], list[str], list[str]]:
    """Two independent schedules, one block type per layer.

    Mirrors ``DeepseekV4Config.__post_init__``: the attention default is a 2-layer HCA
    bootstrap followed by an interleave over the remaining ``n - 2`` positions (CSA on odd
    interleave slots, HCA on even) — which lands as HCA on layers 0/1/2 and then
    CSA on odd / HCA on even layer indices. The MoE default is ``num_hash_layers``
    ``hash_moe`` layers followed by ``moe``.
    """
    if layer_types is None:
        interleave = [
            "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
            for i in range(max(n_layers - 2, 0))
        ]
        layer_types = ["heavily_compressed_attention"] * min(n_layers, 2) + interleave
    layer_types = list(layer_types[:n_layers])
    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) must match n_layers ({n_layers})")

    if mlp_layer_types is None:
        mlp_layer_types = ["hash_moe"] * min(n_layers, num_hash_layers) + [
            "moe"
        ] * max(0, n_layers - num_hash_layers)
    mlp_layer_types = list(mlp_layer_types[:n_layers])
    if len(mlp_layer_types) != n_layers:
        raise ValueError(f"mlp_layer_types length ({len(mlp_layer_types)}) must match n_layers ({n_layers})")

    block_types = []
    for attention, mlp in zip(layer_types, mlp_layer_types, strict=True):
        if attention not in DEEPSEEK_V4_ATTENTION_KINDS:
            raise ValueError(
                f"Unsupported DeepSeek-V4 layer type '{attention}'. "
                f"Expected one of {sorted(DEEPSEEK_V4_ATTENTION_KINDS)}"
            )
        if mlp not in DEEPSEEK_V4_MLP_KINDS:
            raise ValueError(
                f"Unsupported DeepSeek-V4 mlp layer type '{mlp}'. Expected one of {sorted(DEEPSEEK_V4_MLP_KINDS)}"
            )
        block_types.append(f"{DEEPSEEK_V4_ATTENTION_KINDS[attention]}_{DEEPSEEK_V4_MLP_KINDS[mlp]}")
    return block_types, layer_types, mlp_layer_types


@nn.hf_config(
    architecture="DeepseekV4ForCausalLM",
    model_type="deepseek_v4",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    head_size="head_dim",
    q_lora_rank="q_lora_rank",
    o_groups="o_groups",
    o_lora_rank="o_lora_rank",
    d_ff="moe_intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    eps="rms_norm_eps",
    num_experts="n_routed_experts",
    num_experts_per_tok="num_experts_per_tok",
    n_shared_experts="n_shared_experts",
    scoring_func="scoring_func",
    norm_topk_prob="norm_topk_prob",
    routed_scaling_factor="routed_scaling_factor",
    swiglu_limit="swiglu_limit",
    sliding_window="sliding_window",
    layer_types="layer_types",
    mlp_layer_types="mlp_layer_types",
    num_hash_layers="num_hash_layers",
    compress_rates="compress_rates",
    rope_theta="rope_theta",
    compress_rope_theta="compress_rope_theta",
    partial_rotary_factor="partial_rotary_factor",
    hc_mult="hc_mult",
    hc_sinkhorn_iters="hc_sinkhorn_iters",
    hc_eps="hc_eps",
    index_n_heads="index_n_heads",
    index_head_dim="index_head_dim",
    index_topk="index_topk",
    num_nextn_predict_layers="num_nextn_predict_layers",
    tie_word_embeddings="tie_word_embeddings",
)
class DeepseekV4Model(nn.Model):
    """DeepSeek-V4 text model: shared-KV MQA + sliding window, sqrt-softplus MoE everywhere."""

    _name_remap_ = STANDARD_MODEL_NAME_REMAP
    _serve_blocks_ = {
        "sliding": DeepseekV4SlidingMoEBlock,
        "csa": DeepseekV4CsaMoEBlock,
        "hca": DeepseekV4HcaMoEBlock,
    }

    _hf_block_mappings_ = {
        # --- norms ---
        "ln1_weight": f"{_LAYER}.input_layernorm.weight",
        "ln2_weight": f"{_LAYER}.post_attention_layernorm.weight",
        # --- attention ---
        "q_a_proj_weight": f"{_ATTN}.q_a_proj.weight",
        "q_a_norm_weight": f"{_ATTN}.q_a_norm.weight",
        "q_b_proj_weight": f"{_ATTN}.q_b_proj.weight",
        # `q_b_norm` is an UNWEIGHTED RMSNorm — no tensor in the checkpoint.
        "kv_proj_weight": f"{_ATTN}.kv_proj.weight",
        "kv_norm_weight": f"{_ATTN}.kv_norm.weight",
        "o_a_proj_weight": f"{_ATTN}.o_a_proj.weight",
        "o_b_proj_weight": f"{_ATTN}.o_b_proj.weight",
        "sinks": f"{_ATTN}.sinks",
        # --- MoE ---
        "router_weight": f"{_MLP}.gate.weight",
        # Present only on `moe` layers...
        "e_score_correction_bias": f"{_MLP}.gate.e_score_correction_bias",
        # ...and only on `hash_moe` layers. A union dict is fine: an entry is emitted per
        # layer only for a parameter that layer's block actually declares.
        "tid2eid": f"{_MLP}.gate.tid2eid",
        # DeepSeek-V4 ships the experts already batched as [E, 2*I, C], but with GATE in
        # the first half; surogate's fused layout and `kernels/swiglu.cu` are [up; gate].
        # Naming the swap keeps the mapping truthful — the loader raises "unsupported
        # transform" until it is implemented, which is the right failure mode; a
        # passthrough would silently train with gate and up exchanged.
        "experts_gate_up": transform(f"{_MLP}.experts.gate_up_proj", fn="swap_gate_up_halves"),
        "experts_down": f"{_MLP}.experts.down_proj",
        # Shared expert — note the plural `shared_experts` in the V4 checkpoint.
        "shared_expert_gate": f"{_MLP}.shared_experts.gate_proj.weight",
        "shared_expert_up": f"{_MLP}.shared_experts.up_proj.weight",
        "shared_expert_down": f"{_MLP}.shared_experts.down_proj.weight",
        # --- model level ---
        "embedding": "model.embed_tokens.weight",
        # V4 keeps a real final norm; the mHC head collapses the streams *before* it.
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }

    def __init__(
        self,
        vocab_size: int = 129280,
        d_model: int = 4096,
        n_layers: int = 43,
        num_query_heads: int = 64,
        num_kv_heads: int = 1,
        head_size: int = 512,
        q_lora_rank: int = 1024,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        d_ff: int = 2048,
        max_seq: int = 1048576,
        eps: float = 1e-6,
        num_experts: int = 256,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 1,
        scoring_func: str = "sqrtsoftplus",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        sliding_window: int = 128,
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        num_hash_layers: int = 3,
        compress_rates: dict | None = None,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        partial_rotary_factor: float = 64 / 512,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        num_nextn_predict_layers: int = 1,
        tie_word_embeddings: bool = False,
        ep_size: int = 1,
    ):
        super().__init__()
        if num_kv_heads != 1:
            raise ValueError(
                "DeepSeek-V4 is shared-KV MQA: num_key_value_heads must be 1 "
                f"(K and V are the same tensor), got {num_kv_heads}"
            )
        if scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"DeepSeek-V4 declares scoring_func='sqrtsoftplus'; got {scoring_func!r}. "
                "Other scoring functions are not declared."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.q_lora_rank = q_lora_rank
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.tie_word_embeddings = tie_word_embeddings
        self.ep_size = ep_size
        self.use_qk_norm = False
        self.use_qkv_bias = False
        self.use_out_bias = False

        # mHC geometry: captured so the serve-spec generator and any future lowering read
        # it from one place. Not in the training graph (see the module docstring).
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        # Lightning Indexer / compressor geometry: same deal.
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        rates = dict(compress_rates or {})
        self.compress_rate_csa = int(rates.get("compressed_sparse_attention", 4))
        self.compress_rate_hca = int(rates.get("heavily_compressed_attention", 128))
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # This declaration is structural, not numerical. Say so in the IR: booleans reach
        # the runtime config, so a consumer can refuse the model instead of training a
        # lookalike. `DEEPSEEK_V4_DEFERRED_MECHANISMS` is the same list in prose.
        self.numerically_faithful = False
        self.mhc_deferred = True
        self.compressor_deferred = True
        self.indexer_deferred = True
        self.rope_deferred = True
        self.hash_routing_deferred = True
        self.router_sqrt_deferred = True
        self.swiglu_clamp_deferred = True
        self.mtp_deferred = True

        # The shared expert is one LlamaMLP at `moe_intermediate_size` — `n_shared_experts`
        # does NOT widen it in the reference (LlamaMLP reads `intermediate_size`, which
        # `attribute_map` routes to `moe_intermediate_size`).
        self.shared_expert_intermediate = d_ff

        # Derived dims
        self.D = head_size
        self.Hq = num_query_heads
        self.Hkv = 1
        self.M = d_ff
        self.SharedM = self.shared_expert_intermediate
        self.E = num_experts
        self.K = num_experts_per_tok
        self.AttnDim = num_query_heads * head_size
        self.QKV = (num_query_heads + 2) * head_size
        self.QLoraRank = q_lora_rank
        self.OaIn = self.AttnDim // o_groups
        self.OaOut = o_groups * o_lora_rank
        # V4 rotates `head_dim * partial_rotary_factor` channels (64 of 512 by default).
        self.rotary_dim = int(head_size * partial_rotary_factor)
        self.RotaryDim = self.rotary_dim

        self.block_types, self.layer_types, self.mlp_layer_types = _resolve_deepseek_v4_layer_types(
            n_layers=n_layers,
            layer_types=layer_types,
            mlp_layer_types=mlp_layer_types,
            num_hash_layers=num_hash_layers,
        )
        self.num_hash_layers = sum(1 for t in self.mlp_layer_types if t == "hash_moe")
        self.hybrid_pattern = "".join(
            {
                "sliding_moe": "S",
                "sliding_hash": "s",
                "csa_moe": "C",
                "csa_hash": "c",
                "hca_moe": "H",
                "hca_hash": "h",
            }[t]
            for t in self.block_types
        )

        block_kwargs = dict(
            d_model=d_model,
            num_query_heads=num_query_heads,
            head_size=head_size,
            q_lora_rank=q_lora_rank,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
            d_ff=d_ff,
            shared_expert_intermediate=self.shared_expert_intermediate,
            max_seq=max_seq,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            vocab_size=vocab_size,
            routed_scaling_factor=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            sliding_window=sliding_window,
            rotary_dim=self.rotary_dim,
            eps=eps,
            ep_size=ep_size,
        )

        block_configs = []
        for stem, block_cls in DEEPSEEK_V4_BLOCK_CLASSES.items():
            count = sum(1 for t in self.block_types if t == stem)
            # HybridBlockStack derives the block type from the param name, and the shape
            # resolver reads `n_<type>_blocks` off the model.
            setattr(self, f"n_{stem}_blocks", count)
            if count:
                block_configs.append((f"{stem}_blocks", block_cls, count, dict(block_kwargs)))
        self.has_attn_blocks = True

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

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        self._register_activation(
            "freq_cis", ("max_seq", "rotary_dim // 2", 2), dtype="fp32", scope=G, aliases=["rope_freqs"]
        )

        # Global intermediate slots. Unlike qwen4_exp, the residual is ONE stream wide
        # (mHC deferred) and there IS a final norm (`model.norm`), so residual_final /
        # ln_final_rstd are real slots here.
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
        # DEFERRED (mHC): the reference broadcasts the embedding into `hc_mult` parallel
        # residual streams here and collapses them again in `model.hc_head` before the
        # final norm. Single stream until Sinkhorn is expressible.
        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss
