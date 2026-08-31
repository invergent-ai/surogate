"""Qwen3.8-Flash-Next (qwen4_exp) models — hyper-connected hybrid GDN/attention MoE.

The single-source-of-truth declaration for the unified train/serve programme: a
qwen3_5_moe core (GDN mixers 3-of-4 layers, gated full attention every 4th, 512-expert
top-10 MoE with a sigmoid-gated shared expert) wrapped in hyper-connections — ``hc_count``
parallel residual streams whose mix/combine pairs REPLACE every layer norm. The final mix
before the LM head (``output_hc``, no inject weights) is the output norm; there is no
``model.norm`` tensor in the checkpoint.

Differences from Qwen3.5-MoE expressed here:
  * hyper-connections instead of RMSNormPlus1 pre-norms (and no final norm);
  * GDN output gate is **sigmoid** (``output_gate_type``), not SiLU;
  * router renormalises the top-k probabilities (``norm_topk_prob=True``).

Captured in config but NOT yet part of the training graph (deferred, in priority order):
  * **PLE n-gram memory** (layer ``ple_layer_ids``): load-bearing at inference — a
    training run of the real checkpoint is NOT numerically faithful until it lands.
    Needs an n-gram row-index input, a signed-sqrt gate primitive and a dilated causal
    conv; the ~320M-row table also needs offload-aware handling. All its hyperparameters
    are declared below so the serve-spec generator sees them.
  * **QSA indexer**: inference-time sparse selection; training runs dense attention,
    which is the exact semantics (identical below the 2048-token budget, and the
    reference modelling code trains dense). Indexer projections are not declared.
  * **MTP draft head**: dropped by every serving stack (no tensors in the GGUF export);
    no DSL precedent. Not declared.
  * **Vision tower**: text-only declaration, like the serve target.
"""

from __future__ import annotations

from .. import nn
from ..block_schema import ServeObject
from ..blocks.qwen4_exp import _PLE_SERVE_OBJECTS, Qwen4ExpAttentionBlock, Qwen4ExpLinearBlock
from ..hf import expand_module_mapping
from ..modules import Embedding, HyperConnection, LMHead, StreamBroadcast
from ..modules.attention import _resolve_rotary_dim
from ..modules.moe import MoESharedExpert
from ..specs import ActivationScope
from .qwen3_5 import _parse_qwen3_5_layer_types


#: Model-level objects as a serving artifact stores them. The per-layer objects are
#: declared on the block schemas; these are the ones outside the stack. There is no
#: final norm — the output hyper-connection mix is it — and the n-gram table travels
#: as a raw resource rather than a tensor, so only its hash parameters appear here.
QWEN4_EXP_MODEL_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("text/token_embedding", "w8", ("Vocab", "C"), ("embedding",), scope="model"),
    ServeObject("text/output_hc/norm", "fp32", ("HcWidth",), ("output_hc_norm",), scope="model"),
    ServeObject("text/output_hc/down", "bf16", ("HcLowRank", "HcWidth"),
                ("output_hc_down",), scope="model"),
    ServeObject("text/output_hc/up", "bf16", ("HcWidth", "HcLowRank"),
                ("output_hc_up",), scope="model"),
    ServeObject("text/output_head", "w8", ("Vocab", "C"), ("lm_head",), scope="model"),
    ServeObject("text/ple/multipliers", "i32", ("PleMultipliers",), scope="model"),
    ServeObject("text/ple/head_offsets", "i32", ("PleHeads",), scope="model"),
    ServeObject("text/ple/head_vocab_sizes", "i32", ("PleHeads",), scope="model"),
)

#: Objects that exist only on the layer carrying the n-gram memory.
QWEN4_EXP_PLE_SERVE_OBJECTS = _PLE_SERVE_OBJECTS

QWEN4_EXP_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- stream_init (StreamBroadcast) ---
    "stream_init_res": "residual0",
    # --- output_hc (final HyperConnection mix, no inject) — this IS the output norm ---
    "output_hc_mixed": "xF",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}


def _build_qwen4_exp_block_mappings(layer_prefix: str, model_prefix: str) -> dict[str, object]:
    """HF mappings for qwen4_exp. Tensor names from the HF checkpoint layout
    (``attn_hyper_connection``/``mlp_hyper_connection`` per layer,
    ``hyper_connection_mixer`` at the head, ``linear_attn``/``self_attn``/``mlp`` as in
    Qwen3.5-MoE, batched pre-fused experts)."""

    moe_prefix = f"{layer_prefix}.mlp"
    return {
        # Hyper-connections (per layer, two sets)
        "hc_attn_norm": f"{layer_prefix}.attn_hyper_connection.hc_norm.weight",
        "hc_attn_down": f"{layer_prefix}.attn_hyper_connection.input_mix_weight_down.weight",
        "hc_attn_up": f"{layer_prefix}.attn_hyper_connection.input_mix_weight_up.weight",
        "hc_attn_inject": f"{layer_prefix}.attn_hyper_connection.block_inject_weight.weight",
        "hc_ffn_norm": f"{layer_prefix}.mlp_hyper_connection.hc_norm.weight",
        "hc_ffn_down": f"{layer_prefix}.mlp_hyper_connection.input_mix_weight_down.weight",
        "hc_ffn_up": f"{layer_prefix}.mlp_hyper_connection.input_mix_weight_up.weight",
        "hc_ffn_inject": f"{layer_prefix}.mlp_hyper_connection.block_inject_weight.weight",
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
        # Linear-attention params (identical layout to Qwen3.5-MoE)
        "lin_in_proj_qkv_weight": f"{layer_prefix}.linear_attn.in_proj_qkv.weight",
        "lin_in_proj_z_weight": f"{layer_prefix}.linear_attn.in_proj_z.weight",
        "lin_in_proj_b_weight": f"{layer_prefix}.linear_attn.in_proj_b.weight",
        "lin_in_proj_a_weight": f"{layer_prefix}.linear_attn.in_proj_a.weight",
        "lin_conv_weight": f"{layer_prefix}.linear_attn.conv1d.weight",
        "lin_A_log": f"{layer_prefix}.linear_attn.A_log",
        "lin_dt_bias": f"{layer_prefix}.linear_attn.dt_bias",
        "lin_norm_weight": f"{layer_prefix}.linear_attn.norm.weight",
        "lin_out_weight": f"{layer_prefix}.linear_attn.out_proj.weight",
        # MoE: batched pre-fused experts (Qwen3-Next-style layout, passthrough)
        "router_weight": f"{moe_prefix}.gate.weight",
        "experts_gate_up": f"{moe_prefix}.experts.gate_up_proj",
        "experts_down": f"{moe_prefix}.experts.down_proj",
        **expand_module_mapping(
            MoESharedExpert._hf_mapping_defaults_,
            hf_prefix=moe_prefix,
            param_prefix="shared_expert_",
        ),
        "shared_expert_gate_proj_weight": f"{moe_prefix}.shared_expert_gate.weight",
        # Model-level weights. NO final_norm — the output_hc mix replaces it.
        "embedding": f"{model_prefix}.embed_tokens.weight",
        "output_hc_norm": f"{model_prefix}.hyper_connection_mixer.hc_norm.weight",
        "output_hc_down": f"{model_prefix}.hyper_connection_mixer.input_mix_weight_down.weight",
        "output_hc_up": f"{model_prefix}.hyper_connection_mixer.input_mix_weight_up.weight",
        "lm_head": "lm_head.weight",
    }


_QWEN4_EXP_TEXT_CONFIG_MAPPING: dict[str, str] = {
    "d_model": "hidden_size",
    "n_layers": "num_hidden_layers",
    "num_query_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads",
    "d_ff": "moe_intermediate_size",
    "vocab_size": "vocab_size",
    "max_seq": "max_position_embeddings",
    "head_size": "head_dim",
    "eps": "rms_norm_eps",
    "use_qkv_bias": "attention_bias",
    "num_experts": "num_experts",
    "num_experts_per_tok": "num_experts_per_tok",
    "shared_expert_intermediate": "shared_expert_intermediate_size",
    "partial_rotary_factor": "rope_parameters.partial_rotary_factor",
    "mrope_section": "rope_parameters.mrope_section",
    "linear_conv_kernel_dim": "linear_conv_kernel_dim",
    "linear_key_head_dim": "linear_key_head_dim",
    "linear_value_head_dim": "linear_value_head_dim",
    "linear_num_key_heads": "linear_num_key_heads",
    "linear_num_value_heads": "linear_num_value_heads",
    "layer_types": "layer_types",
    "full_attention_interval": "full_attention_interval",
    "hc_count": "hc_count",
    "hc_lowrank": "hc_lowrank",
    "output_gate_type": "output_gate_type",
    # Captured for the serve-spec generator; not yet in the training graph (see module
    # docstring): PLE n-gram memory, QSA indexer, MTP.
    "ple_layer_ids": "ple_layer_ids",
    "ngram_size": "ngram_size",
    "heads_per_ngram": "heads_per_ngram",
    "ple_conv_kernel_size": "ple_conv_kernel_size",
    "ple_embed_dim": "ple_embed_dim",
    "ngram_vocab_size_base": "ngram_vocab_size_base",
    "make_ngram_vocab_size_divisible_by": "make_ngram_vocab_size_divisible_by",
    "split_ngram_parts": "split_ngram_parts",
    "indexer_n_heads": "indexer_n_heads",
    "indexer_kv_heads": "indexer_kv_heads",
    "indexer_head_dim": "indexer_head_dim",
    "indexer_budget": "indexer_budget",
    "indexer_compress_ratio": "indexer_compress_ratio",
    "mtp_num_hidden_layers": "mtp_num_hidden_layers",
}


def _with_text_config_prefix(mapping: dict[str, str]) -> dict[str, str]:
    return {k: f"text_config.{v}" for k, v in mapping.items()}


class _Qwen4ExpBase(nn.Model):
    """Shared constructor/forward for the CausalLM and ConditionalGeneration variants."""

    def _init_qwen4_exp(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        num_query_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq: int,
        head_size: int,
        eps: float,
        use_qkv_bias: bool,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        partial_rotary_factor: float,
        mrope_section,
        linear_conv_kernel_dim: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        layer_types,
        full_attention_interval: int,
        hc_count: int,
        hc_lowrank: int,
        output_gate_type: str,
        ple_layer_ids,
        ngram_size: int,
        heads_per_ngram: int,
        ple_conv_kernel_size: int,
        ple_embed_dim: int,
        ngram_vocab_size_base: int,
        make_ngram_vocab_size_divisible_by: int,
        split_ngram_parts: int,
        indexer_n_heads: int,
        indexer_kv_heads: int,
        indexer_head_dim: int,
        indexer_budget: int,
        indexer_compress_ratio: int,
        mtp_num_hidden_layers: int,
        chunk_size: int,
        ep_size: int,
    ) -> None:
        if output_gate_type not in ("silu", "sigmoid"):
            raise ValueError(f"qwen4_exp output_gate_type must be 'silu' or 'sigmoid', got {output_gate_type!r}")

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
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate = shared_expert_intermediate

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

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.output_gate_type = output_gate_type
        self.gdn_gate_sigmoid = output_gate_type == "sigmoid"

        # Deferred subsystems: hyperparameters captured (ints/bools reach the runtime
        # config and the serve-spec generator), graph participation pending.
        self.ple_layer_ids = list(ple_layer_ids) if ple_layer_ids else []
        self.has_ple = bool(self.ple_layer_ids)
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ple_embed_dim = ple_embed_dim
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.split_ngram_parts = split_ngram_parts
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.mtp_num_hidden_layers = mtp_num_hidden_layers

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

        block_configs = []
        if self.n_linear_blocks > 0:
            block_configs.append(
                (
                    "mamba_blocks",
                    Qwen4ExpLinearBlock,
                    self.n_linear_blocks,
                    dict(
                        d_model=d_model,
                        d_ff=d_ff,
                        num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok,
                        shared_expert_intermediate=shared_expert_intermediate,
                        hc_count=hc_count,
                        hc_lowrank=hc_lowrank,
                        linear_conv_kernel_dim=linear_conv_kernel_dim,
                        linear_key_head_dim=linear_key_head_dim,
                        linear_value_head_dim=linear_value_head_dim,
                        linear_num_key_heads=linear_num_key_heads,
                        linear_num_value_heads=linear_num_value_heads,
                        chunk_size=chunk_size,
                        eps=eps,
                        gate_activation=output_gate_type,
                        ep_size=ep_size,
                    ),
                )
            )
        if self.n_attn_blocks > 0:
            block_configs.append(
                (
                    "attn_blocks",
                    Qwen4ExpAttentionBlock,
                    self.n_attn_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok,
                        shared_expert_intermediate=shared_expert_intermediate,
                        hc_count=hc_count,
                        hc_lowrank=hc_lowrank,
                        eps=eps,
                        use_qkv_bias=use_qkv_bias,
                        partial_rotary_factor=partial_rotary_factor,
                        mrope_section=mrope_section,
                        ep_size=ep_size,
                    ),
                )
            )

        self.embedding = Embedding(vocab_size, d_model)
        self.stream_init = StreamBroadcast(d_model, hc_count)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=self.block_types,
            n_layers=n_layers,
        )
        # The final mix before the LM head IS the output norm; it has no inject weights.
        self.output_hc = HyperConnection(d_model, hc_count, hc_lowrank, eps=eps, include_inject=False)
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

        # Global intermediate slots. The residual is hc_count streams wide; there is no
        # residual_final/ln_final_rstd — the output_hc mix replaces the final norm.
        _h = ("B", "T", "d_model")
        _wide = ("B", "T", "hc_count * d_model")
        self._register_activation("residual0", _wide, scope=G)
        self._register_activation("x0", _h, aliases=["encoded"], scope=G)
        self._register_activation("xN", _h, scope=G)
        self._register_activation("residualN", _wide, scope=G)
        self._register_activation("xF", _h, aliases=["ln_final"], scope=G)
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

        x = self.embedding(token_ids)
        residual = self.stream_init(x)
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        xf = self.output_hc(residual)
        loss = self.lm_head(xf, targets)
        return loss


_QWEN4_EXP_INIT_DEFAULTS = dict(
    vocab_size=248320,
    d_model=2560,
    n_layers=48,
    num_query_heads=24,
    num_kv_heads=2,
    d_ff=640,
    max_seq=262144,
    head_size=256,
    eps=1e-6,
    use_qkv_bias=False,
    num_experts=512,
    num_experts_per_tok=10,
    shared_expert_intermediate=640,
)


@nn.hf_config(
    architecture="Qwen4ExpForCausalLM",
    model_type="qwen4_exp_text",
    **_QWEN4_EXP_TEXT_CONFIG_MAPPING,
)
class Qwen4ExpCausalModel(_Qwen4ExpBase):
    """Qwen3.8-Flash-Next text model for ``Qwen4ExpForCausalLM``."""

    _name_remap_ = QWEN4_EXP_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_qwen4_exp_block_mappings("model.layers.{layer}", "model")

    def __init__(
        self,
        vocab_size: int = 248320,
        d_model: int = 2560,
        n_layers: int = 48,
        num_query_heads: int = 24,
        num_kv_heads: int = 2,
        d_ff: int = 640,
        max_seq: int = 262144,
        head_size: int = 256,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        shared_expert_intermediate: int = 640,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        output_gate_type: str = "sigmoid",
        ple_layer_ids: list[int] | None = None,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ple_conv_kernel_size: int = 4,
        ple_embed_dim: int = 0,
        ngram_vocab_size_base: int = 0,
        make_ngram_vocab_size_divisible_by: int = 128,
        split_ngram_parts: int = 128,
        indexer_n_heads: int = 0,
        indexer_kv_heads: int = 0,
        indexer_head_dim: int = 0,
        indexer_budget: int = 0,
        indexer_compress_ratio: int = 0,
        mtp_num_hidden_layers: int = 0,
        chunk_size: int = 64,
        ep_size: int = 1,
    ):
        super().__init__()
        self._init_qwen4_exp(
            vocab_size, d_model, n_layers, num_query_heads, num_kv_heads, d_ff, max_seq,
            head_size, eps, use_qkv_bias, num_experts, num_experts_per_tok,
            shared_expert_intermediate, partial_rotary_factor, mrope_section,
            linear_conv_kernel_dim, linear_key_head_dim, linear_value_head_dim,
            linear_num_key_heads, linear_num_value_heads, layer_types,
            full_attention_interval, hc_count, hc_lowrank, output_gate_type,
            ple_layer_ids, ngram_size, heads_per_ngram, ple_conv_kernel_size,
            ple_embed_dim, ngram_vocab_size_base, make_ngram_vocab_size_divisible_by,
            split_ngram_parts, indexer_n_heads, indexer_kv_heads, indexer_head_dim,
            indexer_budget, indexer_compress_ratio, mtp_num_hidden_layers,
            chunk_size, ep_size,
        )


@nn.hf_config(
    architecture="Qwen4ExpForConditionalGeneration",
    model_type="qwen4_exp",
    **_with_text_config_prefix(_QWEN4_EXP_TEXT_CONFIG_MAPPING),
)
class Qwen4ExpConditionalModel(_Qwen4ExpBase):
    """Qwen3.8-Flash-Next text model for ``Qwen4ExpForConditionalGeneration``.

    Text-only, like the serve target: the checkpoint's vision tower is not declared and
    its tensors are simply unused at import."""

    _name_remap_ = QWEN4_EXP_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_qwen4_exp_block_mappings(
        "model.language_model.layers.{layer}", "model.language_model"
    )

    def __init__(
        self,
        vocab_size: int = 248320,
        d_model: int = 2560,
        n_layers: int = 48,
        num_query_heads: int = 24,
        num_kv_heads: int = 2,
        d_ff: int = 640,
        max_seq: int = 262144,
        head_size: int = 256,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        shared_expert_intermediate: int = 640,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        output_gate_type: str = "sigmoid",
        ple_layer_ids: list[int] | None = None,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ple_conv_kernel_size: int = 4,
        ple_embed_dim: int = 0,
        ngram_vocab_size_base: int = 0,
        make_ngram_vocab_size_divisible_by: int = 128,
        split_ngram_parts: int = 128,
        indexer_n_heads: int = 0,
        indexer_kv_heads: int = 0,
        indexer_head_dim: int = 0,
        indexer_budget: int = 0,
        indexer_compress_ratio: int = 0,
        mtp_num_hidden_layers: int = 0,
        chunk_size: int = 64,
        ep_size: int = 1,
    ):
        super().__init__()
        self._init_qwen4_exp(
            vocab_size, d_model, n_layers, num_query_heads, num_kv_heads, d_ff, max_seq,
            head_size, eps, use_qkv_bias, num_experts, num_experts_per_tok,
            shared_expert_intermediate, partial_rotary_factor, mrope_section,
            linear_conv_kernel_dim, linear_key_head_dim, linear_value_head_dim,
            linear_num_key_heads, linear_num_value_heads, layer_types,
            full_attention_interval, hc_count, hc_lowrank, output_gate_type,
            ple_layer_ids, ngram_size, heads_per_ngram, ple_conv_kernel_size,
            ple_embed_dim, ngram_vocab_size_base, make_ngram_vocab_size_divisible_by,
            split_ngram_parts, indexer_n_heads, indexer_kv_heads, indexer_head_dim,
            indexer_budget, indexer_compress_ratio, mtp_num_hidden_layers,
            chunk_size, ep_size,
        )
