"""Gemma3 Transformer Blocks.

Gemma3 uses the same sandwich-norm pattern as Gemma4 -- 4 RMSNorms per block:

  input_layernorm -> attention -> post_attention_layernorm -> residual_add
  pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> residual_add

and the same alternation of local (sliding-window) and global (full) attention.
What it does *not* have is everything Gemma4 added on top: no ``layer_scalar``,
no per-layer inputs, no k_eq_v, no KV sharing, no logit softcapping. Nor does it
carry Gemma4's V-norm -- the checkpoint ships ``q_norm`` and ``k_norm`` only --
so these blocks use the config-driven ``GenericGQAttention`` (what Qwen3 uses)
rather than ``Gemma4Attention``, which would normalise a V that has no weight.

Two numeric details separate Gemma3 from Gemma4 and both are easy to get
silently wrong:

* **Softmax scale.** Gemma4 passes ``softmax_scale=1.0`` because its QK-norm
  yields unit-RMS Q and K. Gemma3 does not: it scales by
  ``query_pre_attn_scalar ** -0.5``. For a checkpoint where that scalar equals
  ``head_dim`` (embeddinggemma-300m: both 256) this coincides with the kernel's
  default, which is exactly why it must be passed explicitly -- Gemma3-27B has
  ``query_pre_attn_scalar 168`` against ``head_dim 128`` and would inherit a
  wrong scale in silence.

* **Direction.** ``use_bidirectional_attention`` is a property of the
  checkpoint, not of the family: EmbeddingGemma sets it, the generative Gemma3
  models do not. It reaches the kernel through ``AttentionConfig.causal``.

**What ``sliding_window`` means once attention is bidirectional.** A causal layer
masks ``i - j >= W``: a left window of W. A bidirectional one masks
``abs(i - j) >= W``: W on *each* side. The convention follows from ``causal`` and
needs no separate declaration, but it must be got right in the kernel, because
the two reference implementations disagree:

* transformers, ``modeling_gemma3.py:491`` -- ``abs(q_idx - kv_idx) < sliding_window``,
  so +-512.
* llama.cpp, ``LLAMA_SWA_TYPE_SYMMETRIC`` in ``llama-hparams.h:462`` -- masks when
  ``abs(pos_diff) > n_swa / 2``, so +-256, half as wide.

**Follow transformers.** It is the reference implementation for the published
checkpoint and what produced Google's own numbers. The difference is invisible on
short inputs -- under 256 tokens the two masks are identical -- and measurable
beyond: on a 369-token document the pooled embeddings sit at cosine 0.998534, and
the gap widens with length. Anything that validates only on short strings will
not see this.

Block variants:
  - Gemma3SlidingBlock: sliding-window attention, gated-GELU MLP
  - Gemma3FullBlock:    full attention, gated-GELU MLP
"""

from __future__ import annotations

from .. import nn
from ..activations import Activation
from ..attention import AttentionConfig
from ..block_schema import BlockSchema, ServeObject, SlotDecl
from ..mlp import MLPConfig
from ..modules import GenericGQAttention, GenericMLP, RMSNorm


GEMMA3_BLOCK_NAME_REMAP: dict[str, str] = {
    # --- input_layernorm (standalone rmsnorm) -> ln1 ---
    "input_layernorm_weight": "ln1_weight",
    "input_layernorm_y": "ln1",
    "input_layernorm_rstd": "ln1_rstd",
    # --- self_attn (GenericGQAttention) -> strip prefix ---
    "self_attn_qkv_weight": "qkv_weight",
    "self_attn_out_weight": "out_weight",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_flat": "qkv_flat",
    "self_attn_qkv_norm": "qkv_norm",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    "self_attn_x_flat": "x_flat",
    # --- post_attn_layernorm (standalone rmsnorm) ---
    "post_attn_layernorm_weight": "ln_post_attn_weight",
    "post_attn_layernorm_y": "ln_post_attn",
    "post_attn_layernorm_rstd": "ln_post_attn_rstd",
    # --- pre_ff_layernorm (fused_residual_rmsnorm: res_att + pre_ff_norm) -> ln2 ---
    "pre_ff_layernorm_weight": "ln2_weight",
    "pre_ff_layernorm_y": "ln2",
    "pre_ff_layernorm_rstd": "ln2_rstd",
    "pre_ff_layernorm_res": "res_att",
    # --- mlp (GenericMLP with gelu, separate gate/up) ---
    "mlp_gate_weight": "mlp_gate_weight",
    "mlp_up_weight": "mlp_up_weight",
    "mlp_down_weight": "mlp_down_weight",
    "mlp_x_flat": "mlp_x_flat",
    "mlp_gate_flat": "mlp_gate_flat",
    "mlp_up_flat": "mlp_up_flat",
    "mlp_gate_act": "mlp_gate_act",
    "mlp_act_flat": "swiglu_flat",
    "mlp_down_flat": "mlp_down_flat",
    "mlp_down": "mlp_down",
    # --- post_ff_layernorm (standalone rmsnorm) ---
    "post_ff_layernorm_weight": "ln_post_ff_weight",
    "post_ff_layernorm_y": "ln_post_ff",
    "post_ff_layernorm_rstd": "ln_post_ff_rstd",
    # --- res_att (canonical residual-after-attention slot) ---
    "res_att": "res_att",
}


# Gemma3 uses a separate-gate GELU MLP (no gate/up fusion, GELU on the gate).
_GEMMA3_GELU_MLP_CONFIG = MLPConfig(
    activation=Activation.GELU,
    gated=True,
    fuse_gate_up=False,
)


#: How a serving artifact stores one Gemma 3 block.
#:
#: Every norm carries ``unfold_unit_offset``. Gemma stores RMSNorm weights
#: zero-centred as ``w`` and applies them as ``1 + w``; the runtime does the same,
#: reading these with ``rmsnorm(..., unit_offset=true)``, so the artifact must
#: hold the *unfolded* ``w``. The transform names that requirement.
#:
#: What it costs depends on the source, and the GGUF is the expensive one:
#: an HF safetensors checkpoint already stores ``w`` and passes through, while a
#: GGUF stores the folded ``1 + w`` and the converter must subtract the one.
#: Measured on embeddinggemma-300M-Q8_0, ``cos(gguf - 1, hf) == 1.000000`` for
#: every norm tensor. qwen4exp's converter does exactly this subtraction for its
#: query/key norms (``convert.py:268``).
#:
#: Forgetting it is quiet and expensive: the artifact holds ``1 + w``, the runtime
#: makes it ``2 + w``, and nothing raises. That is the live qwen3_6 indexer bug,
#: where the same omission on ``indexer.q_norm`` only bites past 2,051 cached
#: tokens. Here it would bite everywhere -- loading GGUF norms unsubtracted drops
#: pooled embeddings to cosine ~0 against the reference with retrieval inverted,
#: at plausible-looking similarities around 0.7.
_GEMMA3_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("input_norm", "bf16", ("C",), ("ln1_weight",), transform="unfold_unit_offset"),
    ServeObject("post_attention_norm", "bf16", ("C",), ("ln_post_attn_weight",),
                transform="unfold_unit_offset"),
    ServeObject("pre_feedforward_norm", "bf16", ("C",), ("ln2_weight",),
                transform="unfold_unit_offset"),
    ServeObject("post_feedforward_norm", "bf16", ("C",), ("ln_post_ff_weight",),
                transform="unfold_unit_offset"),
    ServeObject("attention/query_key_value", "quantised", ("QKV", "C"), ("qkv_weight",)),
    ServeObject("attention/query_norm", "bf16", ("HeadDim",), ("q_norm_weight",),
                transform="unfold_unit_offset"),
    ServeObject("attention/key_norm", "bf16", ("HeadDim",), ("k_norm_weight",),
                transform="unfold_unit_offset"),
    ServeObject("attention/output", "quantised", ("C", "AttnDim"), ("out_weight",)),
    ServeObject("mlp/gate", "quantised", ("M", "C"), ("mlp_gate_weight",)),
    ServeObject("mlp/up", "quantised", ("M", "C"), ("mlp_up_weight",)),
    ServeObject("mlp/down", "quantised", ("C", "M"), ("mlp_down_weight",)),
)


def _gemma3_schema(block_family: str) -> BlockSchema:
    return BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_gate_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_up_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("res_att", shape=("B", "T", "C")),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        serve_objects=_GEMMA3_SERVE_OBJECTS,
        attrs={"block_family": block_family},
    )


def _sandwich_attn_phase(block, x, residual, position_ids):
    """input_ln -> attn -> post_attn_ln, leaving the residual add to the caller.

    The ``residual + h_post_attn`` sum is deferred so the MLP phase can fuse it
    with ``pre_ff_layernorm`` through RMSNorm's two-argument fused_residual path,
    saving one HBM round-trip per block.

    Like Gemma4, state flows between blocks through ``x`` rather than through a
    running residual, so the incoming hidden state is materialised into the
    canonical ``res_ffn`` slot for dumps and backward replay while the norm
    itself follows HF's standalone RMSNorm path.
    """
    fresh_zeros = block._zeros(["B", "T", "d_model"], name="fresh_zero")
    block._register_activation(
        "res_ffn",
        ("B", "T", "d_model"),
        aliases=["residual_ffn", "res_in"],
        share_policy="when_recomputed",
    )
    residual = block._add(fresh_zeros, x, name="res_ffn")
    h = block.input_layernorm(residual)
    h = block.self_attn(h, position_ids)
    h = block.post_attn_layernorm(h)
    return residual, h


def _sandwich_mlp_phase(block, residual, h_post_attn):
    """pre_ff_ln(res_att) -> MLP -> post_ff_ln -> residual add."""
    residual, h = block.pre_ff_layernorm(residual, h_post_attn)
    h = block.mlp(h)
    h = block.post_ff_layernorm(h)
    residual = block._add(residual, h, name="res_mlp")
    return residual


def _finalize(block, residual):
    """Publish the block output into its own per-layer ``h_out`` slot.

    Gemma4 scales by ``layer_scalar`` here; Gemma3 has no such parameter, so the
    residual passes through unchanged. The slot itself still matters: writing
    into the MLP's own ``mlp_down`` buffer instead would collide with the MLP's
    output name, and the autodiff's produced-by map keeps only the last writer --
    which silently drops the MLP -> post_ff_ln edge from the backward graph and
    zeroes every MLP LoRA gradient.
    """
    block._register_activation("h_out", ("B", "T", "d_model"), share_policy="per_layer", save=True)
    zero_copy = block._zeros(["B", "T", "d_model"], name="copy_zero")
    h_out = block._add(residual, zero_copy, name="h_out")
    return h_out, h_out


def _make_dims(block, d_model, head_size, num_query_heads, num_kv_heads, d_ff, max_seq):
    block.C = d_model
    block.D = head_size
    block.Hq = num_query_heads
    block.Hkv = num_kv_heads
    block.M = d_ff
    block.MaxSeq = max_seq
    block.AttnDim = num_query_heads * head_size
    block.QKV = (num_query_heads + 2 * num_kv_heads) * head_size


def _attention_config(*, sliding_window, causal, query_pre_attn_scalar, head_size, eps):
    # Gemma3 scales by query_pre_attn_scalar ** -0.5, which is NOT always
    # 1/sqrt(head_dim): the 27B has scalar 168 against head_dim 128. Passing it
    # explicitly keeps a checkpoint where they coincide from hiding the case
    # where they do not.
    scalar = query_pre_attn_scalar if query_pre_attn_scalar else head_size
    return AttentionConfig(
        qk_norm=True,
        sliding_window=sliding_window or 0,
        causal=causal,
        softmax_scale=float(scalar) ** -0.5,
        eps=eps,
    )


class _Gemma3BlockBase(nn.Block):
    """Shared body; the two variants differ only in their attention config."""

    _name_remap_ = GEMMA3_BLOCK_NAME_REMAP

    def _build(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        *,
        sliding_window,
        causal,
        query_pre_attn_scalar,
        eps,
    ):
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads, d_ff, max_seq)
        self.input_layernorm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=_attention_config(
                sliding_window=sliding_window,
                causal=causal,
                query_pre_attn_scalar=query_pre_attn_scalar,
                head_size=head_size,
                eps=eps,
            ),
        )
        self.post_attn_layernorm = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff, config=_GEMMA3_GELU_MLP_CONFIG)
        self.post_ff_layernorm = RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids):
        residual, h_post_attn = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_mlp_phase(self, residual, h_post_attn)
        return _finalize(self, residual)


class Gemma3SlidingBlock(_Gemma3BlockBase):
    """Local attention over a ``sliding_window`` neighbourhood."""

    schema = _gemma3_schema("gemma3_sliding")

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        sliding_window=512,
        causal=True,
        query_pre_attn_scalar=0,
        eps=1e-6,
    ):
        super().__init__()
        self._build(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            d_ff,
            max_seq,
            sliding_window=sliding_window,
            causal=causal,
            query_pre_attn_scalar=query_pre_attn_scalar,
            eps=eps,
        )


class Gemma3FullBlock(_Gemma3BlockBase):
    """Global attention over the whole sequence."""

    schema = _gemma3_schema("gemma3_full")

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        causal=True,
        query_pre_attn_scalar=0,
        eps=1e-6,
    ):
        super().__init__()
        self._build(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            d_ff,
            max_seq,
            sliding_window=None,
            causal=causal,
            query_pre_attn_scalar=query_pre_attn_scalar,
            eps=eps,
        )
