"""Attention modules for the Python DSL.

The ``_hf_mapping_defaults_`` dicts are exposed at the class level so
``surogate.dsl.hf`` can read them without instantiating a module.
"""

from __future__ import annotations

from typing import Any

from ..attention import AttentionConfig
from ..dim import B, Dim, T
from ..hf import fuse
from ..nn import Module, Proxy, Tracer
from ..rope import RoPE
from ..specs import LoRATarget


def _resolve_rotary_dim(head_size: int, partial_rotary_factor: float) -> int:
    """Compute the rotary embedding dimension from head_size and partial_rotary_factor."""
    rotary = int(round(float(head_size) * float(partial_rotary_factor)))
    rotary = max(2, min(rotary, head_size))
    if rotary % 2 != 0:
        rotary -= 1
    return max(2, rotary)


def _base_qkv_hf_mapping() -> dict[str, Any]:
    """Fresh copy of the fused QKV + out projection HF weight template."""
    return dict(_BASE_QKV_HF_MAPPING)


_BASE_QKV_HF_MAPPING: dict[str, Any] = {
    "qkv_weight": fuse(
        "{prefix}.q_proj.weight",
        "{prefix}.k_proj.weight",
        "{prefix}.v_proj.weight",
        dim=0,
    ),
    "qkv_bias": fuse(
        "{prefix}.q_proj.bias",
        "{prefix}.k_proj.bias",
        "{prefix}.v_proj.bias",
        dim=0,
    ),
    "out_weight": "{prefix}.o_proj.weight",
    "out_bias": "{prefix}.o_proj.bias",
}


class GenericGQAttention(Module):
    """Config-driven Grouped-Query Attention.

    Covers the common shape of dense attention across Llama / Qwen3 /
    GPT-OSS: fused QKV projection, optional QK-norm, optional biases,
    optional sinks, optional sliding window, standard or MRoPE
    position encoding.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        config: Any = None,
    ) -> None:
        super().__init__()
        self.config: AttentionConfig = config if config is not None else AttentionConfig()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq

        # Flat attrs so the existing ``when=`` condition machinery
        # (which uses ``getattr(self, ...)``) keeps working.
        self.use_qkv_bias = self.config.qkv_bias
        # The two bias flags are independent: GPT-OSS, for example, has a
        # QKV bias but no o_proj bias. Models that need both must set both
        # explicitly on their AttentionConfig.
        self.use_out_bias = self.config.out_bias
        self.use_qk_norm = self.config.qk_norm
        self.use_sinks = self.config.has_sinks
        self.sliding_window = self.config.sliding_window
        self.is_mrope = self.config.rope is RoPE.MROPE or self.config.rope.cpp_op == "mrope"
        self.rotary_dim = _resolve_rotary_dim(head_size, self.config.partial_rotary_factor)

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

        mapping = _base_qkv_hf_mapping()
        if self.use_qk_norm:
            mapping["q_norm_weight"] = "{prefix}.q_norm.weight"
            mapping["k_norm_weight"] = "{prefix}.k_norm.weight"
        if self.use_sinks:
            mapping["sinks"] = "{prefix}.sinks"
        self._hf_mapping_defaults_ = mapping

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids, *_rest = args
        mrope_section = kwargs.get("mrope_section")

        cfg = self.config

        # LoRA target layout for the fused QKV projection. The fused weight
        # packs Q, then K, then V along the output dim with sizes Hq*D,
        # Hkv*D, Hkv*D (matches the HF weight fusion above). Each logical
        # projection is a separately-addressable LoRA target.
        _hq = self.num_query_heads * self.head_size
        _hkv = self.num_kv_heads * self.head_size
        _hidden = self.d_model
        qkv_targets = [
            LoRATarget(name="q", offset=0, size=_hq),
            LoRATarget(name="k", offset=_hq, size=_hkv),
            LoRATarget(name="v", offset=_hq + _hkv, size=_hkv),
        ]
        out_targets = [LoRATarget(name="o", offset=0, size=_hidden)]

        # -- params -----------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"), lora_targets=qkv_targets)
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"), lora_targets=out_targets)
        out_b = tracer.register_param("out_bias", ("C",), when="use_out_bias")
        # q_norm / k_norm sit before rope_freqs to match the legacy
        # Qwen3Attention param ordering (preserves existing weight init /
        # checkpoint offsets).
        if self.use_qk_norm:
            tracer.register_param(
                "q_norm_weight",
                ("D",),
                quantizable=False,
                when="use_qk_norm",
            )
            tracer.register_param(
                "k_norm_weight",
                ("D",),
                quantizable=False,
                when="use_qk_norm",
            )
        tracer.register_param(
            "rope_freqs",
            ("MaxSeq", "D // 2", 2),
            dtype="fp32",
            frozen=True,
        )
        if self.use_sinks:
            tracer.register_param("sinks", ("Hq",))

        # -- activation slots -------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv",
            ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True,
            share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope",
            ("B", "T", "QKV"),
            save=True,
            share_policy="when_recomputed",
        )
        if self.use_qk_norm:
            tracer.register_activation(
                "q_rstd",
                ("B", "T", "Hq"),
                dtype="fp32",
                save=True,
                share_policy="when_recomputed",
                when="use_qk_norm",
            )
            tracer.register_activation(
                "k_rstd",
                ("B", "T", "Hkv"),
                dtype="fp32",
                save=True,
                share_policy="when_recomputed",
                when="use_qk_norm",
            )
        att_slot = tracer.register_activation(
            "att",
            ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True,
            share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse",
            ("B", "Hq", "T"),
            dtype="fp32",
            save=True,
            share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out",
            ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat,
                qkv_w,
                qkv_b,
                transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat,
                qkv_w,
                transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat,
            shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        # -- QK-norm + RoPE ---------------------------------------------
        # Fused qkv_qk_norm_rope is only registered for standard RoPE;
        # MRoPE or QK-norm-only paths fall through to separate ops.
        rope_kwargs: dict[str, Any] = {}
        if self.is_mrope and mrope_section is not None:
            rope_kwargs["mrope_section"] = mrope_section
        rope_kwargs["rotary_dim"] = self.rotary_dim if self.rotary_dim != self.head_size else "D"

        if self.use_qk_norm and not self.is_mrope:
            qkv_rope, _q_rstd, _k_rstd = g.qkv_qk_norm_rope(
                qkv,
                tracer.prefixed("q_norm_weight"),
                tracer.prefixed("k_norm_weight"),
                tracer.prefixed("rope_freqs"),
                position_ids.ref,
                eps=cfg.eps,
                out_name=qkv_rope_slot,
                q_rstd_name=tracer.prefixed("q_rstd"),
                k_rstd_name=tracer.prefixed("k_rstd"),
            )
        else:
            if self.use_qk_norm:
                qkv, _q_rstd, _k_rstd = g.qkv_qk_norm(
                    qkv,
                    tracer.prefixed("q_norm_weight"),
                    tracer.prefixed("k_norm_weight"),
                    eps=cfg.eps,
                )
            rope_op = g.mrope if self.is_mrope else g.rope
            qkv_rope = rope_op(
                qkv,
                tracer.prefixed("rope_freqs"),
                position_ids.ref,
                out_name=qkv_rope_slot,
                **rope_kwargs,
            )

        # -- Flash attention --------------------------------------------
        fa_kwargs: dict[str, Any] = {"causal": True}
        if self.sliding_window:
            fa_kwargs["window_size"] = self.sliding_window
        if cfg.softmax_scale is not None:
            fa_kwargs["softmax_scale"] = cfg.softmax_scale
        if self.use_sinks:
            fa_kwargs["sinks"] = tracer.prefixed("sinks")

        attn_out, _lse = g.flash_attention(
            qkv_rope,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
            **fa_kwargs,
        )

        # -- Output projection ------------------------------------------
        attn_flat = g.view(
            attn_out,
            shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_out_bias:
            out_flat = g.matmul_bias(
                attn_flat,
                out_w,
                out_b,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat,
                out_w,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Qwen3VLAttention(Module):
    """Qwen3-VL attention with QK-Norm + MRoPE (separate, not fused)."""

    _hf_mapping_defaults_ = {
        "qkv_weight": fuse(
            "{prefix}.q_proj.weight",
            "{prefix}.k_proj.weight",
            "{prefix}.v_proj.weight",
            dim=0,
        ),
        "qkv_bias": fuse(
            "{prefix}.q_proj.bias",
            "{prefix}.k_proj.bias",
            "{prefix}.v_proj.bias",
            dim=0,
        ),
        "out_weight": "{prefix}.o_proj.weight",
        "out_bias": "{prefix}.o_proj.bias",
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
        eps: float = 1e-6,
        mrope_section: tuple[int, int, int] | list[int] = (24, 20, 20),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias
        self.eps = eps
        self.mrope_section = list(mrope_section)

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # Fused QKV layout: Q, K, V along dim-0 of the weight with sizes
        # Hq*D, Hkv*D, Hkv*D. Same layout as GenericGQAttention.
        _hq = self.num_query_heads * self.head_size
        _hkv = self.num_kv_heads * self.head_size
        _hidden = self.d_model
        qkv_targets = [
            LoRATarget(name="q", offset=0, size=_hq),
            LoRATarget(name="k", offset=_hq, size=_hkv),
            LoRATarget(name="v", offset=_hq + _hkv, size=_hkv),
        ]
        out_targets = [LoRATarget(name="o", offset=0, size=_hidden)]

        # -- params --------------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"), lora_targets=qkv_targets)
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"), lora_targets=out_targets)
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param("q_norm_weight", ("D",), quantizable=False)
        tracer.register_param("k_norm_weight", ("D",), quantizable=False)
        tracer.register_param(
            "rope_freqs",
            ("MaxSeq", "D // 2", 2),
            dtype="fp32",
            frozen=True,
        )

        # -- activation slots ----------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv",
            ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True,
            share_policy="when_recomputed",
        )
        qkv_norm_slot = tracer.register_activation(
            "qkv_norm",
            ("B", "T", "QKV"),
            save=True,
            share_policy="when_recomputed",
            description="QKV after QK-Norm",
        )
        tracer.register_activation(
            "q_rstd",
            ("B", "T", "Hq"),
            dtype="fp32",
            save=True,
            share_policy="when_recomputed",
        )
        tracer.register_activation(
            "k_rstd",
            ("B", "T", "Hkv"),
            dtype="fp32",
            save=True,
            share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope",
            ("B", "T", "QKV"),
            save=True,
            share_policy="when_recomputed",
            description="QKV after QK-Norm + MRoPE",
        )
        att_slot = tracer.register_activation(
            "att",
            ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True,
            share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse",
            ("B", "Hq", "T"),
            dtype="fp32",
            save=True,
            share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out",
            ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph ---------------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat,
                qkv_w,
                qkv_b,
                transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat,
                qkv_w,
                transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat,
            shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        # QK-Norm (separate from RoPE)
        qkv_norm, q_rstd, k_rstd = g.qkv_qk_norm(
            qkv,
            tracer.prefixed("q_norm_weight"),
            tracer.prefixed("k_norm_weight"),
            eps=self.eps,
            out_name=qkv_norm_slot,
            q_rstd_name=tracer.prefixed("q_rstd"),
            k_rstd_name=tracer.prefixed("k_rstd"),
        )

        # MRoPE (separate)
        qkv_rope = g.mrope(
            qkv_norm,
            tracer.prefixed("rope_freqs"),
            position_ids.ref,
            rotary_dim="D",
            mrope_section=self.mrope_section,
            out_name=qkv_rope_slot,
        )

        # Flash Attention
        attn_out, lse = g.flash_attention(
            qkv_rope,
            causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Output projection
        attn_flat = g.view(
            attn_out,
            shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat,
                out_w,
                out_b,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat,
                out_w,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Qwen3_5Attention(Module):
    """Qwen3.5 full-attention with separate Q/K/V projections, QK-Norm, partial MRoPE, and gated output.

    Unlike the standard GenericGQAttention path (fused QKV), Qwen3.5 uses:
    - Separate Q projection (outputs 2*Hq*D for Q + gate)
    - Separate K projection
    - Separate V projection
    - QK-Norm with weight+1 bias
    - Partial MRoPE (only rotary_dim of head_dim is rotated)
    - Sigmoid-gated attention output
    """

    _hf_mapping_defaults_ = {
        "q_proj_weight": "{prefix}.q_proj.weight",
        "q_proj_bias": "{prefix}.q_proj.bias",
        "k_proj_weight": "{prefix}.k_proj.weight",
        "k_proj_bias": "{prefix}.k_proj.bias",
        "v_proj_weight": "{prefix}.v_proj.weight",
        "v_proj_bias": "{prefix}.v_proj.bias",
        "out_weight": "{prefix}.o_proj.weight",
        "out_bias": "{prefix}.o_proj.bias",
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
        eps: float = 1e-6,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias
        self.eps = eps
        self.partial_rotary_factor = partial_rotary_factor
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.M = Dim("M")
        self.MaxSeq = Dim("MaxSeq")
        self.AttnDim = self.Hq * self.D
        self.QProjDim = 2 * self.AttnDim
        self.KVDim = self.Hkv * self.D
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.RotaryDim = _resolve_rotary_dim(head_size, partial_rotary_factor)

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # Qwen3.5 uses unfused Q/K/V projections. Each weight has a single
        # LoRA target covering its full output dimension (size=0 = full).
        # The Q projection emits 2*Hq*D (Q + gate) and receives a single
        # "q" LoRA target over the whole 2*Hq*D output — matching the legacy
        # C++ allocation that uses ``q_lora_out = 2 * q_out`` for Qwen3.5.
        _hq2 = 2 * self.num_query_heads * self.head_size
        _hkv = self.num_kv_heads * self.head_size
        _hq = self.num_query_heads * self.head_size
        _hidden = self.d_model

        # -- params ----------------------------------------------------------
        q_proj_w = tracer.register_param(
            "q_proj_weight",
            ("QProjDim", "C"),
            lora_targets=[LoRATarget(name="q", size=_hq2)],
        )
        q_proj_b = tracer.register_param("q_proj_bias", ("QProjDim",), when="use_qkv_bias")
        k_proj_w = tracer.register_param(
            "k_proj_weight",
            ("KVDim", "C"),
            lora_targets=[LoRATarget(name="k", size=_hkv)],
        )
        k_proj_b = tracer.register_param("k_proj_bias", ("KVDim",), when="use_qkv_bias")
        v_proj_w = tracer.register_param(
            "v_proj_weight",
            ("KVDim", "C"),
            lora_targets=[LoRATarget(name="v", size=_hkv)],
        )
        v_proj_b = tracer.register_param("v_proj_bias", ("KVDim",), when="use_qkv_bias")
        out_w = tracer.register_param(
            "out_weight",
            ("C", "AttnDim"),
            lora_targets=[LoRATarget(name="o", size=_hidden)],
        )
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param("q_norm_weight", ("D",), quantizable=False)
        tracer.register_param("k_norm_weight", ("D",), quantizable=False)
        tracer.register_param(
            "rope_freqs",
            ("MaxSeq", "RotaryDim // 2", 2),
            dtype="fp32",
            frozen=True,
            quantizable=False,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv",
            ("B", "T", "QKV"),
            save=True,
            share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope",
            ("B", "T", "QKV"),
            save=True,
            share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att",
            ("B", "T", "AttnDim"),
            aliases=["att_flat"],
            save=True,
            share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse",
            ("B", "Hq", "T"),
            dtype="fp32",
            save=True,
            share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out",
            ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # Separate Q/K/V projections
        if self.use_qkv_bias:
            q_proj = g.matmul_bias(
                x_flat,
                q_proj_w,
                q_proj_b,
                transpose="NT",
                out_name=tracer.prefixed("q_proj"),
            )
            k_proj = g.matmul_bias(
                x_flat,
                k_proj_w,
                k_proj_b,
                transpose="NT",
                out_name=tracer.prefixed("k_proj"),
            )
            v_proj = g.matmul_bias(
                x_flat,
                v_proj_w,
                v_proj_b,
                transpose="NT",
                out_name=tracer.prefixed("v_proj"),
            )
        else:
            q_proj = g.matmul(
                x_flat,
                q_proj_w,
                transpose="NT",
                out_name=tracer.prefixed("q_proj"),
            )
            k_proj = g.matmul(
                x_flat,
                k_proj_w,
                transpose="NT",
                out_name=tracer.prefixed("k_proj"),
            )
            v_proj = g.matmul(
                x_flat,
                v_proj_w,
                transpose="NT",
                out_name=tracer.prefixed("v_proj"),
            )

        # Split Q into Q + gate, reshape all to 4D
        q_proj_4d = g.view(
            q_proj,
            shape=[B, T, self.Hq, 2 * self.D],
            out_name=tracer.prefixed("q_proj_4d"),
        )
        q, gate_4d = g.split(q_proj_4d, split_size=[self.head_size, self.head_size], dim=3)
        q = g.view(q, shape=[B, T, self.Hq, self.D], out_name=tracer.prefixed("q"))
        gate_4d = g.view(gate_4d, shape=[B, T, self.Hq, self.D], out_name=tracer.prefixed("gate"))
        k = g.view(k_proj, shape=[B, T, self.Hkv, self.D], out_name=tracer.prefixed("k"))
        v = g.view(v_proj, shape=[B, T, self.Hkv, self.D], out_name=tracer.prefixed("v"))
        qkv = g.concat(q, k, v, dim=2)

        # QK-Norm with weight+1 bias
        ones_d = g.ones(shape=[self.D], dtype="bf16")
        q_norm_eff = g.add(
            tracer.prefixed("q_norm_weight"),
            ones_d,
            out_name=tracer.prefixed("q_norm_weight_eff"),
        )
        k_norm_eff = g.add(
            tracer.prefixed("k_norm_weight"),
            ones_d,
            out_name=tracer.prefixed("k_norm_weight_eff"),
        )
        qkv_norm, _, _ = g.qkv_qk_norm(
            qkv,
            q_norm_eff,
            k_norm_eff,
            eps=self.eps,
        )

        # Partial MRoPE
        qkv_rope = g.mrope(
            qkv_norm,
            tracer.prefixed("rope_freqs"),
            position_ids.ref,
            rotary_dim=self.RotaryDim,
            mrope_section=self.mrope_section,
            out_name=qkv_rope_slot,
        )

        # Flash Attention
        attn_out, lse = g.flash_attention(
            qkv_rope,
            causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Sigmoid-gated output
        attn_4d = g.view(attn_out, shape=[B, T, self.Hq, self.D], out_name=tracer.prefixed("att_4d"))
        gate_sigmoid = g.sigmoid(gate_4d)
        gated_attn_4d = g.mul(attn_4d, gate_sigmoid)
        gated_attn_flat = g.view(
            gated_attn_4d,
            shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )

        # Output projection
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                gated_attn_flat,
                out_w,
                out_b,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                gated_attn_flat,
                out_w,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Gemma4Attention(Module):
    """Gemma4-style attention with QKV-norm and RoPE.

    Supports two modes:
    - Standard (k_eq_v=False): Fused Q+K+V projection, separate norms.
    - K-equals-V (k_eq_v=True): Fused Q+K projection only, V reuses raw K
      output (before K-norm and RoPE). V-norm still applies.

    Both modes use:
    - Q-norm and K-norm with (1 + weight) scale pattern
    - V-norm without learnable scale (RMS normalization only)
    - RoPE applied after norms (only to Q and K)
    - Optional sliding window for local attention layers
    - Optional partial rotary factor for full attention layers
    """

    _hf_mapping_defaults_ = {
        "qkv_weight": fuse(
            "{prefix}.q_proj.weight",
            "{prefix}.k_proj.weight",
            "{prefix}.v_proj.weight",
            dim=0,
        ),
        "out_weight": "{prefix}.o_proj.weight",
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    # Alternate mapping for k_eq_v mode (no v_proj)
    _hf_mapping_k_eq_v_ = {
        "qkv_weight": fuse(
            "{prefix}.q_proj.weight",
            "{prefix}.k_proj.weight",
            dim=0,
        ),
        "out_weight": "{prefix}.o_proj.weight",
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        sliding_window: int | None = None,
        partial_rotary_factor: float = 1.0,
        k_eq_v: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        self.k_eq_v = k_eq_v
        self.eps = eps

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        if k_eq_v:
            # Q+K only (no V projection)
            self.QKV = (self.Hq + self.Hkv) * self.D
        else:
            # Q+K+V
            self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D
        self.RotaryDim = _resolve_rotary_dim(head_size, partial_rotary_factor)

        # Override HF mapping for k_eq_v
        if k_eq_v:
            self._hf_mapping_defaults_ = self._hf_mapping_k_eq_v_

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # Use numeric dimensions (not DimExpr) to avoid global shape env
        # collisions in hybrid models with different head_size/QKV per block type.
        if self.k_eq_v:
            _qkv_dim = (self.num_query_heads + self.num_kv_heads) * self.head_size
        else:
            _qkv_dim = (self.num_query_heads + 2 * self.num_kv_heads) * self.head_size
        _attn_dim = self.num_query_heads * self.head_size
        _full_qkv_dim = (self.num_query_heads + 2 * self.num_kv_heads) * self.head_size

        # LoRA targets for the fused weight.
        # - Standard mode (k_eq_v=False): Q + K + V packed with sizes
        #   Hq*D, Hkv*D, Hkv*D.
        # - k_eq_v mode: Q + K packed with sizes Hq*D, Hkv*D (V reuses K,
        #   so there is no LoRA target over the missing V).
        _hq = self.num_query_heads * self.head_size
        _hkv = self.num_kv_heads * self.head_size
        _hidden = self.d_model
        if self.k_eq_v:
            qkv_targets = [
                LoRATarget(name="q", offset=0, size=_hq),
                LoRATarget(name="k", offset=_hq, size=_hkv),
            ]
        else:
            qkv_targets = [
                LoRATarget(name="q", offset=0, size=_hq),
                LoRATarget(name="k", offset=_hq, size=_hkv),
                LoRATarget(name="v", offset=_hq + _hkv, size=_hkv),
            ]
        out_targets = [LoRATarget(name="o", offset=0, size=_hidden)]

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", (_qkv_dim, "C"), lora_targets=qkv_targets)
        out_w = tracer.register_param("out_weight", ("C", _attn_dim), lora_targets=out_targets)
        tracer.register_param("q_norm_weight", (self.head_size,), quantizable=False)
        tracer.register_param("k_norm_weight", (self.head_size,), quantizable=False)
        tracer.register_param(
            "rope_freqs",
            (self.max_seq, self.RotaryDim // 2, 2),
            dtype="fp32",
            frozen=True,
            quantizable=False,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv",
            ("B", "T", _qkv_dim),
            aliases=["qkv_flat"],
            save=True,
            share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope",
            ("B", "T", _full_qkv_dim),
            save=True,
            share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att",
            ("B", "T", _attn_dim),
            aliases=["att_flat"],
            save=True,
            share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse",
            ("B", self.num_query_heads, "T"),
            dtype="fp32",
            save=True,
            share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out",
            ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # Fused QKV projection
        qkv_flat = g.matmul(
            x_flat,
            qkv_w,
            transpose="NT",
            out_name=tracer.prefixed("qkv_flat"),
        )

        ones_d = g.ones(shape=[self.head_size], dtype="bf16")
        # Save ones for backward (used as V-norm weight; rmsnorm_backward needs it).
        g.save(ones_d)

        if self.k_eq_v:
            # --- k_eq_v mode: Q+K projection only, V = raw K ---
            qk = g.view(
                qkv_flat,
                shape=[B, T, self.num_query_heads + self.num_kv_heads, self.head_size],
                out_name=qkv_slot,
            )
            q_part, k_raw = g.split(
                qk,
                split_size=[self.num_query_heads, self.num_kv_heads],
                dim=2,
            )
            v_raw = k_raw

            # Q-norm: direct weight (Gemma4 stores the full scale, no +1 offset)
            q_flat_2d = g.view(
                q_part, shape=[B * T * self.num_query_heads, self.head_size], out_name=tracer.prefixed("q_flat")
            )
            q_normed_flat, _ = g.rmsnorm(
                q_flat_2d, tracer.prefixed("q_norm_weight"), eps=self.eps, y_name=tracer.prefixed("q_normed_flat")
            )
            q_normed = g.view(
                q_normed_flat, shape=[B, T, self.num_query_heads, self.head_size], out_name=tracer.prefixed("q_normed")
            )

            # K-norm: direct weight
            k_flat_2d = g.view(
                k_raw, shape=[B * T * self.num_kv_heads, self.head_size], out_name=tracer.prefixed("k_flat")
            )
            k_normed_flat, _ = g.rmsnorm(
                k_flat_2d, tracer.prefixed("k_norm_weight"), eps=self.eps, y_name=tracer.prefixed("k_normed_flat")
            )
            k_normed = g.view(
                k_normed_flat, shape=[B, T, self.num_kv_heads, self.head_size], out_name=tracer.prefixed("k_normed")
            )

            # V-norm: RMS-only (no learnable scale), on raw K
            v_flat_2d = g.view(
                v_raw, shape=[B * T * self.num_kv_heads, self.head_size], out_name=tracer.prefixed("v_flat_2d")
            )
            v_normed_flat, _ = g.rmsnorm(v_flat_2d, ones_d, eps=self.eps, y_name=tracer.prefixed("v_normed_2d"))
            v_normed = g.view(
                v_normed_flat, shape=[B, T, self.num_kv_heads, self.head_size], out_name=tracer.prefixed("v_normed")
            )

            qk_normed = g.concat(q_normed, k_normed, dim=2, split_size=[self.num_query_heads, self.num_kv_heads])
            qkv_normed = g.concat(
                qk_normed, v_normed, dim=2, split_size=[self.num_query_heads + self.num_kv_heads, self.num_kv_heads]
            )

        else:
            # --- Standard mode: Q+K+V projection ---
            qkv = g.view(
                qkv_flat,
                shape=[B, T, self.num_query_heads + 2 * self.num_kv_heads, self.head_size],
                out_name=qkv_slot,
            )

            # QK-norm: direct weight (Gemma4 stores the full scale, no +1 offset)
            qkv_qk_normed, _, _ = g.qkv_qk_norm(
                qkv,
                tracer.prefixed("q_norm_weight"),
                tracer.prefixed("k_norm_weight"),
                eps=self.eps,
            )

            # V-norm: split V from QKV in flattened space, normalize, rejoin
            qk_dim = (self.num_query_heads + self.num_kv_heads) * self.head_size
            v_dim = self.num_kv_heads * self.head_size
            qkv_normed_flat = g.view(
                qkv_qk_normed,
                shape=[B, T, qk_dim + v_dim],
                out_name=tracer.prefixed("qkv_normed_flat"),
            )
            qk_flat_3d, v_flat_3d = g.split(
                qkv_normed_flat,
                split_size=[qk_dim, v_dim],
                dim=2,
            )
            v_flat_2d = g.view(
                v_flat_3d,
                shape=[B * T * self.num_kv_heads, self.head_size],
                out_name=tracer.prefixed("v_flat_2d"),
            )
            v_normed_2d, _ = g.rmsnorm(
                v_flat_2d,
                ones_d,
                eps=self.eps,
                y_name=tracer.prefixed("v_normed_2d"),
            )
            v_normed = g.view(
                v_normed_2d,
                shape=[B, T, self.num_kv_heads * self.head_size],
                out_name=tracer.prefixed("v_normed"),
            )
            qk_dim = (self.num_query_heads + self.num_kv_heads) * self.head_size
            v_dim = self.num_kv_heads * self.head_size
            qkv_normed_3d = g.concat(qk_flat_3d, v_normed, dim=2, split_size=[qk_dim, v_dim])
            qkv_normed = g.view(
                qkv_normed_3d,
                shape=[B, T, self.num_query_heads + 2 * self.num_kv_heads, self.head_size],
                out_name=tracer.prefixed("qkv_normed_4d"),
            )

        # RoPE (only rotates Q and K heads, leaves V as-is)
        qkv_rope = g.rope(
            qkv_normed,
            tracer.prefixed("rope_freqs"),
            position_ids.ref,
            rotary_dim=self.RotaryDim,
            out_name=qkv_rope_slot,
        )

        # Flash attention with optional sliding window.
        # Gemma4 uses softmax_scale=1.0 (not 1/sqrt(head_dim)) because Q/K-norm
        # already produces unit-RMS Q and K, giving attention scores O(1)
        # without an explicit 1/sqrt(head_dim) divisor. Matches HF:
        # transformers/models/gemma4/modeling_gemma4.py
        #   self.scaling = 1.0; attn_weights = Q @ K^T * self.scaling
        fa_kwargs: dict[str, Any] = {"causal": True, "softmax_scale": 1.0}
        if self.sliding_window is not None:
            fa_kwargs["window_size"] = self.sliding_window

        attn_out, lse = g.flash_attention(
            qkv_rope,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
            **fa_kwargs,
        )

        # Output projection
        attn_flat = g.view(
            attn_out,
            shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        out_flat = g.matmul(
            attn_flat,
            out_w,
            transpose="NT",
            out_name=tracer.prefixed("att_out_flat"),
        )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Gemma4SharedKVAttention(Module):
    """Gemma4 Q-only attention for KV-shared layers.

    Shared layers have no k_proj/v_proj/k_norm/v_norm. They compute only Q
    and read pre-computed K,V states from an earlier (source) layer's
    ``qkv_rope`` output.

    The ``kv_source`` input is the source layer's packed QKV tensor after
    norms and RoPE: shape ``[B, T, Hq + 2*Hkv, D]``. K and V are extracted
    from positions ``[Hq : Hq+Hkv]`` and ``[Hq+Hkv : Hq+2*Hkv]``.
    """

    _hf_mapping_defaults_ = {
        "q_weight": "{prefix}.q_proj.weight",
        "out_weight": "{prefix}.o_proj.weight",
        "q_norm_weight": "{prefix}.q_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        partial_rotary_factor: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.partial_rotary_factor = partial_rotary_factor
        self.eps = eps

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QDim = self.Hq * self.D
        self.AttnDim = self.Hq * self.D
        self.RotaryDim = _resolve_rotary_dim(head_size, partial_rotary_factor)

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids, kv_source = args

        # Use numeric dims to avoid global shape env collisions in hybrid models
        _qdim = self.num_query_heads * self.head_size
        _attn_dim = _qdim

        _hq = self.num_query_heads * self.head_size
        _hidden = self.d_model

        # -- params (Q-only, no K/V weights) ---------------------------------
        q_w = tracer.register_param(
            "q_weight",
            (_qdim, "C"),
            lora_targets=[LoRATarget(name="q", size=_hq)],
        )
        out_w = tracer.register_param(
            "out_weight",
            ("C", _attn_dim),
            lora_targets=[LoRATarget(name="o", size=_hidden)],
        )
        tracer.register_param("q_norm_weight", (self.head_size,), quantizable=False)
        tracer.register_param(
            "rope_freqs",
            (self.max_seq, self.RotaryDim // 2, 2),
            dtype="fp32",
            frozen=True,
            quantizable=False,
        )

        # -- activation slots ------------------------------------------------
        att_slot = tracer.register_activation(
            "att",
            ("B", "T", _attn_dim),
            aliases=["att_flat"],
            save=True,
            share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse",
            ("B", self.num_query_heads, "T"),
            dtype="fp32",
            save=True,
            share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out",
            ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))

        # Q projection
        q_flat = g.matmul(x_flat, q_w, transpose="NT", out_name=tracer.prefixed("q_flat"))
        q = g.view(q_flat, shape=[B, T, self.num_query_heads, self.head_size], out_name=tracer.prefixed("q_4d"))

        # Q-norm: direct weight scale (Gemma4 stores the full scale, no +1 offset)
        q_rn_flat = g.view(
            q, shape=[B * T * self.num_query_heads, self.head_size], out_name=tracer.prefixed("q_rn_flat")
        )
        q_normed_flat, _ = g.rmsnorm(
            q_rn_flat, tracer.prefixed("q_norm_weight"), eps=self.eps, y_name=tracer.prefixed("q_normed_flat")
        )
        q_normed = g.view(
            q_normed_flat, shape=[B, T, self.num_query_heads, self.head_size], out_name=tracer.prefixed("q_normed")
        )

        # Extract K,V from kv_source. Use dimension sizes (not head counts)
        # because the source tensor may be 3D (B,T,QKV) during backward replay.
        q_dim = self.num_query_heads * self.head_size
        kv_dim = 2 * self.num_kv_heads * self.head_size
        kv_source_flat = g.view(
            kv_source.ref,
            shape=[B, T, q_dim + kv_dim],
            out_name=tracer.prefixed("kv_source_flat"),
        )
        _, kv_flat = g.split(
            kv_source_flat,
            split_size=[q_dim, kv_dim],
            dim=2,
        )
        kv_part = g.view(
            kv_flat,
            shape=[B, T, 2 * self.num_kv_heads, self.head_size],
            out_name=tracer.prefixed("kv_4d"),
        )

        # Apply RoPE to Q only (K,V already have RoPE from source layer).
        # The rope kernel applies to all heads of the input tensor, so we
        # pass Q alone [B, T, Hq, D] — no K/V contamination.
        q_roped = g.rope(
            q_normed,
            tracer.prefixed("rope_freqs"),
            position_ids.ref,
            rotary_dim=self.RotaryDim,
            out_name=tracer.prefixed("q_roped"),
        )

        # Reassemble with source K,V for packed flash attention
        qkv_final = g.concat(q_roped, kv_part, dim=2, split_size=[self.num_query_heads, 2 * self.num_kv_heads])

        # Gemma4 uses softmax_scale=1.0 (see Gemma4Attention above for why).
        attn_out, lse = g.flash_attention(
            qkv_final,
            causal=True,
            softmax_scale=1.0,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Output projection
        attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim], out_name=tracer.prefixed("att_flat"))
        out_flat = g.matmul(attn_flat, out_w, transpose="NT", out_name=tracer.prefixed("att_out_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=att_out_slot)

        return Proxy(att_out_slot, out)


class NemotronAttention(Module):
    """GQA attention with optional RoPE (Nemotron-H style)."""

    _hf_mapping_defaults_ = {
        "qkv_weight": fuse(
            "{prefix}.q_proj.weight",
            "{prefix}.k_proj.weight",
            "{prefix}.v_proj.weight",
            dim=0,
        ),
        "qkv_bias": fuse(
            "{prefix}.q_proj.bias",
            "{prefix}.k_proj.bias",
            "{prefix}.v_proj.bias",
            dim=0,
        ),
        "out_weight": "{prefix}.o_proj.weight",
        "out_bias": "{prefix}.o_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        attention_bias: bool = False,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.attention_bias = attention_bias
        self.use_rope = use_rope

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        _hq = self.num_query_heads * self.head_size
        _hkv = self.num_kv_heads * self.head_size
        _hidden = self.d_model
        qkv_targets = [
            LoRATarget(name="q", offset=0, size=_hq),
            LoRATarget(name="k", offset=_hq, size=_hkv),
            LoRATarget(name="v", offset=_hq + _hkv, size=_hkv),
        ]
        out_targets = [LoRATarget(name="o", offset=0, size=_hidden)]

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"), lora_targets=qkv_targets)
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="attention_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"), lora_targets=out_targets)
        out_b = tracer.register_param("out_bias", ("C",), when="attention_bias")
        if self.use_rope:
            tracer.register_param(
                "rope_freqs",
                ("MaxSeq", "D // 2", 2),
                dtype="fp32",
                frozen=True,
            )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv",
            ("B", "T", "QKV"),
            aliases=["qkv_flat"],
            save=True,
            share_policy="per_layer",
        )
        if self.use_rope:
            tracer.register_activation(
                "qkv_rope",
                ("B", "T", "QKV"),
                save=True,
                share_policy="when_recomputed",
            )
        att_slot = tracer.register_activation(
            "att",
            ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True,
            share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse",
            ("B", "Hq", "T"),
            dtype="fp32",
            save=True,
            share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out",
            ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="fft_share",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref,
            shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.attention_bias:
            qkv_flat = g.matmul_bias(
                x_flat,
                qkv_w,
                qkv_b,
                transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat,
                qkv_w,
                transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat,
            shape=[B, T, self.QKV],
            out_name=qkv_slot,
        )

        # RoPE (optional - Nemotron-H attention does not use positional encoding)
        if self.use_rope:
            attn_input = g.rope(
                qkv,
                tracer.prefixed("rope_freqs"),
                position_ids.ref,
                rotary_dim=self.head_size,
                out_name=tracer.prefixed("qkv_rope"),
            )
        else:
            attn_input = qkv

        # FlashAttention
        attn_out, lse = g.flash_attention(
            attn_input,
            causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Output projection
        attn_flat = g.view(
            attn_out,
            shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.attention_bias:
            out_flat = g.matmul_bias(
                attn_flat,
                out_w,
                out_b,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat,
                out_w,
                transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat,
            shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


# Class-level default mapping so ``hf.py`` can read
# ``GenericGQAttention._hf_mapping_defaults_`` without instantiating.
# Instances may override via ``self._hf_mapping_defaults_`` in __init__
# when the config enables qk_norm / sinks.
GenericGQAttention._hf_mapping_defaults_ = _base_qkv_hf_mapping()

# ----------------------------------------------------------------------------
# Backwards-compat aliases
# ----------------------------------------------------------------------------
# These preserve the names used by ``surogate.dsl.hf`` and the per-model HF
# weight-mapping helpers. They are now thin subclasses / aliases of
# ``GenericGQAttention`` with the appropriate preset ``AttentionConfig`` and
# a class-level ``_hf_mapping_defaults_`` matching each variant.


class GQAAttention(GenericGQAttention):
    """Base grouped-query attention (Llama / Qwen2 layout)."""

    _hf_mapping_defaults_ = _base_qkv_hf_mapping()


class Qwen3Attention(GenericGQAttention):
    """GQA + QK-norm (Qwen3 layout)."""

    _hf_mapping_defaults_ = {
        **_base_qkv_hf_mapping(),
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        *,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(
                qk_norm=use_qk_norm,
                qkv_bias=use_qkv_bias,
                eps=eps,
            ),
        )


class GptOssAttention(GenericGQAttention):
    """GPT-OSS: GQA with sink tokens and QKV biases."""

    _hf_mapping_defaults_ = {
        **_base_qkv_hf_mapping(),
        "sinks": "{prefix}.sinks",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        *,
        use_qkv_bias: bool = True,
    ) -> None:
        super().__init__(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(
                qkv_bias=use_qkv_bias,
                has_sinks=True,
            ),
        )


# ============================================================================
# Registry population
# ============================================================================
#
# Populate the ``Attention`` namespace with concrete ``AttentionSpec``s.
# This runs at module-load time because the factory classes (defined
# above) are now available. The specs are also registered by name in
# ``_BY_NAME`` so ``attention_from_name(...)`` works.

from ..attention import (  # noqa: E402
    Attention,
    AttentionSpec,
    _register as _register_attention,
)

Attention.GQA = _register_attention(AttentionSpec(name="gqa", factory=GenericGQAttention, config=AttentionConfig()))
Attention.QWEN3 = _register_attention(
    AttentionSpec(
        name="qwen3",
        factory=GenericGQAttention,
        config=AttentionConfig(qk_norm=True),
    )
)
Attention.GPT_OSS = _register_attention(
    AttentionSpec(
        name="gpt_oss",
        factory=GenericGQAttention,
        config=AttentionConfig(qkv_bias=True, has_sinks=True),
    )
)

# Specialized kinds that keep their dedicated class.
for _name, _cls in (
    ("qwen3_vl", Qwen3VLAttention),
    ("qwen3_5", Qwen3_5Attention),
    ("gemma4", Gemma4Attention),
    ("gemma4_shared_kv", Gemma4SharedKVAttention),
    ("nemotron", NemotronAttention),
):
    _register_attention(AttentionSpec(name=_name, factory=_cls))

del _name, _cls
