"""
Torch.nn-like module system for Surogate DSL.

Provides a familiar torch.nn-style API for defining models that compiles
to the same IR as the @block/@model decorated classes.

Example:
    from surogate.dsl import nn

    class Qwen3Block(nn.Block):
        def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                     d_ff, max_seq, eps=1e-6, use_qkv_bias=False, use_qk_norm=True):
            super().__init__()
            self.attn_norm = nn.RMSNorm(d_model, eps=eps)
            self.self_attn = nn.Qwen3Attention(
                d_model, num_query_heads, num_kv_heads, head_size,
                max_seq, use_qkv_bias=use_qkv_bias, use_qk_norm=use_qk_norm, eps=eps,
            )
            self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
            self.mlp = nn.SwiGLUMLP(d_model, d_ff)

        def forward(self, x, residual, position_ids):
            residual, h = self.attn_norm(residual, x)
            h = self.self_attn(h, position_ids)
            residual, h = self.mlp_norm(residual, h)
            h = self.mlp(h)
            return h, residual
"""

from __future__ import annotations

import contextvars
import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .tensor_type import Tensor, Array
from .graph_builder import GraphBuilder, GraphRef, GraphNode
from .dim import Dim, DimExpr, B, T
from .specs import (
    BlockSpec,
    ModelSpec,
    ModuleSpec,
    ParamSpec,
    ParamKind,
    ForwardSpec,
    IOSpec,
    ActivationSlotSpec,
    ActivationLayoutSpec,
    ActivationScope,
    SharePolicy,
    HFConfigSpec,
)
from .tensor_type import TensorAnnotation
from .hf import FuseMapping, fuse, stack_experts, transform
from .decorators import (
    _block_registry,
    _model_registry,
    _extract_constructor_params,
)

# ============================================================================
# Tracing context
# ============================================================================

_current_tracer: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar(
    "_current_tracer", default=None
)


# ============================================================================
# Proxy — represents a tensor during forward tracing
# ============================================================================

@dataclass
class Proxy:
    """A tensor reference during forward tracing.

    Wraps a GraphRef so it can be passed to graph builder methods,
    while carrying a human-readable name for activation slot registration.
    """
    name: str
    ref: GraphRef

    def __repr__(self) -> str:
        return f"Proxy({self.name!r})"


# ============================================================================
# Tracer — accumulates graph, params, activation slots
# ============================================================================

class Tracer:
    """Records graph construction during Block/Model forward tracing."""

    def __init__(self) -> None:
        self.graph = GraphBuilder()
        self.params: OrderedDict[str, ParamSpec] = OrderedDict()
        self.forward_slots: list[ActivationSlotSpec] = []
        self.gradient_slots: list[ActivationSlotSpec] = []
        self.hf_mappings: dict[str, Any] = {}
        self._prefix_stack: list[str] = []
        self._slot_names: set[str] = set()
        self._name_remap: dict[str, str] = {}

    # -- prefix management ---------------------------------------------------

    @property
    def prefix(self) -> str:
        return "_".join(self._prefix_stack) if self._prefix_stack else ""

    def prefixed(self, name: str) -> str:
        p = self.prefix
        raw = f"{p}_{name}" if p else name
        return self._name_remap.get(raw, raw)

    def push_prefix(self, name: str) -> None:
        self._prefix_stack.append(name)

    def pop_prefix(self) -> None:
        self._prefix_stack.pop()

    # -- param registration --------------------------------------------------

    def register_param(
        self,
        local_name: str,
        shape: tuple[str | int, ...],
        *,
        dtype: str = "bf16",
        frozen: bool = False,
        quantizable: bool = True,
        when: str | None = None,
        offload_group: int | str = -1,
    ) -> str:
        """Register a parameter and return its fully-qualified name."""
        full_name = self.prefixed(local_name)
        if full_name not in self.params:
            spec = ParamSpec(
                name=full_name,
                kind=ParamKind.TENSOR,
                shape=shape,
                dtype=dtype,
                frozen=frozen,
                quantizable=quantizable,
                offload_group=offload_group,
            )
            if when is not None:
                spec.optional = True
                spec.condition = lambda self_, _w=when: getattr(self_, _w, False)
            self.params[full_name] = spec
        return full_name

    # -- activation slot registration ----------------------------------------

    def register_activation(
        self,
        local_name: str,
        shape: tuple[str | int, ...],
        *,
        dtype: str | None = None,
        save: bool = False,
        share_policy: str | SharePolicy = SharePolicy.PER_LAYER,
        when: str | None = None,
        aliases: list[str] | None = None,
        description: str | None = None,
        scope: ActivationScope = ActivationScope.BLOCK,
    ) -> str:
        """Register an activation slot and return its fully-qualified name."""
        full_name = self.prefixed(local_name)
        if full_name in self._slot_names:
            return full_name

        if isinstance(share_policy, str):
            share_policy = SharePolicy(share_policy)

        self.forward_slots.append(ActivationSlotSpec(
            name=full_name,
            scope=scope,
            shape=shape,
            dtype=dtype,
            aliases=[self.prefixed(a) for a in (aliases or [])],
            save_for_backward=save,
            share_policy=share_policy,
            condition_expr=when,
            description=description,
        ))
        self._slot_names.add(full_name)
        return full_name

    # -- gradient slot registration ------------------------------------------

    def auto_gradients(self) -> None:
        """Auto-generate gradient slots for forward activations.

        Block-scope slots always get gradients. Global-scope slots get
        gradients only if they are computed (i.e., appear as a graph op
        output), not if they are pure inputs (token_ids, targets, etc.)
        or precomputed constants (freq_cis).
        """
        # Collect names that appear as graph operation outputs.
        # Also include global slots that are NOT graph inputs — these are
        # intermediate computation results (e.g., xN, residualN declared
        # as global slots but produced by StackedBlocks under different names).
        computed = set()
        for node in self.graph.nodes:
            for out in node.outputs:
                computed.add(out)
        graph_inputs = set(self.graph._inputs)
        for slot in self.forward_slots:
            if slot.scope == ActivationScope.GLOBAL and slot.name not in graph_inputs:
                # Skip IO/precomputed slots that have specific dtypes (int32, fp32 constants)
                if slot.dtype in ("int32",):
                    continue
                # Skip freq_cis (precomputed constant)
                if "freq" in slot.name:
                    continue
                computed.add(slot.name)

        existing = {s.name for s in self.gradient_slots}

        for slot in self.forward_slots:
            if slot.scope == ActivationScope.BLOCK:
                grad_scope = ActivationScope.GRADIENT
            elif slot.scope == ActivationScope.GLOBAL and slot.name in computed:
                grad_scope = ActivationScope.GLOBAL_GRADIENT
            else:
                continue

            grad_name = f"d_{slot.name}"
            if grad_name in existing:
                continue
            self.gradient_slots.append(ActivationSlotSpec(
                name=grad_name,
                scope=grad_scope,
                shape=slot.shape,
                dtype=slot.dtype if slot.name == "loss" else None,
                gradient_of=slot.name,
            ))

    # -- HF mapping registration --------------------------------------------

    def register_hf_mapping(self, param_name: str, mapping: Any) -> None:
        """Register a HuggingFace weight mapping for a parameter."""
        self.hf_mappings[param_name] = mapping

    # -- proxy helpers -------------------------------------------------------

    def make_proxy(self, local_name: str, ref: GraphRef) -> Proxy:
        return Proxy(self.prefixed(local_name), ref)


# ============================================================================
# Module base class
# ============================================================================

class Module:
    """Base class for nn-style modules.

    Tracks child modules via attribute assignment. During block compilation,
    ``__call__`` delegates to ``_trace()`` which emits graph nodes.
    """

    # Override in subclasses for HuggingFace weight path templates.
    _hf_mapping_defaults_: dict[str, Any] = {}

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_name", "")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module) and name != "_modules":
            if not hasattr(self, "_modules"):
                object.__setattr__(self, "_modules", OrderedDict())
            self._modules[name] = value
            value._name = name
        object.__setattr__(self, name, value)

    # -- tracing entry point -------------------------------------------------

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy | tuple[Proxy, ...]:
        """Emit graph nodes for this module. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._trace()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tracer = _current_tracer.get()
        if tracer is None:
            raise RuntimeError(
                f"{type(self).__name__} can only be called during block/model "
                "compilation (inside compile())"
            )
        tracer.push_prefix(self._name)
        try:
            # Register HF mappings for this module
            for local_param, mapping in self._hf_mapping_defaults_.items():
                tracer.register_hf_mapping(tracer.prefixed(local_param), mapping)
            result = self._trace(tracer, *args, **kwargs)
        finally:
            tracer.pop_prefix()
        return result

    # -- inline graph op helpers (usable in forward) -------------------------

    @staticmethod
    def _view(x: Proxy, shape: list, *, name: str | None = None) -> Proxy:
        """Reshape a proxy tensor."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("view")
        ref = g.view(x.ref, shape=shape, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _add(a: Proxy, b: Proxy, *, name: str | None = None) -> Proxy:
        """Element-wise add."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("add")
        ref = g.add(a.ref, b.ref, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _mul(a: Proxy, b: Proxy, *, name: str | None = None) -> Proxy:
        """Element-wise multiply."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("mul")
        ref = g.mul(a.ref, b.ref)
        return Proxy(out_name, ref)

    @staticmethod
    def _sigmoid(x: Proxy, *, name: str | None = None) -> Proxy:
        """Sigmoid activation."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("sigmoid")
        ref = g.sigmoid(x.ref, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _matmul(x: Proxy, weight_name: str, *, transpose: str = "NT",
                name: str | None = None) -> Proxy:
        """Matmul with a named weight parameter."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("matmul")
        ref = g.matmul(x.ref, weight_name, transpose=transpose, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _register_param(name: str, shape: tuple, **kwargs) -> str:
        """Register a weight parameter from within forward()."""
        tracer = _current_tracer.get()
        return tracer.register_param(name, shape, **kwargs)

    @staticmethod
    def _zeros(shape: list, *, dtype: str = "bf16", name: str | None = None) -> Proxy:
        """Create a zero-filled tensor."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("zeros")
        ref = g.zeros(shape=shape, dtype=dtype, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _ones(shape: list, *, dtype: str = "bf16", name: str | None = None) -> Proxy:
        """Create a ones-filled tensor."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("ones")
        ref = g.ones(shape=shape, dtype=dtype, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _mask_scatter(
        x: Proxy, mask: Proxy, values: Proxy, *, name: str | None = None,
    ) -> Proxy:
        """Scatter values into x at positions indicated by mask."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("mask_scatter")
        ref = g.mask_scatter(x.ref, mask.ref, values.ref, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _register_activation(
        local_name: str,
        shape: tuple[str | int, ...],
        **kwargs: Any,
    ) -> str:
        """Register an activation slot."""
        tracer = _current_tracer.get()
        return tracer.register_activation(local_name, shape, **kwargs)

    @staticmethod
    def _register_gradient(
        name: str,
        shape: tuple[str | int, ...],
        gradient_of: str,
        *,
        dtype: str | None = None,
        scope: ActivationScope = ActivationScope.GRADIENT,
    ) -> None:
        """Register a gradient slot."""
        tracer = _current_tracer.get()
        # Map scope for global gradients
        if scope == ActivationScope.GLOBAL:
            scope = ActivationScope.GLOBAL_GRADIENT
        tracer.gradient_slots.append(ActivationSlotSpec(
            name=name,
            scope=scope,
            shape=shape,
            dtype=dtype,
            gradient_of=gradient_of,
        ))


# ============================================================================
# Concrete modules
# ============================================================================


class RMSNorm(Module):
    """RMS Layer Normalization.

    When called with one argument ``(x,)`` performs standard rmsnorm.
    When called with two arguments ``(residual, x)`` performs fused
    residual-add + rmsnorm.
    """

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.C = Dim("C")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy | tuple[Proxy, ...]:
        g = tracer.graph
        weight = tracer.register_param("weight", ("C",), quantizable=False)

        if len(args) == 2:
            # Fused residual + rmsnorm
            residual, x = args

            res_slot = tracer.register_activation(
                "res", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            y_slot = tracer.register_activation(
                "y", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd", ("B", "T"),
                dtype="fp32", save=True,
                share_policy="per_layer",
            )

            res_ref, y_ref, rstd_ref = g.fused_residual_rmsnorm(
                residual.ref, x.ref, weight,
                eps=self.eps,
                res_out_name=res_slot,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return (
                Proxy(res_slot, res_ref),
                Proxy(y_slot, y_ref),
            )

        elif len(args) == 1:
            x = args[0]

            y_slot = tracer.register_activation(
                "y", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd", ("B", "T"),
                dtype="fp32", save=True,
                share_policy="per_layer",
            )

            y_ref, rstd_ref = g.rmsnorm(
                x.ref, weight, eps=self.eps,
                y_name=y_slot, rstd_name=rstd_slot,
            )
            return Proxy(y_slot, y_ref)

        raise ValueError(f"RMSNorm expects 1 or 2 inputs, got {len(args)}")


class GQAAttention(Module):
    """Grouped-Query Attention with RoPE and FlashAttention."""

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
        use_qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        qkv_rope = g.rope(
            qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
            rotary_dim="D",
            out_name=qkv_rope_slot,
        )

        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class GptOssAttention(Module):
    """GPT-OSS attention with RoPE and sink tokens."""

    _hf_mapping_defaults_ = {
        **GQAAttention._hf_mapping_defaults_,
        "sinks": "{prefix}.sinks",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params --------------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )
        tracer.register_param("sinks", ("Hq",))

        # -- activation slots ----------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph ---------------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        qkv_rope = g.rope(
            qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
            rotary_dim="D",
            out_name=qkv_rope_slot,
        )

        # FlashAttention with sink tokens
        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            sinks=tracer.prefixed("sinks"),
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Qwen3VLAttention(Module):
    """Qwen3-VL attention with QK-Norm + MRoPE (separate, not fused)."""

    _hf_mapping_defaults_ = {
        **GQAAttention._hf_mapping_defaults_,
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

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params --------------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param("q_norm_weight", ("D",), quantizable=False)
        tracer.register_param("k_norm_weight", ("D",), quantizable=False)
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )

        # -- activation slots ----------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_norm_slot = tracer.register_activation(
            "qkv_norm", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
            description="QKV after QK-Norm",
        )
        tracer.register_activation(
            "q_rstd", ("B", "T", "Hq"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "k_rstd", ("B", "T", "Hkv"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
            description="QKV after QK-Norm + MRoPE",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph ---------------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
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
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Output projection
        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Qwen3Attention(Module):
    """Qwen3-style attention with QK-Norm + RoPE."""

    _hf_mapping_defaults_ = {
        **GQAAttention._hf_mapping_defaults_,
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
        use_qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.eps = eps

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params (always declared, condition controls allocation) ----------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param(
            "q_norm_weight", ("D",), quantizable=False, when="use_qk_norm",
        )
        tracer.register_param(
            "k_norm_weight", ("D",), quantizable=False, when="use_qk_norm",
        )
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
            description="QKV after QK-Norm + RoPE",
        )
        tracer.register_activation(
            "q_rstd", ("B", "T", "Hq"),
            dtype="fp32", save=True, share_policy="when_recomputed",
            when="use_qk_norm",
        )
        tracer.register_activation(
            "k_rstd", ("B", "T", "Hkv"),
            dtype="fp32", save=True, share_policy="when_recomputed",
            when="use_qk_norm",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        if self.use_qk_norm:
            qkv_rope, q_rstd, k_rstd = g.qkv_qk_norm_rope(
                qkv,
                tracer.prefixed("q_norm_weight"),
                tracer.prefixed("k_norm_weight"),
                tracer.prefixed("rope_freqs"),
                position_ids.ref,
                eps=self.eps,
                out_name=qkv_rope_slot,
                q_rstd_name=tracer.prefixed("q_rstd"),
                k_rstd_name=tracer.prefixed("k_rstd"),
            )
        else:
            qkv_rope = g.rope(
                qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
                out_name=qkv_rope_slot,
            )

        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


def _resolve_rotary_dim(head_size: int, partial_rotary_factor: float) -> int:
    """Compute the rotary embedding dimension from head_size and partial_rotary_factor."""
    rotary = int(round(float(head_size) * float(partial_rotary_factor)))
    rotary = max(2, min(rotary, head_size))
    if rotary % 2 != 0:
        rotary -= 1
    return max(2, rotary)


class RMSNormPlus1(Module):
    """RMS Layer Normalization with weight + 1 bias (Qwen3.5 style).

    Like RMSNorm, but adds 1.0 to the weight before applying normalization.
    When called with one argument ``(x,)`` performs standard rmsnorm(x, weight+1).
    When called with two arguments ``(residual, x)`` performs fused
    residual-add + rmsnorm(x, weight+1).
    """

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.C = Dim("C")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy | tuple[Proxy, ...]:
        g = tracer.graph
        weight = tracer.register_param("weight", ("C",), quantizable=False)

        # Create ones and add to weight for the +1 bias
        ones_ref = g.ones(shape=[self.C], dtype="bf16")
        weight_eff = g.add(weight, ones_ref, out_name=tracer.prefixed("weight_eff"))

        if len(args) == 2:
            # Fused residual + rmsnorm
            residual, x = args

            res_slot = tracer.register_activation(
                "res", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            y_slot = tracer.register_activation(
                "y", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd", ("B", "T"),
                dtype="fp32", save=True,
                share_policy="per_layer",
            )

            res_ref, y_ref, rstd_ref = g.fused_residual_rmsnorm(
                residual.ref, x.ref, weight_eff,
                eps=self.eps,
                res_out_name=res_slot,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return (
                Proxy(res_slot, res_ref),
                Proxy(y_slot, y_ref),
            )

        elif len(args) == 1:
            x = args[0]

            y_slot = tracer.register_activation(
                "y", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd", ("B", "T"),
                dtype="fp32", save=True,
                share_policy="per_layer",
            )

            y_ref, rstd_ref = g.rmsnorm(
                x.ref, weight_eff, eps=self.eps,
                y_name=y_slot, rstd_name=rstd_slot,
            )
            return Proxy(y_slot, y_ref)

        raise ValueError(f"RMSNormPlus1 expects 1 or 2 inputs, got {len(args)}")


class Qwen3_5Attention(Module):
    """Qwen3.5 full-attention with separate Q/K/V projections, QK-Norm, partial MRoPE, and gated output.

    Unlike Qwen3Attention which uses fused QKV, Qwen3.5 uses:
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

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params ----------------------------------------------------------
        q_proj_w = tracer.register_param("q_proj_weight", ("QProjDim", "C"))
        q_proj_b = tracer.register_param("q_proj_bias", ("QProjDim",), when="use_qkv_bias")
        k_proj_w = tracer.register_param("k_proj_weight", ("KVDim", "C"))
        k_proj_b = tracer.register_param("k_proj_bias", ("KVDim",), when="use_qkv_bias")
        v_proj_w = tracer.register_param("v_proj_weight", ("KVDim", "C"))
        v_proj_b = tracer.register_param("v_proj_bias", ("KVDim",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param("q_norm_weight", ("D",), quantizable=False)
        tracer.register_param("k_norm_weight", ("D",), quantizable=False)
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "RotaryDim // 2", 2), dtype="fp32",
            frozen=True, quantizable=False,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # Separate Q/K/V projections
        if self.use_qkv_bias:
            q_proj = g.matmul_bias(
                x_flat, q_proj_w, q_proj_b, transpose="NT",
                out_name=tracer.prefixed("q_proj"),
            )
            k_proj = g.matmul_bias(
                x_flat, k_proj_w, k_proj_b, transpose="NT",
                out_name=tracer.prefixed("k_proj"),
            )
            v_proj = g.matmul_bias(
                x_flat, v_proj_w, v_proj_b, transpose="NT",
                out_name=tracer.prefixed("v_proj"),
            )
        else:
            q_proj = g.matmul(
                x_flat, q_proj_w, transpose="NT",
                out_name=tracer.prefixed("q_proj"),
            )
            k_proj = g.matmul(
                x_flat, k_proj_w, transpose="NT",
                out_name=tracer.prefixed("k_proj"),
            )
            v_proj = g.matmul(
                x_flat, v_proj_w, transpose="NT",
                out_name=tracer.prefixed("v_proj"),
            )

        # Split Q into Q + gate, reshape all to 4D
        q_proj_4d = g.view(
            q_proj, shape=[B, T, self.Hq, 2 * self.D],
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
            tracer.prefixed("q_norm_weight"), ones_d,
            out_name=tracer.prefixed("q_norm_weight_eff"),
        )
        k_norm_eff = g.add(
            tracer.prefixed("k_norm_weight"), ones_d,
            out_name=tracer.prefixed("k_norm_weight_eff"),
        )
        qkv_norm, _, _ = g.qkv_qk_norm(
            qkv, q_norm_eff, k_norm_eff, eps=self.eps,
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
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Sigmoid-gated output
        attn_4d = g.view(attn_out, shape=[B, T, self.Hq, self.D], out_name=tracer.prefixed("att_4d"))
        gate_sigmoid = g.sigmoid(gate_4d)
        gated_attn_4d = g.mul(attn_4d, gate_sigmoid)
        gated_attn_flat = g.view(
            gated_attn_4d, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )

        # Output projection
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                gated_attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                gated_attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class GatedDeltaNetMixer(Module):
    """Gated DeltaNet linear attention mixer (Qwen3.5 style).

    Implements the linear attention path:
    - in_proj_qkv: project to QKV space (then conv1d + silu)
    - in_proj_z: gate projection
    - in_proj_b: beta projection (sigmoid)
    - in_proj_a: decay projection
    - A_log, dt_bias: decay parameters (FP32)
    - chunk_gated_delta_rule: fused chunked attention
    - gated_rmsnorm: gated normalization with Z gate
    - out_proj: output projection
    """

    _hf_mapping_defaults_ = {
        "in_proj_qkv_weight": "{prefix}.in_proj_qkv.weight",
        "in_proj_z_weight": "{prefix}.in_proj_z.weight",
        "in_proj_b_weight": "{prefix}.in_proj_b.weight",
        "in_proj_a_weight": "{prefix}.in_proj_a.weight",
        "conv_weight": "{prefix}.conv1d.weight",
        "A_log": "{prefix}.A_log",
        "dt_bias": "{prefix}.dt_bias",
        "norm_weight": "{prefix}.norm.weight",
        "out_weight": "{prefix}.out_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.chunk_size = chunk_size
        self.eps = eps

        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(
                "GatedDeltaNetMixer requires linear_num_value_heads to be "
                "divisible by linear_num_key_heads"
            )

        self.C = Dim("C")
        # Use concrete integers for linear-attention dims to avoid conflicts
        self.Hk = linear_num_key_heads
        self.Hv = linear_num_value_heads
        self.Kd = linear_key_head_dim
        self.Vd = linear_value_head_dim
        self.KeyDim = self.Hk * self.Kd
        self.ValueDim = self.Hv * self.Vd
        self.ConvK = linear_conv_kernel_dim
        self.ConvDim = self.KeyDim * 2 + self.ValueDim
        self.HeadRepeat = self.Hv // self.Hk

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        # -- params ----------------------------------------------------------
        tracer.register_param("in_proj_qkv_weight", (self.ConvDim, "C"))
        tracer.register_param("in_proj_z_weight", (self.ValueDim, "C"))
        tracer.register_param("in_proj_b_weight", (self.Hv, "C"))
        tracer.register_param("in_proj_a_weight", (self.Hv, "C"))
        tracer.register_param(
            "conv_weight", (self.ConvDim, 1, self.ConvK), quantizable=False,
        )
        tracer.register_param("A_log", (self.Hv,), dtype="fp32", quantizable=False)
        tracer.register_param("dt_bias", (self.Hv,), dtype="fp32", quantizable=False)
        tracer.register_param("norm_weight", (self.Vd,), quantizable=False)
        out_proj_w = tracer.register_param("out_weight", ("C", self.ValueDim))

        # -- activation slots ------------------------------------------------
        out_slot = tracer.register_activation(
            "out", ("B", "T", "C"),
            share_policy="per_layer",
            description="GatedDeltaNet mixer output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # QKV projection -> conv1d
        mixed_qkv_flat = g.matmul(
            x_flat, tracer.prefixed("in_proj_qkv_weight"), transpose="NT",
            out_name=tracer.prefixed("mixed_qkv_flat"),
        )
        mixed_qkv = g.view(
            mixed_qkv_flat, shape=[B, T, self.ConvDim],
            out_name=tracer.prefixed("mixed_qkv"),
        )
        mixed_qkv_cf = g.transpose(mixed_qkv, dim0=1, dim1=2)

        conv_weight_2d = g.view(
            tracer.prefixed("conv_weight"),
            shape=[self.ConvDim, self.ConvK],
            out_name=tracer.prefixed("conv_w2d"),
        )
        conv_out_cf = g.mamba_conv1d(
            mixed_qkv_cf, conv_weight_2d, None,
            activation="silu",
            out_name=tracer.prefixed("conv_out_cf"),
        )
        conv_out = g.transpose(conv_out_cf, dim0=1, dim1=2)

        # Split Q/K/V
        q_flat, k_flat, v_flat = g.split(
            conv_out,
            split_size=[self.KeyDim, self.KeyDim, self.ValueDim],
            dim=2,
        )
        query = g.view(
            q_flat, shape=[B, T, self.Hk, self.Kd],
            out_name=tracer.prefixed("query"),
        )
        key = g.view(
            k_flat, shape=[B, T, self.Hk, self.Kd],
            out_name=tracer.prefixed("key"),
        )
        value = g.view(
            v_flat, shape=[B, T, self.Hv, self.Vd],
            out_name=tracer.prefixed("value"),
        )

        # Z gate
        z_flat = g.matmul(
            x_flat, tracer.prefixed("in_proj_z_weight"), transpose="NT",
            out_name=tracer.prefixed("z_flat"),
        )
        z = g.view(z_flat, shape=[B, T, self.Hv, self.Vd], out_name=tracer.prefixed("z"))

        # Beta (sigmoid)
        b_flat = g.matmul(
            x_flat, tracer.prefixed("in_proj_b_weight"), transpose="NT",
            out_name=tracer.prefixed("b_flat"),
        )
        b = g.view(b_flat, shape=[B, T, self.Hv], out_name=tracer.prefixed("b"))
        beta = g.sigmoid(b)

        # Decay
        a_flat = g.matmul(
            x_flat, tracer.prefixed("in_proj_a_weight"), transpose="NT",
            out_name=tracer.prefixed("a_flat"),
        )
        a = g.view(a_flat, shape=[B, T, self.Hv], out_name=tracer.prefixed("a"))
        g_decay = g.qwen3_5_decay(
            a, tracer.prefixed("A_log"), tracer.prefixed("dt_bias"),
            out_name=tracer.prefixed("decay"),
        )

        # Repeat heads if needed
        if self.HeadRepeat > 1:
            query = g.repeat_interleave_heads(
                query, repeats=self.HeadRepeat,
                out_name=tracer.prefixed("query_rep"),
            )
            key = g.repeat_interleave_heads(
                key, repeats=self.HeadRepeat,
                out_name=tracer.prefixed("key_rep"),
            )

        # Chunked gated delta rule
        core_attn_out, _ = g.custom(
            "chunk_gated_delta_rule",
            query, key, value, g_decay, beta,
            num_outputs=2,
            scale=0.0,
            chunk_size=self.chunk_size,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Gated RMSNorm
        core_flat = g.view(
            core_attn_out,
            shape=[B * T * self.Hv, self.Vd],
            out_name=tracer.prefixed("core_flat"),
        )
        z_norm_flat = g.view(
            z, shape=[B * T * self.Hv, self.Vd],
            out_name=tracer.prefixed("z_norm_flat"),
        )
        gated_flat = g.mamba_gated_rmsnorm(
            core_flat, z_norm_flat,
            tracer.prefixed("norm_weight"),
            eps=self.eps,
            n_groups=1,
            norm_before_gate=True,
            out_name=tracer.prefixed("gated_flat"),
        )
        gated = g.view(
            gated_flat, shape=[B, T, self.ValueDim],
            out_name=tracer.prefixed("gated"),
        )

        # Output projection
        gated_bt_flat = g.view(
            gated, shape=[B * T, self.ValueDim],
            out_name=tracer.prefixed("gated_bt_flat"),
        )
        out_flat = g.matmul(
            gated_bt_flat, out_proj_w, transpose="NT",
            out_name=tracer.prefixed("out_flat"),
        )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=out_slot,
        )

        return Proxy(out_slot, out)


class SwiGLUMLP(Module):
    """SwiGLU MLP: down(swiglu(up(x)))."""

    _hf_mapping_defaults_ = {
        "up_weight": fuse(
            "{prefix}.up_proj.weight",
            "{prefix}.gate_proj.weight",
            dim=0,
        ),
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = Dim("C")
        self.M = Dim("M")
        self.MUp = 2 * self.M

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        # -- params ----------------------------------------------------------
        up_w = tracer.register_param("up_weight", ("MUp", "C"))
        down_w = tracer.register_param("down_weight", ("C", "M"))

        # -- activation slots ------------------------------------------------
        up_slot = tracer.register_activation(
            "up", ("B", "T", "MUp"),
            aliases=["up_flat"],
            share_policy="when_recomputed",
        )
        act_slot = tracer.register_activation(
            "act", ("B", "T", "M"),
            aliases=["act_flat"],
            share_policy="when_recomputed",
            description="SwiGLU activation output",
        )
        down_slot = tracer.register_activation(
            "down", ("B", "T", "C"),
            aliases=["down_flat"],
            share_policy="when_recomputed",
            description="MLP down projection output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )
        up_flat = g.matmul(
            x_flat, up_w, transpose="NT",
            out_name=tracer.prefixed("up_flat"),
        )
        up = g.view(
            up_flat, shape=[B, T, self.MUp],
            out_name=up_slot,
        )
        act = g.swiglu(up, out_name=act_slot)
        act_flat = g.view(
            act, shape=[B * T, self.M],
            out_name=tracer.prefixed("act_flat"),
        )
        out_flat = g.matmul(
            act_flat, down_w, transpose="NT",
            out_name=tracer.prefixed("down_flat"),
        )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=down_slot,
        )

        return Proxy(down_slot, out)


class GatedMLP(Module):
    """Gated MLP with configurable activation."""

    _hf_mapping_defaults_ = {
        "gate_weight": "{prefix}.gate_proj.weight",
        "up_weight": "{prefix}.up_proj.weight",
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.C = Dim("C")
        self.M = Dim("M")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        gate_w = tracer.register_param("gate_weight", ("M", "C"))
        up_w = tracer.register_param("up_weight", ("M", "C"))
        down_w = tracer.register_param("down_weight", ("C", "M"))

        tracer.register_activation(
            "gate", ("B", "T", "M"), share_policy="when_recomputed",
        )
        tracer.register_activation(
            "up", ("B", "T", "M"), share_policy="when_recomputed",
        )
        down_slot = tracer.register_activation(
            "down", ("B", "T", "C"), share_policy="when_recomputed",
        )

        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))
        gate_flat = g.matmul(x_flat, gate_w, transpose="NT", out_name=tracer.prefixed("gate_flat"))
        up_flat = g.matmul(x_flat, up_w, transpose="NT", out_name=tracer.prefixed("up_flat"))

        act_map = {"silu": g.silu, "relu": g.relu, "relu2": g.relu2, "gelu": g.gelu}
        act_fn = act_map.get(self.activation, g.silu)
        gate_act = act_fn(gate_flat, out_name=tracer.prefixed("gate_act"))
        hidden = g.mul(gate_act, up_flat)

        out_flat = g.matmul(hidden, down_w, transpose="NT", out_name=tracer.prefixed("down_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=down_slot)

        return Proxy(down_slot, out)


class MoEExpertsGated(Module):
    """MoE with gated expert activation (SwiGLU-style: gate+up fused).

    Handles: router → top-k → permute → grouped GEMM (gate+up) → SwiGLU →
    grouped GEMM (down) → unpermute.  Optionally supports expert parallelism
    (ep_size > 1) with all-to-all dispatch/combine.
    """

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "experts_gate_up": stack_experts(
            "{prefix}.experts.{expert}.gate_proj.weight",
            fuse_gate_up=True,
        ),
        "experts_down": stack_experts(
            "{prefix}.experts.{expert}.down_proj.weight",
        ),
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        # -- params ----------------------------------------------------------
        tracer.register_param("router_weight", ("E", "C"))
        tracer.register_param("experts_gate_up", ("E", "MUp", "C"), offload_group="moe_experts")
        tracer.register_param("experts_down", ("E", "C", "M"), offload_group="moe_experts")

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "router_logits", ("B * T", "E"),
            save=True, share_policy="fft_share",
            description="Router logits before softmax",
        )
        tracer.register_activation(
            "router_probs", ("B * T", "E"),
            save=True, share_policy="fft_share",
            description="Router probabilities",
        )
        tracer.register_activation(
            "routing_weights", ("B * T", "K"),
            save=True, share_policy="fft_share",
            description="Routing weights for selected experts",
        )
        tracer.register_activation(
            "routing_indices", ("B * T", "K"),
            dtype="int32", save=True, share_policy="fft_share",
            description="Expert indices for each token",
        )
        tracer.register_activation(
            "permuted_input", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
            description="Permuted input for grouped GEMM",
        )
        tracer.register_activation(
            "scatter_indices", ("B * T * K",),
            dtype="int32", save=True, share_policy="fft_share",
            description="Indices for scattering back",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
                description="EP-dispatched input tokens",
            )
            tracer.register_activation(
                "ep_recv_scatter", ("B * T * K",),
                dtype="int32", save=True, share_policy="per_layer",
                when="ep_size > 1",
                description="EP-dispatched scatter indices",
            )
        tracer.register_activation(
            "expert_gate_up", ("B * T * K", "MUp"),
            save=True, share_policy="fft_share",
            description="Expert gate+up projection output",
        )
        tracer.register_activation(
            "expert_act", ("B * T * K", "M"),
            save=True, share_policy="fft_share",
            description="Expert SwiGLU activation output",
        )
        tracer.register_activation(
            "expert_down", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
            description="Expert down projection output",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
                description="EP-combined expert output",
            )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"),
            aliases=["out_flat"],
            save=True, share_policy="fft_share",
            description="Combined MoE output",
        )

        # -- graph -----------------------------------------------------------
        router_logits = g.matmul(
            x.ref, tracer.prefixed("router_weight"), transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )
        router_probs = g.moe_softmax(
            router_logits, out_name=tracer.prefixed("router_probs"),
        )
        routing_weights, routing_indices = g.moe_topk(
            router_probs, top_k=self.num_experts_per_tok,
            normalize=self.norm_topk_prob,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )
        permuted_input, scatter_indices = g.moe_permute(
            x.ref, routing_indices, top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input, routing_indices, scatter_indices,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input, tracer.prefixed("experts_gate_up"), gemm_scatter,
            out_name=tracer.prefixed("expert_gate_up"),
        )
        expert_act = g.swiglu(
            expert_gate_up, out_name=tracer.prefixed("expert_act"),
        )
        expert_down = g.moe_grouped_gemm_down(
            expert_act, tracer.prefixed("experts_down"), gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        if self.ep_size > 1:
            expert_down = g.ep_combine(
                expert_down,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        moe_out = g.moe_unpermute(
            expert_down, routing_weights, scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class MoESharedExpert(Module):
    """Shared expert for MoE models (SwiGLU-style: gate + up → silu*mul → down)."""

    _hf_mapping_defaults_ = {
        "gate": "{prefix}.shared_expert.gate_proj.weight",
        "up": "{prefix}.shared_expert.up_proj.weight",
        "down": "{prefix}.shared_expert.down_proj.weight",
    }

    def __init__(self, d_model: int, shared_expert_intermediate: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")
        self.SharedM = Dim("SharedM")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        tracer.register_param("gate", ("SharedM", "C"))
        tracer.register_param("up", ("SharedM", "C"))
        tracer.register_param("down", ("C", "SharedM"))

        tracer.register_activation(
            "gate_out", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "up_out", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "gate_act", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "hidden", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"), share_policy="fft_share",
            when="use_shared_expert",
        )

        shared_gate = g.matmul(
            x.ref, tracer.prefixed("gate"), transpose="NT",
            out_name=tracer.prefixed("gate_out"),
        )
        shared_up = g.matmul(
            x.ref, tracer.prefixed("up"), transpose="NT",
            out_name=tracer.prefixed("up_out"),
        )
        shared_gate_act = g.silu(shared_gate, out_name=tracer.prefixed("gate_act"))
        shared_hidden = g.mul(shared_gate_act, shared_up)
        shared_out = g.matmul(
            shared_hidden, tracer.prefixed("down"), transpose="NT",
            out_name=out_slot,
        )

        return Proxy(out_slot, shared_out)


class GptOssMoEExperts(Module):
    """GPT-OSS MoE with router bias, per-expert biases, and gpt_oss_moe_act."""

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.router.weight",
        "router_bias": "{prefix}.router.bias",
        "experts_gate_up": transform("{prefix}.experts.gate_up_proj", fn="transpose"),
        "experts_gate_up_bias": "{prefix}.experts.gate_up_proj_bias",
        "experts_down": transform("{prefix}.experts.down_proj", fn="transpose"),
        "experts_down_bias": "{prefix}.experts.down_proj_bias",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 4,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        # -- params --------------------------------------------------------------
        tracer.register_param("router_weight", ("E", "C"))
        tracer.register_param("router_bias", ("E",))
        tracer.register_param(
            "experts_gate_up", ("E", "MUp", "C"), offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_gate_up_bias", ("E", "MUp"), offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_down", ("E", "C", "M"), offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_down_bias", ("E", "C"), offload_group="moe_experts",
        )

        # -- activation slots ----------------------------------------------------
        tracer.register_activation(
            "router_logits", ("B * T", "E"),
            save=True, share_policy="per_layer",
            description="Router logits",
        )
        tracer.register_activation(
            "routing_weights", ("B * T", "K"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "routing_indices", ("B * T", "K"),
            dtype="int32", save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "permuted_input", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "scatter_indices", ("B * T * K",),
            dtype="int32", save=True, share_policy="fft_share",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
            tracer.register_activation(
                "ep_recv_scatter", ("B * T * K",),
                dtype="int32", save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
        tracer.register_activation(
            "expert_gate_up", ("B * T * K", "MUp"),
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_gate_up_bias", ("B * T * K", "MUp"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_act", ("B * T * K", "M"),
            save=True, share_policy="fft_share",
            description="GPT-OSS activation output",
        )
        tracer.register_activation(
            "expert_down", ("B * T * K", "C"),
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_down_bias", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"),
            aliases=["out_flat"],
            save=True, share_policy="fft_share",
        )

        # -- graph ---------------------------------------------------------------
        # Router (with bias)
        router_logits = g.matmul_bias(
            x.ref, tracer.prefixed("router_weight"),
            tracer.prefixed("router_bias"), transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )

        # Top-k (softmax + sort_by_index, no normalize)
        routing_weights, routing_indices = g.moe_topk(
            router_logits, top_k=self.num_experts_per_tok,
            normalize=False, softmax=True, sort_by_index=True,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )

        # Permute
        permuted_input, scatter_indices = g.moe_permute(
            x.ref, routing_indices, top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        # EP dispatch
        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input, routing_indices, scatter_indices,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        # Gate+up GEMM (interleaved)
        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input, tracer.prefixed("experts_gate_up"), gemm_scatter,
            gate_up_interleaved=True,
            out_name=tracer.prefixed("expert_gate_up"),
        )

        # Gate+up bias
        expert_gate_up_bias = g.moe_expert_bias_add(
            expert_gate_up, tracer.prefixed("experts_gate_up_bias"),
            out_name=tracer.prefixed("expert_gate_up_bias"),
        )

        # GPT-OSS activation
        expert_act = g.gpt_oss_moe_act(
            expert_gate_up_bias, alpha=1.702, limit=7.0,
            out_name=tracer.prefixed("expert_act"),
        )

        # Down GEMM
        expert_down = g.moe_grouped_gemm_down(
            expert_act, tracer.prefixed("experts_down"), gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        # Down bias
        expert_down_bias = g.moe_expert_bias_add(
            expert_down, tracer.prefixed("experts_down_bias"),
            out_name=tracer.prefixed("expert_down_bias"),
        )

        # EP combine
        if self.ep_size > 1:
            expert_down_bias = g.ep_combine(
                expert_down_bias,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        # Unpermute
        moe_out = g.moe_unpermute(
            expert_down_bias, routing_weights, scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class Mamba2Mixer(Module):
    """Mamba2 SSM module for hybrid architectures like Nemotron-H."""

    _hf_mapping_defaults_ = {
        "in_proj_weight": "{prefix}.in_proj.weight",
        "in_proj_bias": "{prefix}.in_proj.bias",
        "conv_weight": "{prefix}.conv1d.weight",
        "conv_bias": "{prefix}.conv1d.bias",
        "A_log": "{prefix}.A_log",
        "D_param": "{prefix}.D",
        "dt_bias": "{prefix}.dt_bias",
        "gated_norm_weight": "{prefix}.norm.weight",
        "out_proj_weight": "{prefix}.out_proj.weight",
        "out_proj_bias": "{prefix}.out_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        ssm_state_size: int = 128,
        n_groups: int = 8,
        conv_kernel: int = 4,
        chunk_size: int = 256,
        eps: float = 1e-5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        time_step_limit: tuple[float, float] | None = None,
        use_conv_bias: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.eps = eps
        self.dt_min = dt_min
        self.dt_max = dt_max
        import math as _math
        dt_max_default = 1e9
        if time_step_limit is None:
            time_step_limit = (0.0, dt_max_default)
        elif isinstance(time_step_limit, (list, tuple)) and len(time_step_limit) == 2:
            lo = float(time_step_limit[0])
            hi = float(time_step_limit[1])
            if not _math.isfinite(lo):
                lo = 0.0
            if not _math.isfinite(hi):
                hi = dt_max_default
            time_step_limit = (lo, hi)
        self.time_step_limit = time_step_limit
        self.use_conv_bias = use_conv_bias
        self.use_bias = use_bias

        # Derived dimensions
        self.intermediate_size = mamba_num_heads * mamba_head_dim
        self.conv_dim = self.intermediate_size + 2 * n_groups * ssm_state_size
        self.projection_size = self.intermediate_size + self.conv_dim + mamba_num_heads

        # Use concrete integers for Mamba-specific dims to avoid conflicts
        # with attention dims (D, K) in hybrid models.
        self.C = Dim("C")  # d_model — shared with attention

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        P = self.projection_size
        I = self.intermediate_size
        H = self.mamba_num_heads
        D_conv = self.conv_dim
        CK = self.conv_kernel

        # -- params ----------------------------------------------------------
        in_proj_w = tracer.register_param("in_proj_weight", (P, "C"))
        in_proj_b = tracer.register_param("in_proj_bias", (P,), when="use_bias")
        tracer.register_param("conv_weight", (D_conv, CK), quantizable=False)
        tracer.register_param("conv_bias", (D_conv,), when="use_conv_bias", quantizable=False)
        tracer.register_param("A_log", (H,), dtype="fp32", quantizable=False)
        tracer.register_param("D_param", (H,), dtype="fp32", quantizable=False)
        tracer.register_param("dt_bias", (H,), dtype="fp32", quantizable=False)
        tracer.register_param("gated_norm_weight", (I,), quantizable=False)
        out_proj_w = tracer.register_param("out_proj_weight", ("C", I))
        out_proj_b = tracer.register_param("out_proj_bias", ("C",), when="use_bias")

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "projected", ("B", "T", P),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "gate", ("B", "T", I),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "conv_out", ("B", D_conv, "T"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "hidden_states", ("B", I, "T"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "ssm_out", ("B", "T", I),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "ssm_state", ("B", H, self.mamba_head_dim, self.ssm_state_size),
            save=True, share_policy="fft_share",
            description="Final SSM state for caching",
        )
        tracer.register_activation(
            "gated_out", ("B", "T", I),
            save=True, share_policy="fft_share",
        )
        out_slot = tracer.register_activation(
            "out", ("B", "T", "C"),
            share_policy="per_layer",
            description="Mamba2 mixer output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )
        if self.use_bias:
            projected_flat = g.matmul_bias(
                x_flat, in_proj_w, in_proj_b, transpose="NT",
                out_name=tracer.prefixed("projected_flat"),
            )
        else:
            projected_flat = g.matmul(
                x_flat, in_proj_w, transpose="NT",
                out_name=tracer.prefixed("projected_flat"),
            )
        projected = g.view(
            projected_flat, shape=[B, T, P],
            out_name=tracer.prefixed("projected"),
        )

        # Split projection
        gate, conv_input, dt = g.mamba_split_proj(
            projected,
            intermediate_size=self.intermediate_size,
            conv_dim=self.conv_dim,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            gate_name=tracer.prefixed("gate"),
            conv_input_name=tracer.prefixed("conv_input"),
            dt_name=tracer.prefixed("dt"),
        )

        # Causal 1D convolution
        if self.use_conv_bias:
            conv_out = g.mamba_conv1d(
                conv_input, tracer.prefixed("conv_weight"),
                tracer.prefixed("conv_bias"),
                activation="silu",
                out_name=tracer.prefixed("conv_out"),
            )
        else:
            conv_out = g.mamba_conv1d(
                conv_input, tracer.prefixed("conv_weight"), None,
                activation="silu",
                out_name=tracer.prefixed("conv_out"),
            )

        # Split conv output
        hidden_states, ssm_B, ssm_C = g.mamba_split_conv_out(
            conv_out,
            intermediate_size=self.intermediate_size,
            groups_state_size=self.n_groups * self.ssm_state_size,
            n_groups=self.n_groups,
            ssm_state_size=self.ssm_state_size,
            hidden_name=tracer.prefixed("hidden_states"),
            B_name=tracer.prefixed("ssm_B"),
            C_name=tracer.prefixed("ssm_C"),
        )

        # SSM scan
        ssm_out, ssm_state = g.mamba_ssm_scan(
            hidden_states, dt, tracer.prefixed("A_log"),
            ssm_B, ssm_C, tracer.prefixed("D_param"),
            dt_bias=tracer.prefixed("dt_bias"),
            dt_softplus=True,
            dt_min=self.time_step_limit[0],
            dt_max=self.time_step_limit[1],
            chunk_size=self.chunk_size,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            ssm_state_size=self.ssm_state_size,
            n_groups=self.n_groups,
            out_name=tracer.prefixed("ssm_out"),
            state_name=tracer.prefixed("ssm_state"),
        )

        # Reshape SSM output
        ssm_out_flat = g.view(
            ssm_out, shape=[B, T, I],
            out_name=tracer.prefixed("ssm_out_flat"),
        )

        # Gated RMSNorm
        gated_out = g.mamba_gated_rmsnorm(
            ssm_out_flat, gate, tracer.prefixed("gated_norm_weight"),
            eps=self.eps,
            n_groups=self.n_groups,
            out_name=tracer.prefixed("gated_out"),
        )

        # Output projection
        gated_flat = g.view(
            gated_out, shape=[B * T, I],
            out_name=tracer.prefixed("gated_flat"),
        )
        if self.use_bias:
            out_flat = g.matmul_bias(
                gated_flat, out_proj_w, out_proj_b, transpose="NT",
                out_name=tracer.prefixed("out_flat"),
            )
        else:
            out_flat = g.matmul(
                gated_flat, out_proj_w, transpose="NT",
                out_name=tracer.prefixed("out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=out_slot,
        )

        return Proxy(out_slot, out)


class SimpleMLP(Module):
    """Simple MLP with configurable activation (relu2/silu/gelu)."""

    _hf_mapping_defaults_ = {
        "up_weight": "{prefix}.up_proj.weight",
        "up_bias": "{prefix}.up_proj.bias",
        "down_weight": "{prefix}.down_proj.weight",
        "down_bias": "{prefix}.down_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu2",
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.use_bias = use_bias
        self.C = Dim("C")
        self.M = Dim("M")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        # -- params ----------------------------------------------------------
        up_w = tracer.register_param("up_weight", ("M", "C"))
        up_b = tracer.register_param("up_bias", ("M",), when="use_bias")
        down_w = tracer.register_param("down_weight", ("C", "M"))
        down_b = tracer.register_param("down_bias", ("C",), when="use_bias")

        # -- activation slots ------------------------------------------------
        up_slot = tracer.register_activation(
            "up", ("B", "T", "M"),
            aliases=["up_flat"],
            save=True, share_policy="when_recomputed",
        )
        act_slot = tracer.register_activation(
            "act", ("B", "T", "M"),
            aliases=["act_flat"],
            save=True, share_policy="when_recomputed",
            description="MLP activation output",
        )
        down_slot = tracer.register_activation(
            "down", ("B", "T", "C"),
            aliases=["down_flat"],
            share_policy="when_recomputed",
            description="MLP down projection output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        # Up projection
        if self.use_bias:
            up_flat = g.matmul_bias(
                x_flat, up_w, up_b, transpose="NT",
                out_name=tracer.prefixed("up_flat"),
            )
        else:
            up_flat = g.matmul(
                x_flat, up_w, transpose="NT",
                out_name=tracer.prefixed("up_flat"),
            )
        up = g.view(
            up_flat, shape=[B, T, self.M],
            out_name=up_slot,
        )

        # Activation
        act_map = {"relu2": g.relu2, "silu": g.silu, "gelu": g.gelu}
        act_fn = act_map.get(self.activation, g.relu2)
        act = act_fn(up, out_name=act_slot)

        # Down projection
        act_flat = g.view(
            act, shape=[B * T, self.M],
            out_name=tracer.prefixed("act_flat"),
        )
        if self.use_bias:
            out_flat = g.matmul_bias(
                act_flat, down_w, down_b, transpose="NT",
                out_name=tracer.prefixed("down_flat"),
            )
        else:
            out_flat = g.matmul(
                act_flat, down_w, transpose="NT",
                out_name=tracer.prefixed("down_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=down_slot,
        )

        return Proxy(down_slot, out)


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

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="attention_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="attention_bias")
        if self.use_rope:
            tracer.register_param(
                "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
            )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat"],
            save=True, share_policy="per_layer",
        )
        if self.use_rope:
            tracer.register_activation(
                "qkv_rope", ("B", "T", "QKV"),
                save=True, share_policy="when_recomputed",
            )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="fft_share",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.attention_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.QKV],
            out_name=qkv_slot,
        )

        # RoPE (optional - Nemotron-H attention does not use positional encoding)
        if self.use_rope:
            attn_input = g.rope(
                qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
                rotary_dim=self.head_size,
                out_name=tracer.prefixed("qkv_rope"),
            )
        else:
            attn_input = qkv

        # FlashAttention
        attn_out, lse = g.flash_attention(
            attn_input, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Output projection
        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.attention_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class NemotronMoEExperts(Module):
    """Nemotron MoE with sigmoid routing, correction bias, relu2 activation."""

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "e_score_correction_bias": "{prefix}.gate.e_score_correction_bias",
        "experts_up": stack_experts(
            "{prefix}.experts.{expert}.up_proj.weight",
        ),
        "experts_down": stack_experts(
            "{prefix}.experts.{expert}.down_proj.weight",
        ),
    }

    def __init__(
        self,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        activation: str = "relu2",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.activation = activation
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        # -- params ----------------------------------------------------------
        tracer.register_param("router_weight", ("E", "C"), quantizable=False)
        tracer.register_param("e_score_correction_bias", ("E",), dtype="fp32", quantizable=False)
        tracer.register_param("experts_up", ("E", "M", "C"), offload_group="moe_experts")
        tracer.register_param("experts_down", ("E", "C", "M"), offload_group="moe_experts")

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "router_logits", ("B * T", "E"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "router_probs", ("B * T", "E"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "routing_weights", ("B * T", "K"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "routing_indices", ("B * T", "K"),
            dtype="int32", save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "permuted_input", ("B * T * K", "C"),
            save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "scatter_indices", ("B * T * K",),
            dtype="int32", save=True, share_policy="when_recomputed",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
            tracer.register_activation(
                "ep_recv_scatter", ("B * T * K",),
                dtype="int32", save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
        tracer.register_activation(
            "expert_up", ("B * T * K", "M"),
            save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "expert_act", ("B * T * K", "M"),
            save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "expert_down", ("B * T * K", "C"),
            save=True, share_policy="when_recomputed",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"),
            aliases=["out_flat"],
            save=True, share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        # Router (no bias on router matmul)
        router_logits = g.matmul(
            x.ref, tracer.prefixed("router_weight"), transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )

        # Sigmoid routing
        router_probs = g.moe_sigmoid(
            router_logits, out_name=tracer.prefixed("router_probs"),
        )

        # Top-k with correction bias and scaling factor
        routing_weights, routing_indices = g.moe_topk(
            router_probs, top_k=self.num_experts_per_tok,
            normalize=self.norm_topk_prob,
            scaling_factor=self.routed_scaling_factor,
            correction_bias=tracer.prefixed("e_score_correction_bias"),
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )

        # Permute
        permuted_input, scatter_indices = g.moe_permute(
            x.ref, routing_indices, top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        # EP dispatch
        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input, routing_indices, scatter_indices,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        # Expert up (simple grouped GEMM, NOT gate_up)
        expert_up = g.moe_grouped_gemm(
            gemm_input, tracer.prefixed("experts_up"), gemm_scatter,
        )

        # Activation (relu2 by default)
        if self.activation == "relu2":
            expert_act = g.relu2(
                expert_up, out_name=tracer.prefixed("expert_act"),
            )
        else:
            expert_act = g.silu(
                expert_up, out_name=tracer.prefixed("expert_act"),
            )

        # Expert down
        expert_down = g.moe_grouped_gemm_down(
            expert_act, tracer.prefixed("experts_down"), gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        # EP combine
        if self.ep_size > 1:
            expert_down = g.ep_combine(
                expert_down,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        # Unpermute
        moe_out = g.moe_unpermute(
            expert_down, routing_weights, scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class NemotronSharedExpert(Module):
    """Shared expert for Nemotron MoE (simple up -> activation -> down)."""

    _hf_mapping_defaults_ = {
        "up": "{prefix}.shared_experts.up_proj.weight",
        "down": "{prefix}.shared_experts.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        shared_expert_intermediate: int,
        activation: str = "relu2",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.shared_expert_intermediate = shared_expert_intermediate
        self.activation = activation
        self.C = Dim("C")
        self.SharedM = Dim("SharedM")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        tracer.register_param("up", ("SharedM", "C"))
        tracer.register_param("down", ("C", "SharedM"))

        tracer.register_activation(
            "up_out", ("B * T", "SharedM"),
            share_policy="when_recomputed",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "act", ("B * T", "SharedM"),
            share_policy="when_recomputed",
            when="use_shared_expert",
        )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"),
            share_policy="when_recomputed",
            when="use_shared_expert",
        )

        shared_up = g.matmul(
            x.ref, tracer.prefixed("up"), transpose="NT",
            out_name=tracer.prefixed("up_out"),
        )
        if self.activation == "relu2":
            shared_act = g.relu2(
                shared_up, out_name=tracer.prefixed("act"),
            )
        else:
            shared_act = g.silu(
                shared_up, out_name=tracer.prefixed("act"),
            )
        shared_out = g.matmul(
            shared_act, tracer.prefixed("down"), transpose="NT",
            out_name=out_slot,
        )

        return Proxy(out_slot, shared_out)


class Linear(Module):
    """Linear projection: y = x @ W^T (+ bias)."""

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
        "bias": "{prefix}.bias",
    }

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.C = Dim("in_dim")
        self.O = Dim("out_dim")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        w = tracer.register_param("weight", ("O", "C"))
        if self.use_bias:
            b = tracer.register_param("bias", ("O",), when="use_bias")

        out_slot = tracer.register_activation(
            "out", ("B", "T", "O"), share_policy="when_recomputed",
        )

        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))
        if self.use_bias:
            y_flat = g.matmul_bias(x_flat, w, b, transpose="NT", out_name=tracer.prefixed("y_flat"))
        else:
            y_flat = g.matmul(x_flat, w, transpose="NT", out_name=tracer.prefixed("y_flat"))
        out = g.view(y_flat, shape=[B, T, self.O], out_name=out_slot)

        return Proxy(out_slot, out)


class Embedding(Module):
    """Embedding lookup table."""

    _hf_mapping_defaults_ = {
        "weight": "model.embed_tokens.weight",
    }

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (token_ids,) = args

        w = tracer.register_param("weight", ("vocab_size", "d_model"))

        out_slot = tracer.register_activation(
            "out", ("B", "T", "d_model"),
            scope=ActivationScope.GLOBAL,
            description="Embedded input",
        )

        out = g.embedding(token_ids.ref, w, out_name=out_slot)
        return Proxy(out_slot, out)


class LMHead(Module):
    """Fused LM head projection + cross-entropy loss."""

    _hf_mapping_defaults_ = {
        "weight": "lm_head.weight",
    }

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, targets = args

        w = tracer.register_param("weight", ("vocab_size", "d_model"))

        loss_slot = tracer.register_activation(
            "loss", ("B * T",),
            dtype="fp32",
            scope=ActivationScope.GLOBAL,
            description="Cross-entropy loss per token",
        )

        x_flat = g.view(
            x.ref, shape=["B * T", "d_model"],
            out_name=tracer.prefixed("x_flat"),
        )
        loss = g.fused_lm_head_loss(
            x_flat, w, targets.ref,
            compute_accuracy=True,
            out_name=loss_slot,
        )
        return Proxy(loss_slot, loss)


# ============================================================================
# Canonical name remaps — map auto-prefixed nn names to C++ runtime names
# ============================================================================

# Dense transformer block: attn_norm / self_attn / mlp_norm / mlp
DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNorm) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (Attention) -> strip prefix ---
    "self_attn_qkv_weight": "qkv_weight",
    "self_attn_qkv_bias": "qkv_bias",
    "self_attn_out_weight": "out_weight",
    "self_attn_out_bias": "out_bias",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_flat": "qkv_flat",
    "self_attn_qkv_biased": "qkv_biased",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_attn": "attn",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    "self_attn_x_flat": "x_flat",
    # --- mlp_norm (RMSNorm) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (SwiGLUMLP) ---
    # mlp_up_weight, mlp_down_weight, mlp_up, mlp_down, mlp_up_flat,
    # mlp_down_flat are already correct (mlp_ prefix matches canonical names)
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",  # intermediate, kept as-is
}

# MoE block: same attn + norm, but moe instead of mlp
MOE_BLOCK_NAME_REMAP: dict[str, str] = {
    # Inherit all attn_norm and self_attn mappings
    **{k: v for k, v in DENSE_BLOCK_NAME_REMAP.items()
       if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # --- moe (MoEExpertsGated) -> strip moe_ prefix ---
    # Params
    "moe_router_weight": "router_weight",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_down": "experts_down",
    "moe_experts_up": "experts_up",
    # Activations
    "moe_router_logits": "router_logits",
    "moe_router_probs": "router_probs",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_ep_combined": "ep_combined",
    # moe_out / moe_out_flat already match canonical names
    # --- shared_expert (MoESharedExpert) ---
    # Params use local names gate/up/down; after shared_expert_ prefix they
    # become shared_expert_gate etc. which are already the canonical names.
    # Activation intermediates keep their prefixed names (no remap needed).
}

# GPT-OSS MoE block: GptOssAttention (with sinks) + GptOssMoEExperts (with biases)
GPT_OSS_BLOCK_NAME_REMAP: dict[str, str] = {
    # Inherit attn_norm, self_attn, mlp_norm mappings from dense
    **{k: v for k, v in DENSE_BLOCK_NAME_REMAP.items()
       if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # GPT-OSS attention extra: sinks
    "self_attn_sinks": "sinks",
    # --- moe (GptOssMoEExperts) -> strip moe_ prefix ---
    # Params
    "moe_router_weight": "router_weight",
    "moe_router_bias": "router_bias",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_gate_up_bias": "experts_gate_up_bias",
    "moe_experts_down": "experts_down",
    "moe_experts_down_bias": "experts_down_bias",
    # Activations
    "moe_router_logits": "router_logits",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_gate_up_bias": "expert_gate_up_bias",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_expert_down_bias": "expert_down_bias",
    "moe_ep_combined": "ep_combined",
    # moe_out / moe_out_flat already match canonical names
}

# Nemotron-H block remaps: each block type has norm_y -> ln, and strips mixer_ prefix
# All Nemotron blocks use "norm" -> "ln" convention (not "ln1")

# Mamba block: norm + Mamba2Mixer
NEMOTRON_MAMBA_BLOCK_REMAP: dict[str, str] = {
    # norm -> ln
    "norm_weight": "norm_weight",
    "norm_res": "res_in",
    "norm_y": "ln",
    "norm_rstd": "ln_rstd",
    # Strip mixer_ prefix from Mamba2Mixer params
    "mixer_in_proj_weight": "in_proj_weight",
    "mixer_in_proj_bias": "in_proj_bias",
    "mixer_conv_weight": "conv_weight",
    "mixer_conv_bias": "conv_bias",
    "mixer_A_log": "A_log",
    "mixer_D_param": "D_param",
    "mixer_dt_bias": "dt_bias",
    "mixer_gated_norm_weight": "gated_norm_weight",
    "mixer_out_proj_weight": "out_proj_weight",
    "mixer_out_proj_bias": "out_proj_bias",
    # Strip mixer_ prefix from Mamba2Mixer activations
    "mixer_x_flat": "x_flat",
    "mixer_projected_flat": "projected_flat",
    "mixer_projected": "projected",
    "mixer_gate": "gate",
    "mixer_conv_input": "conv_input",
    "mixer_dt": "dt",
    "mixer_conv_out": "conv_out",
    "mixer_hidden_states": "hidden_states",
    "mixer_ssm_B": "ssm_B",
    "mixer_ssm_C": "ssm_C",
    "mixer_ssm_out": "ssm_out",
    "mixer_ssm_out_flat": "ssm_out_flat",
    "mixer_ssm_state": "ssm_state",
    "mixer_gated_out": "gated_out",
    "mixer_gated_flat": "gated_flat",
    "mixer_out_flat": "out_flat",
    "mixer_out": "out",
}

# Attention block: norm + NemotronAttention
NEMOTRON_ATTN_BLOCK_REMAP: dict[str, str] = {
    # norm -> ln (Nemotron uses ln1 naming for attention block)
    "norm_weight": "norm_weight",
    "norm_res": "res_att",
    "norm_y": "ln1",
    "norm_rstd": "ln1_rstd",
    # Strip mixer_ prefix from NemotronAttention params
    "mixer_qkv_weight": "qkv_weight",
    "mixer_qkv_bias": "qkv_bias",
    "mixer_out_weight": "out_weight",
    "mixer_out_bias": "out_bias",
    "mixer_rope_freqs": "rope_freqs",
    # Strip mixer_ prefix from NemotronAttention activations
    "mixer_x_flat": "x_flat",
    "mixer_qkv_flat": "qkv_flat",
    "mixer_qkv": "qkv",
    "mixer_qkv_rope": "qkv_rope",
    "mixer_att": "att",
    "mixer_att_flat": "att_flat",
    "mixer_attn": "attn",
    "mixer_lse": "lse",
    "mixer_att_out_flat": "att_out_flat",
    "mixer_att_out": "att_out",
}

# MLP block: norm + SimpleMLP
NEMOTRON_MLP_BLOCK_REMAP: dict[str, str] = {
    # norm -> ln
    "norm_weight": "norm_weight",
    "norm_res": "res_in",
    "norm_y": "ln",
    "norm_rstd": "ln_rstd",
    # Strip mixer_ prefix from SimpleMLP params
    "mixer_up_weight": "up_weight",
    "mixer_up_bias": "up_bias",
    "mixer_down_weight": "down_weight",
    "mixer_down_bias": "down_bias",
    # Strip mixer_ prefix from SimpleMLP activations
    "mixer_x_flat": "mlp_x_flat",
    "mixer_up_flat": "mlp_up_flat",
    "mixer_up": "mlp_up",
    "mixer_act": "swiglu",
    "mixer_act_flat": "swiglu_flat",
    "mixer_down_flat": "mlp_down_flat",
    "mixer_down": "mlp_down",
}

# MoE block: norm + NemotronMoEExperts (+ optional NemotronSharedExpert)
NEMOTRON_MOE_BLOCK_REMAP: dict[str, str] = {
    # norm -> ln
    "norm_weight": "norm_weight",
    "norm_res": "res_in",
    "norm_y": "ln",
    "norm_rstd": "ln_rstd",
    # Strip mixer_ prefix from NemotronMoEExperts params
    "mixer_router_weight": "router_weight",
    "mixer_e_score_correction_bias": "e_score_correction_bias",
    "mixer_experts_up": "experts_up",
    "mixer_experts_down": "experts_down",
    # Strip mixer_ prefix from NemotronMoEExperts activations
    "mixer_router_logits": "router_logits",
    "mixer_router_probs": "router_probs",
    "mixer_routing_weights": "routing_weights",
    "mixer_routing_indices": "routing_indices",
    "mixer_permuted_input": "permuted_input",
    "mixer_scatter_indices": "scatter_indices",
    "mixer_ep_recv_input": "ep_recv_input",
    "mixer_ep_recv_scatter": "ep_recv_scatter",
    "mixer_expert_up": "expert_up",
    "mixer_expert_act": "expert_act",
    "mixer_expert_down": "expert_down",
    "mixer_ep_combined": "ep_combined",
    "mixer_out": "moe_out",
    "mixer_out_flat": "moe_out_flat",
    # Strip shared_expert_ prefix from NemotronSharedExpert params
    "shared_expert_up": "shared_expert_up",
    "shared_expert_down": "shared_expert_down",
    # Strip shared_expert_ prefix from activations
    "shared_expert_up_out": "shared_up_out",
    "shared_expert_act": "shared_act",
    "shared_expert_out": "shared_out",
}

# Nemotron-H model-level remap (backbone.* prefix)
NEMOTRON_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- final_norm (RMSNorm) ---
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

# VL dense block: same as dense but with QK-Norm + MRoPE (separate, extra qkv_norm slot)
VL_DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    **DENSE_BLOCK_NAME_REMAP,
    "self_attn_qkv_norm": "qkv_norm",
}

# Model-level remap for embedding / final_norm / lm_head
STANDARD_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- final_norm (RMSNorm) ---
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

# VL model-level: embedding_out stays auto-named (mask_scatter output is x0)
VL_MODEL_NAME_REMAP: dict[str, str] = {
    "embedding_weight": "embedding",
    # No "embedding_out" → "x0" — mask_scatter output takes that name
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

# Qwen3.5 attention block: RMSNormPlus1 norms, Qwen3_5Attention (separate Q/K/V), SwiGLUMLP
QWEN3_5_ATTN_BLOCK_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNormPlus1) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_weight_eff": "ln1_weight_eff",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (Qwen3_5Attention) -> full_* canonical names ---
    "self_attn_q_proj_weight": "full_q_proj_weight",
    "self_attn_q_proj_bias": "full_q_proj_bias",
    "self_attn_k_proj_weight": "full_k_proj_weight",
    "self_attn_k_proj_bias": "full_k_proj_bias",
    "self_attn_v_proj_weight": "full_v_proj_weight",
    "self_attn_v_proj_bias": "full_v_proj_bias",
    "self_attn_out_weight": "full_out_weight",
    "self_attn_out_bias": "full_out_bias",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    # Activations from Qwen3_5Attention
    "self_attn_x_flat": "x_flat",
    "self_attn_q_proj": "full_q_proj",
    "self_attn_k_proj": "full_k_proj",
    "self_attn_v_proj": "full_v_proj",
    "self_attn_q_proj_4d": "full_q_proj_4d",
    "self_attn_q": "full_q",
    "self_attn_gate": "full_gate",
    "self_attn_k": "full_k",
    "self_attn_v": "full_v",
    "self_attn_q_norm_weight_eff": "q_norm_weight_eff",
    "self_attn_k_norm_weight_eff": "k_norm_weight_eff",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_att": "att",
    "self_attn_att_4d": "att_4d",
    "self_attn_att_flat": "att_flat",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    # --- mlp_norm (RMSNormPlus1) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_weight_eff": "ln2_weight_eff",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (SwiGLUMLP) ---
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",
}

# Qwen3.5 linear block: RMSNormPlus1 norms, GatedDeltaNetMixer, SwiGLUMLP
QWEN3_5_LINEAR_BLOCK_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNormPlus1) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_weight_eff": "ln1_weight_eff",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- mixer (GatedDeltaNetMixer) -> lin_* canonical names ---
    "mixer_in_proj_qkv_weight": "lin_in_proj_qkv_weight",
    "mixer_in_proj_z_weight": "lin_in_proj_z_weight",
    "mixer_in_proj_b_weight": "lin_in_proj_b_weight",
    "mixer_in_proj_a_weight": "lin_in_proj_a_weight",
    "mixer_conv_weight": "lin_conv_weight",
    "mixer_A_log": "lin_A_log",
    "mixer_dt_bias": "lin_dt_bias",
    "mixer_norm_weight": "lin_norm_weight",
    "mixer_out_weight": "lin_out_weight",
    # Activations from GatedDeltaNetMixer
    "mixer_x_flat": "lin_x_flat",
    "mixer_mixed_qkv_flat": "lin_mixed_qkv_flat",
    "mixer_mixed_qkv": "lin_mixed_qkv",
    "mixer_conv_w2d": "lin_conv_w2d",
    "mixer_conv_out_cf": "lin_conv_out_cf",
    "mixer_query": "lin_query",
    "mixer_key": "lin_key",
    "mixer_value": "lin_value",
    "mixer_z_flat": "lin_z_flat",
    "mixer_z": "lin_z",
    "mixer_b_flat": "lin_b_flat",
    "mixer_b": "lin_b",
    "mixer_a_flat": "lin_a_flat",
    "mixer_a": "lin_a",
    "mixer_decay": "lin_decay",
    "mixer_query_rep": "lin_query_rep",
    "mixer_key_rep": "lin_key_rep",
    "mixer_core_flat": "lin_core_flat",
    "mixer_z_norm_flat": "lin_z_norm_flat",
    "mixer_gated_flat": "lin_gated_flat",
    "mixer_gated": "lin_gated",
    "mixer_gated_bt_flat": "lin_gated_bt_flat",
    "mixer_out_flat": "lin_att_out_flat",
    "mixer_out": "lin_att_out",
    # --- mlp_norm (RMSNormPlus1) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_weight_eff": "ln2_weight_eff",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (SwiGLUMLP) ---
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",
}

# Qwen3.5 model-level remap (uses RMSNormPlus1 for final_norm)
QWEN3_5_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- final_norm (RMSNormPlus1) ---
    "final_norm_weight": "final_norm",
    "final_norm_weight_eff": "final_norm_eff",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

# Qwen3.5 VL model-level: embedding_out stays auto-named (mask_scatter output is x0)
QWEN3_5_VL_MODEL_NAME_REMAP: dict[str, str] = {
    "embedding_weight": "embedding",
    # No "embedding_out" -> "x0" -- mask_scatter output takes that name
    "final_norm_weight": "final_norm",
    "final_norm_weight_eff": "final_norm_eff",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}


# Qwen3.5 MoE attention block: Qwen3_5Attention + MoE (replaces SwiGLUMLP)
QWEN3_5_MOE_ATTN_BLOCK_REMAP: dict[str, str] = {
    # Inherit norm + attention mappings from Qwen3.5 attention block
    **{k: v for k, v in QWEN3_5_ATTN_BLOCK_REMAP.items()
       if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # --- moe (MoEExpertsGated) -> strip moe_ prefix ---
    "moe_router_weight": "router_weight",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_down": "experts_down",
    "moe_experts_up": "experts_up",
    "moe_router_logits": "router_logits",
    "moe_router_probs": "router_probs",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_ep_combined": "ep_combined",
    # shared_expert: prefixed names are already canonical
    # shared_expert_gate_proj: the sigmoid gate param
}

# Qwen3.5 MoE linear block: GatedDeltaNet + MoE (replaces SwiGLUMLP)
QWEN3_5_MOE_LINEAR_BLOCK_REMAP: dict[str, str] = {
    # Inherit norm + linear attention mappings from Qwen3.5 linear block
    **{k: v for k, v in QWEN3_5_LINEAR_BLOCK_REMAP.items()
       if k.startswith(("attn_norm_", "mixer_", "mlp_norm_"))},
    # --- moe (MoEExpertsGated) -> strip moe_ prefix ---
    "moe_router_weight": "router_weight",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_down": "experts_down",
    "moe_experts_up": "experts_up",
    "moe_router_logits": "router_logits",
    "moe_router_probs": "router_probs",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_ep_combined": "ep_combined",
    # shared_expert: prefixed names are already canonical
    # shared_expert_gate_proj: the sigmoid gate param
}


# ============================================================================
# Block base class
# ============================================================================

class Block(Module):
    """Base class for transformer blocks.

    Subclasses define ``__init__`` (composing nn modules) and ``forward``
    (calling them).  ``compile()`` traces the forward pass and generates a
    ``BlockSpec`` compatible with the existing DSL compiler pipeline.

    Subclasses may define ``_name_remap_`` to map auto-generated prefixed
    names to canonical names expected by the C++ runtime::

        class Qwen3Block(nn.Block):
            _name_remap_ = {
                "attn_norm_y": "ln1",
                "attn_norm_rstd": "ln1_rstd",
                ...
            }

    Example::

        class Qwen3Block(nn.Block):
            def __init__(self, d_model, num_query_heads, ...):
                super().__init__()
                self.attn_norm = nn.RMSNorm(d_model, eps=eps)
                self.self_attn = nn.Qwen3Attention(...)
                self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
                self.mlp = nn.SwiGLUMLP(d_model, d_ff)

            def forward(self, x, residual, position_ids):
                residual, h = self.attn_norm(residual, x)
                h = self.self_attn(h, position_ids)
                residual, h = self.mlp_norm(residual, h)
                h = self.mlp(h)
                return h, residual
    """

    # Override in subclasses to remap auto-generated prefixed names to
    # canonical names expected by the C++ runtime.
    _name_remap_: dict[str, str] = {}

    def forward(self, *args: Proxy, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward()")

    def compile(self) -> BlockSpec:
        """Trace forward and produce a BlockSpec."""
        tracer = Tracer()
        tracer._name_remap = getattr(type(self), '_name_remap_', {})
        token = _current_tracer.set(tracer)
        try:
            # Build proxy inputs from forward() signature
            sig = inspect.signature(self.forward)
            input_proxies = {}
            for name in sig.parameters:
                ref = tracer.graph.input(name)
                input_proxies[name] = Proxy(name, ref)

            # Trace
            result = self.forward(**input_proxies)

            # Auto-generate gradient slots
            tracer.auto_gradients()
        finally:
            _current_tracer.reset(token)

        # Build ForwardSpec with input/output specs for the compiler
        # Create a generic TensorAnnotation placeholder for each IO
        _generic_tensor = Tensor["B", "T", "C"]
        _slot_map_model = {s.name: s for s in tracer.forward_slots}

        input_specs = []
        for name in sig.parameters:
            slot = _slot_map_model.get(name)
            if slot and slot.shape:
                tt = TensorAnnotation(dims=tuple(slot.shape), dtype=slot.dtype or "bf16")
            else:
                tt = _generic_tensor
            input_specs.append(IOSpec(name=name, tensor_type=tt))

        output_specs = []
        if isinstance(result, tuple):
            for i, proxy in enumerate(result):
                out_name = proxy.name if isinstance(proxy, Proxy) else f"out_{i}"
                output_specs.append(IOSpec(name=out_name, tensor_type=_generic_tensor))
        elif isinstance(result, Proxy):
            output_specs.append(IOSpec(name=result.name, tensor_type=_generic_tensor))

        forward_spec = ForwardSpec(
            inputs=input_specs,
            outputs=output_specs,
            graph_fn=lambda self_, g: None,  # graph already built
        )
        forward_spec._traced_graph = tracer.graph
        # Set explicit output names so the compiler doesn't need to infer them
        tracer.graph._returned_outputs = [o.name for o in output_specs]

        # Build ActivationLayoutSpec
        activation_layout = ActivationLayoutSpec(
            name=f"{type(self).__name__}Activations",
            slots=tracer.forward_slots,
            gradient_slots=tracer.gradient_slots,
        )

        # Build BlockSpec
        spec = BlockSpec(
            name=type(self).__name__,
            python_class=type(self),
            docstring=type(self).__doc__,
            constructor_params=_extract_constructor_params(type(self)),
            params=tracer.params,
            forward=forward_spec,
            activations=activation_layout,
        )

        # Register in block registry
        _block_registry[type(self).__name__] = spec

        return spec


# ============================================================================
# Model base class
# ============================================================================

class Model(Module):
    """Base class for full models.

    Subclasses define ``__init__`` (composing nn modules) and ``forward``
    (calling them).  ``compile()`` traces the forward pass and generates a
    ``ModelSpec`` compatible with the existing DSL compiler pipeline.
    """

    # Override in subclasses via @nn.hf_config decorator
    _hf_config_: HFConfigSpec | None = None

    # Override in subclasses to remap auto-generated prefixed names to
    # canonical names expected by the C++ runtime.
    _name_remap_: dict[str, str] = {}

    def forward(self, *args: Proxy, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward()")

    def compile(self) -> ModelSpec:
        """Trace forward and produce a ModelSpec."""
        tracer = Tracer()
        tracer._name_remap = getattr(type(self), '_name_remap_', {})
        token = _current_tracer.set(tracer)
        try:
            sig = inspect.signature(self.forward)
            input_proxies = {}
            for name in sig.parameters:
                ref = tracer.graph.input(name)
                input_proxies[name] = Proxy(name, ref)

            result = self.forward(**input_proxies)
            tracer.auto_gradients()
        finally:
            _current_tracer.reset(token)

        # Build IO specs using activation slot shapes when available so the
        # C++ graph compiler gets correct input shapes (e.g., token_ids is 2D int32,
        # not 3D bf16).
        _generic_tensor = Tensor["B", "T", "C"]
        _slot_map = {s.name: s for s in tracer.forward_slots}

        input_specs = []
        for name in sig.parameters:
            slot = _slot_map.get(name)
            if slot and slot.shape:
                tensor_type = TensorAnnotation(dims=tuple(slot.shape), dtype=slot.dtype or "bf16")
            else:
                tensor_type = _generic_tensor
            input_specs.append(IOSpec(name=name, tensor_type=tensor_type))

        output_specs = []
        if isinstance(result, Proxy):
            slot = _slot_map.get(result.name)
            tt = Tensor[tuple(slot.shape)] if slot and slot.shape else _generic_tensor
            output_specs.append(IOSpec(name=result.name, tensor_type=tt))
        elif isinstance(result, tuple):
            for i, p in enumerate(result):
                nm = p.name if isinstance(p, Proxy) else f"out_{i}"
                slot = _slot_map.get(nm)
                tt = Tensor[tuple(slot.shape)] if slot and slot.shape else _generic_tensor
                output_specs.append(IOSpec(name=nm, tensor_type=tt))

        forward_spec = ForwardSpec(
            inputs=input_specs,
            outputs=output_specs,
            graph_fn=lambda self_, g: None,
        )
        forward_spec._traced_graph = tracer.graph
        # Set explicit output names so the compiler doesn't need to infer them
        tracer.graph._returned_outputs = [o.name for o in output_specs]

        activation_layout = ActivationLayoutSpec(
            name=f"{type(self).__name__}Activations",
            slots=tracer.forward_slots,
            gradient_slots=tracer.gradient_slots,
        )

        hf_mapping_spec = None
        if tracer.hf_mappings:
            from .specs import HFMappingSpec
            hf_mapping_spec = HFMappingSpec(mappings=tracer.hf_mappings)

        spec = ModelSpec(
            name=type(self).__name__,
            python_class=type(self),
            docstring=type(self).__doc__,
            constructor_params=_extract_constructor_params(type(self)),
            params=tracer.params,
            forward=forward_spec,
            activations=activation_layout,
            hf_config=getattr(type(self), "_hf_config_", None),
            hf_mapping=hf_mapping_spec,
        )

        _model_registry[type(self).__name__] = spec
        return spec


# ============================================================================
# hf_config decorator for Model subclasses
# ============================================================================

def hf_config(
    architecture: str,
    model_type: str,
    **field_mappings: str,
) -> Any:
    """Decorator that attaches HuggingFace config mapping to a Model class.

    Usage::

        @nn.hf_config(
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            d_model="hidden_size",
            n_layers="num_hidden_layers",
            ...
        )
        class Qwen3(nn.Model):
            ...
    """
    def decorator(cls: type) -> type:
        hf_cfg = HFConfigSpec(
            architecture=architecture,
            model_type=model_type,
            param_mapping=field_mappings,
        )
        cls._hf_config_ = hf_cfg

        # Register a lazy ModelSpec in the registry so compile_model_for_hf
        # can find this model by architecture name.  The spec carries
        # hf_config for lookup and a graph_fn that triggers nn compilation.
        def _lazy_graph_fn(self_instance: Any, g: Any) -> None:
            pass  # placeholder — real graph is built by nn.Model.compile()

        lazy_spec = ModelSpec(
            name=cls.__name__,
            python_class=cls,
            docstring=cls.__doc__,
            constructor_params=_extract_constructor_params(cls),
            forward=ForwardSpec(graph_fn=_lazy_graph_fn),
            hf_config=hf_cfg,
        )
        # Mark as nn-style so the compiler knows to call compile()
        lazy_spec._nn_model_class = cls
        _model_registry[cls.__name__] = lazy_spec

        return cls
    return decorator


# ============================================================================
# BlockStack — helper for stacking blocks in a model
# ============================================================================

class BlockStack(Module):
    """Calls StackedBlocks with the given block type.

    Used in Model definitions::

        self.blocks = nn.BlockStack(n_layers, Qwen3Block, d_model=..., ...)
    """

    def __init__(self, n_layers: int, block_cls: type, **block_kwargs: Any) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.block_cls = block_cls
        self.block_kwargs = block_kwargs
        # Instantiate one block to get its spec
        self._block_instance = block_cls(**block_kwargs)

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> tuple[Proxy, ...]:
        g = tracer.graph

        # Register the block array as a param (ARRAY kind, not TENSOR).
        # Use the module attribute name (tracer.prefix) as the param name,
        # matching the original DSL convention (e.g., "blocks").
        block_name = self.block_cls.__name__
        param_name = tracer.prefix  # e.g., "blocks"
        if param_name not in tracer.params:
            tracer.params[param_name] = ParamSpec(
                name=param_name,
                kind=ParamKind.ARRAY,
                array_size="n_layers",
                element_type=block_name,
            )

        # Compile the inner block to register it
        self._block_instance.compile()

        # Emit StackedBlocks call — num_outputs matches block return count.
        # The block returns (out, residual) = 2 outputs, while receiving
        # (x, residual, position_ids) = 3 inputs.
        block_sig = inspect.signature(self._block_instance.forward)
        block_params = [p for p in block_sig.parameters]
        num_outputs = len(block_params) - 1  # exclude position_ids

        input_refs = [a.ref for a in args]
        result = g.call(
            "StackedBlocks",
            *input_refs,
            num_outputs=num_outputs,
            blocks=param_name,
            n_layers=self.n_layers,
            **kwargs,
        )

        if isinstance(result, GraphRef):
            return (Proxy("stacked_out", result),)
        return tuple(
            Proxy(f"stacked_out_{i}", r) for i, r in enumerate(result)
        )


class HybridBlockStack(Module):
    """Calls HybridStackedBlocks with multiple block types.

    Used for hybrid architectures like Nemotron-H where different layers
    use different block types (Mamba, Attention, MLP, MoE).

    Usage::

        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=[
                ("mamba_blocks", NemotronHMamba2Block, n_mamba, mamba_kwargs),
                ("attn_blocks", NemotronHAttentionBlock, n_attn, attn_kwargs),
                ("mlp_blocks", NemotronHMLPBlock, n_mlp, mlp_kwargs),
            ],
            block_types=["mamba", "attention", "mlp", ...],
            n_layers=52,
        )
    """

    def __init__(
        self,
        block_configs: list[tuple[str, type, int, dict]],
        block_types: list[str],
        n_layers: int,
    ) -> None:
        super().__init__()
        self.block_configs = block_configs
        self.block_types = block_types
        self.n_layers = n_layers
        # Instantiate one block of each type to get its spec
        self._block_instances: dict[str, tuple[str, Any]] = {}
        for param_name, block_cls, count, kwargs in block_configs:
            if count > 0:
                instance = block_cls(**kwargs)
                # Map block type name to (param_name, instance)
                # Infer block_type from param_name convention
                btype = param_name.replace("_blocks", "")
                self._block_instances[btype] = (param_name, instance, count)

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> tuple[Proxy, ...]:
        g = tracer.graph

        # Register each block array as a param and compile block specs
        call_kwargs = {}
        for btype, (param_name, instance, count) in self._block_instances.items():
            block_name = type(instance).__name__
            size_attr = f"n_{btype}_blocks"
            if param_name not in tracer.params:
                tracer.params[param_name] = ParamSpec(
                    name=param_name,
                    kind=ParamKind.ARRAY,
                    array_size=size_attr,
                    element_type=block_name,
                )
            # Compile the inner block to register it
            instance.compile()
            call_kwargs[param_name] = param_name

        # Emit HybridStackedBlocks call
        input_refs = [a.ref for a in args]
        result = g.call(
            "HybridStackedBlocks",
            *input_refs,
            num_outputs=2,
            block_types=self.block_types,
            n_layers=self.n_layers,
            **call_kwargs,
        )

        if isinstance(result, GraphRef):
            return (Proxy("stacked_out", result),)
        return tuple(
            Proxy(f"stacked_out_{i}", r) for i, r in enumerate(result)
        )
