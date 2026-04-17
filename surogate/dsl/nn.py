"""
Torch.nn-like module system for Surogate DSL.

Provides a familiar torch.nn-style API for defining models that compiles
to the same IR as the @block/@model decorated classes.

Example:
    from surogate.dsl import nn
    from surogate.dsl.attention import AttentionConfig
    from surogate.dsl.mlp import MLPConfig

    class Qwen3Block(nn.Block):
        def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                     d_ff, max_seq, eps=1e-6, use_qkv_bias=False, use_qk_norm=True):
            super().__init__()
            self.attn_norm = nn.RMSNorm(d_model, eps=eps)
            self.self_attn = nn.GenericGQAttention(
                d_model, num_query_heads, num_kv_heads, head_size, max_seq,
                config=AttentionConfig(
                    qk_norm=use_qk_norm, qkv_bias=use_qkv_bias, eps=eps,
                ),
            )
            self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
            self.mlp = nn.GenericMLP(d_model, d_ff, config=MLPConfig())

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
from dataclasses import dataclass
from typing import Any

from .decorators import (
    _block_registry,
    _extract_constructor_params,
    _model_registry,
)
from .graph_builder import GraphBuilder, GraphRef

# Name-remap dictionaries used to live in this module. They now live
# alongside the block/model definitions in ``surogate.dsl.blocks.*``;
# callers should import them from there directly.
from .specs import (
    ActivationLayoutSpec,
    ActivationScope,
    ActivationSlotSpec,
    BlockSpec,
    ForwardSpec,
    HFConfigSpec,
    IOSpec,
    LoRATarget,
    ModelSpec,
    ParamKind,
    ParamSpec,
    SharePolicy,
)
from .tensor_type import Tensor, TensorAnnotation

# ============================================================================
# Tracing context
# ============================================================================

_current_tracer: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("_current_tracer", default=None)


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
        lora_targets: list[LoRATarget] | None = None,
    ) -> str:
        """Register a parameter and return its fully-qualified name.

        ``lora_targets`` declares named slices of the parameter's output
        dimension that can receive LoRA adapters. For unfused weights pass a
        single target with ``size=0`` (meaning the full output). For fused
        projections (e.g. fused QKV), pass one target per logical projection
        with explicit ``offset`` and ``size`` in elements.
        """
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
                lora_targets=list(lora_targets) if lora_targets else [],
            )
            if when is not None:
                spec.optional = True
                spec.condition = lambda self_, _w=when: getattr(self_, _w, False)
            self.params[full_name] = spec
        elif lora_targets:
            # Update existing spec's targets (e.g. re-registration path).
            self.params[full_name].lora_targets = list(lora_targets)
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

        self.forward_slots.append(
            ActivationSlotSpec(
                name=full_name,
                scope=scope,
                shape=shape,
                dtype=dtype,
                aliases=[self.prefixed(a) for a in (aliases or [])],
                save_for_backward=save,
                share_policy=share_policy,
                condition_expr=when,
                description=description,
            )
        )
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
            self.gradient_slots.append(
                ActivationSlotSpec(
                    name=grad_name,
                    scope=grad_scope,
                    shape=slot.shape,
                    dtype=slot.dtype if slot.name == "loss" else None,
                    gradient_of=slot.name,
                )
            )

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

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy | tuple[Proxy, ...]:
        """Emit graph nodes for this module. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._trace()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tracer = _current_tracer.get()
        if tracer is None:
            raise RuntimeError(
                f"{type(self).__name__} can only be called during block/model compilation (inside compile())"
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
    def _scale_by_param(
        x: Proxy,
        param_name: str,
        *,
        name: str | None = None,
    ) -> Proxy:
        """Element-wise multiply a proxy by a named parameter (e.g. a frozen scalar)."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("scaled")
        ref = g.mul(x.ref, tracer.prefixed(param_name))
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
    def _gelu(x: Proxy, *, name: str | None = None) -> Proxy:
        """GeLU activation (tanh approximation)."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("gelu")
        ref = g.gelu(x.ref, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _matmul(x: Proxy, weight_name: str, *, transpose: str = "NT", name: str | None = None) -> Proxy:
        """Matmul with a named weight parameter."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("matmul")
        ref = g.matmul(x.ref, weight_name, transpose=transpose, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _scale_by_const(x: Proxy, factor: float, *, name: str | None = None) -> Proxy:
        """Scale a proxy by a constant factor."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("scaled")
        ref = g.scale(x.ref, factor=factor)
        return Proxy(out_name, ref)

    @staticmethod
    def _rmsnorm(
        x: Proxy,
        weight_name: str,
        *,
        eps: float = 1e-6,
        name: str | None = None,
    ) -> Proxy:
        """Apply RMSNorm with a named weight parameter."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("rmsnorm")
        y_ref, _ = g.rmsnorm(x.ref, tracer.prefixed(weight_name), eps=eps)
        return Proxy(out_name, y_ref)

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
        x: Proxy,
        mask: Proxy,
        values: Proxy,
        *,
        name: str | None = None,
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
        tracer.gradient_slots.append(
            ActivationSlotSpec(
                name=name,
                scope=scope,
                shape=shape,
                dtype=dtype,
                gradient_of=gradient_of,
            )
        )


# ============================================================================
# Concrete modules
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
                self.self_attn = nn.GenericGQAttention(
                    ..., config=AttentionConfig(qk_norm=True),
                )
                self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
                self.mlp = nn.GenericMLP(d_model, d_ff)

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

    def compile(self, name_override: str | None = None) -> BlockSpec:
        """Trace forward and produce a BlockSpec."""
        tracer = Tracer()
        tracer._name_remap = getattr(type(self), "_name_remap_", {})
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
            name=name_override or type(self).__name__,
            python_class=type(self),
            docstring=type(self).__doc__,
            constructor_params=_extract_constructor_params(type(self)),
            params=tracer.params,
            forward=forward_spec,
            activations=activation_layout,
        )

        # Register in block registry (both class name and override name)
        _block_registry[type(self).__name__] = spec
        if name_override and name_override != type(self).__name__:
            _block_registry[name_override] = spec

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
        tracer._name_remap = getattr(type(self), "_name_remap_", {})
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

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> tuple[Proxy, ...]:
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
        return tuple(Proxy(f"stacked_out_{i}", r) for i, r in enumerate(result))


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
        per_layer_input_name: str | None = None,
        kv_sharing_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.block_configs = block_configs
        self.block_types = block_types
        self.n_layers = n_layers
        self.per_layer_input_name = per_layer_input_name
        self.kv_sharing_map = kv_sharing_map or {}
        # Instantiate one block of each type to get its spec
        self._block_instances: dict[str, tuple[str, Any]] = {}
        for param_name, block_cls, count, kwargs in block_configs:
            if count > 0:
                instance = block_cls(**kwargs)
                # Map block type name to (param_name, instance)
                # Infer block_type from param_name convention
                btype = param_name.replace("_blocks", "")
                self._block_instances[btype] = (param_name, instance, count)

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> tuple[Proxy, ...]:
        g = tracer.graph

        # Register each block array as a param and compile block specs
        call_kwargs = {}
        for btype, (param_name, instance, count) in self._block_instances.items():
            base_name = type(instance).__name__
            # Use param_name as a unique key when the same Block class is used
            # with different configs (e.g., Gemma4SharedKVBlock with different head sizes)
            block_name = param_name.replace("_blocks", "").replace("_", "__") + "__" + base_name
            size_attr = f"n_{btype}_blocks"
            if param_name not in tracer.params:
                tracer.params[param_name] = ParamSpec(
                    name=param_name,
                    kind=ParamKind.ARRAY,
                    array_size=size_attr,
                    element_type=block_name,
                )
            # Compile the inner block to register it with a unique name
            instance.compile(name_override=block_name)
            call_kwargs[param_name] = param_name

        # Emit HybridStackedBlocks call
        input_refs = [a.ref for a in args]
        extra_attrs = {}
        if self.per_layer_input_name:
            extra_attrs["per_layer_input_name"] = self.per_layer_input_name
        if self.kv_sharing_map:
            extra_attrs["kv_sharing_map"] = self.kv_sharing_map
        result = g.call(
            "HybridStackedBlocks",
            *input_refs,
            num_outputs=2,
            block_types=self.block_types,
            n_layers=self.n_layers,
            **call_kwargs,
            **extra_attrs,
        )

        if isinstance(result, GraphRef):
            return (Proxy("stacked_out", result),)
        return tuple(Proxy(f"stacked_out_{i}", r) for i, r in enumerate(result))


# ============================================================================
# Module registry bootstrap
# ============================================================================
#
# Import the ``modules`` package at the bottom of ``nn.py`` (after
# ``Module`` / ``Tracer`` / ``Proxy`` / ``Block`` / ``Model`` etc. are
# defined, so ``modules/*.py`` can import from a fully-initialized ``nn``).
# Each ``modules/*.py`` file self-registers its ``Attention`` / ``MLP``
# spec on import, so this import also populates those registries.
#
# This module does *not* re-export runtime module classes. Import them
# directly from ``surogate.dsl.modules`` (or a submodule) where needed.

from . import modules as _modules  # noqa: E402, F401
