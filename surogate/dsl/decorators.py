"""
Decorators for Python DSL

Provides decorators to define modules, blocks, models, and primitives using
native Python class syntax with type annotations.

Example:
    @module
    class Linear:
        def __init__(self, in_dim: int, out_dim: int, use_bias: bool = False):
            ...

        @param
        def weight(self) -> Tensor["out_dim", "in_dim"]:
            ...

        @forward
        def forward(self, x: Tensor["B", "T", "in_dim"]) -> Tensor["B", "T", "out_dim"]:
            ...
"""

from __future__ import annotations
import inspect
import functools
from typing import (
    Any,
    Callable,
    TypeVar,
    overload,
    get_type_hints,
    TYPE_CHECKING,
)

from .tensor_type import (
    TensorAnnotation,
    ArrayAnnotation,
    extract_tensor_annotation,
    extract_array_annotation,
)
from .specs import (
    ModuleSpec,
    BlockSpec,
    ModelSpec,
    PrimitiveSpec,
    PrimitiveIOSpec,
    ParamSpec,
    ParamKind,
    ForwardSpec,
    BackwardSpec,
    IOSpec,
    LetBindingSpec,
    ConstraintSpec,
    HFConfigSpec,
    HFMappingSpec,
    HFTransformSpec,
    ActivationSlotSpec,
    ActivationLayoutSpec,
    ActivationScope,
    ActivationMemoryHint,
    SharePolicy,
)

if TYPE_CHECKING:
    from .graph_builder import GraphBuilder


T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Registry (forward declaration, actual implementation in py_registry.py)
# =============================================================================

_module_registry: dict[str, ModuleSpec] = {}
_block_registry: dict[str, BlockSpec] = {}
_model_registry: dict[str, ModelSpec] = {}
_primitive_registry: dict[str, PrimitiveSpec] = {}


# =============================================================================
# Module/Block/Model Decorators
# =============================================================================


def _extract_constructor_params(cls: type) -> dict[str, tuple[type | None, Any]]:
    """Extract constructor parameters from __init__ signature."""
    params: dict[str, tuple[type | None, Any]] = {}

    if not hasattr(cls, "__init__"):
        return params

    sig = inspect.signature(cls.__init__)
    hints = {}
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        pass

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        type_hint = hints.get(name)
        default = param.default if param.default is not inspect.Parameter.empty else None

        params[name] = (type_hint, default)

    return params


def _extract_param_specs(cls: type) -> dict[str, ParamSpec]:
    """Extract parameter specs from @param decorated methods and Param class attributes."""
    params: dict[str, ParamSpec] = {}

    for name in dir(cls):
        if name.startswith("_"):
            continue

        attr = getattr(cls, name, None)
        if attr is None:
            continue

        # Check for Param class attribute (new style)
        if isinstance(attr, Param):
            spec = attr.to_spec(name)
            params[name] = spec
            continue

        # Check if it has _param_spec attached by @param decorator (old style)
        if hasattr(attr, "_param_spec"):
            spec: ParamSpec = attr._param_spec
            spec.name = name  # Ensure name matches method name
            params[name] = spec

    return params


def _extract_forward_spec(cls: type) -> ForwardSpec | None:
    """Extract forward spec from @forward decorated method."""
    for name in dir(cls):
        attr = getattr(cls, name, None)
        if attr is not None and hasattr(attr, "_forward_spec"):
            return attr._forward_spec
    return None


def _extract_backward_spec(cls: type) -> BackwardSpec | None:
    """Extract backward spec from @backward decorated method."""
    for name in dir(cls):
        attr = getattr(cls, name, None)
        if attr is not None and hasattr(attr, "_backward_spec"):
            return attr._backward_spec
    return None


def _extract_let_bindings(cls: type) -> list[LetBindingSpec]:
    """Extract let bindings from class-level annotations or _let_ dict."""
    bindings: list[LetBindingSpec] = []

    if hasattr(cls, "_let_"):
        let_dict = cls._let_
        for name, expr in let_dict.items():
            bindings.append(LetBindingSpec(name=name, expression=str(expr)))

    return bindings


def _extract_constraints(cls: type) -> list[ConstraintSpec]:
    """Extract constraints from _constraints_ list."""
    constraints: list[ConstraintSpec] = []

    if hasattr(cls, "_constraints_"):
        for item in cls._constraints_:
            if isinstance(item, tuple) and len(item) == 2:
                constraints.append(ConstraintSpec(condition=item[0], message=item[1]))

    return constraints


def _extract_activation_layout(cls: type) -> ActivationLayoutSpec | None:
    """Extract activation layout from Activation and Gradient class attributes.

    Scans the class for Activation and Gradient descriptors and builds an
    ActivationLayoutSpec containing all declared slots.
    """
    forward_slots = []
    gradient_slots = []

    for name in dir(cls):
        if name.startswith("_"):
            continue

        attr = getattr(cls, name, None)
        if attr is None:
            continue

        if isinstance(attr, Activation):
            forward_slots.append(attr.to_spec(name))
        elif isinstance(attr, Gradient):
            gradient_slots.append(attr.to_spec(name))

    # Only create layout if we have any slots
    if not forward_slots and not gradient_slots:
        return None

    return ActivationLayoutSpec(
        name=f"{cls.__name__}Activations",
        slots=forward_slots,
        gradient_slots=gradient_slots,
    )


def _process_module_class(cls: type, spec_class: type) -> Any:
    """Process a class decorated with @module, @block, or @model."""

    # Build the spec
    spec = spec_class(
        name=cls.__name__,
        python_class=cls,
        docstring=cls.__doc__,
        constructor_params=_extract_constructor_params(cls),
        let_bindings=_extract_let_bindings(cls),
        constraints=_extract_constraints(cls),
        params=_extract_param_specs(cls),
        forward=_extract_forward_spec(cls),
        backward=_extract_backward_spec(cls),
    )

    # Handle extends
    if hasattr(cls, "_extends_"):
        spec.extends = cls._extends_

    # Handle abstract
    if hasattr(cls, "_abstract_") and spec_class == ModuleSpec:
        spec.is_abstract = cls._abstract_

    # Handle HF config/mapping for models
    if spec_class == ModelSpec:
        if hasattr(cls, "_hf_config_"):
            spec.hf_config = cls._hf_config_
        if hasattr(cls, "_hf_mapping_"):
            spec.hf_mapping = cls._hf_mapping_
        if hasattr(cls, "_hf_export_"):
            spec.hf_export = cls._hf_export_

    # Handle pattern for blocks
    if spec_class == BlockSpec:
        if hasattr(cls, "_pattern_"):
            spec.pattern = cls._pattern_
        if hasattr(cls, "_pattern_config_"):
            spec.pattern_config = cls._pattern_config_

    # Extract activation layout for blocks and models
    if spec_class in (BlockSpec, ModelSpec):
        activation_layout = _extract_activation_layout(cls)
        if activation_layout:
            spec.activations = activation_layout

    # Attach spec to class
    cls._dsl_spec = spec

    # Register
    if spec_class == ModuleSpec:
        _module_registry[cls.__name__] = spec
    elif spec_class == BlockSpec:
        _block_registry[cls.__name__] = spec
    elif spec_class == ModelSpec:
        _model_registry[cls.__name__] = spec

    return cls


def module(cls: T) -> T:
    """Decorator to define a module.

    Example:
        @module
        class Linear:
            def __init__(self, in_dim: int, out_dim: int):
                self.in_dim = in_dim
                self.out_dim = out_dim

            @param
            def weight(self) -> Tensor["out_dim", "in_dim"]:
                ...
    """
    return _process_module_class(cls, ModuleSpec)


def block(cls: T) -> T:
    """Decorator to define a block (transformer block pattern).

    Example:
        @block
        class DenseTransformerBlock:
            def __init__(self, d_model: int, num_heads: int, d_ff: int):
                ...
    """
    return _process_module_class(cls, BlockSpec)


def model(cls: T) -> T:
    """Decorator to define a model (top-level architecture).

    Example:
        @model
        @hf_config(architecture="Qwen3ForCausalLM", ...)
        class Qwen3Model:
            ...
    """
    return _process_module_class(cls, ModelSpec)


def abstract(cls: T) -> T:
    """Mark a module as abstract (no implementation, just interface)."""
    cls._abstract_ = True
    return cls


def extends(base_name: str) -> Callable[[T], T]:
    """Decorator to specify module inheritance."""
    def decorator(cls: T) -> T:
        cls._extends_ = base_name
        return cls
    return decorator


# =============================================================================
# Parameter Declaration (Class Attribute Style)
# =============================================================================


class Param:
    """Lightweight parameter declaration for class attributes.

    Allows declaring parameters as class attributes instead of methods:

        class Qwen3Block:
            # Tensor parameters
            ln1_weight = Param(Tensor["C"])
            qkv_weight = Param(Tensor["QKV", "C"])
            qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
            rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

        class Qwen3Model:
            # Array parameters (stacked blocks)
            blocks = Param(Array["n_layers", "Qwen3Block"])

    Args:
        param_type: A Tensor[...] or Array[...] annotation
        when: Condition for optional parameters. Can be:
            - str: attribute name to check (e.g., "use_bias")
            - callable: lambda self: self.use_bias
        frozen: If True, this parameter is precomputed and not trained
        hf_mapping: HuggingFace weight path for import
    """

    def __init__(
        self,
        param_type: TensorAnnotation | ArrayAnnotation,
        *,
        when: str | Callable[[Any], bool] | None = None,
        frozen: bool = False,
        hf_mapping: str | None = None,
    ):
        if not isinstance(param_type, (TensorAnnotation, ArrayAnnotation)):
            raise TypeError(
                f"Param() requires a Tensor[...] or Array[...] annotation, "
                f"got {type(param_type).__name__}. "
                f"Usage: Param(Tensor[\"C\", \"D\"]) or Param(Array[\"n_layers\", \"Block\"])"
            )
        self.param_type = param_type
        self.when = when
        self.frozen = frozen
        self.hf_mapping = hf_mapping
        # Name will be set by __set_name__
        self._name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the attribute is assigned to a class."""
        self._name = name

    def __repr__(self) -> str:
        parts = [f"Param({self.param_type!r}"]
        if self.when is not None:
            parts.append(f", when={self.when!r}")
        if self.frozen:
            parts.append(", frozen=True")
        if self.hf_mapping:
            parts.append(f", hf_mapping={self.hf_mapping!r}")
        parts.append(")")
        return "".join(parts)

    def to_spec(self, name: str) -> ParamSpec:
        """Convert to ParamSpec for the compilation pipeline."""
        spec = ParamSpec(name=name)

        if isinstance(self.param_type, TensorAnnotation):
            spec.kind = ParamKind.TENSOR
            spec.shape = self.param_type.dims
            spec.dtype = self.param_type.dtype
            spec.optional = self.param_type.optional
        elif isinstance(self.param_type, ArrayAnnotation):
            spec.kind = ParamKind.ARRAY
            spec.array_size = self.param_type.size
            spec.element_type = self.param_type.element_type

        spec.frozen = self.frozen
        spec.hf_path = self.hf_mapping

        # Handle condition
        if self.when is not None:
            if isinstance(self.when, str):
                # Convert string attribute name to lambda that accesses self.attr directly
                # (no default - let it raise AttributeError like @param style, which
                # causes the exception handler to include the param)
                attr_name = self.when
                spec.condition = lambda self, attr=attr_name: getattr(self, attr)
            else:
                spec.condition = self.when

        return spec


class Activation:
    """Lightweight activation slot declaration for class attributes.

    Allows declaring activation slots as class attributes in @block definitions.
    These generate pre-allocated tensor buffers for forward/backward passes,
    eliminating hardcoded tensor name → slot mappings in the C++ runtime.

    Example:
        @block
        class TransformerBlock:
            # Forward activation slots
            ln1 = Activation(Tensor["B", "T", "C"])
            ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)
            qkv = Activation(Tensor["B", "T", "QKV"], aliases=["qkv_flat", "qkv_biased"])
            lse = Activation(Tensor["B", "Hq", "T"], dtype="fp32", save=True)
            att = Activation(Tensor["B", "T", "AttnDim"])
            mlp_up = Activation(Tensor["B", "T", "MUp"])
            swiglu = Activation(Tensor["B", "T", "M"])

    Args:
        tensor_type: A Tensor[...] annotation specifying the shape
        dtype: Override dtype (default: inherit from runtime)
        aliases: Alternative names that map to this slot (e.g., "qkv_flat" -> "qkv")
        save: If True, save this tensor for backward pass
        recompute: If True, this tensor can be recomputed in backward
        shares_with: Name of another slot to share memory with
        when: Condition for optional slots (e.g., "use_qk_norm")
        scope: "block" (default), "global", "gradient", or "global_gradient"
        description: Documentation for this slot
    """

    def __init__(
        self,
        tensor_type: TensorAnnotation,
        *,
        dtype: str | None = None,
        aliases: list[str] | None = None,
        save: bool = False,
        recompute: bool = False,
        shares_with: str | None = None,
        recompute_from: list[str] | None = None,
        recompute_op: str | None = None,
        recompute_attrs: dict[str, Any] | None = None,
        recompute_policy: str = "always",
        recompute_group: str | None = None,
        recompute_outputs: list[str] | None = None,
        lora_targets: list[str] | None = None,
        share_policy: str | SharePolicy = "when_recomputed",
        when: str | Callable[[Any], bool] | None = None,
        scope: str = "block",
        description: str | None = None,
    ):
        if not isinstance(tensor_type, TensorAnnotation):
            raise TypeError(
                f"Activation() requires a Tensor[...] annotation, "
                f"got {type(tensor_type).__name__}. "
                f"Usage: Activation(Tensor[\"B\", \"T\", \"C\"])"
            )
        self.tensor_type = tensor_type
        self.dtype = dtype
        self.aliases = aliases or []
        self.save = save
        self.recompute = recompute
        self.shares_with = shares_with
        self.recompute_from = recompute_from or []
        self.recompute_op = recompute_op
        self.recompute_attrs = recompute_attrs or {}
        self.recompute_policy = recompute_policy
        self.recompute_group = recompute_group
        self.recompute_outputs = recompute_outputs or []
        self.lora_targets = lora_targets or []
        # Normalize share_policy to string
        if isinstance(share_policy, SharePolicy):
            self.share_policy = share_policy.value
        else:
            self.share_policy = share_policy
        self.when = when
        self.scope = scope
        self.description = description
        # Name will be set by __set_name__
        self._name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the attribute is assigned to a class."""
        self._name = name

    def __repr__(self) -> str:
        parts = [f"Activation({self.tensor_type!r}"]
        if self.dtype:
            parts.append(f", dtype={self.dtype!r}")
        if self.aliases:
            parts.append(f", aliases={self.aliases!r}")
        if self.save:
            parts.append(", save=True")
        if self.recompute:
            parts.append(", recompute=True")
        if self.recompute_from:
            parts.append(f", recompute_from={self.recompute_from!r}")
        if self.recompute_op:
            parts.append(f", recompute_op={self.recompute_op!r}")
        if self.recompute_attrs:
            parts.append(f", recompute_attrs={self.recompute_attrs!r}")
        if self.recompute_policy != "always":
            parts.append(f", recompute_policy={self.recompute_policy!r}")
        if self.recompute_group:
            parts.append(f", recompute_group={self.recompute_group!r}")
        if self.recompute_outputs:
            parts.append(f", recompute_outputs={self.recompute_outputs!r}")
        if self.lora_targets:
            parts.append(f", lora_targets={self.lora_targets!r}")
        if self.share_policy != "when_recomputed":
            parts.append(f", share_policy={self.share_policy!r}")
        if self.shares_with:
            parts.append(f", shares_with={self.shares_with!r}")
        if self.when:
            parts.append(f", when={self.when!r}")
        if self.scope != "block":
            parts.append(f", scope={self.scope!r}")
        parts.append(")")
        return "".join(parts)

    def to_spec(self, name: str) -> ActivationSlotSpec:
        """Convert to ActivationSlotSpec for the compilation pipeline."""
        # Map scope string to enum
        scope_map = {
            "block": ActivationScope.BLOCK,
            "global": ActivationScope.GLOBAL,
            "gradient": ActivationScope.GRADIENT,
            "global_gradient": ActivationScope.GLOBAL_GRADIENT,
        }
        scope_enum = scope_map.get(self.scope, ActivationScope.BLOCK)

        # Map share_policy string to enum
        share_policy_map = {
            "per_layer": SharePolicy.PER_LAYER,
            "when_recomputed": SharePolicy.WHEN_RECOMPUTED,
            "always_share": SharePolicy.ALWAYS_SHARE,
            "fft_share": SharePolicy.FFT_SHARE,
            "lora_share": SharePolicy.LORA_SHARE,
        }
        share_policy_enum = share_policy_map.get(self.share_policy, SharePolicy.WHEN_RECOMPUTED)

        # Determine memory hint
        if self.shares_with:
            memory_hint = ActivationMemoryHint.SHARED
        elif self.save:
            memory_hint = ActivationMemoryHint.SAVE
        elif self.recompute:
            memory_hint = ActivationMemoryHint.RECOMPUTE
        else:
            memory_hint = ActivationMemoryHint.PERSISTENT

        # Build condition lambda if needed
        condition = None
        condition_expr = None
        if self.when is not None:
            if isinstance(self.when, str):
                attr_name = self.when
                condition_expr = attr_name
                condition = lambda self, attr=attr_name: getattr(self, attr)
            else:
                condition = self.when

        return ActivationSlotSpec(
            name=name,
            scope=scope_enum,
            shape=self.tensor_type.dims or (),
            dtype=self.dtype or self.tensor_type.dtype,
            aliases=list(self.aliases),
            memory_hint=memory_hint,
            shares_with=self.shares_with,
            save_for_backward=self.save,
            recompute_in_backward=self.recompute,
            recompute_from=list(self.recompute_from),
            recompute_op=self.recompute_op,
            recompute_attrs=dict(self.recompute_attrs),
            recompute_policy=self.recompute_policy,
            recompute_group=self.recompute_group,
            recompute_outputs=list(self.recompute_outputs),
            lora_targets=list(self.lora_targets),
            share_policy=share_policy_enum,
            condition=condition,
            condition_expr=condition_expr,
            description=self.description,
        )


class Gradient:
    """Lightweight gradient slot declaration for class attributes.

    Similar to Activation but for gradient buffers. Use this in @block definitions
    to declare gradient tensors needed during backward pass.

    Example:
        @block
        class TransformerBlock:
            # Gradient slots (typically match activation shapes)
            d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
            d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
            d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")

    Args:
        tensor_type: A Tensor[...] annotation specifying the shape
        gradient_of: Name of the forward activation this is the gradient of
        dtype: Override dtype (default: inherit from runtime)
        shares_with: Name of another slot to share memory with
        alias_of: Name of an existing activation slot to alias (reuse) at runtime
        when: Condition for optional slots
        scope: "gradient" (default) or "global" for model-level gradient slots
        description: Documentation for this slot
    """

    def __init__(
        self,
        tensor_type: TensorAnnotation,
        *,
        gradient_of: str,
        dtype: str | None = None,
        shares_with: str | None = None,
        alias_of: str | None = None,
        when: str | Callable[[Any], bool] | None = None,
        scope: str = "gradient",
        description: str | None = None,
    ):
        if not isinstance(tensor_type, TensorAnnotation):
            raise TypeError(
                f"Gradient() requires a Tensor[...] annotation, "
                f"got {type(tensor_type).__name__}."
            )
        self.tensor_type = tensor_type
        self.gradient_of = gradient_of
        self.dtype = dtype
        self.shares_with = shares_with
        self.alias_of = alias_of
        self.when = when
        self.scope = scope
        self.description = description
        self._name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        parts = [f"Gradient({self.tensor_type!r}, gradient_of={self.gradient_of!r}"]
        if self.dtype:
            parts.append(f", dtype={self.dtype!r}")
        if self.shares_with:
            parts.append(f", shares_with={self.shares_with!r}")
        if self.alias_of:
            parts.append(f", alias_of={self.alias_of!r}")
        if self.when:
            parts.append(f", when={self.when!r}")
        parts.append(")")
        return "".join(parts)

    def to_spec(self, name: str) -> ActivationSlotSpec:
        """Convert to ActivationSlotSpec for the compilation pipeline."""
        memory_hint = ActivationMemoryHint.SHARED if self.shares_with else ActivationMemoryHint.PERSISTENT

        condition = None
        condition_expr = None
        if self.when is not None:
            if isinstance(self.when, str):
                attr_name = self.when
                condition_expr = attr_name
                condition = lambda self, attr=attr_name: getattr(self, attr)
            else:
                condition = self.when

        # Map scope string to enum
        scope_map = {
            "gradient": ActivationScope.GRADIENT,
            "global": ActivationScope.GLOBAL_GRADIENT,
            "global_gradient": ActivationScope.GLOBAL_GRADIENT,
        }
        scope_enum = scope_map.get(self.scope, ActivationScope.GRADIENT)

        return ActivationSlotSpec(
            name=name,
            scope=scope_enum,
            shape=self.tensor_type.dims or (),
            dtype=self.dtype or self.tensor_type.dtype,
            memory_hint=memory_hint,
            shares_with=self.shares_with,
            alias_of=self.alias_of,
            gradient_of=self.gradient_of,
            condition=condition,
            condition_expr=condition_expr,
            description=self.description,
        )


# =============================================================================
# Parameter Decorator (Method Style)
# =============================================================================


@overload
def param(fn: F) -> F: ...

@overload
def param(
    *,
    condition: Callable[[Any], bool] | None = None,
    frozen: bool = False,
    hf_mapping: str | None = None,
) -> Callable[[F], F]: ...


def param(
    fn: F | None = None,
    *,
    condition: Callable[[Any], bool] | None = None,
    frozen: bool = False,
    hf_mapping: str | None = None,
) -> F | Callable[[F], F]:
    """Decorator to define a module parameter (weight, bias, submodule).

    Example:
        @param
        def weight(self) -> Tensor["out_dim", "in_dim"]:
            ...

        @param(condition=lambda self: self.use_bias)
        def bias(self) -> Tensor["out_dim"]:
            ...

        @param(frozen=True)
        def rope_freqs(self) -> Tensor["max_seq", "D // 2", 2, "fp32"]:
            ...

        @param(hf_mapping="model.embed_tokens.weight")
        def embedding(self) -> Tensor["vocab_size", "d_model"]:
            ...

        @param
        def blocks(self) -> Array["n_layers", "DenseTransformerBlock"]:
            ...
    """
    def decorator(fn: F) -> F:
        # Get return type annotation
        hints = {}
        try:
            hints = get_type_hints(fn)
        except Exception:
            pass

        return_hint = hints.get("return")
        spec = ParamSpec(name=fn.__name__)

        # Determine kind from return type
        tensor_ann = extract_tensor_annotation(return_hint)
        array_ann = extract_array_annotation(return_hint)

        if tensor_ann is not None:
            spec.kind = ParamKind.TENSOR
            spec.shape = tensor_ann.dims
            spec.dtype = tensor_ann.dtype
            spec.optional = tensor_ann.optional
        elif array_ann is not None:
            spec.kind = ParamKind.ARRAY
            spec.array_size = array_ann.size
            spec.element_type = array_ann.element_type
        elif isinstance(return_hint, str):
            # Module type reference
            spec.kind = ParamKind.MODULE
            spec.module_type = return_hint
        elif return_hint is not None:
            # Could be a class reference
            spec.kind = ParamKind.MODULE
            spec.module_type = getattr(return_hint, "__name__", str(return_hint))

        spec.condition = condition
        spec.frozen = frozen
        spec.hf_path = hf_mapping

        fn._param_spec = spec
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator


def tied_to(target: str) -> Callable[[F], F]:
    """Decorator to tie a parameter to another parameter.

    Example:
        @param
        @tied_to("embedding")
        def lm_head(self) -> Tensor["vocab_size", "d_model"]:
            ...
    """
    def decorator(fn: F) -> F:
        # Get or create param spec
        if not hasattr(fn, "_param_spec"):
            fn._param_spec = ParamSpec(name=fn.__name__)

        fn._param_spec.kind = ParamKind.TIED
        fn._param_spec.tied_to = target
        return fn

    return decorator


# =============================================================================
# Forward/Backward Decorators
# =============================================================================


def _extract_io_specs(fn: Callable) -> tuple[list[IOSpec], list[IOSpec]]:
    """Extract input and output specs from function signature."""
    inputs: list[IOSpec] = []
    outputs: list[IOSpec] = []

    sig = inspect.signature(fn)
    hints = {}
    try:
        hints = get_type_hints(fn)
    except Exception:
        pass

    # Extract inputs from parameters
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        type_hint = hints.get(name)
        tensor_ann = extract_tensor_annotation(type_hint)

        if tensor_ann is not None:
            inputs.append(IOSpec(
                name=name,
                tensor_type=tensor_ann,
                is_optional=tensor_ann.optional,
                default=param.default if param.default is not inspect.Parameter.empty else None,
            ))

    # Extract outputs from return type
    return_hint = hints.get("return")
    if return_hint is not None:
        # Could be single tensor or tuple
        origin = getattr(return_hint, "__origin__", None)
        if origin is tuple:
            # Multiple outputs
            args = getattr(return_hint, "__args__", ())
            for i, arg in enumerate(args):
                tensor_ann = extract_tensor_annotation(arg)
                if tensor_ann is not None:
                    outputs.append(IOSpec(
                        name=f"out_{i}",
                        tensor_type=tensor_ann,
                    ))
        else:
            # Single output
            tensor_ann = extract_tensor_annotation(return_hint)
            if tensor_ann is not None:
                outputs.append(IOSpec(
                    name="out",
                    tensor_type=tensor_ann,
                ))

    return inputs, outputs


def forward(fn: F) -> F:
    """Decorator to mark the forward pass method.

    The decorated method should use the graph() context manager to define
    the computation graph.

    Example:
        @forward
        def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
            with graph() as g:
                x_flat = g.view(x, shape=["B * T", "C"])
                y_flat = g.matmul(x_flat, self.weight, transpose="NT")
                y = g.view(y_flat, shape=["B", "T", "C"])
                return y
    """
    inputs, outputs = _extract_io_specs(fn)

    spec = ForwardSpec(
        inputs=inputs,
        outputs=outputs,
        graph_fn=fn,
    )

    # If @save / @recompute ran before @forward, they stored lists on the function.
    # Apply them now so either decorator order works.
    if hasattr(fn, "_save_list"):
        spec.save = list(fn._save_list)
    if hasattr(fn, "_recompute_list"):
        spec.recompute = list(fn._recompute_list)

    fn._forward_spec = spec
    return fn


def backward(fn: F) -> F:
    """Decorator to mark the backward pass method.

    Example:
        @backward
        def backward(self, d_out: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
            with graph() as g:
                ...
    """
    inputs, outputs = _extract_io_specs(fn)

    spec = BackwardSpec(
        gradient_inputs=inputs,
        gradient_outputs=outputs,
        graph_fn=fn,
    )

    fn._backward_spec = spec
    return fn


def save(*tensor_names: str) -> Callable[[F], F]:
    """Decorator to specify tensors to save for backward pass.

    Example:
        @forward
        @save("x", "weight")
        def forward(self, x):
            ...
    """
    def decorator(fn: F) -> F:
        if hasattr(fn, "_forward_spec"):
            fn._forward_spec.save = list(tensor_names)
        else:
            # Store for later when @forward is applied
            fn._save_list = list(tensor_names)
        return fn

    return decorator


def recompute(*tensor_names: str) -> Callable[[F], F]:
    """Decorator to specify tensors to recompute in backward pass.

    Example:
        @forward
        @recompute("hidden", "gate")
        def forward(self, x):
            ...
    """
    def decorator(fn: F) -> F:
        if hasattr(fn, "_forward_spec"):
            fn._forward_spec.recompute = list(tensor_names)
        else:
            fn._recompute_list = list(tensor_names)
        return fn

    return decorator


# =============================================================================
# HuggingFace Decorators
# =============================================================================


def hf_config(
    architecture: str,
    model_type: str | None = None,
    config_class: str | None = None,
    **param_mapping: str,
) -> Callable[[T], T]:
    """Decorator to specify HuggingFace config mapping.

    Example:
        @model
        @hf_config(
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            d_model="hidden_size",
            n_layers="num_hidden_layers",
        )
        class Qwen3Model:
            ...
    """
    def decorator(cls: T) -> T:
        cls._hf_config_ = HFConfigSpec(
            architecture=architecture,
            model_type=model_type,
            config_class=config_class,
            param_mapping=param_mapping,
        )
        return cls

    return decorator


def hf_mapping(**mappings: str) -> Callable[[T], T]:
    """Decorator to specify HuggingFace weight import mappings.

    Example:
        @model
        @hf_mapping(
            embedding="model.embed_tokens.weight",
            lm_head="lm_head.weight",
        )
        class Qwen3Model:
            ...

    For indexed mappings (blocks), use hf_mapping.indexed().
    """
    def decorator(cls: T) -> T:
        if not hasattr(cls, "_hf_mapping_"):
            cls._hf_mapping_ = HFMappingSpec()

        for internal_name, external_path in mappings.items():
            cls._hf_mapping_.mappings[internal_name] = external_path

        return cls

    return decorator


def hf_export(**mappings: str) -> Callable[[T], T]:
    """Decorator to specify HuggingFace weight export mappings.

    Example:
        @model
        @hf_export(
            embedding="model.embed_tokens.weight",
        )
        class Qwen3Model:
            ...
    """
    def decorator(cls: T) -> T:
        if not hasattr(cls, "_hf_export_"):
            cls._hf_export_ = HFMappingSpec()

        for internal_name, external_path in mappings.items():
            cls._hf_export_.mappings[internal_name] = external_path

        return cls

    return decorator


class _HFMappingIndexed:
    """Helper for indexed HF mappings (for block arrays)."""

    def __call__(
        self,
        param_name: str,
        index_var: str = "layer",
        **mappings: str,
    ) -> Callable[[T], T]:
        """Define indexed mappings for array parameters.

        Example:
            @hf_mapping.indexed("blocks", layer="layer",
                ln1_weight="model.layers.{layer}.input_layernorm.weight",
                qkv_weight=fuse(
                    "model.layers.{layer}.self_attn.q_proj.weight",
                    "model.layers.{layer}.self_attn.k_proj.weight",
                    "model.layers.{layer}.self_attn.v_proj.weight",
                    dim=0
                ),
            )
            class Qwen3Model:
                ...
        """
        def decorator(cls: T) -> T:
            if not hasattr(cls, "_hf_mapping_"):
                cls._hf_mapping_ = HFMappingSpec()

            for sub_param, external_path in mappings.items():
                # Create indexed key: "blocks[{layer}].ln1_weight"
                indexed_key = f"{param_name}[{{{index_var}}}].{sub_param}"
                cls._hf_mapping_.mappings[indexed_key] = external_path

            return cls

        return decorator


# Attach indexed helper to hf_mapping
hf_mapping.indexed = _HFMappingIndexed()


# =============================================================================
# Primitive Decorator
# =============================================================================


def _extract_primitive_io(hints: dict, prefix: str) -> PrimitiveIOSpec | None:
    """Extract primitive IO spec from type hints."""
    # Look for in_A, in_B style or just the return type
    named: dict[str, TensorAnnotation] = {}

    for name, hint in hints.items():
        if name.startswith(prefix + "_"):
            tensor_name = name[len(prefix) + 1:]
            tensor_ann = extract_tensor_annotation(hint)
            if tensor_ann:
                named[tensor_name] = tensor_ann

    if named:
        return PrimitiveIOSpec(named_tensors=named)

    return None


@overload
def primitive(fn: F) -> F: ...

@overload
def primitive(
    *,
    impl: str | None = None,
    backward_impl: str | None = None,
) -> Callable[[F], F]: ...


def primitive(
    fn: F | None = None,
    *,
    impl: str | None = None,
    backward_impl: str | None = None,
) -> F | Callable[[F], F]:
    """Decorator to define a primitive operation.

    Example:
        @primitive(impl="kernels.matmul")
        def matmul(
            A: Tensor["M", "K"],
            B: Tensor["K", "N"],
            *,
            transpose: TransposeMode = TransposeMode.NN,
        ) -> Tensor["M", "N"]:
            '''Matrix multiplication.'''
            ...

        @matmul.backward
        @save("A", "B")
        def matmul_backward(
            d_C: Tensor["M", "N"],
            A: Tensor["M", "K"],
            B: Tensor["K", "N"],
        ) -> tuple[Tensor["M", "K"], Tensor["K", "N"]]:
            ...
    """
    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)
        hints = {}
        try:
            hints = get_type_hints(fn)
        except Exception:
            pass

        # Extract primitive parameters (keyword-only args)
        params: dict[str, tuple[type | None, Any]] = {}
        input_tensors: dict[str, TensorAnnotation] = {}

        for name, param in sig.parameters.items():
            type_hint = hints.get(name)
            default = param.default if param.default is not inspect.Parameter.empty else None

            tensor_ann = extract_tensor_annotation(type_hint)
            if tensor_ann:
                input_tensors[name] = tensor_ann
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                params[name] = (type_hint, default)

        # Extract output
        return_hint = hints.get("return")
        output_tensors: dict[str, TensorAnnotation] = {}

        if return_hint is not None:
            origin = getattr(return_hint, "__origin__", None)
            if origin is tuple:
                args = getattr(return_hint, "__args__", ())
                for i, arg in enumerate(args):
                    tensor_ann = extract_tensor_annotation(arg)
                    if tensor_ann:
                        output_tensors[f"out_{i}"] = tensor_ann
            else:
                tensor_ann = extract_tensor_annotation(return_hint)
                if tensor_ann:
                    output_tensors["out"] = tensor_ann

        spec = PrimitiveSpec(
            name=fn.__name__,
            python_fn=fn,
            docstring=fn.__doc__,
            params=params,
            forward_in=PrimitiveIOSpec(named_tensors=input_tensors) if input_tensors else None,
            forward_out=PrimitiveIOSpec(named_tensors=output_tensors) if output_tensors else None,
            forward_impl=impl,
            backward_impl=backward_impl,
        )

        fn._primitive_spec = spec

        # Add backward method to allow @fn.backward
        def add_backward(backward_fn: F) -> F:
            backward_hints = {}
            try:
                backward_hints = get_type_hints(backward_fn)
            except Exception:
                pass

            backward_inputs: dict[str, TensorAnnotation] = {}
            backward_outputs: dict[str, TensorAnnotation] = {}

            backward_sig = inspect.signature(backward_fn)
            for name, param in backward_sig.parameters.items():
                type_hint = backward_hints.get(name)
                tensor_ann = extract_tensor_annotation(type_hint)
                if tensor_ann:
                    backward_inputs[name] = tensor_ann

            backward_return = backward_hints.get("return")
            if backward_return is not None:
                origin = getattr(backward_return, "__origin__", None)
                if origin is tuple:
                    args = getattr(backward_return, "__args__", ())
                    for i, arg in enumerate(args):
                        tensor_ann = extract_tensor_annotation(arg)
                        if tensor_ann:
                            backward_outputs[f"d_{i}"] = tensor_ann
                else:
                    tensor_ann = extract_tensor_annotation(backward_return)
                    if tensor_ann:
                        backward_outputs["d_out"] = tensor_ann

            spec.backward_in = PrimitiveIOSpec(named_tensors=backward_inputs)
            spec.backward_out = PrimitiveIOSpec(named_tensors=backward_outputs)

            # Handle @save decorator on backward
            if hasattr(backward_fn, "_save_list"):
                spec.save = backward_fn._save_list

            return backward_fn

        fn.backward = add_backward

        # Register
        _primitive_registry[fn.__name__] = spec

        return fn

    if fn is not None:
        return decorator(fn)
    return decorator
