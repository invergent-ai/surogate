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
    """Extract parameter specs from @param decorated methods."""
    params: dict[str, ParamSpec] = {}

    for name in dir(cls):
        if name.startswith("_"):
            continue

        attr = getattr(cls, name, None)
        if attr is None:
            continue

        # Check if it has _param_spec attached by @param decorator
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
# Parameter Decorator
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
