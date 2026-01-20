"""
Specification Dataclasses for Python DSL

These specs are the intermediate representation between decorated Python classes
and the final IR. They capture all information needed to generate GraphIR.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .tensor_type import TensorAnnotation, ArrayAnnotation
    from .graph_builder import GraphBuilder


class ParamKind(str, Enum):
    """Kind of module parameter."""
    TENSOR = "tensor"           # Weight tensor [shape]
    MODULE = "module"           # Submodule instance
    ARRAY = "array"             # Array of tensors/modules [n] x Type
    TIED = "tied"               # Tied to another parameter


@dataclass
class ParamSpec:
    """Specification for a module parameter (weight, bias, submodule, etc.)."""

    name: str
    kind: ParamKind = ParamKind.TENSOR

    # For TENSOR kind
    shape: tuple[str | int, ...] | None = None
    dtype: str = "bf16"

    # For MODULE kind
    module_type: str | None = None
    module_args: tuple[Any, ...] = ()
    module_kwargs: dict[str, Any] = field(default_factory=dict)

    # For ARRAY kind
    array_size: str | int | None = None
    element_type: str | None = None

    # For TIED kind
    tied_to: str | None = None

    # Common attributes
    condition: Callable[[Any], bool] | None = None  # lambda self: self.use_bias
    optional: bool = False
    frozen: bool = False  # @frozen - precomputed, not trained

    # HuggingFace mapping
    hf_path: str | None = None
    hf_transform: HFTransformSpec | None = None

    # Annotations
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class HFTransformSpec:
    """Specification for HuggingFace weight transformation."""
    kind: str  # "fuse", "split", "transform", "tied_to"
    sources: list[str] = field(default_factory=list)
    dim: int = 0
    fn: str | None = None  # For transform
    ranges: list[tuple[int, int]] | None = None  # For split


@dataclass
class IOSpec:
    """Input/output specification for forward/backward."""
    name: str
    tensor_type: TensorAnnotation
    is_optional: bool = False
    default: Any = None


@dataclass
class ForwardSpec:
    """Specification for a forward pass."""

    # Input/output signatures (from type hints)
    inputs: list[IOSpec] = field(default_factory=list)
    outputs: list[IOSpec] = field(default_factory=list)

    # The graph builder function (captures the computation)
    graph_fn: Callable[[Any, GraphBuilder], Any] | None = None

    # Memory directives
    save: list[str] = field(default_factory=list)
    recompute: list[str] = field(default_factory=list)


@dataclass
class BackwardSpec:
    """Specification for a backward pass."""

    # Gradient inputs (d_out, etc.) and outputs (d_in, d_weight, etc.)
    gradient_inputs: list[IOSpec] = field(default_factory=list)
    gradient_outputs: list[IOSpec] = field(default_factory=list)

    # The graph builder function
    graph_fn: Callable[[Any, GraphBuilder], Any] | None = None

    # What tensors from forward are available
    saved_tensors: list[str] = field(default_factory=list)


@dataclass
class ConstraintSpec:
    """Compile-time constraint specification."""
    condition: str  # Expression string: "C % H == 0"
    message: str


@dataclass
class LetBindingSpec:
    """Let binding specification."""
    name: str
    expression: str  # "d_model // num_heads"


@dataclass
class HFConfigSpec:
    """HuggingFace config mapping specification."""
    architecture: str  # "Qwen3ForCausalLM"
    model_type: str | None = None  # "qwen3"
    config_class: str | None = None  # "Qwen3Config"
    param_mapping: dict[str, str] = field(default_factory=dict)  # d_model -> hidden_size
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class HFMappingSpec:
    """HuggingFace weight mapping specification."""
    mappings: dict[str, str | HFTransformSpec] = field(default_factory=dict)
    # Key: internal param name (can include {layer} placeholder)
    # Value: HF path or transform spec


@dataclass
class BaseModuleSpec:
    """Base specification for module-like constructs."""

    name: str
    python_class: type | None = None

    # Constructor parameters (d_model: int, use_bias: bool = False)
    constructor_params: dict[str, tuple[type | None, Any]] = field(default_factory=dict)
    # name -> (type_hint, default_value)

    # Let bindings
    let_bindings: list[LetBindingSpec] = field(default_factory=list)

    # Constraints
    constraints: list[ConstraintSpec] = field(default_factory=list)

    # Weight/submodule parameters
    params: dict[str, ParamSpec] = field(default_factory=dict)

    # Forward/backward
    forward: ForwardSpec | None = None
    backward: BackwardSpec | None = None

    # Docstring
    docstring: str | None = None

    # Annotations
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleSpec(BaseModuleSpec):
    """Specification for a module declaration."""

    extends: str | None = None
    is_abstract: bool = False


@dataclass
class BlockSpec(BaseModuleSpec):
    """Specification for a block declaration (transformer block pattern)."""

    extends: str | None = None

    # Pattern-based definition (alternative to explicit forward/backward)
    pattern: str | None = None  # "sequential_residual", "parallel_residual"
    pattern_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec(BaseModuleSpec):
    """Specification for a model declaration (top-level architecture)."""

    # HuggingFace integration
    hf_config: HFConfigSpec | None = None
    hf_mapping: HFMappingSpec | None = None
    hf_export: HFMappingSpec | None = None


@dataclass
class PrimitiveIOSpec:
    """Input/output specification for primitives."""
    # Can be named tuple: (A: [M, K], B: [K, N])
    # Or single tensor: [M, N]
    # Or empty: ()
    named_tensors: dict[str, TensorAnnotation] | None = None
    single_tensor: TensorAnnotation | None = None
    is_empty: bool = False


@dataclass
class PrimitiveSpec:
    """Specification for a primitive operation (CUDA kernel wrapper)."""

    name: str
    python_fn: Callable | None = None

    # Docstring
    docstring: str | None = None

    # Primitive parameters (transpose: enum, accumulate: bool, etc.)
    params: dict[str, tuple[type | None, Any]] = field(default_factory=dict)

    # Forward signature
    forward_in: PrimitiveIOSpec | None = None
    forward_out: PrimitiveIOSpec | None = None

    # Backward signature
    backward_in: PrimitiveIOSpec | None = None
    backward_out: PrimitiveIOSpec | None = None

    # What to save for backward
    save: list[str] = field(default_factory=list)

    # What to recompute
    recompute: list[str] = field(default_factory=list)

    # Implementation references
    forward_impl: str | None = None  # "kernels.matmul"
    backward_impl: str | None = None

    # Invariants
    invariants: dict[str, list[str]] = field(default_factory=dict)

    # Memory info
    memory_info: dict[str, Any] = field(default_factory=dict)

    # Precision info
    precision_info: dict[str, Any] = field(default_factory=dict)

    # Optimization hints
    optimization_info: dict[str, Any] = field(default_factory=dict)

    # Fusion patterns
    fusion_patterns: list[tuple[list[str], str]] = field(default_factory=list)


@dataclass
class RecipeSpec:
    """Specification for a precision recipe."""

    name: str
    settings: dict[str, Any] = field(default_factory=dict)
