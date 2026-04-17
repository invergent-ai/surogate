"""
Module DSL - Domain-Specific Language for Neural Network Architecture Definition

This package implements a Python decorator-based DSL for defining neural network
architectures with explicit forward and backward computation graphs.

Example usage:
    from surogate.dsl import module, block, model, param, forward, Tensor, graph

    @module
    class Linear:
        def __init__(self, in_dim: int, out_dim: int):
            self.in_dim = in_dim
            self.out_dim = out_dim

        @param
        def weight(self) -> Tensor["out_dim", "in_dim"]:
            ...

        @forward
        def forward(self, x: Tensor["B", "T", "in_dim"]) -> Tensor["B", "T", "out_dim"]:
            with graph() as g:
                y = g.matmul(x, "weight", transpose="NT")
                return y

Key components:
- Decorators: @module, @block, @model, @primitive, @param, @forward, @backward
- Types: Tensor[...] annotations for shape specification
- Graph Builder: Context manager for defining computation graphs
- HF Mapping: Utilities for HuggingFace weight mapping (fuse, split, transform)
- IR: Graph IR representation for C++ runtime
"""

# Import subpackages to ensure registration
from . import blocks, models, modules, primitives
from .decorators import (
    Activation,
    Gradient,
    Param,
    abstract,
    backward,
    block,
    extends,
    forward,
    hf_config,
    hf_export,
    hf_mapping,
    model,
    module,
    param,
    primitive,
    recompute,
    save,
    tied_to,
)

# First-class dimension types
from .dim import (
    B,
    ConcreteDimValue,
    Dim,
    DimExpr,
    T,
    dim_to_ir,
)
from .errors import (
    DSLConstraintError,
    DSLError,
    DSLResolutionError,
    DSLShapeError,
    DSLSyntaxError,
    DSLTypeError,
    DSLUndefinedError,
    WarningCollector,
)
from .graph_builder import GraphBuilder, GraphRef, graph
from .hf import fuse, split, transform
from .hf import tied_to as hf_tied_to
from .ir import (
    GraphIR,
    KernelType,
    ModuleIR,
    OpNode,
    ScheduleIR,
    TensorRef,
)
from .py_compiler import (
    compile_model,
    compile_model_for_hf,
    get_block_spec,
    get_model_spec,
    get_module_spec,
    list_registered_blocks,
    list_registered_models,
    list_registered_modules,
)
from .py_lowering import (
    lower_block_spec,
    lower_graph_builder,
    lower_model_spec,
    lower_module_spec,
)
from .py_registry import (
    get_module as py_get_module,
)
from .py_registry import (
    get_primitive as py_get_primitive,
)
from .py_registry import (
    list_modules as py_list_modules,
)
from .py_registry import (
    registry,
)
from .specs import (
    ActivationLayoutSpec,
    ActivationMemoryHint,
    ActivationScope,
    ActivationSlotSpec,
    BackwardSpec,
    BlockSpec,
    ForwardSpec,
    ModelSpec,
    ModuleSpec,
    ParamSpec,
    PrimitiveSpec,
)

# Python DSL (decorator-based)
from .tensor_type import Array, Tensor, TensorType
from .types import (
    ComputedDim,
    ConcreteDim,
    Dtype,
    HookMode,
    HookPoint,
    MemoryMode,
    Shape,
    ShardStrategy,
    SymbolicDim,
    TensorTypeSpec,
    VariadicDim,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Dtype",
    "SymbolicDim",
    "ConcreteDim",
    "ComputedDim",
    "VariadicDim",
    "Shape",
    "TensorTypeSpec",
    "MemoryMode",
    "HookPoint",
    "HookMode",
    "ShardStrategy",
    # First-class dimensions
    "Dim",
    "DimExpr",
    "ConcreteDimValue",
    "B",
    "T",
    "dim_to_ir",
    # IR
    "GraphIR",
    "ModuleIR",
    "ScheduleIR",
    "OpNode",
    "TensorRef",
    "KernelType",
    # Errors
    "DSLError",
    "DSLSyntaxError",
    "DSLTypeError",
    "DSLShapeError",
    "DSLResolutionError",
    "DSLUndefinedError",
    "DSLConstraintError",
    "WarningCollector",
    # Types
    "Tensor",
    "TensorType",
    "Array",
    # Decorators
    "module",
    "block",
    "model",
    "primitive",
    "param",
    "Param",
    "Activation",
    "Gradient",
    "forward",
    "backward",
    "save",
    "recompute",
    "hf_config",
    "hf_mapping",
    "hf_export",
    "abstract",
    "extends",
    "tied_to",
    # Graph
    "graph",
    "GraphBuilder",
    "GraphRef",
    # HF utilities
    "fuse",
    "split",
    "transform",
    "hf_tied_to",
    # Specs
    "ModuleSpec",
    "BlockSpec",
    "ModelSpec",
    "PrimitiveSpec",
    "ParamSpec",
    "ForwardSpec",
    "BackwardSpec",
    "ActivationSlotSpec",
    "ActivationLayoutSpec",
    "ActivationScope",
    "ActivationMemoryHint",
    # Registry
    "registry",
    "py_get_module",
    "py_get_primitive",
    "py_list_modules",
    # Lowering
    "lower_module_spec",
    "lower_block_spec",
    "lower_model_spec",
    "lower_graph_builder",
    # Compiler
    "compile_model",
    "compile_model_for_hf",
    "get_model_spec",
    "get_block_spec",
    "get_module_spec",
    "list_registered_models",
    "list_registered_blocks",
    "list_registered_modules",
    # Subpackages
    "primitives",
    "modules",
    "blocks",
    "models",
]
