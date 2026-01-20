"""
Module DSL - Domain-Specific Language for Neural Network Architecture Definition

This package implements the Module DSL parser, type checker, and compiler
for defining neural network architectures with explicit forward and backward
computation graphs.

Two syntaxes are supported:

1. **Lark DSL** (.module files): Custom grammar with arrow syntax
   ```
   module Linear(in_dim: int, out_dim: int):
       params:
           weight: [out_dim, in_dim]
       forward:
           in: [B, T, in_dim]
           out: [B, T, out_dim]
           graph:
               x -> matmul(weight, transpose=NT) -> y
   ```

2. **Python DSL** (decorators): Native Python with type annotations
   ```python
   from surogate.dsl import module, param, forward, Tensor, graph

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
   ```

Key components:
- Parser: Lark-based parser for .module DSL syntax
- Python DSL: Decorator-based definitions with Tensor[...] annotations
- AST: Typed AST node definitions
- Types: Tensor types, dtypes, symbolic dimensions
- Resolver: Import resolution, type inference, shape validation
- IR: Graph IR and Schedule IR representations
- Compiler: Main compilation entry point

Example usage (Lark DSL):
    from surogate.dsl import compile_module, parse_file

    # Parse a DSL file
    program = parse_file("models/llama.module")

    # Compile to Graph IR
    result = compile_module("models/llama.module")

Example usage (Python DSL):
    from surogate.dsl import module, block, model, param, forward, Tensor, graph

    @module
    class MyModule:
        ...
"""

from .ast_nodes import (
    ModuleNode,
    BlockNode,
    ModelNode,
    PrimitiveNode,
    GraphStatement,
    Operation,
    TensorType,
    Annotation,
    ParamDecl,
    ForwardBlock,
    BackwardBlock,
    Program,
)

from .types import (
    Dtype,
    SymbolicDim,
    ConcreteDim,
    ComputedDim,
    VariadicDim,
    Shape,
    TensorTypeSpec,
    MemoryMode,
    HookPoint,
    HookMode,
    ShardStrategy,
)

from .parser import (
    parse_source,
    parse_file,
    ModuleDSLParser,
)

from .compiler import (
    compile_module,
    compile_and_lower,
    validate_source,
    Compiler,
    CompilerOptions,
    CompilationResult,
    ModuleRegistry,
    get_registry,
)

from .resolver import (
    ModuleResolver,
    ResolverContext,
    ResolvedModule,
    resolve_program,
)

from .ir import (
    GraphIR,
    ModuleIR,
    ScheduleIR,
    OpNode,
    TensorRef,
    KernelType,
)

from .lowering import (
    lower_program,
    ModuleLowerer,
)

from .errors import (
    DSLError,
    DSLSyntaxError,
    DSLTypeError,
    DSLShapeError,
    DSLResolutionError,
    DSLUndefinedError,
    DSLConstraintError,
    WarningCollector,
)

# Python DSL (decorator-based)
from .tensor_type import Tensor, TensorType, Array
from .decorators import (
    module,
    block,
    model,
    primitive,
    param,
    forward,
    backward,
    save,
    recompute,
    hf_config,
    hf_mapping,
    hf_export,
    abstract,
    extends,
    tied_to,
)
from .graph_builder import graph, GraphBuilder, GraphRef
from .hf import fuse, split, transform
from .hf import tied_to as hf_tied_to
from .specs import (
    ModuleSpec,
    BlockSpec,
    ModelSpec,
    PrimitiveSpec,
    ParamSpec,
    ForwardSpec,
    BackwardSpec,
)
from .py_registry import (
    registry,
    get_module as py_get_module,
    get_primitive as py_get_primitive,
    list_modules as py_list_modules,
)
from .py_lowering import (
    lower_module_spec,
    lower_block_spec,
    lower_model_spec,
    lower_graph_builder,
)
from .py_compiler import (
    compile_model,
    compile_model_for_hf,
    get_model_spec,
    get_block_spec,
    get_module_spec,
    list_registered_models,
    list_registered_blocks,
    list_registered_modules,
)

__version__ = "0.1.0"

__all__ = [
    # AST nodes
    "ModuleNode",
    "BlockNode",
    "ModelNode",
    "PrimitiveNode",
    "GraphStatement",
    "Operation",
    "TensorType",
    "Annotation",
    "ParamDecl",
    "ForwardBlock",
    "BackwardBlock",
    "Program",
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
    # Parser
    "parse_source",
    "parse_file",
    "ModuleDSLParser",
    # Compiler
    "compile_module",
    "compile_and_lower",
    "validate_source",
    "Compiler",
    "CompilerOptions",
    "CompilationResult",
    "ModuleRegistry",
    "get_registry",
    # Resolver
    "ModuleResolver",
    "ResolverContext",
    "ResolvedModule",
    "resolve_program",
    # IR
    "GraphIR",
    "ModuleIR",
    "ScheduleIR",
    "OpNode",
    "TensorRef",
    "KernelType",
    # Lowering
    "lower_program",
    "ModuleLowerer",
    # Errors
    "DSLError",
    "DSLSyntaxError",
    "DSLTypeError",
    "DSLShapeError",
    "DSLResolutionError",
    "DSLUndefinedError",
    "DSLConstraintError",
    "WarningCollector",
    # Python DSL - Types
    "Tensor",
    "TensorType",
    "Array",
    # Python DSL - Decorators
    "module",
    "block",
    "model",
    "primitive",
    "param",
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
    # Python DSL - Graph
    "graph",
    "GraphBuilder",
    "GraphRef",
    # Python DSL - HF utilities
    "fuse",
    "split",
    "transform",
    "hf_tied_to",
    # Python DSL - Specs
    "ModuleSpec",
    "BlockSpec",
    "ModelSpec",
    "PrimitiveSpec",
    "ParamSpec",
    "ForwardSpec",
    "BackwardSpec",
    # Python DSL - Registry
    "registry",
    "py_get_module",
    "py_get_primitive",
    "py_list_modules",
    # Python DSL - Lowering
    "lower_module_spec",
    "lower_block_spec",
    "lower_model_spec",
    "lower_graph_builder",
    # Python DSL - Compiler
    "compile_model",
    "compile_model_for_hf",
    "get_model_spec",
    "get_block_spec",
    "get_module_spec",
    "list_registered_models",
    "list_registered_blocks",
    "list_registered_modules",
]
