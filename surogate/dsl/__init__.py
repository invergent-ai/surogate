"""
Module DSL - Domain-Specific Language for Neural Network Architecture Definition

This package implements the Module DSL parser, type checker, and compiler
for defining neural network architectures with explicit forward and backward
computation graphs.

Key components:
- Parser: Lark-based parser for DSL syntax
- AST: Typed AST node definitions
- Types: Tensor types, dtypes, symbolic dimensions
- Resolver: Import resolution, type inference, shape validation
- IR: Graph IR and Schedule IR representations
- Compiler: Main compilation entry point

Example usage:
    from surogate.dsl import compile_module, parse_file

    # Parse a DSL file
    program = parse_file("models/llama.module")

    # Compile to Graph IR
    result = compile_module("models/llama.module")

    # Access compiled modules
    for module_ir in result.modules:
        print(f"Compiled: {module_ir.name}")
        if module_ir.forward_graph:
            print(f"  Forward ops: {len(module_ir.forward_graph.nodes)}")
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
]
