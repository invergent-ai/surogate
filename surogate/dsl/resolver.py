"""
Resolution Phase: Import, Type, and Shape Resolution

This module handles:
1. Import resolution: Loading and linking imported modules
2. Type resolution: Resolving type references to concrete types
3. Shape inference: Inferring tensor shapes through the graph
4. Symbol validation: Checking all identifiers are defined
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from .ast_nodes import (
    Program,
    ModuleNode,
    BlockNode,
    ModelNode,
    PrimitiveNode,
    ImportDecl,
    ImportItem,
    ParamDecl,
    TensorDecl,
    TensorType,
    ForwardBlock,
    BackwardBlock,
    GraphStatement,
    Operation,
    Expression,
    Identifier,
    Literal,
    BinaryOp,
    UnaryOp,
    CallExpr,
    TernaryExpr,
    Annotation,
    LetBinding,
    ConstraintDecl,
)
from .types import (
    Dtype,
    Shape,
    TensorTypeSpec,
    SymbolicDim,
    ConcreteDim,
    ComputedDim,
    VariadicDim,
    Dim,
)
from .errors import (
    DSLError,
    DSLResolutionError,
    DSLTypeError,
    DSLShapeError,
    DSLUndefinedError,
    DSLConstraintError,
    ErrorCode,
    SourceLocation,
    WarningCollector,
    WarningCode,
)
from .primitives import PRIMITIVES


# =============================================================================
# Resolution Context
# =============================================================================


@dataclass
class SymbolInfo:
    """Information about a resolved symbol."""

    name: str
    kind: str  # 'module', 'block', 'model', 'primitive', 'param', 'tensor', 'let'
    type_spec: Optional[TensorTypeSpec] = None
    value: Any = None  # For let bindings and params with defaults
    source: Optional[str] = None  # File or module name


@dataclass
class ResolverContext:
    """Context for the resolution phase."""

    # Symbol tables (layered: global -> module -> local)
    global_symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    module_symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    local_symbols: Dict[str, SymbolInfo] = field(default_factory=dict)

    # Resolved modules
    resolved_modules: Dict[str, "ResolvedModule"] = field(default_factory=dict)

    # Current module being resolved
    current_module: Optional[str] = None

    # Warnings
    warnings: WarningCollector = field(default_factory=WarningCollector)

    # Module parameters (resolved values)
    param_values: Dict[str, Any] = field(default_factory=dict)

    # Search paths for imports
    search_paths: List[Path] = field(default_factory=list)

    def push_scope(self):
        """Push a new local scope."""
        # Move current local to module level
        pass

    def pop_scope(self):
        """Pop local scope."""
        self.local_symbols.clear()

    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """Look up a symbol in all scopes."""
        # Check local first, then module, then global
        if name in self.local_symbols:
            return self.local_symbols[name]
        if name in self.module_symbols:
            return self.module_symbols[name]
        if name in self.global_symbols:
            return self.global_symbols[name]
        return None

    def define_local(self, name: str, info: SymbolInfo):
        """Define a symbol in local scope."""
        if name in self.local_symbols:
            raise DSLError(
                ErrorCode.E017,
                f"Redefinition of '{name}' in same scope (SSA violation)",
            )
        # Check for shadowing
        if name in self.module_symbols:
            self.warnings.warn(
                WarningCode.W002,
                f"Local '{name}' shadows module-level definition",
            )
        self.local_symbols[name] = info

    def define_module(self, name: str, info: SymbolInfo):
        """Define a symbol in module scope."""
        if name in self.module_symbols:
            raise DSLError(
                ErrorCode.E009,
                f"Duplicate definition of '{name}'",
            )
        self.module_symbols[name] = info

    def define_global(self, name: str, info: SymbolInfo):
        """Define a symbol in global scope."""
        if name in self.global_symbols:
            self.warnings.warn(
                WarningCode.W001,
                f"Definition of '{name}' shadows primitive",
            )
        self.global_symbols[name] = info


@dataclass
class ResolvedModule:
    """A fully resolved module definition."""

    name: str
    kind: str  # 'module', 'block', 'model', 'primitive'

    # Resolved parameters
    params: Dict[str, TensorTypeSpec] = field(default_factory=dict)

    # Config parameters with resolved defaults
    config: Dict[str, Any] = field(default_factory=dict)

    # Input/output types
    input_types: Dict[str, TensorTypeSpec] = field(default_factory=dict)
    output_types: Dict[str, TensorTypeSpec] = field(default_factory=dict)

    # Dependencies
    dependencies: Set[str] = field(default_factory=set)

    # Source AST node (for lowering)
    ast_node: Any = None


# =============================================================================
# Type Resolver
# =============================================================================


class TypeResolver:
    """Resolves and validates tensor types."""

    def __init__(self, ctx: ResolverContext):
        self.ctx = ctx

    def resolve_tensor_type(self, tt: TensorType) -> TensorTypeSpec:
        """Resolve a TensorType AST node to TensorTypeSpec."""
        dims = []

        for dim in tt.dims:
            resolved_dim = self._resolve_dim(dim)
            dims.append(resolved_dim)

        dtype = Dtype.BF16  # Default
        if tt.dtype:
            try:
                dtype = Dtype.from_string(tt.dtype)
            except ValueError:
                raise DSLTypeError(
                    f"Unknown dtype: {tt.dtype}",
                )

        return TensorTypeSpec(
            shape=Shape(dims),
            dtype=dtype,
            optional=tt.optional,
        )

    def _resolve_dim(self, dim) -> Dim:
        """Resolve a dimension specification."""
        if dim == "*":
            return VariadicDim()

        if isinstance(dim, int):
            return ConcreteDim(dim)

        if isinstance(dim, str):
            # Check if it's a known parameter
            if dim in self.ctx.param_values:
                value = self.ctx.param_values[dim]
                if isinstance(value, int):
                    return ConcreteDim(value)
            return SymbolicDim(dim)

        if isinstance(dim, Expression):
            # Try to evaluate
            try:
                value = self._evaluate_dim_expr(dim)
                if isinstance(value, int):
                    return ConcreteDim(value)
            except Exception:
                pass
            # Return as computed dimension
            return ComputedDim(str(dim), str(dim))

        return SymbolicDim(str(dim))

    def _evaluate_dim_expr(self, expr: Expression) -> Any:
        """Evaluate a dimension expression to a concrete value."""
        if isinstance(expr, Literal):
            return expr.value

        if isinstance(expr, Identifier):
            if expr.name in self.ctx.param_values:
                return self.ctx.param_values[expr.name]
            raise ValueError(f"Unknown identifier: {expr.name}")

        if isinstance(expr, BinaryOp):
            left = self._evaluate_dim_expr(expr.left)
            right = self._evaluate_dim_expr(expr.right)

            ops = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: a / b,
                "//": lambda a, b: a // b,
                "%": lambda a, b: a % b,
            }

            if expr.op in ops:
                return ops[expr.op](left, right)
            raise ValueError(f"Unknown operator: {expr.op}")

        if isinstance(expr, UnaryOp):
            operand = self._evaluate_dim_expr(expr.operand)
            if expr.op == "-":
                return -operand
            return operand

        raise ValueError(f"Cannot evaluate expression: {expr}")


# =============================================================================
# Shape Inference
# =============================================================================


class ShapeInferencer:
    """Infers tensor shapes through the computation graph."""

    def __init__(self, ctx: ResolverContext):
        self.ctx = ctx
        self.tensor_shapes: Dict[str, Shape] = {}

    def infer_operation(
        self,
        op_name: str,
        input_shapes: Dict[str, Shape],
        attrs: Dict[str, Any],
    ) -> Dict[str, Shape]:
        """Infer output shapes for an operation."""

        # Get primitive spec if available
        if op_name in PRIMITIVES:
            prim = PRIMITIVES[op_name]
            return self._infer_from_primitive(prim, input_shapes, attrs)

        # Fallback: try to infer from operation name
        return self._infer_generic(op_name, input_shapes, attrs)

    def _infer_from_primitive(
        self,
        prim,
        input_shapes: Dict[str, Shape],
        attrs: Dict[str, Any],
    ) -> Dict[str, Shape]:
        """Infer shapes from primitive specification."""
        # Use primitive's shape inference rules
        output_shapes = {}

        # Map input shapes to output using primitive's spec
        for out_name, out_spec in prim.get("outputs", {}).items():
            # Try to resolve output shape
            if "shape" in out_spec:
                shape_spec = out_spec["shape"]
                # Simple case: same as input
                if isinstance(shape_spec, str) and shape_spec in input_shapes:
                    output_shapes[out_name] = input_shapes[shape_spec]
                else:
                    # Keep symbolic
                    output_shapes[out_name] = Shape([SymbolicDim("?")])
            else:
                # Default to first input's shape
                if input_shapes:
                    first_input = next(iter(input_shapes.values()))
                    output_shapes[out_name] = first_input

        return output_shapes

    def _infer_generic(
        self,
        op_name: str,
        input_shapes: Dict[str, Shape],
        attrs: Dict[str, Any],
    ) -> Dict[str, Shape]:
        """Generic shape inference for unknown operations."""

        # Matmul: [*, M, K] x [*, K, N] -> [*, M, N]
        if op_name in ("matmul", "Linear"):
            if len(input_shapes) >= 2:
                shapes = list(input_shapes.values())
                a_shape = shapes[0]
                b_shape = shapes[1]

                # Get output dims
                out_dims = list(a_shape.dims[:-1])  # batch + M
                if b_shape.dims:
                    out_dims.append(b_shape.dims[-1])  # N

                return {"out": Shape(out_dims)}

        # Normalization ops: preserve shape
        if op_name in ("rmsnorm", "layernorm", "RMSNorm", "LayerNorm"):
            if input_shapes:
                first = next(iter(input_shapes.values()))
                return {"out": first}

        # Activation ops: preserve shape
        if op_name in ("swiglu", "silu", "gelu", "relu", "softmax"):
            if input_shapes:
                first = next(iter(input_shapes.values()))
                return {"out": first}

        # Split: output depends on dim and sizes
        if op_name == "split":
            if input_shapes:
                first = next(iter(input_shapes.values()))
                num_splits = attrs.get("num_splits", 2)
                return {f"out_{i}": first for i in range(num_splits)}

        # Default: preserve first input shape
        if input_shapes:
            first = next(iter(input_shapes.values()))
            return {"out": first}

        return {}

    def check_shape_compatibility(
        self,
        shape1: Shape,
        shape2: Shape,
        operation: str,
    ) -> bool:
        """Check if two shapes are compatible for an operation."""
        # Basic checks
        if shape1.has_variadic or shape2.has_variadic:
            return True  # Can't validate variadic shapes statically

        # For elementwise ops, shapes must match
        if operation in ("add", "mul", "sub", "div"):
            return self._shapes_broadcast(shape1, shape2)

        # For matmul, check inner dimensions
        if operation in ("matmul", "batched_matmul"):
            if len(shape1.dims) >= 1 and len(shape2.dims) >= 2:
                k1 = shape1.dims[-1]
                k2 = shape2.dims[-2]
                return self._dims_match(k1, k2)

        return True

    def _shapes_broadcast(self, s1: Shape, s2: Shape) -> bool:
        """Check if shapes can broadcast together."""
        # Align from right
        d1 = list(reversed(s1.dims))
        d2 = list(reversed(s2.dims))

        for dim1, dim2 in zip(d1, d2):
            if not self._dims_broadcast(dim1, dim2):
                return False

        return True

    def _dims_broadcast(self, d1: Dim, d2: Dim) -> bool:
        """Check if two dimensions broadcast."""
        if isinstance(d1, ConcreteDim) and isinstance(d2, ConcreteDim):
            return d1.value == d2.value or d1.value == 1 or d2.value == 1
        # Can't validate symbolic dimensions statically
        return True

    def _dims_match(self, d1: Dim, d2: Dim) -> bool:
        """Check if two dimensions match exactly."""
        if isinstance(d1, ConcreteDim) and isinstance(d2, ConcreteDim):
            return d1.value == d2.value
        if isinstance(d1, SymbolicDim) and isinstance(d2, SymbolicDim):
            return d1.name == d2.name
        # Can't validate mixed dimensions statically
        return True


# =============================================================================
# Import Resolver
# =============================================================================


class ImportResolver:
    """Resolves import statements."""

    def __init__(self, ctx: ResolverContext):
        self.ctx = ctx
        self._loading: Set[str] = set()  # For cycle detection

    def resolve_imports(self, imports: List[ImportDecl]) -> Dict[str, SymbolInfo]:
        """Resolve all imports and return imported symbols."""
        symbols = {}

        for imp in imports:
            module_path = imp.module_path

            # Check for cycles
            if module_path in self._loading:
                raise DSLError(
                    ErrorCode.E007,
                    f"Circular import detected: {module_path}",
                )

            # Try to load module
            resolved = self._load_module(module_path)

            if resolved is None:
                # Check if it's a primitive
                if module_path == "std.primitives" or module_path.startswith("std."):
                    symbols.update(self._load_std_primitives(imp.items))
                    continue

                raise DSLResolutionError(
                    ErrorCode.E002,
                    f"Cannot resolve import: {module_path}",
                )

            # Import specific items or all
            if imp.items:
                for item in imp.items:
                    if item.name not in resolved:
                        raise DSLResolutionError(
                            ErrorCode.E002,
                            f"'{item.name}' not found in '{module_path}'",
                        )

                    alias = item.alias or item.name

                    # Check for conflicts
                    if alias in symbols:
                        raise DSLError(
                            ErrorCode.E023,
                            f"Import name conflict: '{alias}'",
                        )

                    symbols[alias] = SymbolInfo(
                        name=item.name,
                        kind="import",
                        source=module_path,
                    )
            else:
                for name, mod in resolved.items():
                    if name in symbols:
                        raise DSLError(
                            ErrorCode.E023,
                            f"Import name conflict: '{name}'",
                        )
                    symbols[name] = SymbolInfo(
                        name=name,
                        kind=getattr(mod, "kind", "import"),
                        source=module_path,
                    )

        return symbols

    def _load_module(self, module_path: str) -> Optional[Dict[str, Any]]:
        """Load a module from file system."""
        # Return cached resolution if available.
        if module_path in self.ctx.resolved_modules:
            cached = self.ctx.resolved_modules[module_path]
            return {mod.name: mod for mod in cached}

        # Try to find the module file
        for search_path in self.ctx.search_paths:
            module_file = search_path / f"{module_path.replace('.', '/')}.module"
            if module_file.exists():
                # Parse and resolve the module
                from .parser import parse_file

                self._loading.add(module_path)
                try:
                    program = parse_file(str(module_file))
                    # Recursively resolve
                    resolver = ModuleResolver(self.ctx)
                    resolved = resolver.resolve_program(program)

                    # Extract exported symbols
                    exports = {}
                    for mod in resolved:
                        exports[mod.name] = mod

                    self.ctx.resolved_modules[module_path] = resolved
                    return exports
                finally:
                    self._loading.discard(module_path)

        return None

    def _load_std_primitives(self, items: Optional[List[ImportItem]]) -> Dict[str, SymbolInfo]:
        """Load standard library primitives."""
        symbols = {}

        if items is None:
            # Import all primitives
            for name, spec in PRIMITIVES.items():
                symbols[name] = SymbolInfo(
                    name=name,
                    kind="primitive",
                    source="std.primitives",
                )
        else:
            # Import specific primitives
            for item in items:
                if item.name in PRIMITIVES:
                    alias = item.alias or item.name
                    symbols[alias] = SymbolInfo(
                        name=item.name,
                        kind="primitive",
                        source="std.primitives",
                    )
                else:
                    raise DSLResolutionError(
                        ErrorCode.E002,
                        f"Unknown primitive: {item.name}",
                    )

        return symbols


# =============================================================================
# Module Resolver
# =============================================================================


class ModuleResolver:
    """Main resolver for module definitions."""

    def __init__(self, ctx: Optional[ResolverContext] = None):
        self.ctx = ctx or ResolverContext()
        self.type_resolver = TypeResolver(self.ctx)
        self.shape_inferencer = ShapeInferencer(self.ctx)
        self.import_resolver = ImportResolver(self.ctx)

    def resolve_program(self, program: Program) -> List[ResolvedModule]:
        """Resolve all definitions in a program."""
        results = []

        # First pass: resolve imports
        imported_symbols = self.import_resolver.resolve_imports(program.imports)
        for name, info in imported_symbols.items():
            self.ctx.define_global(name, info)

        # Register all primitives
        for name in PRIMITIVES:
            if name not in self.ctx.global_symbols:
                self.ctx.define_global(name, SymbolInfo(
                    name=name,
                    kind="primitive",
                    source="std.primitives",
                ))

        # Second pass: resolve module definitions (forward declarations)
        for prim in program.primitives:
            self._register_primitive(prim)

        for module in program.modules:
            self._register_module(module)

        for block in program.blocks:
            self._register_block(block)

        for model in program.models:
            self._register_model(model)

        # Third pass: fully resolve each definition
        for prim in program.primitives:
            resolved = self._resolve_primitive(prim)
            results.append(resolved)

        for module in program.modules:
            resolved = self._resolve_module(module)
            results.append(resolved)

        for block in program.blocks:
            resolved = self._resolve_block(block)
            results.append(resolved)

        for model in program.models:
            resolved = self._resolve_model(model)
            results.append(resolved)

        return results

    def _resolve_primitive(self, prim: PrimitiveNode) -> ResolvedModule:
        """Resolve a primitive declaration to a minimal module record."""
        return ResolvedModule(
            name=prim.name,
            kind="primitive",
            ast_node=prim,
        )

    def _register_primitive(self, prim: PrimitiveNode):
        """Register a primitive definition."""
        self.ctx.define_global(prim.name, SymbolInfo(
            name=prim.name,
            kind="primitive",
            source=self.ctx.current_module,
        ))

    def _register_module(self, module: ModuleNode):
        """Register a module definition (forward declaration)."""
        self.ctx.define_global(module.name, SymbolInfo(
            name=module.name,
            kind="module",
            source=self.ctx.current_module,
        ))

    def _register_block(self, block: BlockNode):
        """Register a block definition."""
        self.ctx.define_global(block.name, SymbolInfo(
            name=block.name,
            kind="block",
            source=self.ctx.current_module,
        ))

    def _register_model(self, model: ModelNode):
        """Register a model definition."""
        self.ctx.define_global(model.name, SymbolInfo(
            name=model.name,
            kind="model",
            source=self.ctx.current_module,
        ))

    def _resolve_module(self, node: ModuleNode) -> ResolvedModule:
        """Fully resolve a module definition."""
        self.ctx.module_symbols.clear()
        self.ctx.local_symbols.clear()

        # Resolve extends
        if node.extends:
            base_info = self.ctx.lookup(node.extends)
            if base_info is None:
                raise DSLResolutionError(
                    ErrorCode.E002,
                    f"Base module not found: {node.extends}",
                )
            if base_info.kind != "module":
                raise DSLError(
                    ErrorCode.E011,
                    f"Cannot extend non-module: {node.extends}",
                )

        # Resolve parameters
        resolved = ResolvedModule(
            name=node.name,
            kind="module",
            ast_node=node,
        )

        # Process config params
        for param in node.params:
            self._resolve_param(param, resolved)

        # Process let bindings
        for binding in node.let_bindings:
            self._resolve_let_binding(binding, resolved)

        # Check constraints
        for constraint in node.constraints:
            self._check_constraint(constraint)

        # Process weight params
        for param_decl in node.param_decls:
            self._resolve_param_decl(param_decl, resolved)

        # Resolve forward block
        if node.forward:
            self._resolve_forward(node.forward, resolved)

        # Resolve backward block
        if node.backward:
            self._resolve_backward(node.backward, resolved)

        return resolved

    def _resolve_block(self, node: BlockNode) -> ResolvedModule:
        """Fully resolve a block definition."""
        self.ctx.module_symbols.clear()
        self.ctx.local_symbols.clear()

        resolved = ResolvedModule(
            name=node.name,
            kind="block",
            ast_node=node,
        )

        for param in node.params:
            self._resolve_param(param, resolved)

        for binding in node.let_bindings:
            self._resolve_let_binding(binding, resolved)

        for constraint in node.constraints:
            self._check_constraint(constraint)

        for param_decl in node.param_decls:
            self._resolve_param_decl(param_decl, resolved)

        if node.forward:
            self._resolve_forward(node.forward, resolved)

        if node.backward:
            self._resolve_backward(node.backward, resolved)

        return resolved

    def _resolve_model(self, node: ModelNode) -> ResolvedModule:
        """Fully resolve a model definition."""
        self.ctx.module_symbols.clear()
        self.ctx.local_symbols.clear()

        resolved = ResolvedModule(
            name=node.name,
            kind="model",
            ast_node=node,
        )

        for param in node.params:
            self._resolve_param(param, resolved)

        for binding in node.let_bindings:
            self._resolve_let_binding(binding, resolved)

        for constraint in node.constraints:
            self._check_constraint(constraint)

        for param_decl in node.param_decls:
            self._resolve_param_decl(param_decl, resolved)

        if node.forward:
            self._resolve_forward(node.forward, resolved)

        if node.backward:
            self._resolve_backward(node.backward, resolved)

        return resolved

    def _resolve_param(self, param: ParamDecl, resolved: ResolvedModule):
        """Resolve a config parameter."""
        # Evaluate default value if present
        default_value = None
        if param.default:
            default_value = self._evaluate_expr(param.default)

        resolved.config[param.name] = default_value
        self.ctx.param_values[param.name] = default_value

        self.ctx.define_module(param.name, SymbolInfo(
            name=param.name,
            kind="param",
            value=default_value,
        ))

        if isinstance(param.type_annotation, tuple) and param.type_annotation and param.type_annotation[0] == "enum":
            enum_values = param.type_annotation[1]
            for value in enum_values:
                self.ctx.define_module(value, SymbolInfo(
                    name=value,
                    kind="enum",
                    value=value,
                ))

    def _resolve_let_binding(self, binding: LetBinding, resolved: ResolvedModule):
        """Resolve a let binding."""
        value = self._evaluate_expr(binding.value)
        self.ctx.param_values[binding.name] = value

        self.ctx.define_module(binding.name, SymbolInfo(
            name=binding.name,
            kind="let",
            value=value,
        ))

    def _check_constraint(self, constraint: ConstraintDecl):
        """Check a constraint."""
        try:
            result = self._evaluate_expr(constraint.condition)
            if not result:
                raise DSLConstraintError(
                    str(constraint.condition),
                    constraint.message,
                )
        except DSLConstraintError:
            raise
        except Exception as e:
            # Can't evaluate constraint at compile time
            self.ctx.warnings.warn(
                WarningCode.W003,
                f"Constraint '{constraint.message}' cannot be validated at compile time",
            )

    def _resolve_param_decl(self, param: TensorDecl, resolved: ResolvedModule):
        """Resolve a tensor parameter declaration."""
        type_spec = None
        if param.tensor_type:
            type_spec = self.type_resolver.resolve_tensor_type(param.tensor_type)

        resolved.params[param.name] = type_spec

        self.ctx.define_module(param.name, SymbolInfo(
            name=param.name,
            kind="tensor",
            type_spec=type_spec,
        ))

    def _resolve_forward(self, forward: ForwardBlock, resolved: ResolvedModule):
        """Resolve forward block."""
        # Reset local scope for forward graph
        self.ctx.local_symbols.clear()
        # Resolve input types
        if forward.input_type:
            type_spec = self.type_resolver.resolve_tensor_type(forward.input_type)
            resolved.input_types["in"] = type_spec
            self.ctx.define_local("in", SymbolInfo(
                name="in",
                kind="tensor",
                type_spec=type_spec,
            ))
        elif forward.inputs:
            for name, tt in forward.inputs.items():
                type_spec = self.type_resolver.resolve_tensor_type(tt)
                resolved.input_types[name] = type_spec
                self.ctx.define_local(name, SymbolInfo(
                    name=name,
                    kind="tensor",
                    type_spec=type_spec,
                ))

        # Resolve output types
        if forward.output_type:
            type_spec = self.type_resolver.resolve_tensor_type(forward.output_type)
            resolved.output_types["out"] = type_spec
        elif forward.outputs:
            for name, tt in forward.outputs.items():
                type_spec = self.type_resolver.resolve_tensor_type(tt)
                resolved.output_types[name] = type_spec

        # Resolve graph body
        if forward.graph:
            self._resolve_graph_body(forward.graph)

    def _resolve_backward(self, backward: BackwardBlock, resolved: ResolvedModule):
        """Resolve backward block."""
        # Reset local scope for backward graph
        self.ctx.local_symbols.clear()
        # Add gradient inputs to scope
        for name, tt in backward.gradient_inputs.items():
            type_spec = self.type_resolver.resolve_tensor_type(tt)
            self.ctx.define_local(name, SymbolInfo(
                name=name,
                kind="tensor",
                type_spec=type_spec,
            ))

        # Resolve graph body
        if backward.graph:
            self._resolve_graph_body(backward.graph)

    def _resolve_graph_body(self, graph):
        """Resolve a graph body."""
        from .ast_nodes import GraphBody, ConditionalGraph, RecomputeBlock

        for stmt in graph.statements:
            if isinstance(stmt, GraphStatement):
                self._resolve_graph_statement(stmt)
            elif isinstance(stmt, ConditionalGraph):
                self._resolve_conditional(stmt)
            elif isinstance(stmt, RecomputeBlock):
                self._resolve_recompute(stmt)

    def _resolve_graph_statement(self, stmt: GraphStatement):
        """Resolve a graph statement."""
        # Check source tensors are defined
        self._check_tensor_ref(stmt.source)

        # Resolve operations
        for op in stmt.operations:
            self._resolve_operation(op)

        # Define destination tensors
        self._define_tensor_ref(stmt.dest)

    def _resolve_operation(self, op: Operation):
        """Resolve an operation."""
        # Check operation is defined
        info = self.ctx.lookup(op.name)
        if info is None:
            raise DSLUndefinedError(op.name)

        # Resolve arguments
        for arg in op.args:
            self._resolve_expr(arg)

        for value in op.kwargs.values():
            self._resolve_expr(value)

    def _resolve_conditional(self, cond):
        """Resolve conditional graph."""
        # Resolve condition
        self._resolve_expr(cond.condition)

        # Resolve branches (compile-time if possible)
        branch = None
        try:
            value = self._evaluate_expr(cond.condition)
            if isinstance(value, bool):
                branch = cond.true_branch if value else (cond.false_branch or [])
        except Exception:
            branch = None

        if branch is not None:
            for stmt in branch:
                if isinstance(stmt, GraphStatement):
                    self._resolve_graph_statement(stmt)
        else:
            for stmt in cond.true_branch:
                if isinstance(stmt, GraphStatement):
                    self._resolve_graph_statement(stmt)
            if cond.false_branch:
                for stmt in cond.false_branch:
                    if isinstance(stmt, GraphStatement):
                        self._resolve_graph_statement(stmt)

    def _resolve_recompute(self, recompute):
        """Resolve recompute block."""
        for stmt in recompute.statements:
            if isinstance(stmt, GraphStatement):
                self._resolve_graph_statement(stmt)

    def _check_tensor_ref(self, ref):
        """Check that a tensor reference is defined."""
        from .ast_nodes import TensorRef as ASTTensorRef, TupleRef

        if isinstance(ref, str):
            info = self.ctx.lookup(ref)
            if info is None:
                raise DSLUndefinedError(ref)
        elif isinstance(ref, ASTTensorRef):
            info = self.ctx.lookup(ref.name)
            if info is None and not ref.is_saved:
                raise DSLUndefinedError(ref.name)
        elif isinstance(ref, TupleRef):
            for elem in ref.elements:
                self._check_tensor_ref(elem)

    def _define_tensor_ref(self, ref):
        """Define tensor from reference."""
        from .ast_nodes import TensorRef as ASTTensorRef, TupleRef

        if isinstance(ref, str):
            if ref == "_":
                return
            self.ctx.define_local(ref, SymbolInfo(
                name=ref,
                kind="tensor",
            ))
        elif isinstance(ref, ASTTensorRef):
            if ref.name == "_":
                return
            self.ctx.define_local(ref.name, SymbolInfo(
                name=ref.name,
                kind="tensor",
            ))
        elif isinstance(ref, TupleRef):
            for elem in ref.elements:
                self._define_tensor_ref(elem)

    def _resolve_expr(self, expr):
        """Resolve an expression (check all identifiers are defined)."""
        if isinstance(expr, Identifier):
            info = self.ctx.lookup(expr.name)
            if info is None:
                # Allow dtype literals (bf16, fp32, int32, etc.) as built-ins.
                if expr.name in {dtype.value for dtype in Dtype}:
                    return
                if expr.name in {"NN", "NT", "TN", "TT"}:
                    return
                raise DSLUndefinedError(expr.name)
        elif isinstance(expr, BinaryOp):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
        elif isinstance(expr, UnaryOp):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, CallExpr):
            # Check function exists
            info = self.ctx.lookup(expr.func)
            if info is None:
                # Could be builtin
                builtins = {"sqrt", "ceil_div", "min", "max", "abs"}
                if expr.func not in builtins:
                    raise DSLUndefinedError(expr.func)
            for arg in expr.args:
                self._resolve_expr(arg)
        elif isinstance(expr, TernaryExpr):
            self._resolve_expr(expr.condition)
            self._resolve_expr(expr.true_value)
            self._resolve_expr(expr.false_value)

    def _evaluate_expr(self, expr) -> Any:
        """Evaluate an expression to a value."""
        if isinstance(expr, Literal):
            return expr.value

        if isinstance(expr, Identifier):
            if expr.name in self.ctx.param_values:
                return self.ctx.param_values[expr.name]
            info = self.ctx.lookup(expr.name)
            if info and info.value is not None:
                return info.value
            if expr.name in {dtype.value for dtype in Dtype}:
                return Dtype.from_string(expr.name)
            if expr.name in {"NN", "NT", "TN", "TT"}:
                return expr.name
            return None

        if isinstance(expr, BinaryOp):
            left = self._evaluate_expr(expr.left)
            right = self._evaluate_expr(expr.right)

            if left is None or right is None:
                return None

            ops = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: a / b,
                "//": lambda a, b: a // b,
                "%": lambda a, b: a % b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                "<": lambda a, b: a < b,
                ">": lambda a, b: a > b,
                "<=": lambda a, b: a <= b,
                ">=": lambda a, b: a >= b,
                "and": lambda a, b: a and b,
                "or": lambda a, b: a or b,
            }

            if expr.op in ops:
                return ops[expr.op](left, right)
            return None

        if isinstance(expr, UnaryOp):
            operand = self._evaluate_expr(expr.operand)
            if operand is None:
                return None
            if expr.op == "-":
                return -operand
            if expr.op == "not":
                return not operand
            return operand

        if isinstance(expr, CallExpr):
            args = [self._evaluate_expr(arg) for arg in expr.args]
            if None in args:
                return None

            builtins = {
                "sqrt": lambda x: x ** 0.5,
                "ceil_div": lambda a, b: (a + b - 1) // b,
                "min": min,
                "max": max,
                "abs": abs,
            }

            if expr.func in builtins:
                return builtins[expr.func](*args)

            return None

        if isinstance(expr, TernaryExpr):
            cond = self._evaluate_expr(expr.condition)
            if isinstance(cond, bool):
                return self._evaluate_expr(expr.true_value if cond else expr.false_value)
            return None

        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def resolve_program(
    program: Program,
    search_paths: Optional[List[str]] = None,
) -> Tuple[List[ResolvedModule], WarningCollector]:
    """Resolve a program, returning resolved modules and warnings."""
    ctx = ResolverContext()

    if search_paths:
        ctx.search_paths = [Path(p) for p in search_paths]

    resolver = ModuleResolver(ctx)
    resolved = resolver.resolve_program(program)

    return resolved, ctx.warnings
