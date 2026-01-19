"""
AST Node Definitions for Module DSL

This module defines all AST node types used to represent parsed DSL programs.
All nodes are immutable dataclasses with optional source location tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum

from .types import (
    Dtype,
    Shape,
    TensorTypeSpec,
    TupleType,
    ArrayType,
    TypeSpec,
    SymbolicDim,
    ConcreteDim,
    VariadicDim,
    Dim,
    ConstExpr,
    Constraint,
    MemoryMode,
    HookPoint,
    HookMode,
    ShardStrategy,
    TransposeMode,
)
from .errors import SourceLocation


# =============================================================================
# Base Node Types
# =============================================================================


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    # kw_only=True allows subclasses to have required positional fields
    location: Optional[SourceLocation] = field(default=None, repr=False, compare=False, kw_only=True)


# =============================================================================
# Type Nodes
# =============================================================================


@dataclass
class TensorType(ASTNode):
    """Tensor type specification in the AST.

    Example DSL:
        [B, T, d_model]
        [B, T, 4096, bf16]
        [*, C]?
    """

    dims: List[Union[str, int, Tuple[str, str]]] = field(default_factory=list)  # Dimension specs
    dtype: Optional[str] = None  # None means auto-infer
    optional: bool = False

    def to_type_spec(self) -> TensorTypeSpec:
        """Convert to resolved TensorTypeSpec."""
        parsed_dims = []
        for d in self.dims:
            if d == "*":
                parsed_dims.append(VariadicDim())
            elif isinstance(d, int):
                parsed_dims.append(ConcreteDim(d))
            elif isinstance(d, str):
                parsed_dims.append(SymbolicDim(d))
            elif isinstance(d, tuple):
                # Computed dimension (name, expression)
                from .types import ComputedDim
                parsed_dims.append(ComputedDim(d[0], d[1]))

        dtype = Dtype.from_string(self.dtype) if self.dtype else Dtype.AUTO
        return TensorTypeSpec(Shape(parsed_dims), dtype, self.optional)


@dataclass
class TupleTypeNode(ASTNode):
    """Tuple type node.

    Example: (Tensor[B, T, C], Tensor[B, T, C])
    """

    elements: List[TensorType]


@dataclass
class ArrayTypeNode(ASTNode):
    """Array type node for repeated elements.

    Example: [n_layers] x ModuleType
    """

    size: Union[str, int]  # Can be symbolic or concrete
    element_type: Union[str, "ModuleInstantiation"]


# =============================================================================
# Expression Nodes
# =============================================================================


@dataclass
class Literal(ASTNode):
    """Literal value."""

    value: Union[int, float, bool, str, None]

    def __str__(self) -> str:
        if self.value is None:
            return "None"
        return repr(self.value)


@dataclass
class Identifier(ASTNode):
    """Identifier reference."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class BinaryOp(ASTNode):
    """Binary operation expression."""

    left: "Expression"
    op: str  # +, -, *, /, //, %, ==, !=, <, >, <=, >=, and, or
    right: "Expression"

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryOp(ASTNode):
    """Unary operation expression."""

    op: str  # -, not
    operand: "Expression"

    def __str__(self) -> str:
        return f"{self.op} {self.operand}"


@dataclass
class CallExpr(ASTNode):
    """Function call expression.

    Example: sqrt(d_head), fuse(...), tied_to(embedding)
    """

    func: str
    args: List["Expression"]
    kwargs: Dict[str, "Expression"] = field(default_factory=dict)

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{self.func}({all_args})"


@dataclass
class IndexExpr(ASTNode):
    """Index/slice expression.

    Example: x[0], x[:, :seq_len], x[..., -1]
    """

    base: "Expression"
    indices: List[Union["Expression", "SliceExpr"]]


@dataclass
class SliceExpr(ASTNode):
    """Slice expression for indexing.

    Example: :, 0:10, start:end
    """

    start: Optional["Expression"] = None
    stop: Optional["Expression"] = None
    step: Optional["Expression"] = None


@dataclass
class AttributeExpr(ASTNode):
    """Attribute access expression.

    Example: saved.x, module.param
    """

    base: "Expression"
    attr: str


@dataclass
class TernaryExpr(ASTNode):
    """Ternary/conditional expression.

    Example: bias if use_bias else None
    """

    condition: "Expression"
    true_value: "Expression"
    false_value: "Expression"


# Type alias for any expression
Expression = Union[
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    CallExpr,
    IndexExpr,
    SliceExpr,
    AttributeExpr,
    TernaryExpr,
]


# =============================================================================
# Annotation Nodes
# =============================================================================


@dataclass
class Annotation(ASTNode):
    """Annotation on operations or parameters.

    Example:
        @memory(save)
        @hook(AfterQKVProjection, mode=modify)
        @shard(column, tp_size=8)
    """

    name: str
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = []
        parts.extend(str(a) for a in self.args)
        parts.extend(f"{k}={v}" for k, v in self.kwargs.items())
        if parts:
            return f"@{self.name}({', '.join(parts)})"
        return f"@{self.name}"


# =============================================================================
# Declaration Nodes
# =============================================================================


@dataclass
class ParamDecl(ASTNode):
    """Module/function parameter declaration.

    Example:
        d_model: int = 4096
        use_bias: bool = true
    """

    name: str
    type_annotation: Optional[str] = None  # "int", "float", "bool", etc.
    default: Optional[Expression] = None


@dataclass
class TensorDecl(ASTNode):
    """Tensor parameter declaration.

    Example:
        weight: [out_dim, in_dim]
        bias: [out_dim] if use_bias
        qkv_weight: [qkv_dim, d_model] @shard(column)
    """

    name: str
    tensor_type: TensorType
    condition: Optional[Expression] = None  # "if use_bias"
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class LetBinding(ASTNode):
    """Let binding for computed values.

    Example:
        d_head = d_model // num_heads
        q_dim = num_heads * d_head
    """

    name: str
    value: Expression


@dataclass
class ConstraintDecl(ASTNode):
    """Constraint declaration.

    Example:
        C % H == 0, "d_model must be divisible by num_heads"
    """

    condition: Expression
    message: str


# =============================================================================
# Graph Nodes
# =============================================================================


@dataclass
class TensorRef(ASTNode):
    """Reference to a tensor in a graph.

    Example: in, out, saved.x, x[0]
    """

    name: str
    is_saved: bool = False  # True if saved.name


@dataclass
class TupleRef(ASTNode):
    """Tuple of tensor references.

    Example: (q, k, v)
    """

    elements: List[TensorRef]


@dataclass
class Operation(ASTNode):
    """Operation in a graph statement.

    Example:
        Linear(weight)
        matmul(transpose=TN)
        flash_attention(causal=true)
    """

    name: str
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = []
        parts.extend(str(a) for a in self.args)
        parts.extend(f"{k}={v}" for k, v in self.kwargs.items())
        if parts:
            return f"{self.name}({', '.join(parts)})"
        return f"{self.name}()"


@dataclass
class GraphStatement(ASTNode):
    """A data flow statement in a graph.

    Example:
        in -> Linear(weight) -> out
        (q, k, v) -> flash_attention() -> (attn_out, lse) @memory(save)
    """

    source: Union[TensorRef, TupleRef, str]
    operations: List[Operation]
    dest: Union[TensorRef, TupleRef, str]
    annotations: List[Annotation] = field(default_factory=list)

    def __str__(self) -> str:
        ops_str = " -> ".join(str(op) for op in self.operations)
        anns_str = " ".join(str(a) for a in self.annotations)
        if anns_str:
            return f"{self.source} -> {ops_str} -> {self.dest} {anns_str}"
        return f"{self.source} -> {ops_str} -> {self.dest}"


@dataclass
class ConditionalGraph(ASTNode):
    """Conditional graph statement.

    Example:
        if use_bias:
            (y, bias) -> add() -> out
        else:
            y -> out
    """

    condition: Expression
    true_branch: List["GraphNode"]
    false_branch: Optional[List["GraphNode"]] = None


@dataclass
class RecomputeBlock(ASTNode):
    """Recompute block in backward pass.

    Example:
        recompute:
            saved.in -> Linear(weight) -> gate_up
            gate_up -> split() -> (gate, up)
    """

    statements: List["GraphNode"]


# Type alias for graph nodes
GraphNode = Union[GraphStatement, ConditionalGraph, RecomputeBlock]


# =============================================================================
# Forward/Backward Block Nodes
# =============================================================================


@dataclass
class IOSpec(ASTNode):
    """Input/output specification for forward/backward.

    Example:
        in: [B, T, d_model]
        out: ([B, T, d_model], [B, T])
    """

    name: str
    type_spec: Union[TensorType, TupleTypeNode]


@dataclass
class NamedIOSpec(ASTNode):
    """Named input/output specification.

    Example:
        inputs:
            x: [B, T, d_model]
            residual: [B, T, d_model]
    """

    specs: Dict[str, TensorType]


@dataclass
class GraphBody(ASTNode):
    """Body of a graph section."""

    statements: List[GraphNode]


@dataclass
class ForwardBlock(ASTNode):
    """Forward pass definition.

    Example:
        forward:
            in: [B, T, d_model]
            out: [B, T, d_model]

            graph:
                in -> Linear(weight) -> out

            save: [in]
            recompute: [hidden]
    """

    # Simple in/out
    input_type: Optional[Union[TensorType, TupleTypeNode]] = None
    output_type: Optional[Union[TensorType, TupleTypeNode]] = None

    # Named inputs/outputs (for blocks)
    inputs: Optional[Dict[str, TensorType]] = None
    outputs: Optional[Dict[str, TensorType]] = None

    # Graph body
    graph: Optional[GraphBody] = None

    # Memory directives
    save: List[str] = field(default_factory=list)
    recompute: List[str] = field(default_factory=list)


@dataclass
class BackwardBlock(ASTNode):
    """Backward pass definition.

    Example:
        backward:
            d_out: [B, T, d_model]
            d_in: [B, T, d_model]

            graph:
                (d_out, weight) -> matmul(transpose=NN) -> d_in
    """

    # Gradient inputs (from downstream)
    gradient_inputs: Dict[str, TensorType] = field(default_factory=dict)

    # Gradient outputs (to upstream)
    gradient_outputs: Dict[str, TensorType] = field(default_factory=dict)

    # Graph body
    graph: Optional[GraphBody] = None


# =============================================================================
# Module/Block/Model Nodes
# =============================================================================


@dataclass
class ModuleInstantiation(ASTNode):
    """Module instantiation with arguments.

    Example:
        DenseTransformerBlock(d_model, num_heads, d_ff)
        Linear(in_dim=256, out_dim=512)
    """

    module_name: str
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class ModuleNode(ASTNode):
    """Module declaration.

    Example:
        module Linear(in_dim, out_dim, bias: bool = false):
            params:
                weight: [out_dim, in_dim]
                bias: [out_dim] if bias
            forward:
                ...
            backward:
                ...
    """

    name: str
    params: List[ParamDecl]
    extends: Optional[str] = None
    is_abstract: bool = False

    # Documentation
    docstring: Optional[str] = None

    # Let bindings
    let_bindings: List[LetBinding] = field(default_factory=list)
    constraints: List[ConstraintDecl] = field(default_factory=list)

    # Parameter declarations (weights, biases, etc.)
    param_decls: List[TensorDecl] = field(default_factory=list)

    # Forward and backward
    forward: Optional[ForwardBlock] = None
    backward: Optional[BackwardBlock] = None

    # Module-level annotations
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class BlockNode(ASTNode):
    """Block declaration (transformer block pattern).

    Example:
        block DenseTransformerBlock(d_model, num_heads, d_ff, ...):
            params:
                ln1: RMSNormParams(d_model)
                attention: CausalSelfAttention(...)
                ...

            pattern: sequential_residual
                sublayers:
                    - (ln1, attention)
                    - (ln2, mlp)
    """

    name: str
    params: List[ParamDecl]
    extends: Optional[str] = None

    # Documentation
    docstring: Optional[str] = None

    # Let bindings
    let_bindings: List[LetBinding] = field(default_factory=list)
    constraints: List[ConstraintDecl] = field(default_factory=list)

    # Submodule parameters
    param_decls: List[TensorDecl] = field(default_factory=list)

    # Pattern or explicit forward/backward
    pattern: Optional[str] = None  # "sequential_residual", "parallel_residual"
    pattern_config: Dict[str, Any] = field(default_factory=dict)

    # Explicit forward/backward (alternative to pattern)
    forward: Optional[ForwardBlock] = None
    backward: Optional[BackwardBlock] = None

    # Block-level annotations
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class HFConfigMapping(ASTNode):
    """HuggingFace config mapping.

    Example:
        hf_config:
            architecture: "LlamaForCausalLM"
            config_class: "LlamaConfig"
            param_mapping:
                d_model: hidden_size
                n_layers: num_hidden_layers
    """

    architecture: str
    config_class: str
    param_mapping: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightMapping(ASTNode):
    """HuggingFace weight mapping entry.

    Example:
        embedding: "model.embed_tokens.weight"
        blocks[{i}].qkv_weight: fuse(
            "model.layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight",
            dim=0
        )
    """

    internal_name: str  # Our parameter name (can include {i} placeholder)
    external_spec: Union[str, CallExpr]  # HF name or fuse/transform call
    optional: bool = False


@dataclass
class HFMappingSection(ASTNode):
    """HuggingFace weight mapping section."""

    mappings: List[WeightMapping] = field(default_factory=list)


@dataclass
class ModelNode(ASTNode):
    """Model declaration (top-level architecture).

    Example:
        model Llama2(vocab_size: int = 32000, d_model: int = 4096, ...):
            params:
                embedding: [vocab_size, d_model]
                blocks: [n_layers] x DenseTransformerBlock(...)
                ...

            forward:
                ...

            hf_config:
                ...

            hf_mapping:
                ...
    """

    name: str
    params: List[ParamDecl]

    # Documentation
    docstring: Optional[str] = None

    # Let bindings
    let_bindings: List[LetBinding] = field(default_factory=list)
    constraints: List[ConstraintDecl] = field(default_factory=list)

    # Parameter declarations
    param_decls: List[TensorDecl] = field(default_factory=list)

    # Forward and backward
    forward: Optional[ForwardBlock] = None
    backward: Optional[BackwardBlock] = None

    # HuggingFace compatibility
    hf_config: Optional[HFConfigMapping] = None
    hf_mapping: Optional[HFMappingSection] = None
    hf_export: Optional[HFMappingSection] = None

    # Model-level annotations
    annotations: List[Annotation] = field(default_factory=list)


@dataclass
class PrimitiveNode(ASTNode):
    """Primitive operation declaration.

    Example:
        primitive matmul:
            params:
                transpose: enum(NN, NT, TN, TT) = NN
                accumulate: bool = false

            forward:
                in: (A: [M, K], B: [K, N])
                out: C: [M, N]

            backward:
                d_A = matmul(d_C, B, transpose=NT)
                d_B = matmul(A, d_C, transpose=TN)

            impl:
                forward: kernels.matmul
                backward: kernels.matmul
    """

    name: str

    # Documentation
    docstring: Optional[str] = None

    # Primitive parameters
    params: List[ParamDecl] = field(default_factory=list)

    # IO specification
    forward_in: Optional[Union[TensorType, TupleTypeNode, Dict[str, TensorType]]] = None
    forward_out: Optional[Union[TensorType, TupleTypeNode, Dict[str, TensorType]]] = None

    # Backward specification (can be expressions or full graph)
    backward_exprs: Dict[str, Expression] = field(default_factory=dict)
    backward_in: Optional[Union[TensorType, TupleTypeNode, Dict[str, TensorType]]] = None
    backward_out: Optional[Union[TensorType, TupleTypeNode, Dict[str, TensorType]]] = None

    # Implementation references
    forward_impl: Optional[str] = None
    backward_impl: Optional[str] = None

    # What to save for backward
    save: List[str] = field(default_factory=list)

    # Invariants for validation
    invariants: Dict[str, List[str]] = field(default_factory=dict)

    # Memory and precision info
    memory_info: Dict[str, Any] = field(default_factory=dict)
    precision_info: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Import Nodes
# =============================================================================


@dataclass
class ImportItem(ASTNode):
    """Single import item.

    Example:
        Linear
        Linear as Lin
    """

    name: str
    alias: Optional[str] = None


@dataclass
class ImportDecl(ASTNode):
    """Import declaration.

    Example:
        import std.primitives
        import std.primitives.matmul.v1
        from models.llama import LlamaAttention as Attn
    """

    module_path: str
    version: Optional[str] = None  # "v1", "v2", etc.
    alias: Optional[str] = None  # For "import X as Y"
    items: Optional[List[ImportItem]] = None  # For "from X import Y, Z"


# =============================================================================
# Recipe Node
# =============================================================================


@dataclass
class RecipeNode(ASTNode):
    """Precision recipe declaration.

    Example:
        recipe FP8HybridRecipe:
            forward_activation_dtype: fp8_e4m3
            backward_gradient_dtype: fp8_e5m2
            accumulation_dtype: fp32
            ...
    """

    name: str
    settings: Dict[str, Expression] = field(default_factory=dict)


# =============================================================================
# Program Node (Top-level)
# =============================================================================


@dataclass
class Program(ASTNode):
    """Top-level program (a DSL source file).

    Contains all declarations from a single source file.
    """

    imports: List[ImportDecl] = field(default_factory=list)
    modules: List[ModuleNode] = field(default_factory=list)
    blocks: List[BlockNode] = field(default_factory=list)
    models: List[ModelNode] = field(default_factory=list)
    primitives: List[PrimitiveNode] = field(default_factory=list)
    recipes: List[RecipeNode] = field(default_factory=list)

    # Source file info
    source_file: Optional[str] = None
