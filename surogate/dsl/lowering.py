"""
Lowering Phase: AST to Graph IR

Converts parsed AST nodes into Graph IR for execution.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

from .ast_nodes import (
    Program,
    ModuleNode,
    BlockNode,
    ModelNode,
    PrimitiveNode,
    GraphStatement,
    ConditionalGraph,
    RecomputeBlock,
    Operation,
    TensorRef as ASTTensorRef,
    TupleRef,
    ForwardBlock,
    BackwardBlock,
    GraphBody,
    TensorDecl,
    LetBinding,
    Expression,
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    TernaryExpr,
    CallExpr,
    Annotation,
)
from .types import (
    Dtype,
    Shape,
    TensorTypeSpec,
    SymbolicDim,
    ConcreteDim,
    ComputedDim,
    VariadicDim,
    MemoryMode,
    HookPoint,
    HookMode,
    ShardStrategy,
)
from .ir import (
    GraphIR,
    OpNode,
    TensorRef,
    Edge,
    ModuleIR,
    KernelType,
    CompilationContext,
    PrimitiveSpec,
)
from .errors import (
    DSLError,
    DSLResolutionError,
    DSLTypeError,
    ErrorCode,
    SourceLocation,
    WarningCollector,
    WarningCode,
)


# =============================================================================
# Expression Evaluator (for compile-time constants)
# =============================================================================


class ExpressionEvaluator:
    """Evaluates compile-time constant expressions."""

    def __init__(self, env: Dict[str, Any]):
        self.env = env

    def evaluate(self, expr: Expression) -> Any:
        """Evaluate an expression to a Python value."""
        if isinstance(expr, Literal):
            return expr.value

        if isinstance(expr, Identifier):
            if expr.name in self.env:
                return self.env[expr.name]
            if expr.name in {dtype.value for dtype in Dtype}:
                return Dtype.from_string(expr.name)
            if expr.name in {"NN", "NT", "TN", "TT"}:
                return expr.name
            raise DSLResolutionError(
                ErrorCode.E002,
                f"Undefined identifier: {expr.name}",
            )

        if isinstance(expr, BinaryOp):
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            return self._eval_binary(expr.op, left, right)

        if isinstance(expr, UnaryOp):
            operand = self.evaluate(expr.operand)
            return self._eval_unary(expr.op, operand)

        if isinstance(expr, CallExpr):
            return self._eval_call(expr)

        if isinstance(expr, TernaryExpr):
            cond = self.evaluate(expr.condition)
            if isinstance(cond, bool):
                return self.evaluate(expr.true_value if cond else expr.false_value)
            raise ValueError("Ternary condition is not a compile-time boolean")

        # For complex expressions, return as-is for later resolution
        return expr

    def _eval_binary(self, op: str, left: Any, right: Any) -> Any:
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "//": lambda a, b: a // b,
            "%": lambda a, b: a % b,
            "**": lambda a, b: a ** b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
        }
        if op in ops:
            return ops[op](left, right)
        raise ValueError(f"Unknown operator: {op}")

    def _eval_unary(self, op: str, operand: Any) -> Any:
        if op == "-":
            return -operand
        if op == "not":
            return not operand
        if op == "+":
            return +operand
        raise ValueError(f"Unknown unary operator: {op}")

    def _eval_call(self, expr: CallExpr) -> Any:
        func = expr.func
        args = [self.evaluate(arg) for arg in expr.args]

        builtins = {
            "sqrt": lambda x: x ** 0.5,
            "ceil_div": lambda a, b: (a + b - 1) // b,
            "min": min,
            "max": max,
            "abs": abs,
        }

        if func in builtins:
            return builtins[func](*args)

        raise ValueError(f"Unknown function: {func}")


# =============================================================================
# Shape helpers
# =============================================================================


def _dim_from_value(value) -> "Dim":
    if isinstance(value, VariadicDim):
        return value
    if isinstance(value, ConcreteDim):
        return value
    if isinstance(value, ComputedDim):
        return value
    if isinstance(value, SymbolicDim):
        return value
    if value == "*":
        return VariadicDim()
    if isinstance(value, int):
        return ConcreteDim(value)
    return SymbolicDim(str(value))


def _resolve_dim(expr, evaluator: ExpressionEvaluator) -> "Dim":
    if isinstance(expr, VariadicDim):
        return expr
    if expr == "*":
        return VariadicDim()
    if isinstance(expr, ConcreteDim):
        return expr
    if isinstance(expr, SymbolicDim):
        return expr
    if isinstance(expr, int):
        return ConcreteDim(expr)
    if isinstance(expr, str):
        # Resolve known constants in env if possible
        if expr in evaluator.env and isinstance(evaluator.env[expr], int):
            return ConcreteDim(int(evaluator.env[expr]))
        return SymbolicDim(expr)
    try:
        value = evaluator.evaluate(expr)
    except Exception:
        value = None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if value is None:
        return SymbolicDim(str(expr))
    return _dim_from_value(value)


def tensor_type_to_shape(tensor_type, evaluator: ExpressionEvaluator) -> Optional[Shape]:
    if tensor_type is None:
        return None
    dims = []
    for dim in tensor_type.dims:
        dims.append(_resolve_dim(dim, evaluator))
    return Shape(dims)


def tensor_type_to_dtype(tensor_type) -> Dtype:
    if tensor_type and tensor_type.dtype:
        return Dtype.from_string(tensor_type.dtype)
    return Dtype.BF16


# =============================================================================
# Kernel Type Mapping
# =============================================================================


def operation_to_kernel_type(op_name: str) -> KernelType:
    """Map operation name to kernel type."""
    mapping = {
        # Linear algebra
        "matmul": KernelType.MATMUL,
        "Linear": KernelType.MATMUL,
        "batched_matmul": KernelType.BATCHED_MATMUL,
        "grouped_gemm": KernelType.GROUPED_GEMM,

        # Normalization
        "rmsnorm": KernelType.RMSNORM,
        "RMSNorm": KernelType.RMSNORM,
        "fused_residual_rmsnorm": KernelType.FUSED_RESIDUAL_RMSNORM,
        "layernorm": KernelType.LAYERNORM,
        "LayerNorm": KernelType.LAYERNORM,

        # Activations
        "swiglu": KernelType.SWIGLU,
        "SwiGLU": KernelType.SWIGLU,
        "geglu": KernelType.GEGLU,
        "GeGLU": KernelType.GEGLU,
        "silu": KernelType.SILU,
        "relu": KernelType.RELU,
        "relu2": KernelType.RELU2,
        "gelu": KernelType.GELU,
        "softmax": KernelType.SOFTMAX,

        # Attention
        "flash_attention": KernelType.FLASH_ATTENTION,
        "FlashAttention": KernelType.FLASH_ATTENTION,
        "rope": KernelType.ROPE,
        "RoPE": KernelType.ROPE,
        "qk_norm": KernelType.QK_NORM,

        # Tensor manipulation
        "split": KernelType.SPLIT,
        "concat": KernelType.CONCAT,
        "view": KernelType.VIEW,
        "transpose": KernelType.TRANSPOSE,
        "permute": KernelType.PERMUTE,
        "contiguous": KernelType.CONTIGUOUS,
        "copy": KernelType.COPY,

        # Elementwise
        "add": KernelType.ADD,
        "Add": KernelType.ADD,
        "mul": KernelType.MUL,
        "scale": KernelType.SCALE,
        "add3": KernelType.ADD3,

        # Reduction
        "reduce_sum": KernelType.REDUCE_SUM,
        "reduce_mean": KernelType.REDUCE_MEAN,
        "reduce_max": KernelType.REDUCE_MAX,

        # Embedding
        "embedding": KernelType.EMBEDDING,
        "Embedding": KernelType.EMBEDDING,

        # MoE
        "moe_router": KernelType.MOE_ROUTER,
        "moe_permute": KernelType.MOE_PERMUTE,
        "moe_unpermute": KernelType.MOE_UNPERMUTE,

        # Mamba
        "mamba_conv1d": KernelType.MAMBA_CONV1D,
        "mamba_selective_scan": KernelType.MAMBA_SELECTIVE_SCAN,

        # Utility
        "zeros": KernelType.ZEROS,
        "ones": KernelType.ONES,
        "fill": KernelType.FILL,
    }

    return mapping.get(op_name, KernelType.CUSTOM)


# =============================================================================
# Annotation Processing
# =============================================================================


def process_annotations(
    annotations: List[Annotation],
) -> Tuple[MemoryMode, Optional[HookPoint], Optional[HookMode], Optional[ShardStrategy], Dict[str, Any]]:
    """Process annotations and extract relevant information."""
    memory_mode = MemoryMode.TEMPORARY
    hook_point = None
    hook_mode = None
    shard_strategy = None
    extra_attrs = {}

    for ann in annotations:
        if ann.name == "memory":
            if ann.args:
                mode_str = str(ann.args[0])
                if isinstance(ann.args[0], Identifier):
                    mode_str = ann.args[0].name
                try:
                    memory_mode = MemoryMode(mode_str.lower())
                except ValueError:
                    pass

        elif ann.name == "hook":
            if ann.args:
                point_str = str(ann.args[0])
                if isinstance(ann.args[0], Identifier):
                    point_str = ann.args[0].name
                try:
                    hook_point = HookPoint(point_str)
                except ValueError:
                    pass
            if "mode" in ann.kwargs:
                mode_val = ann.kwargs["mode"]
                if isinstance(mode_val, Identifier):
                    try:
                        hook_mode = HookMode(mode_val.name.lower())
                    except ValueError:
                        pass

        elif ann.name == "shard":
            if ann.args:
                strat_str = str(ann.args[0])
                if isinstance(ann.args[0], Identifier):
                    strat_str = ann.args[0].name
                try:
                    shard_strategy = ShardStrategy(strat_str.lower())
                except ValueError:
                    pass

        elif ann.name == "dtype":
            if ann.args:
                dtype_str = str(ann.args[0])
                if isinstance(ann.args[0], Identifier):
                    dtype_str = ann.args[0].name
                extra_attrs["dtype"] = dtype_str

        elif ann.name == "checkpoint":
            memory_mode = MemoryMode.RECOMPUTE

        elif ann.name == "frozen":
            extra_attrs["frozen"] = True

        elif ann.name == "trainable":
            extra_attrs["trainable"] = True

    return memory_mode, hook_point, hook_mode, shard_strategy, extra_attrs


# =============================================================================
# Graph Builder
# =============================================================================


class GraphBuilder:
    """Builds Graph IR from AST graph statements."""

    def __init__(self, ctx: CompilationContext, warnings: WarningCollector):
        self.ctx = ctx
        self.warnings = warnings
        self.current_graph: Optional[GraphIR] = None

    def build_forward_graph(
        self,
        name: str,
        forward: ForwardBlock,
        params: List[TensorDecl],
    ) -> GraphIR:
        """Build Graph IR for forward pass."""
        graph = GraphIR(name=f"{name}_forward")
        self.current_graph = graph
        evaluator = ExpressionEvaluator(self.ctx.module_params)
        param_shapes: Dict[str, Shape] = {}

        # Add parameters
        for param in params:
            shape = tensor_type_to_shape(param.tensor_type, evaluator)
            dtype = tensor_type_to_dtype(param.tensor_type)
            tensor_ref = TensorRef(
                name=param.name,
                dtype=dtype,
                shape=shape,
                is_param=True,
            )
            graph.params[param.name] = tensor_ref
            if shape is not None:
                param_shapes[param.name] = shape

        # Process input specification
        if forward.input_type:
            shape = tensor_type_to_shape(getattr(forward.input_type, "dims", None) and forward.input_type, evaluator)
            dtype = tensor_type_to_dtype(getattr(forward.input_type, "dims", None) and forward.input_type)
            # Simple in: type
            graph.inputs["in"] = TensorRef(
                name="in",
                dtype=dtype if shape else Dtype.BF16,
                shape=shape,
                is_input=True,
            )
        elif forward.inputs:
            # Named inputs
            for name, tensor_type in forward.inputs.items():
                shape = tensor_type_to_shape(tensor_type, evaluator)
                dtype = tensor_type_to_dtype(tensor_type)
                graph.inputs[name] = TensorRef(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    is_input=True,
                )

        # Process output specification
        if forward.output_type:
            shape = tensor_type_to_shape(getattr(forward.output_type, "dims", None) and forward.output_type, evaluator)
            dtype = tensor_type_to_dtype(getattr(forward.output_type, "dims", None) and forward.output_type)
            graph.outputs["out"] = TensorRef(
                name="out",
                dtype=dtype if shape else Dtype.BF16,
                shape=shape,
                is_output=True,
            )
        elif forward.outputs:
            for name, tensor_type in forward.outputs.items():
                shape = tensor_type_to_shape(tensor_type, evaluator)
                dtype = tensor_type_to_dtype(tensor_type)
                graph.outputs[name] = TensorRef(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    is_output=True,
                )

        # Process graph body
        if forward.graph:
            self._process_graph_body(forward.graph, graph)

        # Process save/recompute lists
        graph.save_list = forward.save.copy()
        graph.recompute_list = forward.recompute.copy()

        return graph

    def build_backward_graph(
        self,
        name: str,
        backward: BackwardBlock,
        params: List[TensorDecl],
        forward_graph: GraphIR,
    ) -> GraphIR:
        """Build Graph IR for backward pass."""
        graph = GraphIR(name=f"{name}_backward")
        self.current_graph = graph
        evaluator = ExpressionEvaluator(self.ctx.module_params)
        param_shapes: Dict[str, Shape] = {}

        # Add parameters (weights)
        for param in params:
            shape = tensor_type_to_shape(param.tensor_type, evaluator)
            dtype = tensor_type_to_dtype(param.tensor_type)
            tensor_ref = TensorRef(
                name=param.name,
                dtype=dtype,
                shape=shape,
                is_param=True,
            )
            graph.params[param.name] = tensor_ref
            if shape is not None:
                param_shapes[param.name] = shape

        # Add gradient inputs
        for grad_name, tensor_type in backward.gradient_inputs.items():
            shape = tensor_type_to_shape(tensor_type, evaluator)
            dtype = tensor_type_to_dtype(tensor_type)
            graph.inputs[grad_name] = TensorRef(
                name=grad_name,
                dtype=dtype,
                shape=shape,
                is_input=True,
            )

        # Add gradient outputs
        for grad_name, tensor_type in backward.gradient_outputs.items():
            shape = tensor_type_to_shape(tensor_type, evaluator)
            dtype = tensor_type_to_dtype(tensor_type)
            graph.outputs[grad_name] = TensorRef(
                name=grad_name,
                dtype=dtype,
                shape=shape,
                is_output=True,
            )

        # Add saved tensors from forward (accessible as saved.x)
        for saved_name in forward_graph.save_list:
            graph.params[f"saved.{saved_name}"] = TensorRef(
                name=f"saved.{saved_name}",
                is_param=True,  # Treat as param for access
            )

        # Add parameter gradients
        for param in params:
            d_name = f"d_{param.name}"
            graph.outputs[d_name] = TensorRef(
                name=d_name,
                dtype=tensor_type_to_dtype(param.tensor_type),
                shape=param_shapes.get(param.name),
                is_output=True,
            )

        # Process graph body
        if backward.graph:
            self._process_graph_body(backward.graph, graph)

        return graph

    def _process_graph_body(self, body: GraphBody, graph: GraphIR):
        """Process graph body statements."""
        for stmt in body.statements:
            if isinstance(stmt, GraphStatement):
                self._process_data_flow(stmt, graph)
            elif isinstance(stmt, ConditionalGraph):
                self._process_conditional(stmt, graph)
            elif isinstance(stmt, RecomputeBlock):
                self._process_recompute(stmt, graph)

    def _process_data_flow(self, stmt: GraphStatement, graph: GraphIR):
        """Process a data flow statement."""
        # Get source tensor names
        source_names = self._get_tensor_names(stmt.source)

        # Get destination tensor names
        dest_names = self._get_tensor_names(stmt.dest)

        # Process annotations
        memory_mode, hook_point, hook_mode, shard_strategy, extra_attrs = process_annotations(
            stmt.annotations
        )

        # Process each operation in the chain
        current_inputs = source_names

        for i, op in enumerate(stmt.operations):
            is_last = (i == len(stmt.operations) - 1)
            outputs = dest_names if is_last else [f"_tmp_{self.ctx.new_node_id()}"]

            # Create OpNode
            node = OpNode(
                id=self.ctx.new_node_id(),
                kernel_type=operation_to_kernel_type(op.name),
                name=op.name,
                inputs=current_inputs.copy(),
                outputs=outputs.copy(),
                attrs=self._process_op_attrs(op),
                memory_mode=memory_mode if is_last else MemoryMode.TEMPORARY,
                hook_point=hook_point if is_last else None,
                hook_mode=hook_mode if is_last else None,
                shard_strategy=shard_strategy if is_last else None,
                layer_idx=self.ctx.layer_idx,
            )

            graph.nodes.append(node)

            # Add intermediate tensors
            for out in outputs:
                if out not in graph.inputs and out not in graph.params:
                    if out not in graph.intermediates:
                        graph.intermediates[out] = TensorRef(name=out)

            current_inputs = outputs

    def _process_conditional(self, cond: ConditionalGraph, graph: GraphIR):
        """Process conditional graph."""
        # Evaluate condition
        evaluator = ExpressionEvaluator(self.ctx.module_params)
        try:
            condition_value = evaluator.evaluate(cond.condition)
            is_const = True
        except Exception:
            # Non-constant condition - need to handle dynamically
            is_const = False
            condition_value = True  # Default to true branch for now

        if is_const:
            # Compile-time conditional: only include relevant branch
            if condition_value:
                for stmt in cond.true_branch:
                    if isinstance(stmt, GraphStatement):
                        self._process_data_flow(stmt, graph)
            elif cond.false_branch:
                for stmt in cond.false_branch:
                    if isinstance(stmt, GraphStatement):
                        self._process_data_flow(stmt, graph)
        else:
            # Runtime conditional not supported
            self.warnings.warn(
                WarningCode.W003,
                f"Runtime conditional at {cond.location} - only true branch compiled",
            )
            for stmt in cond.true_branch:
                if isinstance(stmt, GraphStatement):
                    self._process_data_flow(stmt, graph)

    def _process_recompute(self, recompute: RecomputeBlock, graph: GraphIR):
        """Process recompute block."""
        # Mark all operations in recompute block
        for stmt in recompute.statements:
            if isinstance(stmt, GraphStatement):
                # Add recompute annotation
                stmt.annotations.append(Annotation(name="memory", args=[Identifier(name="recompute")]))
                self._process_data_flow(stmt, graph)

    def _get_tensor_names(self, ref) -> List[str]:
        """Get tensor names from a reference."""
        if isinstance(ref, str):
            return [ref]
        if isinstance(ref, ASTTensorRef):
            prefix = "saved." if ref.is_saved else ""
            return [f"{prefix}{ref.name}"]
        if isinstance(ref, TupleRef):
            result = []
            for elem in ref.elements:
                result.extend(self._get_tensor_names(elem))
            return result
        return ["unknown"]

    def _process_op_attrs(self, op: Operation) -> Dict[str, Any]:
        """Process operation attributes."""
        attrs = {}
        evaluator = ExpressionEvaluator(self.ctx.module_params)

        # Add positional args
        for i, arg in enumerate(op.args):
            try:
                attrs[f"arg{i}"] = evaluator.evaluate(arg)
            except Exception:
                attrs[f"arg{i}"] = arg

        # Add keyword args
        for key, value in op.kwargs.items():
            try:
                attrs[key] = evaluator.evaluate(value)
            except Exception:
                attrs[key] = value

        return attrs


# =============================================================================
# Shape Inference (Graph IR)
# =============================================================================


def _attr_to_value(value, evaluator: ExpressionEvaluator):
    if isinstance(value, list):
        return [_attr_to_value(v, evaluator) for v in value]
    if isinstance(value, dict):
        return {k: _attr_to_value(v, evaluator) for k, v in value.items()}
    if isinstance(value, (int, float, str, Dtype)):
        return value
    try:
        return evaluator.evaluate(value)
    except Exception:
        return value


def _shape_dims(shape: Optional[Shape]) -> Optional[List[Any]]:
    if shape is None:
        return None
    return list(shape.dims)


def _shape_from_dims(dims: Optional[List[Any]]) -> Optional[Shape]:
    if dims is None:
        return None
    return Shape(dims)


def _infer_rstd_shape(input_shape: Optional[Shape]) -> Optional[Shape]:
    if input_shape is None:
        return None
    dims = list(input_shape.dims)
    if dims:
        dims = dims[:-1]
    return Shape(dims)


def _infer_matmul_shape(a: Optional[Shape], b: Optional[Shape], transpose: str) -> Optional[Shape]:
    if a is None or b is None:
        return None
    a_dims = list(a.dims)
    b_dims = list(b.dims)
    if len(a_dims) < 2 or len(b_dims) < 2:
        return None
    tA = transpose[0] if transpose else "N"
    tB = transpose[1] if transpose else "N"
    if tA == "N":
        m_dim = a_dims[-2]
        k_dim = a_dims[-1]
    else:
        m_dim = a_dims[-1]
        k_dim = a_dims[-2]
    if tB == "N":
        n_dim = b_dims[-1]
    else:
        n_dim = b_dims[-2]
    batch_dims = a_dims[:-2]
    out_dims = batch_dims + [m_dim, n_dim]
    return Shape(out_dims)


def _infer_swiglu_shape(x: Optional[Shape]) -> Optional[Shape]:
    if x is None:
        return None
    dims = list(x.dims)
    if not dims:
        return x
    last = dims[-1]
    if isinstance(last, ConcreteDim) and last.value % 2 == 0:
        dims[-1] = ConcreteDim(last.value // 2)
        return Shape(dims)
    return Shape(dims[:-1] + [SymbolicDim("D")])


def _infer_flash_attention_shapes(qkv: Optional[Shape], env: Dict[str, Any]) -> Tuple[Optional[Shape], Optional[Shape]]:
    if qkv is None or len(qkv.dims) < 4:
        return None, None
    b_dim, t_dim = qkv.dims[0], qkv.dims[1]
    d_dim = qkv.dims[3]
    hq = env.get("num_query_heads")
    hkv = env.get("num_kv_heads")
    if isinstance(hq, int):
        hq_dim = ConcreteDim(hq)
    else:
        hq_dim = SymbolicDim("Hq")
    out = Shape([b_dim, t_dim, hq_dim, d_dim])
    lse = Shape([b_dim, hq_dim, t_dim])
    return out, lse


def _infer_qk_norm_shapes(qkv: Optional[Shape], env: Dict[str, Any]) -> Tuple[Optional[Shape], Optional[Shape], Optional[Shape]]:
    if qkv is None or len(qkv.dims) < 4:
        return None, None, None
    b_dim, t_dim = qkv.dims[0], qkv.dims[1]
    hq = env.get("num_query_heads")
    hkv = env.get("num_kv_heads")
    hq_dim = ConcreteDim(hq) if isinstance(hq, int) else SymbolicDim("Hq")
    hkv_dim = ConcreteDim(hkv) if isinstance(hkv, int) else SymbolicDim("Hkv")
    q_rstd = Shape([b_dim, hq_dim, t_dim])
    k_rstd = Shape([b_dim, hkv_dim, t_dim])
    return qkv, q_rstd, k_rstd


def infer_graph_shapes(graph: GraphIR, env: Dict[str, Any]) -> None:
    evaluator = ExpressionEvaluator(env)

    # Seed known shapes/dtypes
    shapes: Dict[str, Shape] = {}
    dtypes: Dict[str, Dtype] = {}
    for bucket in (graph.inputs, graph.outputs, graph.params, graph.intermediates):
        for name, tref in bucket.items():
            if tref.shape is not None:
                shapes[name] = tref.shape
            if tref.dtype is not None:
                dtypes[name] = tref.dtype

    def set_tensor(name: str, shape: Optional[Shape], dtype: Optional[Dtype]):
        if shape is not None:
            shapes[name] = shape
        if dtype is not None:
            dtypes[name] = dtype
        tref = graph.get_tensor(name)
        if tref is None:
            graph.intermediates[name] = TensorRef(name=name, shape=shape, dtype=dtype)
        else:
            if shape is not None and tref.shape is None:
                tref.shape = shape
            if dtype is not None and tref.dtype is None:
                tref.dtype = dtype

    for node in graph.nodes:
        op = node.name
        attrs = {k: _attr_to_value(v, evaluator) for k, v in (node.attrs or {}).items()}
        in_shapes = [shapes.get(n) for n in node.inputs]
        in_dtypes = [dtypes.get(n) for n in node.inputs]

        out_shapes: List[Optional[Shape]] = []
        out_dtypes: List[Optional[Dtype]] = []

        if op == "view":
            shape_spec = attrs.get("shape")
            if isinstance(shape_spec, list):
                dims = [_resolve_dim(d, evaluator) for d in shape_spec]
                out_shapes = [Shape(dims)]
            else:
                out_shapes = [None]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op in ("zeros", "ones", "fill_constant", "fill_normal"):
            shape_spec = attrs.get("shape")
            if isinstance(shape_spec, list):
                dims = [_resolve_dim(d, evaluator) for d in shape_spec]
                out_shapes = [Shape(dims)]
            else:
                out_shapes = [None]
            dtype = attrs.get("dtype")
            if isinstance(dtype, Dtype):
                out_dtypes = [dtype]
            else:
                out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op == "matmul":
            transpose = attrs.get("transpose")
            if isinstance(transpose, Identifier):
                transpose = transpose.name
            if isinstance(transpose, str):
                transpose = transpose.upper()
            else:
                transpose = "NN"
            out_shapes = [_infer_matmul_shape(in_shapes[0] if len(in_shapes) > 0 else None,
                                              in_shapes[1] if len(in_shapes) > 1 else None,
                                              transpose)]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op == "embedding":
            # token_ids shape [B,T], weights [V,C]
            if len(in_shapes) >= 2 and in_shapes[1] is not None:
                w_shape = in_shapes[1]
                if w_shape and len(w_shape.dims) >= 2:
                    c_dim = w_shape.dims[1]
                else:
                    c_dim = SymbolicDim("C")
            else:
                c_dim = ConcreteDim(env["d_model"]) if isinstance(env.get("d_model"), int) else SymbolicDim("C")
            if in_shapes and in_shapes[0] is not None and len(in_shapes[0].dims) >= 2:
                b_dim, t_dim = in_shapes[0].dims[0], in_shapes[0].dims[1]
                out_shapes = [Shape([b_dim, t_dim, c_dim])]
            else:
                out_shapes = [None]
            out_dtypes = [in_dtypes[1] if len(in_dtypes) > 1 else (in_dtypes[0] if in_dtypes else None)]
        elif op in ("fused_residual_rmsnorm", "fused_residual_rmsnorm_backward"):
            base_shape = in_shapes[0] if in_shapes else None
            if op == "fused_residual_rmsnorm":
                out_shapes = [base_shape, base_shape, _infer_rstd_shape(base_shape)]
                out_dtypes = [in_dtypes[0] if in_dtypes else None,
                              in_dtypes[0] if in_dtypes else None,
                              Dtype.FP32]
            else:
                out_shapes = [base_shape, base_shape, in_shapes[3] if len(in_shapes) > 3 else None]
                out_dtypes = [in_dtypes[0] if in_dtypes else None,
                              in_dtypes[0] if in_dtypes else None,
                              in_dtypes[3] if len(in_dtypes) > 3 else None]
        elif op == "rmsnorm":
            base_shape = in_shapes[0] if in_shapes else None
            out_shapes = [base_shape, _infer_rstd_shape(base_shape)]
            out_dtypes = [in_dtypes[0] if in_dtypes else None, Dtype.FP32]
        elif op == "swiglu":
            out_shapes = [_infer_swiglu_shape(in_shapes[0] if in_shapes else None)]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op == "flash_attention":
            out, lse = _infer_flash_attention_shapes(in_shapes[0] if in_shapes else None, env)
            out_shapes = [out, lse]
            out_dtypes = [in_dtypes[0] if in_dtypes else None, Dtype.FP32]
        elif op == "rope":
            out_shapes = [in_shapes[0] if in_shapes else None]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op == "qkv_qk_norm_rope":
            qkv_shape, q_rstd, k_rstd = _infer_qk_norm_shapes(in_shapes[0] if in_shapes else None, env)
            out_shapes = [qkv_shape, q_rstd, k_rstd]
            out_dtypes = [in_dtypes[0] if in_dtypes else None, Dtype.FP32, Dtype.FP32]
        elif op == "bias_add":
            out_shapes = [in_shapes[0] if in_shapes else None]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op == "bias_backward":
            if in_shapes and in_shapes[0] is not None and in_shapes[0].dims:
                out_shapes = [Shape([in_shapes[0].dims[-1]])]
            else:
                out_shapes = [None]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op in ("embedding_backward",):
            v = env.get("vocab_size")
            c = env.get("d_model")
            if isinstance(v, int) and isinstance(c, int):
                out_shapes = [Shape([ConcreteDim(v), ConcreteDim(c)])]
            else:
                out_shapes = [Shape([SymbolicDim("V"), SymbolicDim("C")])]
            out_dtypes = [in_dtypes[0] if in_dtypes else None]
        elif op in ("StackedBlocks", "StackedBlocksBackward"):
            out_shapes = [in_shapes[0] if len(in_shapes) > 0 else None,
                          in_shapes[1] if len(in_shapes) > 1 else None]
            out_dtypes = [in_dtypes[0] if len(in_dtypes) > 0 else None,
                          in_dtypes[1] if len(in_dtypes) > 1 else None]
        else:
            # Default: preserve first input shape/dtype
            out_shapes = [in_shapes[0] if in_shapes else None] * len(node.outputs)
            out_dtypes = [in_dtypes[0] if in_dtypes else None] * len(node.outputs)

        # Apply inferred shapes/dtypes to outputs
        for out_name, out_shape, out_dtype in zip(node.outputs, out_shapes, out_dtypes):
            set_tensor(out_name, out_shape, out_dtype)


# =============================================================================
# Module Lowerer
# =============================================================================


class ModuleLowerer:
    """Lowers module AST nodes to Module IR."""

    def __init__(self, warnings: Optional[WarningCollector] = None):
        self.warnings = warnings or WarningCollector()
        self.ctx = CompilationContext()

    def lower_module(self, node: ModuleNode) -> ModuleIR:
        """Lower a module AST node to Module IR."""
        # Build compilation context
        self._build_context(node)

        # Create Module IR
        module_ir = ModuleIR(
            name=node.name,
            config=self.ctx.module_params.copy(),
            extends=node.extends,
        )

        # Add parameters
        for param in node.param_decls:
            tensor_ref = TensorRef(
                name=param.name,
                is_param=True,
            )
            module_ir.params.append(tensor_ref)

        # Build forward graph
        if node.forward:
            builder = GraphBuilder(self.ctx, self.warnings)
            module_ir.forward_graph = builder.build_forward_graph(
                node.name,
                node.forward,
                node.param_decls,
            )
            if module_ir.forward_graph:
                infer_graph_shapes(module_ir.forward_graph, self.ctx.module_params)

        # Build backward graph
        if node.backward and module_ir.forward_graph:
            builder = GraphBuilder(self.ctx, self.warnings)
            module_ir.backward_graph = builder.build_backward_graph(
                node.name,
                node.backward,
                node.param_decls,
                module_ir.forward_graph,
            )
            if module_ir.backward_graph:
                infer_graph_shapes(module_ir.backward_graph, self.ctx.module_params)

        return module_ir

    def lower_block(self, node: BlockNode) -> ModuleIR:
        """Lower a block AST node to Module IR."""
        self._build_context(node)

        module_ir = ModuleIR(
            name=node.name,
            config=self.ctx.module_params.copy(),
            is_block=True,
            extends=node.extends,
        )

        for param in node.param_decls:
            tensor_ref = TensorRef(
                name=param.name,
                is_param=True,
            )
            module_ir.params.append(tensor_ref)

        if node.forward:
            builder = GraphBuilder(self.ctx, self.warnings)
            module_ir.forward_graph = builder.build_forward_graph(
                node.name,
                node.forward,
                node.param_decls,
            )
            if module_ir.forward_graph:
                infer_graph_shapes(module_ir.forward_graph, self.ctx.module_params)

        if node.backward and module_ir.forward_graph:
            builder = GraphBuilder(self.ctx, self.warnings)
            module_ir.backward_graph = builder.build_backward_graph(
                node.name,
                node.backward,
                node.param_decls,
                module_ir.forward_graph,
            )
            if module_ir.backward_graph:
                infer_graph_shapes(module_ir.backward_graph, self.ctx.module_params)

        return module_ir

    def lower_model(self, node: ModelNode) -> ModuleIR:
        """Lower a model AST node to Module IR."""
        self._build_context(node)

        module_ir = ModuleIR(
            name=node.name,
            config=self.ctx.module_params.copy(),
            is_model=True,
        )

        for param in node.param_decls:
            tensor_ref = TensorRef(
                name=param.name,
                is_param=True,
            )
            module_ir.params.append(tensor_ref)

        if node.forward:
            builder = GraphBuilder(self.ctx, self.warnings)
            module_ir.forward_graph = builder.build_forward_graph(
                node.name,
                node.forward,
                node.param_decls,
            )
            if module_ir.forward_graph:
                infer_graph_shapes(module_ir.forward_graph, self.ctx.module_params)

        if node.backward and module_ir.forward_graph:
            builder = GraphBuilder(self.ctx, self.warnings)
            module_ir.backward_graph = builder.build_backward_graph(
                node.name,
                node.backward,
                node.param_decls,
                module_ir.forward_graph,
            )
            if module_ir.backward_graph:
                infer_graph_shapes(module_ir.backward_graph, self.ctx.module_params)

        # Process HuggingFace mappings
        if node.hf_config:
            hf_config = {}
            if node.hf_config.architecture:
                hf_config["architecture"] = node.hf_config.architecture
            if node.hf_config.config_class:
                hf_config["config_class"] = node.hf_config.config_class
            if node.hf_config.param_mapping:
                hf_config["param_mapping"] = node.hf_config.param_mapping
            if node.hf_config.extras:
                for key, value in node.hf_config.extras.items():
                    hf_config.setdefault(key, value)
            module_ir.hf_config_mapping = hf_config

        if node.hf_mapping:
            for mapping in node.hf_mapping.mappings:
                module_ir.hf_weight_mapping[mapping.internal_name] = mapping

        if node.hf_export:
            for mapping in node.hf_export.mappings:
                module_ir.hf_export_mapping[mapping.internal_name] = mapping

        return module_ir

    def _build_context(self, node):
        """Build compilation context from node parameters."""
        # Process module parameters
        for param in node.params:
            if param.default:
                evaluator = ExpressionEvaluator(self.ctx.module_params)
                try:
                    value = evaluator.evaluate(param.default)
                    self.ctx.module_params[param.name] = value
                except Exception:
                    self.ctx.module_params[param.name] = None
            else:
                self.ctx.module_params[param.name] = None

        # Process let bindings
        for binding in node.let_bindings:
            evaluator = ExpressionEvaluator(self.ctx.module_params)
            try:
                value = evaluator.evaluate(binding.value)
                self.ctx.module_params[binding.name] = value
            except Exception:
                pass

        # Check constraints
        for constraint in node.constraints:
            evaluator = ExpressionEvaluator(self.ctx.module_params)
            try:
                result = evaluator.evaluate(constraint.condition)
                if not result:
                    raise DSLError(
                        ErrorCode.E027,
                        f"Constraint failed: {constraint.message}",
                    )
            except DSLError:
                raise
            except Exception:
                # Constraint couldn't be evaluated - defer to runtime
                pass


def lower_program(program: Program, warnings: Optional[WarningCollector] = None) -> List[ModuleIR]:
    """Lower an entire program to Module IRs."""
    warnings = warnings or WarningCollector()
    lowerer = ModuleLowerer(warnings)

    results = []

    # Lower all modules
    for module in program.modules:
        results.append(lowerer.lower_module(module))

    # Lower all blocks
    for block in program.blocks:
        results.append(lowerer.lower_block(block))

    # Lower all models
    for model in program.models:
        results.append(lowerer.lower_model(model))

    return results
