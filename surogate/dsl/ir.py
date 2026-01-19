"""
Intermediate Representation (IR) for Module DSL

Defines two levels of IR:
1. Graph IR: High-level functional representation (DAG of operations)
2. Schedule IR: Low-level imperative representation with buffer assignments

The Graph IR is produced by the lowering phase from AST.
The Schedule IR is produced by the scheduling phase for execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from enum import Enum, auto

from .types import (
    Dtype,
    Shape,
    TensorTypeSpec,
    MemoryMode,
    HookPoint,
    HookMode,
    ShardStrategy,
    TransposeMode,
)


# =============================================================================
# Kernel Types (Operations)
# =============================================================================


class KernelType(str, Enum):
    """Enumeration of available kernel types."""

    # Linear algebra
    MATMUL = "matmul"
    BATCHED_MATMUL = "batched_matmul"
    GROUPED_GEMM = "grouped_gemm"

    # Normalization
    RMSNORM = "rmsnorm"
    FUSED_RESIDUAL_RMSNORM = "fused_residual_rmsnorm"
    LAYERNORM = "layernorm"

    # Activations
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    SILU = "silu"
    RELU = "relu"
    RELU2 = "relu2"
    GELU = "gelu"
    SOFTMAX = "softmax"

    # Attention
    FLASH_ATTENTION = "flash_attention"
    ROPE = "rope"
    QK_NORM = "qk_norm"

    # Tensor manipulation
    SPLIT = "split"
    CONCAT = "concat"
    VIEW = "view"
    TRANSPOSE = "transpose"
    PERMUTE = "permute"
    CONTIGUOUS = "contiguous"
    COPY = "copy"

    # Elementwise
    ADD = "add"
    MUL = "mul"
    SCALE = "scale"
    ADD3 = "add3"

    # Reduction
    REDUCE_SUM = "reduce_sum"
    REDUCE_MEAN = "reduce_mean"
    REDUCE_MAX = "reduce_max"

    # Embedding
    EMBEDDING = "embedding"

    # MoE
    MOE_ROUTER = "moe_router"
    MOE_PERMUTE = "moe_permute"
    MOE_UNPERMUTE = "moe_unpermute"

    # Mamba/SSM
    MAMBA_CONV1D = "mamba_conv1d"
    MAMBA_SELECTIVE_SCAN = "mamba_selective_scan"

    # Utility
    ZEROS = "zeros"
    ONES = "ones"
    FILL = "fill"

    # Identity (no-op)
    IDENTITY = "identity"

    # Custom/user-defined
    CUSTOM = "custom"


# =============================================================================
# Graph IR Nodes
# =============================================================================


@dataclass
class TensorRef:
    """Reference to a tensor in the IR."""

    name: str
    dtype: Dtype = Dtype.BF16
    shape: Optional[Shape] = None
    is_param: bool = False  # True if this is a weight/parameter
    is_input: bool = False  # True if this is a graph input
    is_output: bool = False  # True if this is a graph output


@dataclass
class OpNode:
    """Operation node in Graph IR.

    Represents a single computation operation.
    """

    id: str  # Unique node ID
    kernel_type: KernelType
    name: str  # Operation name (for debugging)

    # Inputs and outputs
    inputs: List[str]  # Names of input tensors
    outputs: List[str]  # Names of output tensors

    # Operation attributes
    attrs: Dict[str, Any] = field(default_factory=dict)

    # Annotations from DSL
    memory_mode: MemoryMode = MemoryMode.TEMPORARY
    hook_point: Optional[HookPoint] = None
    hook_mode: Optional[HookMode] = None
    shard_strategy: Optional[ShardStrategy] = None

    # Layer index (for multi-layer models)
    layer_idx: Optional[int] = None

    # Source location (for debugging)
    source_loc: Optional[str] = None

    def __str__(self) -> str:
        inputs_str = ", ".join(self.inputs)
        outputs_str = ", ".join(self.outputs)
        return f"{outputs_str} = {self.name}({inputs_str})"


@dataclass
class Edge:
    """Edge in Graph IR (data flow between nodes)."""

    source_node: str  # Node ID
    source_output: str  # Output tensor name
    dest_node: str  # Node ID
    dest_input: str  # Input tensor name
    tensor_type: TensorTypeSpec  # Type of data flowing on edge


@dataclass
class GraphIR:
    """Graph IR: Functional representation of computation.

    This is the high-level IR produced by lowering from AST.
    """

    name: str

    # Tensors
    inputs: Dict[str, TensorRef] = field(default_factory=dict)
    outputs: Dict[str, TensorRef] = field(default_factory=dict)
    params: Dict[str, TensorRef] = field(default_factory=dict)
    intermediates: Dict[str, TensorRef] = field(default_factory=dict)

    # Operations (nodes in topological order)
    nodes: List[OpNode] = field(default_factory=list)

    # Edges (explicit data flow)
    edges: List[Edge] = field(default_factory=list)

    # Saved tensors for backward
    save_list: List[str] = field(default_factory=list)

    # Recompute list
    recompute_list: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Optional[OpNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_tensor(self, name: str) -> Optional[TensorRef]:
        """Get tensor reference by name."""
        if name in self.inputs:
            return self.inputs[name]
        if name in self.outputs:
            return self.outputs[name]
        if name in self.params:
            return self.params[name]
        if name in self.intermediates:
            return self.intermediates[name]
        return None

    def topological_sort(self) -> List[OpNode]:
        """Return nodes in topological order."""
        # Build dependency graph
        deps: Dict[str, Set[str]] = {node.id: set() for node in self.nodes}
        tensor_producer: Dict[str, str] = {}

        for node in self.nodes:
            for out in node.outputs:
                tensor_producer[out] = node.id

        for node in self.nodes:
            for inp in node.inputs:
                if inp in tensor_producer:
                    deps[node.id].add(tensor_producer[inp])

        # Kahn's algorithm
        result = []
        ready = [n for n in self.nodes if not deps[n.id]]

        while ready:
            node = ready.pop(0)
            result.append(node)
            for other in self.nodes:
                if node.id in deps[other.id]:
                    deps[other.id].remove(node.id)
                    if not deps[other.id] and other not in result and other not in ready:
                        ready.append(other)

        return result


# =============================================================================
# Schedule IR
# =============================================================================


class BufferKind(Enum):
    """Kind of buffer in Schedule IR."""

    ACTIVATION = auto()  # Activation tensor
    GRADIENT = auto()    # Gradient tensor
    WEIGHT = auto()      # Model parameter
    WORKSPACE = auto()   # Temporary workspace


@dataclass
class BufferDecl:
    """Buffer declaration in Schedule IR."""

    id: str
    size_bytes: int
    kind: BufferKind
    dtype: Dtype = Dtype.BF16

    # Lifetime information
    lifetime_start: int = 0  # First op index using this buffer
    lifetime_end: int = -1   # Last op index using this buffer

    # Aliasing
    aliases: List[str] = field(default_factory=list)
    alias_offset: int = 0  # Offset into aliased buffer


@dataclass
class BufferRef:
    """Reference to a buffer (or portion of a buffer)."""

    buffer_id: str
    offset: int = 0
    size: int = 0


class SyncKind(Enum):
    """Kind of synchronization point."""

    STREAM_SYNC = auto()  # CUDA stream synchronization
    NCCL_WAIT = auto()    # NCCL communication wait
    EVENT_WAIT = auto()   # CUDA event wait


@dataclass
class SyncPoint:
    """Synchronization point in schedule."""

    after_op: int  # Operation index after which to sync
    kind: SyncKind
    comm_group: Optional[str] = None


class StreamAssignment(Enum):
    """CUDA stream assignment for operations."""

    COMPUTE = auto()  # Main compute stream
    NCCL = auto()     # NCCL communication stream
    COPY = auto()     # Memory copy stream
    OVERLAP = auto()  # Overlap stream for async ops


@dataclass
class RecomputeSegment:
    """Segment of operations to recompute."""

    start_op: int  # First op to recompute
    end_op: int    # Last op to recompute
    frontier_buffers: List[str]  # Buffers available at start


@dataclass
class ScheduledOp:
    """Scheduled operation in Schedule IR."""

    order: int  # Execution order
    kernel: str  # Kernel name to invoke
    kernel_type: KernelType

    # Buffer references
    inputs: List[BufferRef]
    outputs: List[BufferRef]

    # Scheduling info
    stream: StreamAssignment = StreamAssignment.COMPUTE

    # Recompute info (if this op is part of recompute)
    recompute: Optional[RecomputeSegment] = None

    # Original node info
    original_node_id: Optional[str] = None
    layer_idx: Optional[int] = None

    # Hook info
    hook_point: Optional[HookPoint] = None


@dataclass
class ScheduleIR:
    """Schedule IR: Imperative execution plan.

    This is the low-level IR ready for execution.
    """

    name: str

    # Buffers
    buffers: Dict[str, BufferDecl] = field(default_factory=dict)

    # Scheduled operations
    ops: List[ScheduledOp] = field(default_factory=list)

    # Synchronization points
    sync_points: List[SyncPoint] = field(default_factory=list)

    # Activation layout (generated struct)
    activation_struct: Dict[str, Tuple[int, int, Dtype]] = field(default_factory=dict)
    total_activation_size: int = 0

    # Gradient layout
    gradient_struct: Dict[str, Tuple[int, int, Dtype]] = field(default_factory=dict)
    total_gradient_size: int = 0

    # Weight mapping (name -> buffer)
    weight_mapping: Dict[str, str] = field(default_factory=dict)

    def compute_memory_usage(self) -> Dict[str, int]:
        """Compute memory usage by category."""
        usage = {
            "activations": 0,
            "gradients": 0,
            "weights": 0,
            "workspace": 0,
        }

        for buf in self.buffers.values():
            if buf.kind == BufferKind.ACTIVATION:
                usage["activations"] += buf.size_bytes
            elif buf.kind == BufferKind.GRADIENT:
                usage["gradients"] += buf.size_bytes
            elif buf.kind == BufferKind.WEIGHT:
                usage["weights"] += buf.size_bytes
            elif buf.kind == BufferKind.WORKSPACE:
                usage["workspace"] += buf.size_bytes

        return usage


# =============================================================================
# Module IR (Container for forward + backward graphs)
# =============================================================================


@dataclass
class ModuleIR:
    """IR for a complete module (forward + backward)."""

    name: str

    # Module parameters
    params: List[TensorRef] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    # Forward graph
    forward_graph: Optional[GraphIR] = None

    # Backward graph (if defined)
    backward_graph: Optional[GraphIR] = None

    # Scheduled versions (after scheduling phase)
    forward_schedule: Optional[ScheduleIR] = None
    backward_schedule: Optional[ScheduleIR] = None

    # HuggingFace mapping
    hf_weight_mapping: Dict[str, str] = field(default_factory=dict)
    hf_config_mapping: Dict[str, str] = field(default_factory=dict)

    # Metadata
    is_block: bool = False
    is_model: bool = False
    extends: Optional[str] = None


# =============================================================================
# Compilation Context
# =============================================================================


@dataclass
class CompilationContext:
    """Context for IR compilation/lowering."""

    # Symbol table (name -> type)
    symbols: Dict[str, TensorTypeSpec] = field(default_factory=dict)

    # Module parameters (resolved values)
    module_params: Dict[str, Any] = field(default_factory=dict)

    # Current layer index (for block instantiation)
    layer_idx: Optional[int] = None

    # Imported modules
    imports: Dict[str, "ModuleIR"] = field(default_factory=dict)

    # Standard library primitives
    primitives: Dict[str, "PrimitiveSpec"] = field(default_factory=dict)

    # Counter for generating unique node IDs
    _node_counter: int = 0

    def new_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def resolve_symbol(self, name: str) -> Optional[TensorTypeSpec]:
        """Resolve a symbol to its type."""
        return self.symbols.get(name)

    def add_symbol(self, name: str, type_spec: TensorTypeSpec):
        """Add a symbol to the table."""
        self.symbols[name] = type_spec


@dataclass
class PrimitiveSpec:
    """Specification for a primitive operation."""

    name: str
    kernel_type: KernelType

    # IO specification
    input_types: Dict[str, TensorTypeSpec] = field(default_factory=dict)
    output_types: Dict[str, TensorTypeSpec] = field(default_factory=dict)

    # Default attributes
    default_attrs: Dict[str, Any] = field(default_factory=dict)

    # What to save for backward
    save: List[str] = field(default_factory=list)

    # Backward specification
    backward_kernel: Optional[KernelType] = None

    # Implementation
    forward_impl: Optional[str] = None
    backward_impl: Optional[str] = None
