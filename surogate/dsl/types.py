"""
DSL Type System

Defines:
- Dtype: Data type specifiers (bf16, fp32, fp8_e4m3, etc.)
- Shape: Tensor shape with symbolic and concrete dimensions
- TensorTypeSpec: Full tensor type specification
- SymbolicDim: Runtime-determined dimension
- ConcreteDim: Compile-time known dimension
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional, List, Dict, Any, Tuple
import math


class Dtype(str, Enum):
    """Data type specifiers supported by the DSL."""

    # Standard floating point
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    # FP8 variants
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"

    # FP4 (Blackwell+)
    FP4_E2M1 = "fp4_e2m1"

    # Integer types
    INT8 = "int8"
    INT32 = "int32"

    # Auto (inferred)
    AUTO = "auto"

    @property
    def bits(self) -> int:
        """Number of bits for this dtype."""
        return {
            Dtype.FP32: 32,
            Dtype.FP16: 16,
            Dtype.BF16: 16,
            Dtype.FP8_E4M3: 8,
            Dtype.FP8_E5M2: 8,
            Dtype.FP4_E2M1: 4,
            Dtype.INT8: 8,
            Dtype.INT32: 32,
            Dtype.AUTO: 16,  # Default to bf16
        }[self]

    @property
    def bytes(self) -> float:
        """Number of bytes for this dtype."""
        return self.bits / 8

    @classmethod
    def from_string(cls, s: str) -> "Dtype":
        """Parse dtype from string."""
        s = s.lower()
        for dtype in cls:
            if dtype.value == s:
                return dtype
        raise ValueError(f"Unknown dtype: {s}")

    def is_fp8(self) -> bool:
        return self in (Dtype.FP8_E4M3, Dtype.FP8_E5M2)

    def is_fp4(self) -> bool:
        return self == Dtype.FP4_E2M1

    def is_quantized(self) -> bool:
        return self.is_fp8() or self.is_fp4() or self == Dtype.INT8

    def is_float(self) -> bool:
        return self in (Dtype.FP32, Dtype.FP16, Dtype.BF16, Dtype.FP8_E4M3, Dtype.FP8_E5M2, Dtype.FP4_E2M1)

    def is_int(self) -> bool:
        return self in (Dtype.INT8, Dtype.INT32)


@dataclass(frozen=True)
class SymbolicDim:
    """A symbolic dimension that is resolved at runtime.

    Examples:
        SymbolicDim("B")      # Batch dimension
        SymbolicDim("T")      # Sequence length
        SymbolicDim("d_model")  # Model dimension (from param)
    """

    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"SymbolicDim({self.name!r})"


@dataclass(frozen=True)
class ConcreteDim:
    """A concrete dimension known at compile time.

    Example:
        ConcreteDim(4096)  # d_model = 4096
    """

    value: int

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"ConcreteDim({self.value})"


@dataclass(frozen=True)
class ComputedDim:
    """A dimension computed from other dimensions.

    Example:
        ComputedDim("d_model // num_heads")
        ComputedDim("Hq + 2 * Hkv", "Hq + 2 * Hkv")  # With optional name
    """

    expression: str
    name: str = ""  # Optional name, defaults to expression

    def __post_init__(self) -> None:
        # If name not provided, use expression as name
        if not self.name:
            object.__setattr__(self, "name", self.expression)

    def __str__(self) -> str:
        return self.expression

    def __repr__(self) -> str:
        return f"ComputedDim({self.expression!r})"


@dataclass(frozen=True)
class VariadicDim:
    """Variadic batch dimensions (leading dimensions).

    Written as * in DSL: [*, T, C] means arbitrary leading batch dims.
    """

    def __str__(self) -> str:
        return "*"

    def __repr__(self) -> str:
        return "VariadicDim()"


# Type alias for any dimension type
Dim = Union[SymbolicDim, ConcreteDim, ComputedDim, VariadicDim]


@dataclass
class Shape:
    """Tensor shape with support for symbolic dimensions.

    Examples:
        Shape([SymbolicDim("B"), SymbolicDim("T"), ConcreteDim(4096)])
        Shape([VariadicDim(), SymbolicDim("C")])  # [*, C]
    """

    dims: List[Dim]

    def __str__(self) -> str:
        return "[" + ", ".join(str(d) for d in self.dims) + "]"

    def __repr__(self) -> str:
        return f"Shape({self.dims!r})"

    def __len__(self) -> int:
        return len(self.dims)

    def __getitem__(self, idx: int) -> Dim:
        return self.dims[idx]

    def __iter__(self):
        return iter(self.dims)

    @property
    def rank(self) -> int:
        """Number of dimensions (excluding variadic)."""
        return sum(1 for d in self.dims if not isinstance(d, VariadicDim))

    @property
    def has_variadic(self) -> bool:
        """Whether shape has variadic leading dimensions."""
        return any(isinstance(d, VariadicDim) for d in self.dims)

    def is_concrete(self) -> bool:
        """Whether all dimensions are concrete."""
        return all(isinstance(d, ConcreteDim) for d in self.dims)

    def concrete_size(self) -> Optional[int]:
        """Total number of elements if fully concrete."""
        if not self.is_concrete():
            return None
        result = 1
        for d in self.dims:
            if isinstance(d, ConcreteDim):
                result *= d.value
        return result


@dataclass
class TensorTypeSpec:
    """Full tensor type specification.

    Combines shape and dtype information.

    Example:
        TensorTypeSpec(
            shape=Shape([SymbolicDim("B"), SymbolicDim("T"), ConcreteDim(4096)]),
            dtype=Dtype.BF16
        )
    """

    shape: Shape
    dtype: Dtype = Dtype.BF16
    optional: bool = False  # Whether tensor can be None

    def __str__(self) -> str:
        dtype_str = f", {self.dtype.value}" if self.dtype != Dtype.AUTO else ""
        opt_str = "?" if self.optional else ""
        return f"{self.shape}{dtype_str}{opt_str}"

    def __repr__(self) -> str:
        return f"TensorTypeSpec({self.shape!r}, {self.dtype!r}, optional={self.optional})"

    @classmethod
    def from_dims(
        cls,
        dims: List[Union[str, int]],
        dtype: Union[str, Dtype] = Dtype.BF16,
        optional: bool = False,
    ) -> "TensorTypeSpec":
        """Create TensorTypeSpec from a list of dimension specs.

        Args:
            dims: List of dimension names (str) or concrete values (int)
            dtype: Data type
            optional: Whether tensor can be None
        """
        parsed_dims = []
        for d in dims:
            if d == "*":
                parsed_dims.append(VariadicDim())
            elif isinstance(d, int):
                parsed_dims.append(ConcreteDim(d))
            elif isinstance(d, str):
                parsed_dims.append(SymbolicDim(d))
            else:
                raise TypeError(f"Invalid dimension type: {type(d)}")

        if isinstance(dtype, str):
            dtype = Dtype.from_string(dtype)

        return cls(Shape(parsed_dims), dtype, optional)


@dataclass
class TupleType:
    """Tuple type for multiple tensors.

    Example:
        TupleType([TensorTypeSpec(...), TensorTypeSpec(...)])
    """

    elements: List[TensorTypeSpec]

    def __str__(self) -> str:
        return "(" + ", ".join(str(e) for e in self.elements) + ")"

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> TensorTypeSpec:
        return self.elements[idx]


@dataclass
class ArrayType:
    """Array type for homogeneous repeated elements (e.g., stacked layers).

    Example:
        ArrayType(32, ModuleRef("DenseTransformerBlock"))
    """

    size: Union[int, str]  # Size can be concrete or symbolic (e.g., "n_layers")
    element_type: Any  # Module reference or TensorTypeSpec

    def __str__(self) -> str:
        return f"[{self.size}] x {self.element_type}"


# Type for any DSL type
TypeSpec = Union[TensorTypeSpec, TupleType, ArrayType, Dtype]


@dataclass
class ConstExpr:
    """Compile-time constant expression.

    Represents expressions that can be evaluated at compile time.
    """

    value: Any  # Can be int, float, bool, str, or expression AST
    expr_str: str  # Original expression string for error messages

    def __str__(self) -> str:
        return self.expr_str

    def evaluate(self, env: Dict[str, Any]) -> Any:
        """Evaluate the expression given an environment of bindings."""
        if isinstance(self.value, (int, float, bool, str, type(None))):
            return self.value
        # For complex expressions, need to evaluate
        return eval(self.expr_str, {"__builtins__": {}}, env)


@dataclass
class Constraint:
    """Compile-time constraint.

    Example:
        Constraint(
            condition=ConstExpr(..., "C % H == 0"),
            message="d_model must be divisible by num_heads"
        )
    """

    condition: ConstExpr
    message: str

    def check(self, env: Dict[str, Any]) -> Tuple[bool, str]:
        """Check the constraint, return (passed, message)."""
        try:
            result = self.condition.evaluate(env)
            if not result:
                return False, self.message
            return True, ""
        except Exception as e:
            return False, f"Error evaluating constraint: {e}"


# Precision-related types

class TransposeMode(str, Enum):
    """Transpose mode for matmul operations."""

    NN = "NN"  # Neither transposed
    NT = "NT"  # B transposed
    TN = "TN"  # A transposed
    TT = "TT"  # Both transposed


class MemoryMode(str, Enum):
    """Memory lifecycle mode for tensors."""

    SAVE = "save"          # Save for backward pass
    RECOMPUTE = "recompute"  # Recompute in backward
    TEMPORARY = "temporary"  # Free immediately after use
    PIN = "pin"            # Keep in GPU memory
    OFFLOAD = "offload"    # Eligible for CPU offload
    STREAM = "stream"      # Stream from CPU/NVMe


class HookPoint(str, Enum):
    """Standard hook points in computation graph."""

    NONE = "none"
    AFTER_EMBEDDING = "AfterEmbedding"
    AFTER_QKV_PROJECTION = "AfterQKVProjection"
    AFTER_QK_NORM = "AfterQKNorm"
    BEFORE_ATTENTION = "BeforeAttention"
    AFTER_ATTENTION = "AfterAttention"
    AFTER_ATTN_OUT_PROJECTION = "AfterAttnOutProjection"
    BEFORE_RESIDUAL_ADD = "BeforeResidualAdd"
    AFTER_RESIDUAL_ADD = "AfterResidualAdd"
    AFTER_MLP_UP_PROJECTION = "AfterMLPUpProjection"
    AFTER_MLP_ACTIVATION = "AfterMLPActivation"
    AFTER_MLP_DOWN_PROJECTION = "AfterMLPDownProjection"
    AFTER_ROUTER_PROJECTION = "AfterRouterProjection"
    BEFORE_EXPERT_COMPUTE = "BeforeExpertCompute"
    AFTER_EXPERT_COMPUTE = "AfterExpertCompute"
    BEFORE_LM_HEAD = "BeforeLMHead"
    AFTER_LM_HEAD = "AfterLMHead"


class HookMode(str, Enum):
    """Hook execution mode."""

    OBSERVE = "observe"  # Read-only
    MODIFY = "modify"    # In-place modification
    REPLACE = "replace"  # Return replacement


class ShardStrategy(str, Enum):
    """Tensor sharding strategy for distributed execution."""

    COLUMN = "column"       # Shard along columns (output dim)
    ROW = "row"             # Shard along rows (input dim)
    REPLICATED = "replicated"  # Full copy on each device
    EXPERT = "expert"       # Distribute experts
    SEQUENCE = "sequence"   # Shard along sequence
    BATCH = "batch"         # Shard along batch
    HEAD = "head"           # Shard along head dimension


class ScalingStrategy(str, Enum):
    """Quantization scaling strategy."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_TOKEN = "per_token"
    PER_BLOCK = "per_block"
    DELAYED = "delayed"
    DYNAMIC = "dynamic"
