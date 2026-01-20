"""
Tensor Type Annotation for Python DSL

Provides Tensor[...] syntax for annotating tensor shapes and dtypes in Python.

Example:
    from surogate.dsl.tensor_type import Tensor, Array

    def forward(x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        ...

    # With explicit dtype
    def f(x: Tensor["M", "K", "bf16"]) -> Tensor["M", "N", "fp32"]:
        ...

    # Optional tensor
    def g(bias: Tensor["O"] | None) -> ...:
        ...

    # Computed dimensions
    def h(qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"]) -> ...:
        ...

    # Array of modules
    blocks: Array["n_layers", "Qwen3Block"]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union, TYPE_CHECKING
import re

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


# Known dtype strings
_DTYPE_STRINGS = frozenset({
    "bf16", "fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", "int32",
})

# Pattern for computed dimensions (contains operators)
_COMPUTED_DIM_PATTERN = re.compile(r'[+\-*/%()\s]')


@dataclass(frozen=True)
class TensorAnnotation:
    """Runtime representation of a Tensor[...] annotation.

    This is what Tensor.__class_getitem__ returns. It carries shape and dtype
    information that can be extracted at decoration time.
    """
    dims: tuple[str | int, ...]
    dtype: str = "bf16"
    optional: bool = False

    def __repr__(self) -> str:
        dims_str = ", ".join(repr(d) if isinstance(d, str) else str(d) for d in self.dims)
        dtype_str = f", {self.dtype}" if self.dtype != "bf16" else ""
        opt_str = "?" if self.optional else ""
        return f"Tensor[{dims_str}{dtype_str}]{opt_str}"

    def __or__(self, other: Any) -> TensorAnnotation:
        """Support Tensor[...] | None for optional tensors."""
        if other is type(None):
            return TensorAnnotation(self.dims, self.dtype, optional=True)
        return NotImplemented

    def __ror__(self, other: Any) -> TensorAnnotation:
        """Support None | Tensor[...] for optional tensors."""
        if other is type(None):
            return TensorAnnotation(self.dims, self.dtype, optional=True)
        return NotImplemented

    def to_type_spec(self) -> TensorTypeSpec:
        """Convert to the existing TensorTypeSpec format."""
        parsed_dims: list[Dim] = []

        for d in self.dims:
            if d == "*":
                parsed_dims.append(VariadicDim())
            elif isinstance(d, int):
                parsed_dims.append(ConcreteDim(d))
            elif isinstance(d, str):
                # Check if it's a computed dimension (contains operators)
                if _COMPUTED_DIM_PATTERN.search(d):
                    # Extract a name from the expression (first identifier)
                    name_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', d)
                    name = name_match.group(1) if name_match else "computed"
                    parsed_dims.append(ComputedDim(name, d))
                else:
                    parsed_dims.append(SymbolicDim(d))
            else:
                raise TypeError(f"Invalid dimension type: {type(d)}")

        dtype = Dtype.from_string(self.dtype) if self.dtype else Dtype.BF16
        return TensorTypeSpec(Shape(parsed_dims), dtype, self.optional)


class _TensorMeta(type):
    """Metaclass for Tensor to support subscript syntax."""

    def __getitem__(cls, params: Any) -> TensorAnnotation:
        """Handle Tensor[dim1, dim2, ...] or Tensor[dim1, dim2, dtype]."""
        if not isinstance(params, tuple):
            params = (params,)

        if len(params) == 0:
            raise TypeError("Tensor requires at least one dimension")

        # Check if last param is a dtype
        dtype = "bf16"
        dims = params

        if params and isinstance(params[-1], str) and params[-1].lower() in _DTYPE_STRINGS:
            dtype = params[-1].lower()
            dims = params[:-1]

        # Validate dimensions
        for d in dims:
            if not isinstance(d, (str, int)):
                raise TypeError(f"Tensor dimensions must be str or int, got {type(d)}")

        return TensorAnnotation(dims=dims, dtype=dtype)

    def __instancecheck__(cls, instance: Any) -> bool:
        """Allow isinstance checks."""
        return isinstance(instance, TensorAnnotation)


class Tensor(metaclass=_TensorMeta):
    """Tensor type annotation with symbolic shape support.

    Use as a type hint to annotate tensor shapes:

        # Basic shapes
        x: Tensor["B", "T", "C"]              # 3D tensor with symbolic dims
        y: Tensor["M", 4096]                   # Mixed symbolic and concrete
        z: Tensor["*", "C"]                    # Variadic batch dimensions

        # With explicit dtype
        w: Tensor["K", "N", "fp32"]           # Float32 tensor
        q: Tensor["B", "T", "H", "D", "fp8_e4m3"]  # FP8 quantized

        # Computed dimensions
        qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"]  # Arithmetic on dims

        # Optional tensors
        bias: Tensor["O"] | None              # Can be None

    Supported dtypes: bf16, fp32, fp16, fp8_e4m3, fp8_e5m2, fp4_e2m1, int8, int32
    """

    # Prevent instantiation
    def __new__(cls, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Tensor is a type annotation, not instantiable. Use Tensor[...] syntax.")


# Alias for backward compatibility
TensorType = TensorAnnotation


@dataclass(frozen=True)
class ArrayAnnotation:
    """Runtime representation of an Array[size, element_type] annotation.

    Used for repeated elements like stacked transformer blocks.
    """
    size: str | int
    element_type: str

    def __repr__(self) -> str:
        return f"Array[{self.size!r}, {self.element_type!r}]"


class _ArrayMeta(type):
    """Metaclass for Array to support subscript syntax."""

    def __getitem__(cls, params: tuple[Any, Any]) -> ArrayAnnotation:
        """Handle Array[size, element_type]."""
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Array requires exactly two parameters: Array[size, element_type]")

        size, element_type = params

        if not isinstance(size, (str, int)):
            raise TypeError(f"Array size must be str or int, got {type(size)}")
        if not isinstance(element_type, str):
            raise TypeError(f"Array element_type must be str, got {type(element_type)}")

        return ArrayAnnotation(size=size, element_type=element_type)


class Array(metaclass=_ArrayMeta):
    """Array type annotation for repeated elements.

    Use for stacked layers or repeated module instances:

        blocks: Array["n_layers", "DenseTransformerBlock"]
        experts: Array[8, "ExpertMLP"]  # Concrete size

    The size can be symbolic (referencing a constructor parameter) or concrete.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Array is a type annotation, not instantiable. Use Array[size, type] syntax.")


def extract_tensor_annotation(annotation: Any) -> TensorAnnotation | None:
    """Extract TensorAnnotation from a type hint.

    Handles:
    - Direct TensorAnnotation
    - Optional types (Tensor[...] | None)
    - typing.Optional[Tensor[...]]
    """
    if isinstance(annotation, TensorAnnotation):
        return annotation

    # Handle Union types (for Optional)
    origin = getattr(annotation, "__origin__", None)
    if origin is Union:
        args = getattr(annotation, "__args__", ())
        for arg in args:
            if isinstance(arg, TensorAnnotation):
                return TensorAnnotation(arg.dims, arg.dtype, optional=True)
            if arg is type(None):
                continue

    return None


def extract_array_annotation(annotation: Any) -> ArrayAnnotation | None:
    """Extract ArrayAnnotation from a type hint."""
    if isinstance(annotation, ArrayAnnotation):
        return annotation
    return None
