"""
Tensor Type Annotation for Python DSL

Provides Tensor[...] syntax for annotating tensor shapes and dtypes in Python.

Example:
    from surogate.dsl.tensor_type import Tensor, Array
    from surogate.dsl.dim import Dim, B, T

    class MyBlock:
        def __init__(self, d_model, num_heads):
            # Define typed dimensions bound to config parameters
            self.C = Dim("d_model")
            self.H = Dim("num_heads")
            self.D = self.C // self.H  # DimExpr for computed dimensions

        # Annotations use strings (evaluated at class definition time)
        @forward
        def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
            # Graph operations use Dim objects (evaluated at runtime)
            with graph() as g:
                x_flat = g.view(x, shape=[B * T, self.C])
                ...

        # With explicit dtype
        @param
        def weight(self) -> Tensor["C", "C", "fp32"]:
            ...

        # Computed dimensions use string expressions
        @param
        def qkv_weight(self) -> Tensor["QKV", "D"]:
            ...

    # Array of modules
    blocks: Array["n_layers", "Qwen3Block"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, TYPE_CHECKING

from .dim import Dim, DimExpr, ConcreteDimValue
from .types import (
    Dtype,
    Shape,
    TensorTypeSpec,
    SymbolicDim,
    ConcreteDim,
    ComputedDim,
    VariadicDim,
)

if TYPE_CHECKING:
    from .types import Dim as TypeDim


# Known dtype strings
_DTYPE_STRINGS = frozenset({
    "bf16", "fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", "int32",
})

# Type for dimension in TensorAnnotation
# Supports: strings, Dim, DimExpr, ConcreteDimValue, or int
DimType = str | Dim | DimExpr | ConcreteDimValue


@dataclass(frozen=True)
class TensorAnnotation:
    """Runtime representation of a Tensor[...] annotation.

    This is what Tensor.__class_getitem__ returns. It carries shape and dtype
    information that can be extracted at decoration time.
    """

    dims: tuple[DimType, ...]
    dtype: str = "bf16"
    optional: bool = False

    def __repr__(self) -> str:
        dims_str = ", ".join(str(d) for d in self.dims)
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
        """Convert to the TensorTypeSpec format."""
        parsed_dims: list[SymbolicDim | ConcreteDim | ComputedDim | VariadicDim] = []

        for d in self.dims:
            if isinstance(d, str):
                # String dimension - could be symbolic like "C" or variadic like "..."
                if d == "...":
                    parsed_dims.append(VariadicDim())
                else:
                    parsed_dims.append(SymbolicDim(d))
            elif isinstance(d, Dim):
                parsed_dims.append(SymbolicDim(d.name))
            elif isinstance(d, DimExpr):
                parsed_dims.append(ComputedDim(d.to_expr_string(), d.to_expr_string()))
            elif isinstance(d, ConcreteDimValue):
                parsed_dims.append(ConcreteDim(d.value))
            else:
                raise TypeError(f"Invalid dimension type: {type(d)}")

        dtype = Dtype.from_string(self.dtype) if self.dtype else Dtype.BF16
        return TensorTypeSpec(Shape(parsed_dims), dtype, self.optional)


class _TensorMeta(type):
    """Metaclass for Tensor to support subscript syntax."""

    def __getitem__(cls, params: Any) -> TensorAnnotation:
        """Handle Tensor[dim1, dim2, ...] or Tensor[dim1, dim2, dtype].

        Dimensions must be Dim, DimExpr, ConcreteDimValue, or int.
        """
        if not isinstance(params, tuple):
            params = (params,)

        if len(params) == 0:
            raise TypeError("Tensor requires at least one dimension")

        # Check if last param is a dtype string
        dtype = "bf16"
        dims = params

        if params and isinstance(params[-1], str) and params[-1].lower() in _DTYPE_STRINGS:
            dtype = params[-1].lower()
            dims = params[:-1]

        # Process dimensions - str, Dim, DimExpr, ConcreteDimValue, or int allowed
        processed_dims: list[DimType] = []
        for d in dims:
            if isinstance(d, str):
                # String dimensions for annotations (e.g., "B", "T", "C")
                processed_dims.append(d)
            elif isinstance(d, (Dim, DimExpr, ConcreteDimValue)):
                processed_dims.append(d)
            elif isinstance(d, int):
                processed_dims.append(ConcreteDimValue(d))
            else:
                raise TypeError(
                    f"Tensor dimensions must be str, Dim, DimExpr, or int, "
                    f"got {type(d).__name__}: {d!r}"
                )

        return TensorAnnotation(dims=tuple(processed_dims), dtype=dtype)

    def __instancecheck__(cls, instance: Any) -> bool:
        """Allow isinstance checks."""
        return isinstance(instance, TensorAnnotation)


class Tensor(metaclass=_TensorMeta):
    """Tensor type annotation with symbolic shape support.

    Use as a type hint to annotate tensor shapes. Annotations use STRING dimensions
    because Python evaluates type hints at class definition time (before `self` exists).

        from surogate.dsl.dim import Dim, B, T

        class MyBlock:
            def __init__(self, d_model, num_heads):
                # Define typed dimensions for graph operations
                self.C = Dim("d_model")
                self.H = Dim("num_heads")
                self.D = self.C // self.H

            # Annotations use strings (resolved at compile time)
            def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
                # Graph operations use Dim objects (evaluated at runtime)
                with graph() as g:
                    x_flat = g.view(x, shape=[B * T, self.C])
                    ...

            # Concrete dimensions
            def f(self, y: Tensor["C", 4096]) -> ...:
                ...

            # With explicit dtype
            def g(self, w: Tensor["K", "N", "fp32"]) -> ...:
                ...

            # Computed dimensions use symbolic strings
            def h(self, qkv: Tensor["B", "T", "QKV", "D"]) -> ...:
                ...

            # Optional tensors
            def i(self, bias: Tensor["O"] | None) -> ...:
                ...

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
