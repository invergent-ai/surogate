"""
First-Class Symbolic Dimensions for DSL

Provides typed dimension objects with operator overloading for building
dimension expressions that serialize to IR format.

Usage:
    from surogate.dsl.dim import Dim, B, T

    class MyBlock:
        def __init__(self, d_model, num_heads, ...):
            self.C = Dim("d_model")
            self.H = Dim("num_heads")
            self.D = self.C // self.H  # DimExpr

        @param
        def weight(self) -> Tensor[self.C, self.C]:
            ...

        @forward
        def forward(self, x: Tensor[B, T, self.C]) -> Tensor[B, T, self.C]:
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ConcreteDimValue:
    """A concrete integer dimension value."""

    value: int

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"ConcreteDimValue({self.value})"

    def to_expr_string(self) -> str:
        return str(self.value)

    def __add__(self, other: DimLike) -> DimExpr:
        return DimExpr("+", self, _to_dim_term(other))

    def __radd__(self, other: DimLike) -> DimExpr:
        return DimExpr("+", _to_dim_term(other), self)

    def __sub__(self, other: DimLike) -> DimExpr:
        return DimExpr("-", self, _to_dim_term(other))

    def __rsub__(self, other: DimLike) -> DimExpr:
        return DimExpr("-", _to_dim_term(other), self)

    def __mul__(self, other: DimLike) -> DimExpr:
        return DimExpr("*", self, _to_dim_term(other))

    def __rmul__(self, other: DimLike) -> DimExpr:
        return DimExpr("*", _to_dim_term(other), self)

    def __truediv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", self, _to_dim_term(other))

    def __rtruediv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", _to_dim_term(other), self)

    def __floordiv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", self, _to_dim_term(other))

    def __rfloordiv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", _to_dim_term(other), self)


@dataclass(frozen=True)
class Dim:
    """A symbolic dimension that can participate in expressions.

    Args:
        name: The config parameter name this dimension binds to.

    Usage:
        self.C = Dim("d_model")           # Bind to config parameter
        self.Hq = Dim("num_query_heads")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D  # Creates DimExpr
    """

    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Dim({self.name!r})"

    def to_expr_string(self) -> str:
        """Convert to string expression for IR."""
        return self.name

    def __add__(self, other: DimLike) -> DimExpr:
        return DimExpr("+", self, _to_dim_term(other))

    def __radd__(self, other: DimLike) -> DimExpr:
        return DimExpr("+", _to_dim_term(other), self)

    def __sub__(self, other: DimLike) -> DimExpr:
        return DimExpr("-", self, _to_dim_term(other))

    def __rsub__(self, other: DimLike) -> DimExpr:
        return DimExpr("-", _to_dim_term(other), self)

    def __mul__(self, other: DimLike) -> DimExpr:
        return DimExpr("*", self, _to_dim_term(other))

    def __rmul__(self, other: DimLike) -> DimExpr:
        return DimExpr("*", _to_dim_term(other), self)

    def __truediv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", self, _to_dim_term(other))

    def __rtruediv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", _to_dim_term(other), self)

    def __floordiv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", self, _to_dim_term(other))

    def __rfloordiv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", _to_dim_term(other), self)


# Type aliases
DimTerm = Union[Dim, ConcreteDimValue, "DimExpr"]
DimLike = Union[Dim, "DimExpr", ConcreteDimValue, int]


def _to_dim_term(value: DimLike) -> DimTerm:
    """Convert a dimension-like value to a DimTerm."""
    if isinstance(value, (Dim, DimExpr, ConcreteDimValue)):
        return value
    if isinstance(value, int):
        return ConcreteDimValue(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to dimension term")


# Operator precedence for parenthesization
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "//": 2}


def _needs_parentheses(inner_op: str, outer_op: str, is_left: bool) -> bool:
    """Determine if parentheses are needed based on precedence."""
    inner_prec = _PRECEDENCE.get(inner_op, 0)
    outer_prec = _PRECEDENCE.get(outer_op, 0)

    if inner_prec < outer_prec:
        return True
    # Right-associativity issues for - and //
    if inner_prec == outer_prec and not is_left and outer_op in ("-", "//"):
        return True
    return False


def _term_to_string(term: DimTerm, parent_op: str, is_left: bool) -> str:
    """Convert a term to string with appropriate parentheses."""
    if isinstance(term, (Dim, ConcreteDimValue)):
        return term.to_expr_string()
    if isinstance(term, DimExpr):
        needs_parens = _needs_parentheses(term.op, parent_op, is_left)
        inner = term.to_expr_string()
        return f"({inner})" if needs_parens else inner
    return str(term)


def _collect_dim_names(term: DimTerm, names: set[str]) -> None:
    """Recursively collect all dimension names from an expression."""
    if isinstance(term, Dim):
        names.add(term.name)
    elif isinstance(term, DimExpr):
        _collect_dim_names(term.left, names)
        _collect_dim_names(term.right, names)
    # ConcreteDimValue has no names to collect


@dataclass(frozen=True)
class DimExpr:
    """An expression combining dimensions.

    Created by arithmetic operations on Dim objects:
        expr = (Hq + 2 * Hkv) * D  # DimExpr

    Internally represented as a binary tree of operations.
    """

    op: str  # "+", "-", "*", "//"
    left: DimTerm
    right: DimTerm

    def __str__(self) -> str:
        return self.to_expr_string()

    def __repr__(self) -> str:
        return f"DimExpr({self.op!r}, {self.left!r}, {self.right!r})"

    def to_expr_string(self) -> str:
        """Convert to string expression for IR.

        Uses minimal parentheses based on operator precedence.
        """
        left_str = _term_to_string(self.left, self.op, is_left=True)
        right_str = _term_to_string(self.right, self.op, is_left=False)
        return f"{left_str} {self.op} {right_str}"

    def get_referenced_dims(self) -> set[str]:
        """Return all dimension names referenced in this expression."""
        names: set[str] = set()
        _collect_dim_names(self, names)
        return names

    def validate(self, config: dict[str, int]) -> None:
        """Validate that all referenced dims exist in config.

        Args:
            config: Dictionary mapping dimension names to concrete values.

        Raises:
            ValueError: If a referenced dimension is not in config.
        """
        for name in self.get_referenced_dims():
            if name not in config:
                raise ValueError(f"Dimension '{name}' not found in config")

    def evaluate(self, config: dict[str, int]) -> int:
        """Evaluate expression to concrete value.

        Args:
            config: Dictionary mapping dimension names to concrete values.

        Returns:
            The computed integer value.

        Raises:
            ValueError: If a referenced dimension is not in config.
        """
        return _evaluate_term(self, config)

    def __add__(self, other: DimLike) -> DimExpr:
        return DimExpr("+", self, _to_dim_term(other))

    def __radd__(self, other: DimLike) -> DimExpr:
        return DimExpr("+", _to_dim_term(other), self)

    def __sub__(self, other: DimLike) -> DimExpr:
        return DimExpr("-", self, _to_dim_term(other))

    def __rsub__(self, other: DimLike) -> DimExpr:
        return DimExpr("-", _to_dim_term(other), self)

    def __mul__(self, other: DimLike) -> DimExpr:
        return DimExpr("*", self, _to_dim_term(other))

    def __rmul__(self, other: DimLike) -> DimExpr:
        return DimExpr("*", _to_dim_term(other), self)

    def __truediv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", self, _to_dim_term(other))

    def __rtruediv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", _to_dim_term(other), self)

    def __floordiv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", self, _to_dim_term(other))

    def __rfloordiv__(self, other: DimLike) -> DimExpr:
        return DimExpr("//", _to_dim_term(other), self)


def _evaluate_term(term: DimTerm, config: dict[str, int]) -> int:
    """Evaluate a dimension term to a concrete value."""
    if isinstance(term, ConcreteDimValue):
        return term.value
    if isinstance(term, Dim):
        if term.name not in config:
            raise ValueError(f"Dimension '{term.name}' not found in config")
        return config[term.name]
    if isinstance(term, DimExpr):
        left_val = _evaluate_term(term.left, config)
        right_val = _evaluate_term(term.right, config)
        if term.op == "+":
            return left_val + right_val
        if term.op == "-":
            return left_val - right_val
        if term.op == "*":
            return left_val * right_val
        if term.op == "//":
            if right_val == 0:
                raise ValueError("Division by zero in dimension expression")
            return left_val // right_val
        raise ValueError(f"Unknown operator: {term.op}")
    raise TypeError(f"Cannot evaluate {type(term).__name__}")


# =============================================================================
# Predefined Runtime Dimensions
# =============================================================================

# Batch dimension (resolved at runtime)
B = Dim("B")

# Sequence length (resolved at runtime)
T = Dim("T")


# =============================================================================
# Utility Functions
# =============================================================================


def dim_to_ir(d: Dim | DimExpr | ConcreteDimValue | int) -> str | int:
    """Convert a dimension to IR format (string or int)."""
    if isinstance(d, int):
        return d
    if isinstance(d, ConcreteDimValue):
        return d.value
    if isinstance(d, (Dim, DimExpr)):
        return d.to_expr_string()
    raise TypeError(f"Cannot convert {type(d).__name__} to IR dimension")
