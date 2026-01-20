"""Common definitions for primitives."""

from enum import Enum


class TransposeMode(str, Enum):
    """Transpose mode for matmul operations."""
    NN = "NN"  # Neither transposed
    NT = "NT"  # B transposed
    TN = "TN"  # A transposed
    TT = "TT"  # Both transposed
