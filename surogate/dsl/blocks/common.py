"""Common definitions for blocks."""

from enum import Enum


class ActivationFunc(str, Enum):
    """Activation function for MLP."""
    SwiGLU = "swiglu"
    SiLU = "silu"
    ReLU2 = "relu2"
    GELU = "gelu"


# Backwards compatibility alias
Activation = ActivationFunc
