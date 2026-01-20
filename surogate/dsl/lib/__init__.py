"""
Standard Library for Python DSL

This package contains the standard library modules, blocks, and primitives.

Example:
    from surogate.dsl.lib.primitives import matmul
    from surogate.dsl.lib.modules import Linear, RMSNorm
    from surogate.dsl.lib.blocks import DenseTransformerBlock
    from surogate.dsl.lib.models import Qwen3Model
"""

# Direct imports to ensure modules are registered when this package is imported
from . import primitives
from . import modules
from . import blocks
from . import models

__all__ = ["primitives", "modules", "blocks", "models"]
