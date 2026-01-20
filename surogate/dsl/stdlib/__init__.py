"""
Standard Library for Python DSL

This package contains the standard library modules, blocks, and primitives
ported from the .module DSL files to Python decorator syntax.

Example:
    from surogate.dsl.stdlib.primitives import matmul
    from surogate.dsl.stdlib.modules import Linear, RMSNorm
    from surogate.dsl.stdlib.blocks import DenseTransformerBlock
    from surogate.dsl.stdlib.models import Qwen3Model
"""

# Direct imports to ensure modules are registered when this package is imported
from . import primitives
from . import modules
from . import blocks
from . import models

__all__ = ["primitives", "modules", "blocks", "models"]
