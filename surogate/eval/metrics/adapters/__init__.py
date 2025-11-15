# surogate/eval/metrics/adapters/__init__.py
"""
Adapters for external metric libraries.
"""

from .deepeval_adapter import DeepEvalAdapter

__all__ = [
    'DeepEvalAdapter',
]