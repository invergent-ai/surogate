# surogate/eval/backend/__init__.py
from .base import ExecutionBackend
from .local import LocalBackend

__all__ = ['ExecutionBackend', 'LocalBackend']