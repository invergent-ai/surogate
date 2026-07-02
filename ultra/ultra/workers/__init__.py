"""Vendored worker-pool stack (standalone copy of the router's shared infra).

Workers are reached through OpenAI-compatible endpoints; ``WorkerPool`` adds disk
caching, budgeting, concurrency control and an n-sample helper. ``FakeProvider``
runs the whole stack offline and for free in tests.

``build_pool`` (live wiring) lives in ``ultra.workers.factory``.
"""

from .budget import BudgetTracker
from .cache import CompletionCache
from .pool import RateGate
from .providers import FakeProvider, OpenRouterProvider, Provider, RoutedOpenAIProvider, WorkerPool
from .types import Completion, Message, Sampling, ToolCall, ToolCompletion

__all__ = [
    "BudgetTracker",
    "CompletionCache",
    "Completion",
    "FakeProvider",
    "Message",
    "OpenRouterProvider",
    "Provider",
    "RateGate",
    "RoutedOpenAIProvider",
    "Sampling",
    "ToolCall",
    "ToolCompletion",
    "WorkerPool",
]
