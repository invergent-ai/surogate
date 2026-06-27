"""Harness adapters: the executor routes a workflow step by ``environment.harness``."""

from .base import HARNESS_REGISTRY, Harness, StepInput, StepResult, register_harness
from . import single_call  # noqa: F401  (registers direct_qa + code_exec harnesses)
from . import scaffolded  # noqa: F401  (registers opencode + claude_code + codex)
from . import opencode  # noqa: F401  (registers concrete opencode + opencode_repo harnesses)
from . import harbor  # noqa: F401  (registers terminal_sandbox harness)

__all__ = [
    "HARNESS_REGISTRY",
    "Harness",
    "StepInput",
    "StepResult",
    "register_harness",
]
