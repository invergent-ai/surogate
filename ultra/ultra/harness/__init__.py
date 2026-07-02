"""Harness adapters: the executor routes a workflow step by ``environment.harness``."""

from .base import HARNESS_REGISTRY, Harness, StepInput, StepResult, register_harness
from . import single_call  # noqa: F401  (registers direct_qa + code_exec harnesses)
from . import scaffolded  # noqa: F401  (registers opencode + claude_code + codex)
from . import code_cli  # noqa: F401  (registers concrete claude_code + codex harnesses)
from . import opencode  # noqa: F401  (registers concrete opencode + opencode_repo harnesses)
from . import harbor  # noqa: F401  (registers terminal_sandbox harness)
from . import tool_dialog  # noqa: F401  (registers tool_dialog harness)
from . import tau_bench  # noqa: F401  (registers tau_bench harness)
from . import long_context  # noqa: F401  (registers long_context harness)

__all__ = [
    "HARNESS_REGISTRY",
    "Harness",
    "StepInput",
    "StepResult",
    "register_harness",
]
