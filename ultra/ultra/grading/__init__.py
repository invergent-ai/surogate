"""Vendored verifiers (standalone copy of the router's judge-free graders).

Pure ``(output: str, solution) -> reward: float`` functions: math_equal, mc_letter,
gsm8k_exact, code_exec, code_exec_stdio, grid_exact.
"""

from .verifiers import REGISTRY, Grader, get_grader

__all__ = ["REGISTRY", "Grader", "get_grader"]
