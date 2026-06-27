"""System prompt for the agentic coding loop."""

from __future__ import annotations

from .actions import SUBMIT_SENTINEL

AGENT_SYSTEM = f"""You are an autonomous software engineer working in a shell.

Rules:
- Each turn, respond with EXACTLY ONE bash command inside a single ```bash code block.
- Use the shell to explore the repo, reproduce the issue, edit files, and run tests.
- Do not explain at length; act. One command per turn.
- When the task is fully solved and tests pass, emit a final block containing only:
  ```bash
  {SUBMIT_SENTINEL}
  ```
"""


def wrap_observation(observation: str, max_chars: int = 4000) -> str:
    obs = observation if len(observation) <= max_chars else observation[:max_chars] + "\n...[truncated]"
    return f"<observation>\n{obs}\n</observation>"
