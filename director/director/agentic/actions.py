"""Parse a worker turn into a single shell action (mini-swe-agent style).

Each turn the worker emits exactly one ```bash ...``` block. A block whose sole content
is the submit sentinel ends the rollout (the agent is done). Anything with no parseable
block is treated as a no-op action that nudges the agent to use the required format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUBMIT_SENTINEL = "DIRECTOR_SUBMIT"

_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class Action:
    command: str | None  # None => no parseable command this turn
    submit: bool = False


def parse_action(text: str) -> Action:
    m = _BLOCK_RE.search(text)
    if not m:
        # tolerate a bare submit mention without a code block
        return Action(command=None, submit=SUBMIT_SENTINEL in text)
    cmd = m.group(1).strip()
    if SUBMIT_SENTINEL in cmd:
        return Action(command=None, submit=True)
    return Action(command=cmd, submit=False)
