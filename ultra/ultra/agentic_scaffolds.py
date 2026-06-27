"""Container-native workflow scaffolds for the agentic-coding step-zero (opencode_repo).

The text scaffolds (solve→critique→revise) don't map to a repo workspace — a "critique" step
that edits nothing is meaningless in a container. These scaffolds are workspace-native: every
step either edits the repo (fresh, or continuing a predecessor's edits) or synthesizes
independent candidate patches. Worker indices index the pool; the defaults wire the
GLM(0)→Gemini(1)→Opus(2) cost ladder so heterogeneity is the default, not an override.

Step semantics map to the executor's workspace lineage (ultra-data §6):
  access=[]      -> fresh container (Rule A)
  access=[j]     -> continue in step j's workspace (Rule B)
  access=[j,k]   -> fresh container + predecessors' patches as artifacts (Rule C)
"""

from __future__ import annotations

from .schemas import Workflow, WorkflowStep

_FIX = "Implement a complete, correct fix for the bug."
_DEBUG = "Find why the tests still fail and finish the fix."
_SYNTH = "Produce one correct fix, combining the strongest ideas from the candidate patches."


def ag_direct(w: int = 0) -> Workflow:
    """One worker fixes the bug (the single-worker baseline; also scaffold A)."""
    return Workflow(steps=[WorkflowStep(worker_id=w, subtask=_FIX, access=[])])


def self_repair(w: int = 0) -> Workflow:
    """Same worker takes a second pass — controls for 'does a 2nd pass help' without heterogeneity."""
    return Workflow(
        steps=[
            WorkflowStep(worker_id=w, subtask=_FIX, access=[]),
            WorkflowStep(worker_id=w, subtask=_DEBUG, access=[0]),
        ]
    )


def builder_debugger(builder: int = 0, debugger: int = 2) -> Workflow:
    """Cheap model builds, premium model debugs the same workspace (the key heterogeneous pair)."""
    return Workflow(
        steps=[
            WorkflowStep(worker_id=builder, subtask=_FIX, access=[]),
            WorkflowStep(worker_id=debugger, subtask=_DEBUG, access=[0]),
        ]
    )


def ladder(a: int = 0, b: int = 1, c: int = 2) -> Workflow:
    """GLM→Gemini→Opus continue-in-place (the cost-escalation ladder, as a static workflow)."""
    return Workflow(
        steps=[
            WorkflowStep(worker_id=a, subtask=_FIX, access=[]),
            WorkflowStep(worker_id=b, subtask=_DEBUG, access=[0]),
            WorkflowStep(worker_id=c, subtask=_DEBUG, access=[1]),
        ]
    )


def debate_synth(a: int = 0, b: int = 1, synth: int = 2) -> Workflow:
    """Two independent attempts (fresh workspaces) → a third synthesizes both patches (Rule C)."""
    return Workflow(
        steps=[
            WorkflowStep(worker_id=a, subtask=_FIX, access=[]),
            WorkflowStep(worker_id=b, subtask=_FIX, access=[]),
            WorkflowStep(worker_id=synth, subtask=_SYNTH, access=[0, 1]),
        ]
    )


# Multi-step scaffolds only — single-worker direct (ag_direct per worker) is the baseline.
AGENTIC_SCAFFOLDS = {
    "self_repair": self_repair,
    "builder_debugger": builder_debugger,
    "ladder": ladder,
    "debate_synth": debate_synth,
}
