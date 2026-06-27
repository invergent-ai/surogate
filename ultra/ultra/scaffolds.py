"""Fixed workflow scaffolds A–F for the step-zero headroom test (ultra-intro §3).

Each builder returns a Workflow parameterized by worker indices (into the pool). Defaults
use a single worker so a scaffold can be run on one worker to isolate orchestration gain
(ultra-intro's "force every step to one worker still improved" ablation); pass distinct
indices to test heterogeneous assignments.
"""

from __future__ import annotations

from .schemas import Workflow, WorkflowStep

_SOLVE = "Solve the task and give the final answer."
_PLAN = (
    "Devise a correct, concrete solution strategy and outline the key steps. "
    "Do NOT produce the final answer yet."
)
_EXECUTE_FROM_PLAN = "Using the strategy above, solve the task and give the final answer."
_CRITIQUE = "Critically review the candidate solution above. Identify any concrete errors or gaps."
_REVISE = "Using the critique above, produce the corrected final answer."
_SYNTHESIZE = (
    "Combine the independent attempts above into one correct final answer. Preserve correct "
    "parts and resolve disagreements."
)


def direct(w: int = 0) -> Workflow:
    return Workflow(steps=[WorkflowStep(worker_id=w, subtask=_SOLVE, access=[])])


def plan_execute(planner: int = 0, executor: int = 0) -> Workflow:
    return Workflow(
        steps=[
            WorkflowStep(worker_id=planner, subtask=_PLAN, access=[]),
            WorkflowStep(worker_id=executor, subtask=_EXECUTE_FROM_PLAN, access=[0]),
        ]
    )


def solve_critique_revise(w: int = 0) -> Workflow:
    return Workflow(
        steps=[
            WorkflowStep(worker_id=w, subtask=_SOLVE, access=[]),
            WorkflowStep(worker_id=w, subtask=_CRITIQUE, access=[0]),
            WorkflowStep(worker_id=w, subtask=_REVISE, access=[0, 1]),
        ]
    )


def debate_synthesize(a: int = 0, b: int = 0, synth: int = 0) -> Workflow:
    return Workflow(
        steps=[
            WorkflowStep(worker_id=a, subtask=_SOLVE, access=[]),
            WorkflowStep(worker_id=b, subtask=_SOLVE, access=[]),
            WorkflowStep(worker_id=synth, subtask=_SYNTHESIZE, access=[0, 1]),
        ]
    )


def specialist_plan_execute(specialist: int = 1, executor: int = 0) -> Workflow:
    return plan_execute(specialist, executor)


def execute_critic_revise(executor: int = 0, critic: int = 1) -> Workflow:
    return Workflow(
        steps=[
            WorkflowStep(worker_id=executor, subtask=_SOLVE, access=[]),
            WorkflowStep(worker_id=critic, subtask=_CRITIQUE, access=[0]),
            WorkflowStep(worker_id=executor, subtask=_REVISE, access=[0, 1]),
        ]
    )


# Step-zero scaffold set (ultra-intro §3 strategies A–F).
SCAFFOLDS = {
    "A_direct": direct,
    "B_plan_execute": plan_execute,
    "C_solve_critique_revise": solve_critique_revise,
    "D_debate_synthesize": debate_synthesize,
    "E_specialist_plan_execute": specialist_plan_execute,
    "F_execute_critic_revise": execute_critic_revise,
}
