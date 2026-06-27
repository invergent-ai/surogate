"""Single-call harnesses: one worker call, then grade the text (ultra-data2 §6).

``direct_qa`` covers math / science / knowledge / factual / short-answer; ``code_exec``
covers function- and program-level coding where the worker emits code in one shot and a
code grader runs it. Both share one body — they differ only by the grader the task names.

Each call assembles the step context C_i = original task + authorized prior-step results +
the Conductor's subtask instruction (ultra-intro execution semantics).
"""

from __future__ import annotations

from ..grading import get_grader
from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness


def _assemble(step: StepInput) -> list[dict]:
    msgs = [dict(m) for m in step.task.input.messages]
    if step.prior_artifacts:
        blocks = [
            f"[Worker {a.get('worker_id')} — subtask]\n{a.get('subtask', '')}\n"
            f"[Worker {a.get('worker_id')} — result]\n{a.get('response', '')}"
            for a in step.prior_artifacts
        ]
        msgs.append(
            {
                "role": "user",
                "content": "Authorized prior-step results you may use:\n\n" + "\n\n".join(blocks),
            }
        )
    sub = (step.subtask or "").strip()
    if sub:
        msgs.append({"role": "user", "content": f"Your subtask: {sub}"})
    return msgs


class _SingleCall:
    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult:
        comp = await pool.call(step.worker_id, _assemble(step), sampling)
        return StepResult(
            text=comp.text,
            input_tokens=comp.prompt_tokens,
            output_tokens=comp.completion_tokens,
            cost_usd=comp.cost_usd,
            cached=comp.cached,
            error=comp.error,
            termination="truncated" if comp.finish_reason == "length" else "completed",
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        score = get_grader(task.grader.type)(final.text, task.grader.expected_answer)
        return Grade(score=score, success=score >= task.grader.success_threshold)


@register_harness
class DirectQAHarness(_SingleCall):
    name = "direct_qa"


@register_harness
class CodeExecHarness(_SingleCall):
    name = "code_exec"
