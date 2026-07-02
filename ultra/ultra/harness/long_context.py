"""Long-context document-pack harness."""

from __future__ import annotations

import json

from ..grading import get_grader
from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness


def _doc_text(doc: object, index: int) -> str:
    if isinstance(doc, dict):
        title = doc.get("title") or doc.get("id") or f"document-{index}"
        body = doc.get("text") or doc.get("content") or json.dumps(doc, sort_keys=True)
        return f"[{title}]\n{body}"
    return f"[document-{index}]\n{doc}"


def _assemble(step: StepInput) -> list[dict]:
    docs = "\n\n".join(_doc_text(doc, i) for i, doc in enumerate(step.task.input.context_documents, start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the provided documents. If the answer depends on a code, "
                "name, number, or date, include it exactly."
            ),
        },
        {"role": "user", "content": "Documents:\n\n" + docs},
    ]
    messages.extend(dict(m) for m in step.task.input.messages)
    if step.prior_artifacts:
        blocks = [
            f"[Worker {a.get('worker_id')} result]\n{a.get('response', '')}"
            for a in step.prior_artifacts
        ]
        messages.append({"role": "user", "content": "Authorized prior-step results:\n\n" + "\n\n".join(blocks)})
    if step.subtask.strip():
        messages.append({"role": "user", "content": f"Your subtask: {step.subtask}"})
    return messages


@register_harness
class LongContextHarness:
    name = "long_context"

    async def run_step(self, step: StepInput, pool: WorkerPool, sampling: Sampling) -> StepResult:
        if not step.task.input.context_documents:
            return StepResult(text="", error="long_context task has no documents", termination="missing_context")
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
