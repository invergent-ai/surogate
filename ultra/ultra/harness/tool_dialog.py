"""Deterministic tool-dialogue harness for custom tau-style tasks."""

from __future__ import annotations

import copy
import json
from typing import Any

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, ToolCall, WorkerPool
from .base import StepInput, StepResult, register_harness


def _get_path(state: dict[str, Any], path: list[str]) -> Any:
    cur: Any = state
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _success_score(state: dict[str, Any], checks: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    if not checks:
        return 0.0, [{"error": "no success checks"}]
    failures = []
    for check in checks:
        path = [str(p) for p in check.get("path", [])]
        actual = _get_path(state, path)
        expected = check.get("equals")
        if actual != expected:
            failures.append({"path": path, "expected": expected, "actual": actual})
    return (1.0 if not failures else 0.0), failures


class ToolDialogSimulator:
    def __init__(self, initial_state: dict[str, Any]) -> None:
        self.state = copy.deepcopy(initial_state)

    def call(self, call: ToolCall) -> dict[str, Any]:
        name = call.name
        args = call.arguments
        if name == "finish":
            return {"ok": True, "finished": True}
        if name == "cancel_order":
            return self._cancel_order(str(args.get("order_id") or ""))
        if name == "update_shipping_address":
            return self._update_shipping_address(
                str(args.get("order_id") or ""),
                str(args.get("address") or ""),
            )
        if name == "assign_seat":
            return self._assign_seat(
                str(args.get("reservation_id") or ""),
                str(args.get("seat") or ""),
            )
        if name == "freeze_card":
            return self._freeze_card(str(args.get("card_id") or ""))
        return {"ok": False, "error": f"unknown tool {name!r}"}

    def _cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self.state.get("orders", {}).get(order_id)
        if not order:
            return {"ok": False, "error": "order_not_found"}
        if order.get("status") == "shipped":
            return {"ok": False, "error": "already_shipped"}
        order["status"] = "cancelled"
        return {"ok": True, "order": order}

    def _update_shipping_address(self, order_id: str, address: str) -> dict[str, Any]:
        order = self.state.get("orders", {}).get(order_id)
        if not order:
            return {"ok": False, "error": "order_not_found"}
        if order.get("status") == "shipped":
            return {"ok": False, "error": "already_shipped"}
        order["address"] = address
        return {"ok": True, "order": order}

    def _assign_seat(self, reservation_id: str, seat: str) -> dict[str, Any]:
        reservation = self.state.get("reservations", {}).get(reservation_id)
        if not reservation:
            return {"ok": False, "error": "reservation_not_found"}
        if reservation.get("status") != "active":
            return {"ok": False, "error": "reservation_not_active"}
        reservation["seat"] = seat
        return {"ok": True, "reservation": reservation}

    def _freeze_card(self, card_id: str) -> dict[str, Any]:
        card = self.state.get("cards", {}).get(card_id)
        if not card:
            return {"ok": False, "error": "card_not_found"}
        card["status"] = "frozen"
        return {"ok": True, "card": card}


def _task_payload(task: TaskSpec) -> dict[str, Any]:
    payload = task.grader.expected_answer
    return payload if isinstance(payload, dict) else {}


def _assemble_messages(step: StepInput, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Use the provided tools to satisfy the user request. "
                "Do not invent state. Call finish when the requested state change is complete."
            ),
        }
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
    messages.extend(transcript)
    return messages


@register_harness
class ToolDialogHarness:
    name = "tool_dialog"

    def __init__(self) -> None:
        self.simulator: ToolDialogSimulator | None = None
        self.transcript: list[dict[str, Any]] = []
        self.final_state: dict[str, Any] | None = None

    async def run_step(self, step: StepInput, pool: WorkerPool, sampling: Sampling) -> StepResult:
        payload = _task_payload(step.task)
        if self.simulator is None:
            self.simulator = ToolDialogSimulator(payload.get("initial_state") or {})
        tools = list(step.task.input.tools)
        if not tools:
            return StepResult(text="", error="tool_dialog task has no tools", termination="missing_tools")

        max_turns = int(payload.get("max_turns") or step.task.metadata.estimated_worker_calls or 4)
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0
        final_content = ""
        finished = False

        for _turn in range(max_turns):
            comp = await pool.call_tools(step.worker_id, _assemble_messages(step, self.transcript), tools, sampling)
            total_prompt += comp.prompt_tokens
            total_completion += comp.completion_tokens
            total_cost += comp.cost_usd
            final_content = comp.content or ""
            if not comp.tool_calls:
                break
            self.transcript.append(
                {
                    "role": "assistant",
                    "content": comp.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                        }
                        for call in comp.tool_calls
                    ],
                }
            )
            for call in comp.tool_calls:
                result = self.simulator.call(call)
                self.transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, sort_keys=True),
                    }
                )
                finished = finished or bool(result.get("finished"))
            if finished:
                break

        assert self.simulator is not None
        self.final_state = copy.deepcopy(self.simulator.state)
        return StepResult(
            text=json.dumps(
                {
                    "content": final_content,
                    "state": self.final_state,
                    "turns": len([m for m in self.transcript if m.get("role") == "assistant"]),
                },
                sort_keys=True,
            ),
            input_tokens=total_prompt,
            output_tokens=total_completion,
            cost_usd=total_cost,
            termination="completed" if finished else "max_turns_or_no_tool_call",
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        payload = _task_payload(task)
        state = self.final_state
        if state is None:
            try:
                data = json.loads(final.text)
                state = data.get("state")
            except json.JSONDecodeError:
                state = None
        if not isinstance(state, dict):
            return Grade(score=0.0, success=False, details={"error": "missing final state"})
        score, failures = _success_score(state, list(payload.get("success") or []))
        return Grade(score=score, success=score >= task.grader.success_threshold, details={"failures": failures})
