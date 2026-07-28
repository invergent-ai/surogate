"""tau2-bench solo-mode harness (office telecom lane).

This drives the installed ``tau2`` package's Orchestrator in *solo mode*: the
worker resolves a ticket operating BOTH the agent tools and the customer's
device tools against tau2's environment, and signals completion with the
``done`` stop tool. No user-simulator LLM exists in this mode, so the
environment side of a rollout is fully deterministic and the reward is tau2's
own programmatic evaluation (env-state assertions + action checks).

Split discipline: the published 114-task ``telecom`` set is the reportable
benchmark and stays sealed for eval; training manifests draw from
``telecom_full`` minus those ids (see ``ultra.tau2_manifest``).

The worker is injected through a queue-agent: the orchestrator asks the agent
for its next message (including the FIRST one, inside ``initialize()``), and
the agent returns whatever the harness enqueued from the worker's completion.
Messages are timestamped at dequeue because the finalized trajectory is
timestamp-sorted; stamping at construction would misorder pre-built messages
relative to tool results and break evaluation replay.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness
from .repo_artifacts import write_json

# Solo flows have no user chatter, but troubleshooting needs reads +
# mutations + the final ``done`` call; gold action counts run 1-15.
_MAX_TURNS_BY_BUDGET = {
    "short": 8,
    "medium": 16,
    "long": 30,
    "max": None,
}

# task_set -> {tau2 task id -> Task}; loading telecom_full parses 2285 tasks,
# so share it across rollouts within the process.
_TASK_CACHE: dict[str, dict[str, Any]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_tau2() -> None:
    """Point tau2 at the vendored data dir BEFORE its first import.

    ``tau2.utils.utils`` resolves TAU2_DATA_DIR at module import time; setting
    it later is a silent no-op.
    """
    if "tau2" not in sys.modules:
        os.environ.setdefault(
            "TAU2_DATA_DIR",
            str(_repo_root() / "director" / "vendor" / "tau2_bench" / "data"),
        )
    import tau2  # noqa: F401


def _payload(task: TaskSpec) -> dict[str, Any]:
    payload = task.grader.expected_answer
    return payload if isinstance(payload, dict) else {}


def _max_turns_for_step(task: TaskSpec, budget: str) -> int:
    payload = _payload(task)
    configured = int(payload.get("max_turns") or task.metadata.estimated_worker_calls or 30)
    cap = _MAX_TURNS_BY_BUDGET.get(budget, _MAX_TURNS_BY_BUDGET["medium"])
    return min(configured, cap) if cap is not None else configured


def _load_task(task_set: str, tau2_task_id: str) -> Any:
    from tau2.registry import registry

    cache = _TASK_CACHE.get(task_set)
    if cache is None:
        tasks = registry.get_tasks_loader(task_set)()
        cache = {t.id: t for t in tasks}
        _TASK_CACHE[task_set] = cache
    if tau2_task_id not in cache:
        raise KeyError(f"tau2 task {tau2_task_id!r} not in task set {task_set!r}")
    return cache[tau2_task_id]


def _openai_transcript(trajectory: list[Any]) -> list[dict[str, Any]]:
    from tau2.data_model.message import AssistantMessage, ToolMessage

    messages: list[dict[str, Any]] = []
    for msg in trajectory:
        if isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in msg.tool_calls
                ]
            messages.append(entry)
        elif isinstance(msg, ToolMessage):
            messages.append({"role": "tool", "tool_call_id": msg.id, "content": str(msg.content)})
        else:  # solo mode has no user; convert defensively
            messages.append({"role": "user", "content": str(getattr(msg, "content", "") or "")})
    return messages


def _assemble_messages(step: StepInput, system_prompt: str, trajectory: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if step.prior_artifacts:
        blocks = [
            f"[Worker {a.get('worker_id')} result]\n{a.get('response', '')}"
            for a in step.prior_artifacts
        ]
        messages.append({"role": "user", "content": "Authorized prior-step results:\n\n" + "\n\n".join(blocks)})
    if step.subtask.strip():
        messages.append({"role": "user", "content": f"Your subtask: {step.subtask}"})
    messages.extend(_openai_transcript(trajectory))
    return messages


@register_harness
class Tau2SoloHarness:
    name = "tau2_solo"

    def __init__(self) -> None:
        self.orch = None
        self.agent = None
        self.tau2_task = None
        self.domain: str = ""
        self.evaluation_type: str = "ALL"
        self.tools: list[dict[str, Any]] = []
        self.reward: float = 0.0
        self.done: bool = False
        # The executor reuses ONE harness instance for every step of a
        # workflow, so later steps must CONTINUE the same orchestrator.
        # Re-initializing would reset the environment DB mid-ticket.
        self._orch_initialized = False

    def _init(self, task: TaskSpec) -> None:
        if self.orch is not None:
            return
        _ensure_tau2()
        from tau2.agent.llm_agent import LLMSoloAgent
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.registry import registry
        from tau2.user.user_simulator import DummyUser
        from tau2.utils.utils import get_now

        payload = _payload(task)
        self.domain = str(payload.get("domain") or "telecom")
        self.evaluation_type = str(payload.get("evaluation_type") or "ALL")
        task_set = str(payload.get("task_set") or "telecom_full")
        self.tau2_task = _load_task(task_set, str(payload.get("tau2_task_id")))

        class _QueueSoloAgent(LLMSoloAgent):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.queue: list[Any] = []

            def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
                if not self.queue:
                    raise RuntimeError("queue agent asked for a message before one was enqueued")
                msg = self.queue.pop(0)
                msg.timestamp = get_now()  # trajectory is timestamp-sorted
                if msg.tool_calls:
                    msg = self._check_if_stop_toolcall(msg)
                return msg, state

        env = registry.get_env_constructor(self.domain)(solo_mode=True)
        self.agent = _QueueSoloAgent(
            tools=env.get_tools() + env.get_user_tools(),
            domain_policy=env.get_policy(),
            task=self.tau2_task,
            llm="queued",
        )
        self.tools = [t.openai_schema for t in self.agent.tools]
        self.orch = Orchestrator(
            domain=self.domain,
            agent=self.agent,
            user=DummyUser(),
            environment=env,
            task=self.tau2_task,
            solo_mode=True,
        )
        # _finalize() needs these; they are normally set by Orchestrator.run(),
        # which we bypass to drive step() ourselves.
        self.orch._run_start_time = get_now()
        self.orch._run_start_perf = time.perf_counter()

    def _to_tau2_assistant(self, comp: Any) -> Any:
        from tau2.data_model.message import AssistantMessage, ToolCall

        tool_calls = [
            ToolCall(id=call.id, name=call.name, arguments=call.arguments)
            for call in comp.tool_calls
        ] or None
        content = comp.content if comp.content else (None if tool_calls else "")
        return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)

    def _advance(self) -> None:
        from tau2.orchestrator.orchestrator import Role

        assert self.orch is not None
        if not self._orch_initialized:
            self.orch.initialize()  # consumes the enqueued first message
            self._orch_initialized = True
        else:
            self.orch.step()  # agent -> env (or stop)
        while not self.orch.done and self.orch.to_role != Role.AGENT:
            self.orch.step()

    def _finalize_and_score(self) -> dict[str, Any]:
        from tau2.data_model.simulation import TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        assert self.orch is not None
        if self.orch.termination_reason is None:
            self.orch.termination_reason = TerminationReason.MAX_STEPS
        sim = self.orch._finalize()
        reward_info = evaluate_simulation(
            sim,
            self.tau2_task,
            EvaluationType[self.evaluation_type],
            solo_mode=True,
            domain=self.domain,
        )
        self.reward = float(reward_info.reward)
        self.done = bool(self.orch.done)
        return {
            "termination_reason": str(self.orch.termination_reason.value),
            "messages": [m.model_dump(mode="json") for m in sim.messages],
            "reward_breakdown": reward_info.model_dump(mode="json"),
        }

    async def run_step(self, step: StepInput, pool: WorkerPool, sampling: Sampling) -> StepResult:
        self._init(step.task)
        assert self.orch is not None and self.agent is not None

        max_turns = _max_turns_for_step(step.task, step.budget)
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        final_content = ""
        turns = 0

        # A prior workflow step may already have finished the ticket
        # (agent stop or solo-mode error); make no worker calls then.
        for _ in range(max_turns if not self.orch.done else 0):
            messages = _assemble_messages(step, self.agent.system_prompt, list(self.orch.trajectory))
            comp = await pool.call_tools(step.worker_id, messages, self.tools, sampling)
            prompt_tokens += comp.prompt_tokens
            completion_tokens += comp.completion_tokens
            cost += comp.cost_usd
            final_content = comp.content or ""
            turns += 1
            self.agent.queue.append(self._to_tau2_assistant(comp))
            await asyncio.to_thread(self._advance)
            if self.orch.done:
                break

        detail = await asyncio.to_thread(self._finalize_and_score)
        termination = {
            "agent_stop": "completed",
            "max_steps": "max_turns",
        }.get(detail["termination_reason"], detail["termination_reason"])

        messages_ref = None
        if step.artifact_dir:
            messages_ref = write_json(
                Path(step.artifact_dir) / "tau2_solo_transcript.json",
                {
                    "reward": self.reward,
                    "termination_reason": detail["termination_reason"],
                    "transcript": detail["messages"],
                    "reward_breakdown": detail["reward_breakdown"],
                },
            )
        return StepResult(
            text=json.dumps(
                {
                    "done": self.done,
                    "reward": self.reward,
                    "content": final_content,
                    "turns": turns,
                },
                sort_keys=True,
            ),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=cost,
            termination=termination,
            messages_ref=messages_ref,
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        return Grade(
            score=self.reward,
            success=self.reward >= task.grader.success_threshold,
            details={"done": self.done},
        )

    def close(self) -> None:
        self.orch = None
        self.agent = None
        self.tau2_task = None
        self._orch_initialized = False
