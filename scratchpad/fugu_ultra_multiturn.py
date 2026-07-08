"""Fugu-Ultra MULTI-TURN env (Stage-2 substrate) — OFFLINE DEV MODULE.

Encodes the report's two memory scopes (3.2.2):
  * Intra-workflow isolation: within ONE workflow, a worker sees other workers ONLY through the
    access list (prevents "orchestration collapse"). ALREADY implemented by execute_workflow's
    `prior_artifacts` (built from step.access) — inherited unchanged.
  * Inter-workflow shared memory: ACROSS turns of a conversation, agents retain the accumulated
    environment state (so they don't rediscover the same artifacts). Lives in state["shared_memory"]
    and is injected into later turns. For single-call workers (no tools yet) this reduces to the
    conductor seeing the prior turn's outcome; the per-worker tool-memory injection point is marked
    for Stage-2c (when workers call tools).

Recursion-lite instance: max_turns=2. Turn 0 = plan; if it fails, env_response hands the outcome
back as shared memory + a revise instruction; Turn 1 = repair plan. Terminal reward = grade of the
LAST executed workflow (turn 0 if it already succeeded, else turn 1).

Trainer/scheduler/spool need ZERO changes: multi-turn is a per-token completion_mask concern the
base MultiTurnEnv handles; a 2-turn rollout is still ONE RolloutOutput downstream.

DEV NOTE: standalone here (imported helpers from the live package via importlib) so the running env
is never touched. Moves into the package only at deploy.
"""
from __future__ import annotations

import importlib.util
import uuid
from typing import Any

import verifiers as vf

_spec = importlib.util.spec_from_file_location(
    "fugu_pkg", "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
_pkg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pkg)

parse_workflow = _pkg.parse_workflow
_extract_workflow_payload = _pkg._extract_workflow_payload
_completion_text = _pkg._completion_text
execute_workflow = _pkg.execute_workflow
WorkflowValidationError = _pkg.WorkflowValidationError


REVISE_INSTRUCTION = (
    "The workflow above was executed and its final answer was evaluated as INCORRECT.\n"
    "Outcome of the previous attempt:\n{outcome}\n\n"
    "Using what the previous attempt revealed, design a NEW workflow that diagnoses the error and "
    "produces a correct solution to the original question. Output the three lists as before."
)


class FuguUltraMultiTurnEnv(vf.MultiTurnEnv):
    """2-turn conductor env: plan -> (execute, feedback) -> repair-plan -> grade.

    Requires from kwargs: `runtime` (a UltraPilotRuntime-like object exposing .pool, .sampling,
    .tasks_by_id, .lane_masks, .worker_harnesses, .max_workflow_steps, .force_step_budget),
    plus the standard vf dataset. `max_turns` defaults to 2.
    """

    def __init__(self, *, runtime: Any, max_turns: int = 2, **kwargs):
        super().__init__(max_turns=max_turns, **kwargs)
        self.rt = runtime

    async def setup_state(self, state: "vf.State") -> "vf.State":
        # inter-workflow shared memory: the accumulated environment state across turns.
        state.setdefault("shared_memory", [])   # list[str], one entry per executed turn
        state.setdefault("turn_records", [])     # list[dict]: {reward, success, outcome_text}
        return state

    async def _execute_last_workflow(self, state: "vf.State") -> dict:
        """Execute the workflow in the most recent conductor completion; grade it; return a record.
        Intra-workflow isolation is inherited from execute_workflow (access-list gated)."""
        info = state.get("info", {}) or {}
        task_id = str(info.get("task_id") or state.get("task_id"))
        lane = str(info.get("lane", "single_turn"))
        task = self.rt.tasks_by_id[task_id]
        worker_ids = self.rt.lane_masks[lane]
        raw = _completion_text(state["trajectory"][-1]["completion"])
        rid = f"mt-{uuid.uuid4().hex[:10]}"
        try:
            workflow = parse_workflow(_extract_workflow_payload(raw))
        except WorkflowValidationError as exc:
            return {"reward": 0.0, "success": False, "parse_valid": False,
                    "outcome_text": f"(unparseable workflow: {exc})"}
        if self.rt.force_step_budget is not None:
            workflow = workflow.model_copy(update={"steps": [
                s.model_copy(update={"budget": self.rt.force_step_budget}) for s in workflow.steps]})
        record = await execute_workflow(
            task, workflow, self.rt.pool, self.rt.sampling, rid,
            worker_ids=worker_ids, worker_harnesses=self.rt.worker_harnesses,
            raw_output=raw, max_steps=self.rt.max_workflow_steps,
            # SHARED-MEMORY HOOK (Stage-2c): when workers call tools, pass
            # shared_context="\n".join(state["shared_memory"]) so every worker in this turn sees
            # prior-turn tool artifacts (inter-workflow memory) while staying access-list-isolated
            # from THIS turn's peers. No-op for single-call workers (no tool artifacts to carry).
        )
        final_text = record.execution.steps[-1].text if record.execution.steps else ""
        return {
            "reward": float(record.reward or 0.0),
            "success": bool(record.grade.success) if record.grade is not None else False,
            "parse_valid": bool(record.conductor.workflow_parse_valid),
            "outcome_text": final_text[:1500],
        }

    async def env_response(self, messages, state, **kwargs):
        """Called before each turn>0: execute the PRIOR turn's workflow, decide continue vs stop.
        On success (or any terminal condition) sets final_env_response to end the rollout."""
        rec = await self._execute_last_workflow(state)
        state["turn_records"].append(rec)
        # inter-workflow shared memory: retain this turn's outcome for later turns.
        state["shared_memory"].append(rec["outcome_text"])

        if rec["success"]:
            # early exit: first plan already solved it. Terminal, reward carried by rubric.
            state["final_env_response"] = [{"role": "user", "content": "Solved."}]
            return [{"role": "user", "content": "Solved."}]
        # failed -> hand the outcome back as the revise turn's context.
        return [{"role": "user",
                 "content": REVISE_INSTRUCTION.format(outcome=rec["outcome_text"])}]

    async def reward(self, state, **kwargs) -> float:
        """Terminal reward = grade of the LAST executed workflow. env_response executed every turn
        EXCEPT the final generated one (unless we early-exited); execute that here, else reuse the
        recorded grade. Cached in turn_records so no double-execution."""
        records = state.get("turn_records", [])
        trajectory = state.get("trajectory", [])
        if len(records) < len(trajectory):
            rec = await self._execute_last_workflow(state)
            state.setdefault("turn_records", []).append(rec)
            records = state["turn_records"]
        return float(records[-1]["reward"]) if records else 0.0
