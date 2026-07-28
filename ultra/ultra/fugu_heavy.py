"""Fugu-Ultra heavy orchestration: each worker runs its own agent loop.

This is the full architecture from the Fugu report (§3.2.1–3.2.2), not the
lightweight per-turn router. A conductor plans a workflow — a sequence of steps,
each an anonymous capability role with a subtask and an access list — and each
step is executed by a worker running ITS OWN function-calling loop against the
shared environment.

Two invariants from §3.2.2, enforced here:

* Intra-workflow isolation. A worker sees only its own trajectory plus the
  outputs named in its access list. It never sees another worker's reasoning.
  This prevents *orchestration collapse* (later workers copying the first's
  path instead of finding their own).
* Persistent shared memory. The ENVIRONMENT (terminal state, files) is shared
  across all workers — a worker builds on what earlier workers did to the world
  without re-discovering it. Only agent trajectories are isolated, not the world.

Model-agnostic throughout: steps carry capability ROLES; the binding resolves
role -> model. Swapping
models is a data edit, never a code change.

The worker loop is injected (`WorkerLoop`), so the same orchestration runs over
a Terminus-2 loop, our own loop, or a stub for offline tests of the semantics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Protocol



def condense_attempt(text: str, limit: int | None) -> str:
    """Optional overflow guard: keep an attempt's tail (where the answer is).

    Disabled by default (`limit is None`) — truncating attempts is a deviation
    from the topology that measured best, not an improvement on it.
    """
    text = (text or "").strip()
    if limit is None or len(text) <= limit:
        return text
    return f"[...earlier reasoning omitted...]\n{text[-limit:]}"


@dataclass(frozen=True)
class WorkflowStep:
    """One planned unit of work: a capability role given a subtask."""

    role: str
    subtask: str
    access: tuple[int, ...] = ()  # indices of earlier steps this worker may see


@dataclass
class StepResult:
    step_index: int
    role: str
    model: str
    output: str


class SharedEnvironment(Protocol):
    """The world every worker acts on — terminal state persists across workers."""

    def snapshot(self) -> str:
        """Current observable environment state (e.g. terminal screen)."""


class WorkerLoop(Protocol):
    """One worker's own agent loop, scoped to a single subtask.

    Implementations run a full function-calling loop (send commands, read
    output, iterate) against `env` until the subtask is done, and return the
    worker's final message. `context` is the ISOLATED view: the subtask plus
    only the access-listed prior outputs — never other workers' trajectories.
    """

    def run(
        self, *, model: str, subtask: str, context: str, env: SharedEnvironment
    ) -> str: ...


class Conductor(Protocol):
    """Plans the workflow. Emits capability-role steps, not model names."""

    def plan(self, instruction: str, roles: list[str]) -> list[WorkflowStep]: ...


@dataclass
class BuildAndDebugConductor:
    """Default planner: the report's validated terminal pattern (§4.4).

    Builder owns the task; a different role verifies/debugs with access to the
    builder's result; the builder applies the fix with access to both. Pure
    topology over roles — no model names, no benchmark-specific tuning.
    """

    builder_role: str = "implementer"
    debugger_role: str = "debugger"

    def plan(self, instruction: str, roles: list[str]) -> list[WorkflowStep]:
        builder = self._pick(self.builder_role, roles)
        debugger = self._pick(self.debugger_role, roles)
        if debugger == builder:
            # Degenerate pool (one usable role): a single builder step.
            return [WorkflowStep(role=builder, subtask=instruction)]
        return [
            WorkflowStep(role=builder, subtask=instruction),
            WorkflowStep(
                role=debugger,
                subtask=(
                    "Independently verify the implementation from step 0: "
                    "enumerate concrete defects, run the checks the task names, "
                    "and report what fails. Do not rewrite it."
                ),
                access=(0,),
            ),
            WorkflowStep(
                role=builder,
                subtask=(
                    "Apply the fixes for the defects reported in step 1 and "
                    "confirm the task's checks pass."
                ),
                access=(0, 1),
            ),
        ]

    @staticmethod
    def _pick(role_tag: str, roles: list[str]) -> str:
        for role in roles:
            if role_tag in role:
                return role
        return roles[0]


@dataclass
class FuguHeavyOrchestrator:
    """Executes a planned workflow with per-worker isolated loops."""

    binding: dict[str, str]  # capability role -> model (from the pool binding)
    worker_loop: WorkerLoop
    conductor: Conductor = field(default_factory=BuildAndDebugConductor)
    # A parallel wave (independent leaves) waits at most this long; slower
    # leaves are DROPPED and later steps aggregate whatever returned. Without
    # this, one stalled provider gates the whole tree past client timeouts
    # (observed: a 47s median question turned into 40+ min of retry churn).
    wave_timeout_s: float | None = 480.0
    # Attempts reach the aggregator IN FULL by default: the measured 89.3%
    # tree arm passed complete attempts, and the middle of a derivation is
    # often the evidence that resolves a disagreement. Set a character budget
    # only to keep a genuinely oversized aggregation inside a context window —
    # it is an overflow guard, not a quality lever.
    attempt_chars: int | None = None

    def resolve_model(self, step: WorkflowStep, *, multi_role: bool) -> str:
        """The binding is the single source of role -> model resolution."""
        return self.binding[step.role]

    def _isolated_context(
        self,
        step: WorkflowStep,
        results: list[StepResult | None],
        instruction: str = "",
    ) -> str:
        """Build the worker's view: subtask + ONLY access-listed prior outputs.

        Other workers' trajectories are never included — that is the isolation
        guarantee. Environment state is not injected here; it reaches the worker
        through the shared environment, which is the shared-memory guarantee.
        """
        # The ORIGINAL request is included verbatim. The conductor's subtask is
        # a paraphrase for routing; a worker that sees only the paraphrase loses
        # the exact question and any answer-format requirement it carried, which
        # silently destroys accuracy on formatted tasks.
        #
        # A LEAF (no access list) is solving the task independently, so it gets
        # the request with MINIMAL framing. Every extra wrapper — workflow
        # preamble, restated subtask — pushes the real question further from the
        # end of a long prompt and measurably dilutes it: on MMLU-Pro the same
        # topology scored 89.3% when leaves received the bare question versus
        # 81.4% through the wrapped router path (2026-07-26).
        parts = []
        if not step.access:
            return instruction or step.subtask
        if instruction:
            parts.append(f"ORIGINAL REQUEST (answer this, honouring its exact "
                         f"format requirements):\n{instruction}")
        parts.append(f"\nYOUR ASSIGNED SUBTASK:\n{step.subtask}")
        for index in step.access:
            prior = results[index] if index < len(results) else None
            if prior is None:  # malformed access (cycle/forward): skip, not crash
                continue
            parts.append(
                f"\nRESULT FROM STEP {index} "
                f"({prior.role} position):\n"
                f"{condense_attempt(prior.output, self.attempt_chars)}"
            )
        if instruction:
            # Restate the format demand at the END: an aggregator reading three
            # long attempts has the request far behind it, and both observed
            # format misses on MMLU-Pro were aggregator outputs.
            parts.append(
                "\nGive your final answer in the exact format the ORIGINAL "
                "REQUEST specifies (if it names a marker such as "
                "'ANSWER: [LETTER]', end with exactly that)."
            )
        return "\n".join(parts)

    def execute(self, instruction: str, env: SharedEnvironment) -> dict[str, Any]:
        roles = list(self.binding)
        workflow = self.conductor.plan(instruction, roles)
        multi_role = len(workflow) > 1
        results: list[StepResult | None] = [None] * len(workflow)

        def run_step(index: int) -> StepResult:
            step = workflow[index]
            model = self.resolve_model(step, multi_role=multi_role)
            # results is indexed by step position; every access-listed entry is
            # complete before this step is scheduled, so lookups cannot miss.
            context = self._isolated_context(step, results, instruction)
            output = self.worker_loop.run(
                model=model, subtask=step.subtask, context=context, env=env
            )
            return StepResult(
                step_index=index, role=step.role, model=model, output=output
            )

        # Wave scheduling: steps whose access lists are fully satisfied run
        # CONCURRENTLY. Independent tree leaves execute in parallel, so a
        # 3-attempt tree costs one leaf's latency plus the aggregator's, not
        # three. Isolation is untouched — each worker still sees only its own
        # access-listed context.
        pending = set(range(len(workflow)))
        while pending:
            ready = sorted(
                index
                for index in pending
                if all(results[dep] is not None for dep in workflow[index].access)
            )
            if not ready:  # malformed access lists; fall back to plan order
                ready = [min(pending)]
            if len(ready) == 1:
                results[ready[0]] = run_step(ready[0])
            else:
                pool = ThreadPoolExecutor(max_workers=len(ready))
                futures = {index: pool.submit(run_step, index) for index in ready}
                done, _ = wait(futures.values(), timeout=self.wave_timeout_s)
                for index, future in futures.items():
                    if future in done and future.exception() is None:
                        results[index] = future.result()
                    # else: leaf dropped — stays None; access lookups skip it
                pool.shutdown(wait=False, cancel_futures=True)
            pending -= set(ready)

        completed = [r for r in results if r is not None]
        # The Conductor's response is the final step's output (§3.2.1) — but
        # never return empty while earlier steps hold real answers. An
        # aggregator that fails (truncation, provider fault) must not discard
        # the leaves it was summarizing: fall back to the last non-empty step.
        final = ""
        for result in reversed(completed):
            if result.output and result.output.strip():
                final = result.output
                break
        return {
            "output": final,
            "steps": [
                {"index": r.step_index, "role": r.role, "model": r.model}
                for r in completed
            ],
        }
