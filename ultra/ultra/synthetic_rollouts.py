"""Outcome-grounded synthetic rollouts for the learned Fugu conductor.

The scenarios in this module simulate only the conductor-facing state machine.
They are derived from recurring live-rollout motifs, while worker progress and
tool outcomes are scripted deterministically.  A real conductor policy still
samples every action, so training can retain exact token IDs and behavior
log-probabilities without making paid worker calls.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .behavior_likelihood import has_full_vocabulary_behavior_likelihood_contract
from .live_control import (
    ControlAction,
    ControlBudget,
    ControlContractError,
    ControlPosition,
    ControlStep,
    LiveControlState,
    WorkerProfile,
    validate_control_action,
)


SyntheticActionName = Literal["continue", "handoff", "replan", "complete"]
MAX_SYNTHETIC_SEQUENCE_TOKENS = 8_192
SYNTHETIC_CURRICULUM_REVISION = "20260724-evidence-motifs-v1"


class SyntheticRolloutError(ValueError):
    """A synthetic conductor scenario or transition is invalid."""


@dataclass(frozen=True)
class SyntheticActionRule:
    """The outcome-preserving action at one scripted live boundary."""

    action: SyntheticActionName
    target_position_id: int | None = None
    required_step_capabilities: tuple[frozenset[str], ...] = ()
    required_step_access: tuple[tuple[int, ...], ...] = ()
    required_step_terms: tuple[frozenset[str], ...] = ()
    oracle_subtasks: tuple[str, ...] = ()

    def matches(
        self,
        action: ControlAction,
        state: LiveControlState,
    ) -> tuple[bool, str]:
        try:
            validate_control_action(action, state)
        except ControlContractError as exc:
            return False, f"invalid_action:{exc}"
        if action.action != self.action:
            return False, f"wrong_action:{action.action}"
        if action.target_position_id != self.target_position_id:
            return False, "wrong_handoff_target"
        if self.action != "replan":
            if action.steps:
                return False, "unexpected_replacement_steps"
            return True, "matched"
        if len(action.steps) != len(self.required_step_capabilities):
            return False, "wrong_replacement_length"
        profiles = {worker.worker_id: worker for worker in state.workers}
        for index, (step, required, access, terms) in enumerate(
            zip(
                action.steps,
                self.required_step_capabilities,
                self.required_step_access,
                self.required_step_terms,
                strict=True,
            )
        ):
            profile = profiles[step.worker_id]
            if not required.issubset(set(profile.capability_tags)):
                return False, f"wrong_step_capability:{index}"
            if step.access != access:
                return False, f"wrong_step_access:{index}"
            normalized_subtask = " ".join(step.subtask.casefold().split())
            if (
                not all(term.casefold() in normalized_subtask for term in terms)
                or any(
                    phrase in normalized_subtask
                    for phrase in ("do not", "don't", "skip ", "omit ")
                )
            ):
                return False, f"wrong_step_subtask:{index}"
        return True, "matched"

    def oracle_action(self, state: LiveControlState) -> ControlAction:
        """Return one model-agnostic action satisfying this boundary."""
        if self.action != "replan":
            action = ControlAction(
                action=self.action,
                reason="Use the observed workflow evidence.",
                target_position_id=self.target_position_id,
            )
            validate_control_action(action, state)
            return action

        if (
            len(self.required_step_capabilities)
            != len(self.required_step_access)
            or len(self.required_step_capabilities) != len(self.required_step_terms)
            or len(self.required_step_capabilities) != len(self.oracle_subtasks)
        ):
            raise SyntheticRolloutError("replan oracle fields have different lengths")
        available = [
            worker
            for worker in state.workers
            if worker.worker_id not in state.unavailable_worker_ids
        ]
        steps: list[ControlStep] = []
        for index, (required, access, subtask) in enumerate(
            zip(
                self.required_step_capabilities,
                self.required_step_access,
                self.oracle_subtasks,
                strict=True,
            )
        ):
            worker = next(
                (
                    candidate
                    for candidate in available
                    if required.issubset(set(candidate.capability_tags))
                ),
                None,
            )
            if worker is None:
                raise SyntheticRolloutError(
                    f"no available profile satisfies oracle step {index}: {sorted(required)}"
                )
            steps.append(
                ControlStep(
                    worker_id=worker.worker_id,
                    subtask=subtask,
                    access=access,
                )
            )
        action = ControlAction(
            action="replan",
            reason="Replace the exhausted topology with the required recovery roles.",
            steps=tuple(steps),
        )
        validate_control_action(action, state)
        return action


@dataclass(frozen=True)
class SyntheticBoundary:
    boundary_id: str
    state: LiveControlState
    oracle: SyntheticActionRule
    next_boundary_id: str | None


@dataclass(frozen=True)
class SyntheticScenario:
    """One deterministic multi-step conductor episode."""

    scenario_id: str
    motif: str
    evidence_basis: tuple[str, ...]
    initial_boundary_id: str
    boundaries: tuple[SyntheticBoundary, ...]

    def boundary_map(self) -> dict[str, SyntheticBoundary]:
        return {boundary.boundary_id: boundary for boundary in self.boundaries}

    def validate(self) -> None:
        if not self.scenario_id or not self.motif or not self.evidence_basis:
            raise SyntheticRolloutError("scenario metadata is incomplete")
        by_id = self.boundary_map()
        if len(by_id) != len(self.boundaries):
            raise SyntheticRolloutError("boundary IDs must be unique")
        if self.initial_boundary_id not in by_id:
            raise SyntheticRolloutError("initial boundary is absent")
        visited: set[str] = set()
        current: str | None = self.initial_boundary_id
        while current is not None:
            if current in visited:
                raise SyntheticRolloutError("scenario boundary chain contains a cycle")
            visited.add(current)
            boundary = by_id[current]
            oracle = boundary.oracle.oracle_action(boundary.state)
            matched, reason = boundary.oracle.matches(oracle, boundary.state)
            if not matched:
                raise SyntheticRolloutError(f"oracle action is invalid: {reason}")
            if (
                boundary.next_boundary_id is not None
                and boundary.next_boundary_id not in by_id
            ):
                raise SyntheticRolloutError("scenario points to an absent boundary")
            is_terminal = boundary.next_boundary_id is None
            if (boundary.oracle.action == "complete") != is_terminal:
                raise SyntheticRolloutError(
                    "only a terminal synthetic boundary may complete"
                )
            if is_terminal:
                active = boundary.state.active_position
                progress = active.progress if active is not None else None
                verification = (
                    progress.get("verification")
                    if isinstance(progress, dict)
                    else None
                )
                if (
                    not isinstance(progress, dict)
                    or progress.get("completion_requested") is not True
                    or not isinstance(verification, dict)
                    or verification.get("passed") is not True
                ):
                    raise SyntheticRolloutError(
                        "terminal synthetic boundary lacks verified outcome evidence"
                    )
            current = boundary.next_boundary_id
        if visited != set(by_id):
            raise SyntheticRolloutError("scenario contains unreachable boundaries")


@dataclass(frozen=True)
class SyntheticStepResult:
    next_state: LiveControlState | None
    done: bool
    reward: float | None
    outcome: str


@dataclass(frozen=True)
class SyntheticDecisionRecord:
    boundary_id: str
    action: ControlAction | None
    matched_oracle: bool
    outcome: str


@dataclass(frozen=True)
class SyntheticPolicyAttestation:
    behavior_policy_revision: str
    runtime_revision: str
    pool_id: str
    pool_binding_revision: str
    sampling_seed: int

    def validate(self) -> None:
        for label, value in (
            ("behavior_policy_revision", self.behavior_policy_revision),
            ("runtime_revision", self.runtime_revision),
            ("pool_id", self.pool_id),
            ("pool_binding_revision", self.pool_binding_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SyntheticRolloutError(f"{label} must be non-empty")
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed,
            int,
        ):
            raise SyntheticRolloutError("sampling_seed must be an integer")


@dataclass(frozen=True)
class SyntheticSampledRollout:
    scenario_id: str
    motif: str
    reward: float
    outcome: str
    decisions: tuple[SyntheticDecisionRecord, ...]
    model_traces: tuple[dict[str, Any], ...]
    policy: SyntheticPolicyAttestation


class SyntheticController(Protocol):
    decision_traces: list[dict[str, Any]]
    supplies_topology: bool
    capability_refs: bool

    def reset_traces(self) -> None: ...

    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction: ...


class SyntheticConductorEnv:
    """Execute policy actions against one scripted conductor scenario."""

    def __init__(self, scenario: SyntheticScenario) -> None:
        scenario.validate()
        self.scenario = scenario
        self._boundaries = scenario.boundary_map()
        self._current_id = scenario.initial_boundary_id
        self._done = False
        self._reward: float | None = None
        self._outcome = "active"
        self._decisions: list[SyntheticDecisionRecord] = []

    @property
    def state(self) -> LiveControlState:
        if self._done:
            raise SyntheticRolloutError("terminal synthetic rollout has no live state")
        return self._boundaries[self._current_id].state

    @property
    def current_boundary(self) -> SyntheticBoundary:
        if self._done:
            raise SyntheticRolloutError(
                "terminal synthetic rollout has no current boundary"
            )
        return self._boundaries[self._current_id]

    @property
    def decisions(self) -> tuple[SyntheticDecisionRecord, ...]:
        return tuple(self._decisions)

    @property
    def reward(self) -> float:
        if self._reward is None:
            raise SyntheticRolloutError("synthetic rollout has not terminated")
        return self._reward

    @property
    def outcome(self) -> str:
        return self._outcome

    def abort(self, reason: str) -> SyntheticStepResult:
        if self._done:
            raise SyntheticRolloutError("synthetic rollout already terminated")
        self._done = True
        self._reward = 0.0
        self._outcome = f"controller_failure:{reason}"
        return SyntheticStepResult(
            next_state=None,
            done=True,
            reward=0.0,
            outcome=self._outcome,
        )

    def reject_policy_output(self, reason: str) -> SyntheticStepResult:
        if self._done:
            raise SyntheticRolloutError("synthetic rollout already terminated")
        boundary = self.current_boundary
        outcome = f"invalid_policy_output:{reason}"
        self._decisions.append(
            SyntheticDecisionRecord(
                boundary_id=boundary.boundary_id,
                action=None,
                matched_oracle=False,
                outcome=outcome,
            )
        )
        self._done = True
        self._reward = 0.0
        self._outcome = outcome
        return SyntheticStepResult(
            next_state=None,
            done=True,
            reward=0.0,
            outcome=outcome,
        )

    def step(self, action: ControlAction) -> SyntheticStepResult:
        if self._done:
            raise SyntheticRolloutError("synthetic rollout already terminated")
        boundary = self._boundaries[self._current_id]
        matched, reason = boundary.oracle.matches(action, boundary.state)
        self._decisions.append(
            SyntheticDecisionRecord(
                boundary_id=boundary.boundary_id,
                action=action,
                matched_oracle=matched,
                outcome=reason,
            )
        )
        if not matched:
            self._done = True
            self._reward = 0.0
            self._outcome = reason
            return SyntheticStepResult(
                next_state=None,
                done=True,
                reward=0.0,
                outcome=reason,
            )
        if boundary.next_boundary_id is None:
            self._done = True
            self._reward = 1.0
            self._outcome = "task_outcome_verified"
            return SyntheticStepResult(
                next_state=None,
                done=True,
                reward=1.0,
                outcome=self._outcome,
            )
        self._current_id = boundary.next_boundary_id
        return SyntheticStepResult(
            next_state=self.state,
            done=False,
            reward=None,
            outcome="scripted_worker_transition",
        )


async def sample_synthetic_rollout(
    controller: SyntheticController,
    scenario: SyntheticScenario,
    *,
    policy: SyntheticPolicyAttestation,
) -> SyntheticSampledRollout:
    """Sample one real exact-token policy trajectory over a synthetic scenario."""
    policy.validate()
    if (
        controller.supplies_topology is not True
        or controller.capability_refs is not True
    ):
        raise SyntheticRolloutError(
            "synthetic training requires anonymous capability topology actions"
        )
    env = SyntheticConductorEnv(scenario)
    controller.reset_traces()
    while True:
        trace_count = len(controller.decision_traces)
        try:
            action = await controller.decide(env.state)
        except ControlContractError as exc:
            if len(controller.decision_traces) != trace_count + 1:
                raise SyntheticRolloutError(
                    "controller failure occurred without exact policy tokens"
                ) from exc
            _validate_exact_trace(
                controller.decision_traces[-1],
                sampling_seed=policy.sampling_seed,
            )
            if controller.decision_traces[-1].get("finish_reason") == "length":
                raise SyntheticRolloutError(
                    "length-truncated policy output cannot enter training"
                ) from exc
            env.reject_policy_output(type(exc).__name__)
            break
        if len(controller.decision_traces) != trace_count + 1:
            raise SyntheticRolloutError(
                "each synthetic decision must produce one exact policy trace"
            )
        _validate_exact_trace(
            controller.decision_traces[-1],
            sampling_seed=policy.sampling_seed,
        )
        result = env.step(action)
        if result.done:
            break
    if len(controller.decision_traces) != len(env.decisions):
        raise SyntheticRolloutError(
            "synthetic decisions and exact policy traces are misaligned"
        )
    return SyntheticSampledRollout(
        scenario_id=scenario.scenario_id,
        motif=scenario.motif,
        reward=env.reward,
        outcome=env.outcome,
        decisions=env.decisions,
        model_traces=tuple(controller.decision_traces),
        policy=policy,
    )


def _validate_exact_trace(
    trace: dict[str, Any],
    *,
    sampling_seed: int,
) -> None:
    if not isinstance(trace, dict):
        raise SyntheticRolloutError("controller trace is not an object")
    prompt_ids = trace.get("prompt_token_ids")
    completion_ids = trace.get("completion_token_ids")
    logprobs = trace.get("completion_logprobs")
    messages = trace.get("messages")
    response = trace.get("response")
    valid_ids = lambda values: (  # noqa: E731
        isinstance(values, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        )
    )
    if (
        not valid_ids(prompt_ids)
        or not prompt_ids
        or not valid_ids(completion_ids)
        or not completion_ids
        or not isinstance(logprobs, list)
        or len(logprobs) != len(completion_ids)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in logprobs
        )
        or trace.get("temperature") != 1.0
        or trace.get("seed") != sampling_seed
        or trace.get("correction") is not None
        or not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
            for message in messages
        )
        or not isinstance(response, str)
        or not response.strip()
        or not has_full_vocabulary_behavior_likelihood_contract(
            trace.get("behavior_likelihood_contract")
        )
    ):
        raise SyntheticRolloutError(
            "controller trace lacks exact full-vocabulary behavior evidence"
        )
    if len(prompt_ids) + len(completion_ids) > MAX_SYNTHETIC_SEQUENCE_TOKENS:
        raise SyntheticRolloutError(
            "controller trace exceeds the proven optimizer window"
        )


_PROFILE_CAPABILITIES = (
    ("planner", "scientist", "aggregator"),
    ("coder", "mathematician", "reasoner"),
    ("debugger", "verifier", "reasoner"),
    ("implementer", "drafter", "fast_pass"),
)
_REQUIRED_MOTIF_CAPABILITIES = frozenset(
    {"debugger", "implementer", "verifier"}
)


@dataclass(frozen=True)
class _SemanticContext:
    deliverable: str
    deliverable_term: str
    artifact_path: str
    check: str
    check_term: str
    defect: str
    defect_term: str


_SEMANTIC_CONTEXTS = (
    _SemanticContext(
        deliverable="JSON dependency inventory",
        deliverable_term="inventory",
        artifact_path="/work/output/dependencies.json",
        check="schema and duplicate-entry checks",
        check_term="schema",
        defect="duplicate package records",
        defect_term="duplicate",
    ),
    _SemanticContext(
        deliverable="command-line conversion utility",
        deliverable_term="conversion",
        artifact_path="/work/output/convert_records",
        check="round-trip conversion suite",
        check_term="round-trip",
        defect="corrupted quoted values",
        defect_term="quoted",
    ),
    _SemanticContext(
        deliverable="service configuration bundle",
        deliverable_term="configuration",
        artifact_path="/work/output/service-config",
        check="offline syntax and smoke checks",
        check_term="smoke",
        defect="omitted environment key",
        defect_term="environment",
    ),
    _SemanticContext(
        deliverable="tabular reconciliation report",
        deliverable_term="reconciliation",
        artifact_path="/work/output/reconciliation.csv",
        check="row-total and key-consistency checks",
        check_term="consistency",
        defect="silently dropped unmatched records",
        defect_term="unmatched",
    ),
    _SemanticContext(
        deliverable="documentation link index",
        deliverable_term="index",
        artifact_path="/work/output/link-index.json",
        check="reference and anchor checks",
        check_term="anchor",
        defect="incorrect relative-anchor resolution",
        defect_term="anchor",
    ),
    _SemanticContext(
        deliverable="archive extraction helper",
        deliverable_term="extraction",
        artifact_path="/work/output/safe-extract",
        check="path-safety and fixture checks",
        check_term="fixture",
        defect="unsafe nested target path",
        defect_term="nested",
    ),
    _SemanticContext(
        deliverable="database migration script",
        deliverable_term="migration",
        artifact_path="/work/output/migrate.sql",
        check="idempotence and rollback checks",
        check_term="rollback",
        defect="duplicate rows on a second execution",
        defect_term="duplicate",
    ),
)


def _validated_profile_capabilities(
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(profile_capabilities, tuple) or not profile_capabilities:
        raise SyntheticRolloutError(
            "profile_capabilities must be a non-empty tuple"
        )
    normalized: list[tuple[str, ...]] = []
    for index, tags in enumerate(profile_capabilities):
        if not isinstance(tags, tuple) or not tags:
            raise SyntheticRolloutError(
                f"profile_capabilities[{index}] must be a non-empty tuple"
            )
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise SyntheticRolloutError(
                f"profile_capabilities[{index}] contains an invalid tag"
            )
        profile = tuple(sorted({" ".join(tag.casefold().split()) for tag in tags}))
        if len(profile) != len(tags):
            raise SyntheticRolloutError(
                f"profile_capabilities[{index}] contains duplicate tags"
            )
        normalized.append(profile)
    if len(set(normalized)) != len(normalized):
        raise SyntheticRolloutError(
            "profile_capabilities must describe unique anonymous profiles"
        )
    available = {tag for profile in normalized for tag in profile}
    missing = sorted(_REQUIRED_MOTIF_CAPABILITIES - available)
    if missing:
        raise SyntheticRolloutError(
            "profile_capabilities lack required motif capabilities: "
            + ", ".join(missing)
        )
    return tuple(normalized)


def _profiles(
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> tuple[WorkerProfile, ...]:
    capabilities = list(_validated_profile_capabilities(profile_capabilities))
    random.Random(seed).shuffle(capabilities)
    return tuple(
        WorkerProfile(
            worker_id=index,
            capability_tags=tuple(tags),
            tool_tags=("terminal", "filesystem", "test_runner"),
        )
        for index, tags in enumerate(capabilities)
    )


def _unavailable_worker_ids(
    profiles: tuple[WorkerProfile, ...],
    *,
    seed: int,
) -> tuple[int, ...]:
    """Sometimes expose a genuine but irrelevant unavailable pool profile."""
    if seed % 3:
        return ()
    irrelevant = [
        profile.worker_id
        for profile in profiles
        if _REQUIRED_MOTIF_CAPABILITIES.isdisjoint(profile.capability_tags)
    ]
    if not irrelevant:
        return ()
    return (irrelevant[(seed // 3) % len(irrelevant)],)


def _semantic_context(seed: int) -> _SemanticContext:
    return _SEMANTIC_CONTEXTS[seed % len(_SEMANTIC_CONTEXTS)]


def _ambient_memory(
    seed: int,
    context: _SemanticContext,
) -> tuple[dict[str, Any], ...]:
    if seed % 2:
        return ()
    notes = (
        {
            "workspace_note": (
                f"Inputs for the {context.deliverable} are already present in "
                "the shared workspace."
            )
        },
        {
            "constraint_note": (
                f"The {context.check} must run locally without fetching new data."
            )
        },
        {
            "handoff_note": (
                f"Preserve {context.artifact_path} when changing ownership."
            )
        },
    )
    return (notes[(seed // 2) % len(notes)],)


def _worker_id(profiles: tuple[WorkerProfile, ...], capability: str) -> int:
    return next(
        profile.worker_id
        for profile in profiles
        if capability in profile.capability_tags
    )


def _budget(*, used: int, limit: int) -> ControlBudget:
    return ControlBudget(
        paid_calls_used=used,
        paid_call_limit=limit,
        elapsed_s=float(used * 37),
        wall_time_limit_s=7_200.0,
    )


def _state(
    *,
    task: str,
    profiles: tuple[WorkerProfile, ...],
    workflow_id: int | None,
    positions: tuple[ControlPosition, ...],
    active_position_id: int | None,
    paid_calls_used: int,
    paid_call_limit: int,
    shared_memory: tuple[dict[str, Any], ...] = (),
    unavailable_worker_ids: tuple[int, ...] = (),
) -> LiveControlState:
    return LiveControlState(
        original_task=task,
        workers=profiles,
        workflow_id=workflow_id,
        positions=positions,
        active_position_id=active_position_id,
        terminal_status="ready",
        terminal_observation="worker terminal is stable",
        shared_memory=shared_memory,
        budget=_budget(used=paid_calls_used, limit=paid_call_limit),
        unavailable_worker_ids=unavailable_worker_ids,
    )


def _verification_chain(
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> SyntheticScenario:
    profiles = _profiles(seed, profile_capabilities)
    context = _semantic_context(seed)
    unavailable = _unavailable_worker_ids(profiles, seed=seed)
    memory = _ambient_memory(seed, context)
    builder = _worker_id(profiles, "implementer")
    verifier = _worker_id(profiles, "verifier")
    builder_position = 10 + seed % 17
    verifier_position = builder_position + 9
    budget_limit = 20 + seed % 9
    calls_used = 2 + seed % 3
    if (seed // len(_SEMANTIC_CONTEXTS)) % 2:
        task = (
            f"Create the {context.deliverable}, exercise it with the "
            f"{context.check}, and require a separate verifier before returning."
        )
    else:
        task = (
            f"Deliver a {context.deliverable} in the shared workspace. Run the "
            f"{context.check} and return only after independent verification."
        )
    builder_done = ControlPosition(
        position_id=builder_position,
        worker_id=builder,
        subtask=(
            f"Implement the {context.deliverable} and run the "
            f"{context.check} locally."
        ),
        access=(),
        status="active",
        progress={
            "completion_requested": True,
            "worker_report": (
                f"The {context.deliverable} is implemented; the "
                f"{context.check} pass locally."
            ),
            "material_progress": {"latest_turn_changed_material_state": True},
        },
        artifacts=(
            {"path": context.artifact_path, "state": "implemented_locally"},
        ),
    )
    verifier_pending = ControlPosition(
        position_id=verifier_position,
        worker_id=verifier,
        subtask=(
            f"Independently inspect the {context.deliverable} and rerun the "
            f"{context.check}."
        ),
        access=(builder_position,),
        status="pending",
    )
    verifying = ControlPosition(
        position_id=verifier_position,
        worker_id=verifier,
        subtask=verifier_pending.subtask,
        access=verifier_pending.access,
        status="active",
        progress={
            "completion_requested": False,
            "worker_report": (
                f"Independent {context.check} are still running against "
                f"{context.artifact_path}."
            ),
            "material_progress": {"latest_turn_changed_material_state": True},
        },
        artifacts=builder_done.artifacts,
    )
    verified = ControlPosition(
        position_id=verifier_position,
        worker_id=verifier,
        subtask=verifier_pending.subtask,
        access=verifier_pending.access,
        status="active",
        progress={
            "completion_requested": True,
            "worker_report": (
                f"The {context.deliverable} passes the independent "
                f"{context.check}."
            ),
            "verification": {
                "passed": True,
                "scope": context.check,
            },
        },
        artifacts=(
            {"path": context.artifact_path, "state": "independently_verified"},
        ),
    )
    return SyntheticScenario(
        scenario_id=f"synthetic-verification-chain-{seed:06d}",
        motif="completion_handoff_continue_complete",
        evidence_basis=(
            "live rollouts hid completion requests during prompt compaction",
            "successful tasks required an independent verifier handoff",
        ),
        initial_boundary_id="builder_done",
        boundaries=(
            SyntheticBoundary(
                boundary_id="builder_done",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(builder_done, verifier_pending),
                    active_position_id=builder_position,
                    paid_calls_used=calls_used,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(
                    action="handoff",
                    target_position_id=verifier_position,
                ),
                next_boundary_id="verifying",
            ),
            SyntheticBoundary(
                boundary_id="verifying",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(
                        ControlPosition(
                            **{
                                **builder_done.__dict__,
                                "status": "completed",
                            }
                        ),
                        verifying,
                    ),
                    active_position_id=verifier_position,
                    paid_calls_used=calls_used + 1,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="continue"),
                next_boundary_id="verified",
            ),
            SyntheticBoundary(
                boundary_id="verified",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(
                        ControlPosition(
                            **{
                                **builder_done.__dict__,
                                "status": "completed",
                            }
                        ),
                        verified,
                    ),
                    active_position_id=verifier_position,
                    paid_calls_used=calls_used + 2,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="complete"),
                next_boundary_id=None,
            ),
        ),
    )


def _failed_verification_recovery(
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> SyntheticScenario:
    profiles = _profiles(seed, profile_capabilities)
    context = _semantic_context(seed)
    unavailable = _unavailable_worker_ids(profiles, seed=seed)
    verifier = _worker_id(profiles, "verifier")
    implementer = _worker_id(profiles, "implementer")
    old_position = 30 + seed % 13
    repair_position = old_position + 7
    verify_position = repair_position + 11
    budget_limit = 27 + seed % 10
    calls_used = 6 + seed % 4
    if (seed // len(_SEMANTIC_CONTEXTS)) % 2:
        task = (
            f"Recover the {context.deliverable} after the {context.check} expose "
            f"{context.defect}. Repair the defect and independently reverify it."
        )
    else:
        task = (
            f"The {context.deliverable} failed the {context.check} because of "
            f"{context.defect}. Correct it, then rerun independent verification "
            "before returning."
        )
    failed = ControlPosition(
        position_id=old_position,
        worker_id=verifier,
        subtask=(
            f"Verify the current {context.deliverable} with the "
            f"{context.check}."
        ),
        access=(),
        status="active",
        progress={
            "completion_requested": True,
            "worker_report": (
                f"The {context.check} reproducibly expose {context.defect}."
            ),
            "verification": {
                "passed": False,
                "failure": context.defect,
            },
        },
        artifacts=(
            {"path": context.artifact_path, "state": "verification_failed"},
        ),
    )
    repair_active = ControlPosition(
        position_id=repair_position,
        worker_id=implementer,
        subtask=(
            f"Repair the {context.defect_term} defect in the "
            f"{context.deliverable} and run focused {context.check}."
        ),
        access=(),
        status="active",
        progress={
            "completion_requested": False,
            "worker_report": (
                f"The {context.defect_term} repair is implemented; focused "
                f"{context.check} are running."
            ),
            "material_progress": {"latest_turn_changed_material_state": True},
        },
        artifacts=(
            {"path": context.artifact_path, "state": "repair_in_progress"},
        ),
    )
    verifier_pending = ControlPosition(
        position_id=verify_position,
        worker_id=verifier,
        subtask=(
            f"Independently rerun the full {context.check} after the repair."
        ),
        access=(repair_position,),
        status="pending",
    )
    repair_done = ControlPosition(
        **{
            **repair_active.__dict__,
            "progress": {
                "completion_requested": True,
                "worker_report": (
                    f"The {context.defect_term} repair is complete and focused "
                    f"{context.check} pass."
                ),
            },
            "artifacts": (
                {"path": context.artifact_path, "state": "repaired"},
            ),
        }
    )
    verified = ControlPosition(
        **{
            **verifier_pending.__dict__,
            "status": "active",
            "progress": {
                "completion_requested": True,
                "worker_report": (
                    f"The repaired {context.deliverable} passes the independent "
                    f"{context.check}."
                ),
                "verification": {
                    "passed": True,
                    "scope": context.check,
                },
            },
            "artifacts": (
                {"path": context.artifact_path, "state": "reverified"},
            ),
        }
    )
    recovery_memory = _ambient_memory(seed, context) + (
        {
            "prior_verification": (
                f"The {context.check} reproduced {context.defect}."
            )
        },
    )
    return SyntheticScenario(
        scenario_id=f"synthetic-failed-verification-{seed:06d}",
        motif="failed_verification_replan_repair_verify_complete",
        evidence_basis=(
            (
                f"failed {context.check} evidence must prevent premature "
                "completion"
            ),
            "successful recovery required a new implementer-to-verifier topology",
        ),
        initial_boundary_id="verification_failed",
        boundaries=(
            SyntheticBoundary(
                boundary_id="verification_failed",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(failed,),
                    active_position_id=old_position,
                    paid_calls_used=calls_used,
                    paid_call_limit=budget_limit,
                    shared_memory=recovery_memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(
                    action="replan",
                    required_step_capabilities=(
                        frozenset({"implementer"}),
                        frozenset({"verifier"}),
                    ),
                    required_step_access=((), (0,)),
                    required_step_terms=(
                        frozenset(
                            {
                                "repair",
                                context.defect_term,
                                context.check_term,
                            }
                        ),
                        frozenset(
                            {
                                "independently",
                                context.check_term,
                                "repair",
                            }
                        ),
                    ),
                    oracle_subtasks=(
                        (
                            f"Repair the {context.defect_term} defect in the "
                            f"{context.deliverable} and run focused "
                            f"{context.check}."
                        ),
                        (
                            f"Independently rerun the full {context.check} "
                            "after the repair."
                        ),
                    ),
                ),
                next_boundary_id="repairing",
            ),
            SyntheticBoundary(
                boundary_id="repairing",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 2,
                    positions=(repair_active, verifier_pending),
                    active_position_id=repair_position,
                    paid_calls_used=calls_used + 1,
                    paid_call_limit=budget_limit,
                    shared_memory=recovery_memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="continue"),
                next_boundary_id="repair_done",
            ),
            SyntheticBoundary(
                boundary_id="repair_done",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 2,
                    positions=(repair_done, verifier_pending),
                    active_position_id=repair_position,
                    paid_calls_used=calls_used + 2,
                    paid_call_limit=budget_limit,
                    shared_memory=recovery_memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(
                    action="handoff",
                    target_position_id=verify_position,
                ),
                next_boundary_id="reverified",
            ),
            SyntheticBoundary(
                boundary_id="reverified",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 2,
                    positions=(
                        ControlPosition(
                            **{
                                **repair_done.__dict__,
                                "status": "completed",
                            }
                        ),
                        verified,
                    ),
                    active_position_id=verify_position,
                    paid_calls_used=calls_used + 3,
                    paid_call_limit=budget_limit,
                    shared_memory=recovery_memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="complete"),
                next_boundary_id=None,
            ),
        ),
    )


def _stalled_owner_recovery(
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> SyntheticScenario:
    profiles = _profiles(seed, profile_capabilities)
    context = _semantic_context(seed)
    unavailable = _unavailable_worker_ids(profiles, seed=seed)
    memory = _ambient_memory(seed, context)
    builder = _worker_id(profiles, "implementer")
    debugger = _worker_id(profiles, "debugger")
    owner_position = 60 + seed % 19
    debugger_position = owner_position + 5
    budget_limit = 22 + seed % 11
    calls_used = 5 + seed % 5
    if (seed // len(_SEMANTIC_CONTEXTS)) % 2:
        task = (
            f"Unblock the stalled {context.deliverable} without discarding the "
            f"workspace. Return it only after the {context.check} pass."
        )
    else:
        task = (
            f"Recover work on the {context.deliverable}, preserve its partial "
            f"artifact, and require the {context.check} before returning."
        )
    stalled = ControlPosition(
        position_id=owner_position,
        worker_id=builder,
        subtask=(
            f"Implement the {context.deliverable} and run the "
            f"{context.check}."
        ),
        access=(),
        status="active",
        progress={
            "completion_requested": False,
            "worker_report": (
                f"Repeated inspection of {context.artifact_path} without "
                "changing the partial result."
            ),
            "material_progress": {
                "latest_turn_changed_material_state": False,
                "turns_without_material_progress": 3 + seed % 3,
            },
        },
        artifacts=(
            {"path": context.artifact_path, "state": "partial_stalled"},
        ),
    )
    debugger_pending = ControlPosition(
        position_id=debugger_position,
        worker_id=debugger,
        subtask=(
            f"Diagnose the stall, finish the {context.deliverable}, and run the "
            f"{context.check}."
        ),
        access=(owner_position,),
        status="pending",
    )
    debugging = ControlPosition(
        **{
            **debugger_pending.__dict__,
            "status": "active",
            "progress": {
                "completion_requested": False,
                "worker_report": (
                    f"The {context.defect_term} issue is corrected; the "
                    f"{context.check} are running."
                ),
                "material_progress": {
                    "latest_turn_changed_material_state": True,
                    "turns_without_material_progress": 0,
                },
            },
            "artifacts": (
                {"path": context.artifact_path, "state": "recovered"},
            ),
        }
    )
    debugged = ControlPosition(
        **{
            **debugging.__dict__,
            "progress": {
                "completion_requested": True,
                "worker_report": (
                    f"The recovered {context.deliverable} passes the "
                    f"{context.check}."
                ),
                "verification": {
                    "passed": True,
                    "scope": context.check,
                },
            },
        }
    )
    return SyntheticScenario(
        scenario_id=f"synthetic-stalled-owner-{seed:06d}",
        motif="stalled_owner_handoff_continue_complete",
        evidence_basis=(
            (
                f"live workers repeatedly inspected a partial "
                f"{context.deliverable} without material change"
            ),
            "successful arms handed the preserved workspace to a debugger",
        ),
        initial_boundary_id="stalled",
        boundaries=(
            SyntheticBoundary(
                boundary_id="stalled",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(stalled, debugger_pending),
                    active_position_id=owner_position,
                    paid_calls_used=calls_used,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(
                    action="handoff",
                    target_position_id=debugger_position,
                ),
                next_boundary_id="debugging",
            ),
            SyntheticBoundary(
                boundary_id="debugging",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(
                        ControlPosition(
                            **{
                                **stalled.__dict__,
                                "status": "interrupted",
                            }
                        ),
                        debugging,
                    ),
                    active_position_id=debugger_position,
                    paid_calls_used=calls_used + 1,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="continue"),
                next_boundary_id="debugged",
            ),
            SyntheticBoundary(
                boundary_id="debugged",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(
                        ControlPosition(
                            **{
                                **stalled.__dict__,
                                "status": "interrupted",
                            }
                        ),
                        debugged,
                    ),
                    active_position_id=debugger_position,
                    paid_calls_used=calls_used + 2,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="complete"),
                next_boundary_id=None,
            ),
        ),
    )


def _budget_limited_initial(
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> SyntheticScenario:
    profiles = _profiles(seed, profile_capabilities)
    context = _semantic_context(seed)
    unavailable = _unavailable_worker_ids(profiles, seed=seed)
    memory = _ambient_memory(seed, context)
    implementer = _worker_id(profiles, "implementer")
    position = 90 + seed % 23
    budget_limit = 5 + seed % 5
    if (seed // len(_SEMANTIC_CONTEXTS)) % 2:
        task = (
            f"Use the short call budget to create the {context.deliverable} and "
            f"run the {context.check}. Return the verified result."
        )
    else:
        task = (
            f"With few calls available, produce the {context.deliverable}; the "
            f"{context.check} must pass before completion."
        )
    implementing = ControlPosition(
        position_id=position,
        worker_id=implementer,
        subtask=(
            f"Create the {context.deliverable} and run the "
            f"{context.check}."
        ),
        access=(),
        status="active",
        progress={
            "completion_requested": False,
            "worker_report": (
                f"The {context.deliverable} exists at {context.artifact_path}; "
                f"the {context.check} are running."
            ),
            "material_progress": {"latest_turn_changed_material_state": True},
        },
        artifacts=(
            {"path": context.artifact_path, "state": "created_unverified"},
        ),
    )
    complete = ControlPosition(
        **{
            **implementing.__dict__,
            "progress": {
                "completion_requested": True,
                "worker_report": (
                    f"The {context.deliverable} is complete and the "
                    f"{context.check} pass."
                ),
                "verification": {
                    "passed": True,
                    "scope": context.check,
                },
            },
            "artifacts": (
                {"path": context.artifact_path, "state": "verified"},
            ),
        }
    )
    return SyntheticScenario(
        scenario_id=f"synthetic-budget-limited-{seed:06d}",
        motif="budget_limited_replan_continue_complete",
        evidence_basis=(
            (
                f"live low-budget plans deferred the {context.deliverable} "
                "behind inspection"
            ),
            (
                f"successful low-budget runs combined creation with the "
                f"{context.check} in the first ownership step"
            ),
        ),
        initial_boundary_id="initial",
        boundaries=(
            SyntheticBoundary(
                boundary_id="initial",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=None,
                    positions=(),
                    active_position_id=None,
                    paid_calls_used=0,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(
                    action="replan",
                    required_step_capabilities=(frozenset({"implementer"}),),
                    required_step_access=((),),
                    required_step_terms=(
                        frozenset(
                            {
                                "create",
                                context.deliverable_term,
                                context.check_term,
                            }
                        ),
                    ),
                    oracle_subtasks=(
                        (
                            f"Create the {context.deliverable} and run the "
                            f"{context.check}."
                        ),
                    ),
                ),
                next_boundary_id="implementing",
            ),
            SyntheticBoundary(
                boundary_id="implementing",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(implementing,),
                    active_position_id=position,
                    paid_calls_used=1,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="continue"),
                next_boundary_id="complete",
            ),
            SyntheticBoundary(
                boundary_id="complete",
                state=_state(
                    task=task,
                    profiles=profiles,
                    workflow_id=seed + 1,
                    positions=(complete,),
                    active_position_id=position,
                    paid_calls_used=2,
                    paid_call_limit=budget_limit,
                    shared_memory=memory,
                    unavailable_worker_ids=unavailable,
                ),
                oracle=SyntheticActionRule(action="complete"),
                next_boundary_id=None,
            ),
        ),
    )


_SCENARIO_FACTORIES = (
    _verification_chain,
    _failed_verification_recovery,
    _stalled_owner_recovery,
    _budget_limited_initial,
)


def build_synthetic_curriculum(
    *,
    count: int,
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...] = _PROFILE_CAPABILITIES,
) -> tuple[SyntheticScenario, ...]:
    """Build a deterministic mix for one anonymous calibrated capability pool."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise SyntheticRolloutError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SyntheticRolloutError("seed must be an integer")
    validated_profiles = _validated_profile_capabilities(profile_capabilities)
    scenarios = tuple(
        _SCENARIO_FACTORIES[index % len(_SCENARIO_FACTORIES)](
            seed + index,
            validated_profiles,
        )
        for index in range(count)
    )
    for scenario in scenarios:
        scenario.validate()
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise SyntheticRolloutError("curriculum scenario IDs are not unique")
    return scenarios


__all__ = [
    "SYNTHETIC_CURRICULUM_REVISION",
    "SyntheticActionRule",
    "SyntheticBoundary",
    "SyntheticConductorEnv",
    "SyntheticDecisionRecord",
    "SyntheticPolicyAttestation",
    "SyntheticRolloutError",
    "SyntheticSampledRollout",
    "SyntheticScenario",
    "SyntheticStepResult",
    "build_synthetic_curriculum",
    "sample_synthetic_rollout",
]
