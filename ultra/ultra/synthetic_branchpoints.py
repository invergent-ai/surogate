"""Event-grounded one-call synthetic branchpoints for conductor GRPO.

Each sample asks the real conductor for exactly one action at a fixed live
state.  A deterministic continuation then executes that action and emits
artifact, check, verification, and budget events.  Training reward is derived
only from those events.  There is no preferred-action label in a scenario.

The simulator is deliberately conservative.  Parsed, legal actions whose
free-form replacement plan cannot be interpreted unambiguously are marked
``unmodeled`` and cannot enter policy optimization.  Invalid policy output is
kept as ``protocol_only`` evidence and is likewise excluded from semantic
GRPO.
"""

from __future__ import annotations

import math
import re
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
    capability_reference_map,
    validate_control_action,
    validate_control_state,
)


BRANCHPOINT_CURRICULUM_REVISION = "20260724-direct-event-branchpoints-v6"
FIXED_CONTINUATION_REVISION = (
    "20260724-position-grounded-uniform-continuation-v8"
)
FIXED_CONTINUATION_MODE = "deterministic_fixed_continuation"
MAX_BRANCHPOINT_SEQUENCE_TOKENS = 8_192

BranchpointDisposition = Literal["eligible", "protocol_only", "unmodeled"]
OperationKind = Literal["produce", "repair", "verify", "inspect", "stall"]


class SyntheticBranchpointError(ValueError):
    """A branchpoint, exact policy trace, or deterministic replay is invalid."""


@dataclass(frozen=True)
class BranchpointFixedStep:
    """One visible workflow-position event executed by the continuation."""

    position_id: int
    operation: OperationKind


@dataclass(frozen=True)
class SyntheticBranchpointScenario:
    """A single sampled decision followed by deterministic event simulation."""

    scenario_id: str
    motif: str
    evidence_basis: tuple[str, ...]
    state: LiveControlState
    required_obligations: tuple[OperationKind, ...]
    initial_artifact_status: Literal["absent", "partial", "ready", "defective"]
    initial_material_worker_id: int | None
    initial_verification_failed: bool
    continuation_budget: int
    fixed_steps: tuple[BranchpointFixedStep, ...]
    artifact_label: str
    check_label: str
    defect_label: str

    def validate(self) -> None:
        if not self.scenario_id or not self.motif or not self.evidence_basis:
            raise SyntheticBranchpointError("branchpoint metadata is incomplete")
        validate_control_state(self.state)
        if not self.required_obligations:
            raise SyntheticBranchpointError("branchpoint has no terminal obligations")
        allowed = {"produce", "repair", "verify"}
        if any(item not in allowed for item in self.required_obligations):
            raise SyntheticBranchpointError("branchpoint has an invalid obligation")
        if len(set(self.required_obligations)) != len(
            self.required_obligations
        ):
            raise SyntheticBranchpointError(
                "branchpoint obligations must be unique"
            )
        if (
            isinstance(self.continuation_budget, bool)
            or not isinstance(self.continuation_budget, int)
            or self.continuation_budget <= 0
        ):
            raise SyntheticBranchpointError(
                "continuation budget must be a positive integer"
            )
        worker_ids = set(self.state.worker_ids)
        worker_profiles = {
            worker.worker_id: worker for worker in self.state.workers
        }
        if (
            self.initial_material_worker_id is not None
            and self.initial_material_worker_id not in worker_ids
        ):
            raise SyntheticBranchpointError(
                "initial material worker is outside the anonymous pool"
            )
        positions = {
            position.position_id: position for position in self.state.positions
        }
        _validate_fixed_steps(
            self.fixed_steps,
            positions=positions,
            worker_profiles=worker_profiles,
        )
        fixed_position_ids = {step.position_id for step in self.fixed_steps}
        eligible_targets: set[int] = set()
        for position in self.state.positions:
            if position.status != "pending":
                continue
            try:
                validate_control_action(
                    ControlAction(
                        action="handoff",
                        reason="Exercise the deterministic continuation.",
                        target_position_id=position.position_id,
                    ),
                    self.state,
                )
            except ControlContractError:
                continue
            eligible_targets.add(position.position_id)
        if not eligible_targets.issubset(fixed_position_ids):
            raise SyntheticBranchpointError(
                "a legal handoff target lacks a fixed position operation"
            )
        try:
            validate_control_action(
                ControlAction(
                    action="continue",
                    reason="Exercise the deterministic continuation.",
                ),
                self.state,
            )
        except ControlContractError:
            pass
        else:
            if (
                self.state.active_position_id is None
                or self.state.active_position_id not in fixed_position_ids
            ):
                raise SyntheticBranchpointError(
                    "a legal continue lacks an active fixed position operation"
                )
        material_workers = {
            positions[step.position_id].worker_id
            for step in self.fixed_steps
            if step.operation in {"produce", "repair"}
        }
        verification_workers = {
            positions[step.position_id].worker_id
            for step in self.fixed_steps
            if step.operation == "verify"
        }
        if (
            material_workers
            and verification_workers
            and not any(
                material_worker != verification_worker
                for material_worker in material_workers
                for verification_worker in verification_workers
            )
        ):
            raise SyntheticBranchpointError(
                "fixed continuation cannot independently verify material"
            )
        for label, value in (
            ("artifact_label", self.artifact_label),
            ("check_label", self.check_label),
            ("defect_label", self.defect_label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SyntheticBranchpointError(f"{label} must be non-empty")


def _validate_fixed_steps(
    steps: tuple[BranchpointFixedStep, ...],
    *,
    positions: dict[int, ControlPosition],
    worker_profiles: dict[int, WorkerProfile],
) -> None:
    position_ids = [step.position_id for step in steps]
    if len(position_ids) != len(set(position_ids)):
        raise SyntheticBranchpointError(
            "fixed continuation position IDs must be unique"
        )
    for step in steps:
        if step.operation not in {
            "produce",
            "repair",
            "verify",
            "inspect",
            "stall",
        }:
            raise SyntheticBranchpointError(
                "fixed continuation has an unknown operation"
            )
        position = positions.get(step.position_id)
        if position is None:
            raise SyntheticBranchpointError(
                "fixed continuation references a position outside the workflow"
            )
        if position.status not in {"active", "pending"}:
            raise SyntheticBranchpointError(
                "fixed continuation position is not executable"
            )
        capabilities = set(
            worker_profiles[position.worker_id].capability_tags
        )
        required = {
            "produce": _IMPLEMENT_CAPABILITIES,
            "repair": _REPAIR_CAPABILITIES,
            "verify": _VERIFY_CAPABILITIES,
            "inspect": _INSPECT_CAPABILITIES,
        }.get(step.operation)
        if required is not None and not capabilities & required:
            raise SyntheticBranchpointError(
                "fixed continuation worker lacks operation capability"
            )


@dataclass(frozen=True)
class BranchpointEvaluation:
    disposition: BranchpointDisposition
    reward: float | None
    outcome: str
    events: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]

    @property
    def training_eligible(self) -> bool:
        return self.disposition == "eligible"


@dataclass(frozen=True)
class BranchpointPolicyAttestation:
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
                raise SyntheticBranchpointError(f"{label} must be non-empty")
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed,
            int,
        ):
            raise SyntheticBranchpointError(
                "sampling_seed must be an integer"
            )


@dataclass(frozen=True)
class SyntheticBranchpointSample:
    scenario_id: str
    motif: str
    action: ControlAction | None
    disposition: BranchpointDisposition
    reward: float | None
    outcome: str
    events: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]
    trace: dict[str, Any]
    policy: BranchpointPolicyAttestation

    @property
    def training_eligible(self) -> bool:
        return self.disposition == "eligible"


class BranchpointController(Protocol):
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


@dataclass
class _World:
    artifact_status: str
    material_revision: int
    material_worker_id: int | None
    verified_revision: int | None
    budget_remaining: int


_IMPLEMENT_CAPABILITIES = frozenset(
    {"implementer", "coder", "drafter", "builder", "engineer"}
)
_REPAIR_CAPABILITIES = frozenset(
    {"implementer", "coder", "debugger", "engineer", "failure_analyst"}
)
_VERIFY_CAPABILITIES = frozenset(
    {"verifier", "reviewer", "independent_reviewer", "auditor"}
)
_INSPECT_CAPABILITIES = frozenset(
    {
        "debugger",
        "planner",
        "scientist",
        "reasoner",
        "failure_analyst",
        "analyst",
    }
)

_PRODUCE_VERBS = frozenset(
    {
        "build",
        "complete",
        "completed",
        "completes",
        "completing",
        "create",
        "deliver",
        "draft",
        "finalise",
        "finalised",
        "finalises",
        "finalising",
        "finalize",
        "finalized",
        "finalizes",
        "finalizing",
        "finish",
        "finished",
        "finishes",
        "finishing",
        "generate",
        "implement",
        "write",
    }
)
_REPAIR_VERBS = frozenset(
    {
        "correct",
        "debug",
        "fix",
        "patch",
        "repair",
        "resolve",
        "remediate",
    }
)
_VERIFY_VERBS = frozenset(
    {
        "audit",
        "check",
        "checked",
        "checking",
        "checks",
        "execute",
        "executed",
        "executing",
        "executes",
        "recheck",
        "rerun",
        "run",
        "running",
        "runs",
        "test",
        "tested",
        "testing",
        "tests",
        "validate",
        "validated",
        "validates",
        "validating",
        "validation",
        "verification",
        "verify",
        "verified",
        "verifies",
        "verifying",
    }
)
_VERIFY_NOMINAL_TERMS = frozenset(
    {
        "check",
        "checks",
        "test",
        "testing",
        "tests",
        "validation",
        "verification",
    }
)
_INSPECT_VERBS = frozenset(
    {"analyze", "diagnose", "explore", "inspect", "investigate", "review"}
)
_NEGATING_DIRECTIVE_PATTERN = re.compile(
    r"(?:^|\b(?:and|but|then)\s+)"
    r"(?:please\s+)?"
    r"(?:"
    r"(?:we|you)\s+"
    r")?"
    r"(?:"
    r"never\b"
    r"|cannot\b"
    r"|can't\b"
    r"|avoid(?:s|ed|ing)?\b"
    r"|omit\b"
    r"|omitting\b"
    r"|skip\b"
    r"|skipping\b"
    r"|refrain(?:s|ed|ing)?\b"
    r"|refus(?:e|es|ed|ing)\b"
    r"|(?:do|does|did|must|will|would|should|can|could)\s+not\b"
    r"|(?:don't|doesn't|didn't|mustn't|won't|wouldn't|shouldn't|"
    r"couldn't)\b"
    r")"
)
_NEGATING_REASON_DIRECTIVE_PATTERN = re.compile(
    r"(?:^|\b(?:and|but|then)\s+)"
    r"(?:please\s+)?"
    r"(?:(?:we|you)\s+)?"
    r"(?:"
    r"never\b"
    r"|avoid(?:s|ed|ing)?\b"
    r"|omit\b"
    r"|omitting\b"
    r"|skip\b"
    r"|skipping\b"
    r"|refrain(?:s|ed|ing)?\b"
    r"|refus(?:e|es|ed|ing)\b"
    r"|(?:do|must|will|would|should)\s+not\b"
    r"|(?:don't|mustn't|won't|wouldn't|shouldn't)\b"
    r")"
)
_NEGATED_REASON_SCOPE_BOUNDARY = re.compile(
    r"\b(?:because|before|after|that|which|while|whereas|without)\b"
    r"|\bdue\s+to\b"
    r"|\bso\s+that\b"
    r"|\b(?:and|then)\s+"
    r"(?:(?:we|you)\s+)?"
    r"(?:(?:must|will|would|should)\s+)?"
    r"(?:complete|continue|handoff|proceed|replan)\b"
)
_CLAUSE_BOUNDARY = re.compile(
    r"(?:[.!?;,]+|\b(?:but|plus|then|while|whereas)\b|"
    r"\b(?:afterwards|subsequently)\b|"
    r"\b(?:as well as|followed by)\b)"
)
_UNMODELED_ACTION_WORDS = frozenset(
    {
        "copy",
        "destroy",
        "delete",
        "download",
        "erase",
        "exfiltrate",
        "expose",
        "install",
        "leak",
        "move",
        "publish",
        "remove",
        "rename",
        "send",
        "share",
        "upload",
    }
)
_BENIGN_VERIFY_WRAPUP = re.compile(
    r"\b(?:and|to)\s+complete\s+(?:the\s+)?(?:task|workflow)\s*$"
)
_ALTERNATIVE_WORK_WORDS = frozenset(
    {"additional", "another", "extra", "second", "separate", "subsequent"}
)
_REASON_OPERATION_WORDS: dict[OperationKind, frozenset[str]] = {
    "produce": _PRODUCE_VERBS
    | {
        "creation",
        "creator",
        "creators",
        "finisher",
        "finishers",
        "produce",
        "produced",
        "producer",
        "producers",
        "produces",
        "producing",
        "production",
    },
    "repair": _REPAIR_VERBS
    | {"repairer", "repairers", "repairing", "repairs"},
    "verify": _VERIFY_VERBS | {"verifier", "verifiers"},
    "inspect": _INSPECT_VERBS | {"inspection", "inspector", "inspectors"},
    "stall": frozenset({"stall", "stalled", "stalling", "stalls"}),
}
_REASON_ACTION_WORDS: dict[str, frozenset[str]] = {
    "complete": frozenset({"complete", "completing", "completion"}),
    "continue": frozenset({"continue", "continued", "continues", "continuing"}),
    "handoff": frozenset({"handoff", "handover"}),
    "replan": frozenset({"replan", "replanned", "replanning"}),
}


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


def _has_negated_operation(value: str) -> bool:
    """Detect instructions to omit work without rejecting observed defects.

    A past-participle description such as "omitted environment key" is
    evidence about the live state, not a request to omit an operation.  Direct
    negative imperatives remain unsupported, as do operation clauses that say
    to proceed without or except the required operation.
    """
    normalized = " ".join(value.casefold().split())
    clauses = [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY.split(normalized)
        if clause.strip()
    ]
    operation_words = (
        _PRODUCE_VERBS
        | _REPAIR_VERBS
        | _VERIFY_VERBS
        | _INSPECT_VERBS
    )
    for clause in clauses:
        if _NEGATING_DIRECTIVE_PATTERN.search(clause):
            return True
        words = _tokens(clause)
        starts_operation = bool(words & operation_words) and bool(
            re.match(
                r"^(?:please\s+)?"
                r"(?:proceed|"
                + "|".join(sorted(operation_words))
                + r")\b",
                clause,
            )
        )
        if starts_operation and (
            re.search(r"\bwithout\b", clause)
            or re.search(r"\bexcept\b", clause)
        ):
            return True
    return False


def _clause_words_for_operation(
    clause: str,
    *,
    operation: OperationKind,
) -> frozenset[str]:
    """Remove only a bounded verifier outcome suffix from clause semantics."""
    normalized = clause
    if operation == "verify":
        # In "run the checks ... and complete the task", verification is the
        # modeled operation and task completion is its ordinary outcome.  A
        # standalone or sequenced completion request is deliberately retained
        # and rejected as a second operation.
        normalized = _BENIGN_VERIFY_WRAPUP.sub("", normalized).strip()
    return _tokens(normalized)


def _classify_replan_step(
    scenario: SyntheticBranchpointScenario,
    step: ControlStep,
) -> OperationKind | None:
    normalized = " ".join(step.subtask.casefold().split())
    if _has_negated_operation(normalized):
        return None
    words = _tokens(normalized)
    profiles = {worker.worker_id: worker for worker in scenario.state.workers}
    capabilities = set(profiles[step.worker_id].capability_tags)

    candidates: list[OperationKind] = []
    if capabilities & _REPAIR_CAPABILITIES and words & _REPAIR_VERBS:
        candidates.append("repair")
    if capabilities & _IMPLEMENT_CAPABILITIES and words & _PRODUCE_VERBS:
        candidates.append("produce")
    if capabilities & _VERIFY_CAPABILITIES and words & _VERIFY_VERBS:
        candidates.append("verify")
    if capabilities & _INSPECT_CAPABILITIES and words & _INSPECT_VERBS:
        candidates.append("inspect")

    # Role evidence disambiguates common phrases such as "implement and test":
    # an implementation-only profile is producing, not independently verifying.
    if len(candidates) > 1:
        if "verify" in candidates and not capabilities & _VERIFY_CAPABILITIES:
            candidates.remove("verify")
        if (
            "verify" in candidates
            and "inspect" in candidates
            and words & _INSPECT_VERBS
            and not words & (_VERIFY_VERBS - _VERIFY_NOMINAL_TERMS)
        ):
            candidates.remove("verify")
        if "inspect" in candidates and (
            "repair" in candidates or "produce" in candidates
        ):
            candidates.remove("inspect")
    if len(candidates) != 1:
        return None
    return candidates[0]


def _selected_reason_operations(
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
) -> frozenset[OperationKind]:
    if action.action == "replan":
        return frozenset(
            operation
            for step in action.steps
            for operation in (_classify_replan_step(scenario, step),)
            if operation is not None
        )
    if action.action == "continue":
        selected_position_id = scenario.state.active_position_id
    elif action.action == "handoff":
        selected_position_id = action.target_position_id
    else:
        selected_position_id = None
    return frozenset(
        step.operation
        for step in scenario.fixed_steps
        if step.position_id == selected_position_id
    )


def _selected_position_mentions(
    value: str,
    *,
    selected_position_id: int | None,
) -> tuple[bool, bool]:
    mentioned = {
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:pos|position|target)"
            r"(?:\s+id)?\s*[\(\[#:]*\s*(\d+)\b",
            value,
        )
    }
    if not mentioned:
        return False, False
    return True, selected_position_id in mentioned


def _reason_scope(
    clause: str,
    *,
    directive_end: int,
) -> str:
    scope = clause[directive_end:].strip()
    boundary = _NEGATED_REASON_SCOPE_BOUNDARY.search(scope)
    if boundary is not None:
        scope = scope[: boundary.start()].strip()
    return scope


def _starts_selected_reason_work(
    value: str,
    *,
    selected_words: frozenset[str],
) -> bool:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    while words and words[0] in {
        "i",
        "must",
        "please",
        "should",
        "we",
        "will",
        "would",
        "you",
    }:
        words.pop(0)
    return bool(words) and (
        words[0] == "proceed" or words[0] in selected_words
    )


def _has_negated_selected_reason_work(
    *,
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
    value: str,
) -> bool:
    """Reject only directives that negate the operation the action selects.

    Free-form reasons routinely explain why a previous owner could not finish
    or why an optional inspection/replan cannot fit the remaining budget.
    Those observations do not negate the operation selected by the structured
    action.  Replan subtasks retain the broader fail-closed detector above.
    """
    selected_operations = _selected_reason_operations(scenario, action)
    selected_words = set(_REASON_ACTION_WORDS.get(action.action, ()))
    for operation in selected_operations:
        selected_words.update(_REASON_OPERATION_WORDS[operation])
    frozen_selected_words = frozenset(selected_words)
    selected_position_id = (
        scenario.state.active_position_id
        if action.action == "continue"
        else action.target_position_id
        if action.action == "handoff"
        else None
    )
    normalized = " ".join(value.casefold().split())
    clauses = [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY.split(normalized)
        if clause.strip()
    ]
    for clause in clauses:
        directive = _NEGATING_REASON_DIRECTIVE_PATTERN.search(clause)
        if directive is not None:
            scope = _reason_scope(
                clause,
                directive_end=directive.end(),
            )
            has_position, mentions_selected_position = (
                _selected_position_mentions(
                    scope,
                    selected_position_id=selected_position_id,
                )
            )
            if has_position:
                if mentions_selected_position:
                    return True
                continue
            scope_words = _tokens(scope)
            if (
                scope_words & _ALTERNATIVE_WORK_WORDS
                and "required" not in scope_words
            ):
                continue
            if scope_words & frozen_selected_words:
                return True

        for marker in ("without", "except"):
            match = re.search(rf"\b{marker}\b", clause)
            if match is None:
                continue
            prefix = clause[: match.start()].strip()
            scope = _reason_scope(
                clause,
                directive_end=match.end(),
            )
            scope_words = _tokens(scope)
            if (
                _starts_selected_reason_work(
                    prefix,
                    selected_words=frozen_selected_words,
                )
                and scope_words & frozen_selected_words
            ):
                return True
    return False


def _unsupported_replan_clauses(
    *,
    scenario: SyntheticBranchpointScenario,
    step: ControlStep,
    operation: OperationKind,
) -> tuple[int, ...]:
    """Fail closed when every semantic clause is not executor-grounded.

    A branchpoint executor models exactly one operation per sampled step.
    Consequently every clause must either name that operation or be a bounded
    object/modifier fragment grounded in the live state.  Unknown actions and
    mixed-operation clauses cannot safely inherit the emitted event.
    """
    normalized = " ".join(step.subtask.casefold().split())
    clauses = [
        clause.strip()
        for clause in _CLAUSE_BOUNDARY.split(normalized)
        if clause.strip()
    ]
    operation_words: dict[OperationKind, frozenset[str]] = {
        "produce": _PRODUCE_VERBS,
        "repair": _REPAIR_VERBS,
        "verify": _VERIFY_VERBS,
        "inspect": _INSPECT_VERBS,
        "stall": frozenset({"stall"}),
    }
    explicit_operation_words: dict[OperationKind, frozenset[str]] = {
        "produce": operation_words["produce"],
        "repair": operation_words["repair"],
        "verify": operation_words["verify"]
        - {"check", "checks", "test", "tests", "validation", "verification"},
        "inspect": operation_words["inspect"] - {"review"},
        "stall": operation_words["stall"],
    }
    unsupported: list[int] = []
    for index, clause in enumerate(clauses):
        clause_operation = _classify_replan_step(
            scenario,
            ControlStep(
                worker_id=step.worker_id,
                subtask=clause,
            ),
        )
        words = _clause_words_for_operation(
            clause,
            operation=operation,
        )
        other_operation_predicate = any(
            words & verbs
            for kind, verbs in explicit_operation_words.items()
            if kind != operation
        )
        if (
            words & _UNMODELED_ACTION_WORDS
            or (
                clause_operation is not None
                and clause_operation != operation
            )
            or other_operation_predicate
        ):
            unsupported.append(index)
    if not clauses:
        unsupported.append(0)
    return tuple(unsupported)


def _replan_step_target_grounded(
    *,
    scenario: SyntheticBranchpointScenario,
    step: ControlStep,
    operation: OperationKind,
) -> bool:
    """Require the sampled operation to name the object the event mutates."""
    normalized = " ".join(step.subtask.casefold().split())
    words = _tokens(step.subtask)
    artifact_generic = frozenset(
        {
            "artifact",
            "artifacts",
            "bundle",
            "code",
            "deliverable",
            "deliverables",
            "file",
            "files",
            "helper",
            "implementation",
            "index",
            "inventory",
            "material",
            "output",
            "outputs",
            "patch",
            "report",
            "script",
            "timeline",
            "utility",
        }
    )
    check_generic = frozenset(
        {
            "check",
            "checks",
            "suite",
            "test",
            "tests",
            "validation",
            "verification",
        }
    )
    defect_generic = frozenset(
        {"bug", "defect", "failure", "issue"}
    )
    nondistinctive = frozenset(
        {"a", "an", "and", "for", "of", "the", "with"}
    )
    artifact_distinctive = (
        _tokens(scenario.artifact_label)
        - artifact_generic
        - nondistinctive
    )
    check_distinctive = (
        _tokens(scenario.check_label) - check_generic - nondistinctive
    )

    def distinctive_match(tokens: frozenset[str]) -> bool:
        return bool(tokens) and len(words & tokens) >= min(2, len(tokens))

    artifact_paths = {
        match.group(0)
        for match in re.finditer(
            r"/[a-zA-Z0-9_./-]+",
            scenario.state.original_task,
        )
    }
    artifact_paths.update(
        path
        for position in scenario.state.positions
        for artifact in position.artifacts
        if isinstance(artifact, dict)
        for path in (artifact.get("path"),)
        if isinstance(path, str) and path
    )
    artifact_path_match = any(
        path.casefold() in normalized for path in artifact_paths
    )
    artifact_match = artifact_path_match or distinctive_match(
        artifact_distinctive
    )
    check_match = distinctive_match(check_distinctive)

    if operation == "produce":
        return artifact_match and not bool(
            words & (check_generic | defect_generic)
        )
    if operation == "repair":
        return artifact_match
    if operation == "verify":
        return check_match
    if operation == "inspect":
        return bool(
            artifact_match
            or check_match
            or words
            & (
                defect_generic
                | _tokens(scenario.defect_label)
                | {"evidence", "input", "inputs"}
            )
        )
    return operation == "stall"


def _event(
    *,
    kind: str,
    source: str,
    actor_profile_ref: str | None = None,
    **payload: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {"kind": kind, "source": source}
    if actor_profile_ref is not None:
        value["actor_profile_ref"] = actor_profile_ref
    value.update(payload)
    return value


def _initial_events(
    scenario: SyntheticBranchpointScenario,
) -> tuple[list[dict[str, Any]], _World]:
    references = capability_reference_map(scenario.state.workers)
    actor_ref = (
        references.worker_id_to_profile_ref[
            scenario.initial_material_worker_id
        ]
        if scenario.initial_material_worker_id is not None
        else None
    )
    events = [
        _event(
            kind="initial_artifact",
            source="observed_live_state",
            actor_profile_ref=actor_ref,
            label=scenario.artifact_label,
            status=scenario.initial_artifact_status,
            revision=0,
        )
    ]
    if scenario.initial_verification_failed:
        events.append(
            _event(
                kind="check_result",
                source="observed_live_state",
                label=scenario.check_label,
                passed=False,
                reason=scenario.defect_label,
                revision=0,
                independent=True,
            )
        )
    return (
        events,
        _World(
            artifact_status=scenario.initial_artifact_status,
            material_revision=0,
            material_worker_id=scenario.initial_material_worker_id,
            verified_revision=None,
            budget_remaining=scenario.continuation_budget,
        ),
    )


def _consume_budget(
    *,
    world: _World,
    events: list[dict[str, Any]],
    source: str,
    actor_ref: str,
    position_id: int | None = None,
) -> bool:
    position_payload = (
        {} if position_id is None else {"position_id": position_id}
    )
    if world.budget_remaining <= 0:
        events.append(
            _event(
                kind="budget_exhausted",
                source=source,
                actor_profile_ref=actor_ref,
                remaining=0,
                **position_payload,
            )
        )
        return False
    world.budget_remaining -= 1
    events.append(
        _event(
            kind="budget_spent",
            source=source,
            actor_profile_ref=actor_ref,
            amount=1,
            remaining=world.budget_remaining,
            **position_payload,
        )
    )
    return True


def _execute_operation(
    *,
    scenario: SyntheticBranchpointScenario,
    operation: OperationKind,
    worker_id: int,
    source: str,
    world: _World,
    events: list[dict[str, Any]],
    material_visible: bool = True,
    position_id: int | None = None,
) -> bool:
    references = capability_reference_map(scenario.state.workers)
    actor_ref = references.worker_id_to_profile_ref[worker_id]
    position_payload = (
        {} if position_id is None else {"position_id": position_id}
    )
    if not _consume_budget(
        world=world,
        events=events,
        source=source,
        actor_ref=actor_ref,
        position_id=position_id,
    ):
        return False

    if operation == "stall":
        events.append(
            _event(
                kind="worker_progress",
                source=source,
                actor_profile_ref=actor_ref,
                changed_material_state=False,
                status="stalled",
                **position_payload,
            )
        )
        return True
    if operation == "inspect":
        events.append(
            _event(
                kind="inspection_result",
                source=source,
                actor_profile_ref=actor_ref,
                label=scenario.artifact_label,
                material_change=False,
                **position_payload,
            )
        )
        return True
    if operation == "produce":
        world.material_revision += 1
        world.artifact_status = "ready"
        world.material_worker_id = worker_id
        world.verified_revision = None
        events.append(
            _event(
                kind="artifact_emitted",
                source=source,
                actor_profile_ref=actor_ref,
                label=scenario.artifact_label,
                status="ready",
                revision=world.material_revision,
                **position_payload,
            )
        )
        return True
    if operation == "repair":
        if world.artifact_status not in {"defective", "partial", "ready"}:
            events.append(
                _event(
                    kind="repair_result",
                    source=source,
                    actor_profile_ref=actor_ref,
                    label=scenario.artifact_label,
                    passed=False,
                    reason="no_repairable_artifact",
                    **position_payload,
                )
            )
            return True
        world.material_revision += 1
        world.artifact_status = "repaired"
        world.material_worker_id = worker_id
        world.verified_revision = None
        events.append(
            _event(
                kind="repair_result",
                source=source,
                actor_profile_ref=actor_ref,
                label=scenario.artifact_label,
                passed=True,
                defect=scenario.defect_label,
                revision=world.material_revision,
                **position_payload,
            )
        )
        return True

    passed = (
        material_visible
        and world.artifact_status in {"ready", "repaired"}
        and world.material_worker_id is not None
        and world.material_worker_id != worker_id
    )
    reason = (
        "passed"
        if passed
        else "material_not_accessible"
        if not material_visible
        else "known_defect_unrepaired"
        if world.artifact_status == "defective"
        else "artifact_not_ready"
        if world.artifact_status not in {"ready", "repaired"}
        else "verification_not_independent"
    )
    if passed:
        world.verified_revision = world.material_revision
    events.append(
        _event(
            kind="verification_result",
            source=source,
            actor_profile_ref=actor_ref,
            label=scenario.check_label,
            passed=passed,
            independent=(
                world.material_worker_id is not None
                and world.material_worker_id != worker_id
            ),
            revision=world.material_revision,
            reason=reason,
            **position_payload,
        )
    )
    return True


def _satisfaction_from_events(
    *,
    scenario: SyntheticBranchpointScenario,
    events: list[dict[str, Any]],
    world: _World,
) -> tuple[list[str], list[str]]:
    successful: set[str] = set()
    for item in events:
        kind = item.get("kind")
        if kind == "artifact_emitted" and item.get("status") == "ready":
            successful.add("produce")
        elif kind == "repair_result" and item.get("passed") is True:
            successful.add("repair")
            # "produce" is the scenario's material-ready obligation, not a
            # demand for one literal verb.  Repairing an existing partial
            # artifact yields the same ready material state.
            successful.add("produce")
    if (
        world.artifact_status in {"ready", "repaired"}
        and world.material_worker_id is not None
        and world.verified_revision is not None
        and world.verified_revision == world.material_revision
    ):
        successful.add("verify")
    satisfied = [
        item for item in scenario.required_obligations if item in successful
    ]
    missing = [
        item for item in scenario.required_obligations if item not in successful
    ]
    if any(item.get("kind") == "budget_exhausted" for item in events):
        missing.append("within_budget")
    return satisfied, missing


def _required_obligations_met(
    *,
    scenario: SyntheticBranchpointScenario,
    events: list[dict[str, Any]],
    world: _World,
) -> bool:
    _, missing = _satisfaction_from_events(
        scenario=scenario,
        events=events,
        world=world,
    )
    return not missing


def _fixed_step_map(
    scenario: SyntheticBranchpointScenario,
) -> dict[int, BranchpointFixedStep]:
    return {step.position_id: step for step in scenario.fixed_steps}


def _execute_fixed_step(
    *,
    scenario: SyntheticBranchpointScenario,
    step: BranchpointFixedStep,
    world: _World,
    events: list[dict[str, Any]],
) -> bool:
    positions = {
        position.position_id: position for position in scenario.state.positions
    }
    position = positions[step.position_id]
    return _execute_operation(
        scenario=scenario,
        operation=step.operation,
        worker_id=position.worker_id,
        source=f"workflow_position:{position.position_id}",
        world=world,
        events=events,
        position_id=position.position_id,
    )


def _execute_fixed_continuation(
    *,
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
    world: _World,
    events: list[dict[str, Any]],
) -> None:
    """Run the selected position first, then one deterministic live DAG."""
    positions = {
        position.position_id: position for position in scenario.state.positions
    }
    fixed = _fixed_step_map(scenario)
    resolved = {
        position.position_id
        for position in scenario.state.positions
        if position.status in {"completed", "interrupted"}
    }
    if action.action == "continue":
        selected_position_id = scenario.state.active_position_id
    else:
        selected_position_id = action.target_position_id
        if scenario.state.active_position_id is not None:
            resolved.add(scenario.state.active_position_id)
    if (
        selected_position_id is None
        or selected_position_id not in fixed
    ):
        raise SyntheticBranchpointError(
            "sampled action has no fixed position operation"
        )

    pending_position_id: int | None = selected_position_id
    executed: set[int] = set()
    while pending_position_id is not None:
        step = fixed[pending_position_id]
        if not _execute_fixed_step(
            scenario=scenario,
            step=step,
            world=world,
            events=events,
        ):
            break
        executed.add(pending_position_id)
        resolved.add(pending_position_id)
        if _required_obligations_met(
            scenario=scenario,
            events=events,
            world=world,
        ):
            break
        pending_position_id = next(
            (
                candidate.position_id
                for candidate in scenario.fixed_steps
                if candidate.position_id not in executed
                and positions[candidate.position_id].status == "pending"
                and set(positions[candidate.position_id].access).issubset(
                    resolved
                )
            ),
            None,
        )


def _executor_evidence_coverage(
    *,
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Conservatively attest that sampled rationale/topology has event support."""
    normalized_reason = " ".join(action.reason.casefold().split())
    reason_words = _tokens(normalized_reason)
    context_words = (
        _tokens(scenario.artifact_label)
        | _tokens(scenario.check_label)
        | _tokens(scenario.defect_label)
    )
    grounding_words = {
        action.action,
        "active",
        "artifact",
        "budget",
        "check",
        "deliverable",
        "evidence",
        "failed",
        "handoff",
        "independent",
        "owner",
        "pending",
        "progress",
        "repair",
        "replan",
        "stalled",
        "verification",
        "verify",
        "workflow",
    }
    reason_has_grounding = bool(
        reason_words & (grounding_words | context_words)
    )
    unsupported_claims: list[str] = []
    if _has_negated_selected_reason_work(
        scenario=scenario,
        action=action,
        value=normalized_reason,
    ):
        unsupported_claims.append("negated_required_work")
    terminal_claim_patterns = (
        r"\ball checks (?:pass|passed)\b",
        r"\bverification (?:passes|passed|is complete)\b",
        r"\btask (?:is )?(?:complete|done|finished)\b",
        r"\bfully verified\b",
    )
    terminal_passed = bool(
        events
        and events[-1].get("kind") == "terminal_verdict"
        and events[-1].get("passed") is True
    )
    has_terminal_claim = any(
        re.search(pattern, normalized_reason)
        for pattern in terminal_claim_patterns
    )
    if has_terminal_claim and (
        not terminal_passed
        or any(
            marker in normalized_reason
            for marker in (" already ", " before ", " currently ")
        )
    ):
        unsupported_claims.append("unsupported_terminal_success_claim")

    step_coverage: list[dict[str, Any]] = []
    for index, step in enumerate(action.steps):
        operation = _classify_replan_step(scenario, step)
        source = f"sampled_replan_step:{index}"
        source_events = [item for item in events if item.get("source") == source]
        step_coverage.append(
            {
                "step_index": index,
                "operation": operation,
                "event_count": len(source_events),
                "supported": operation is not None and bool(source_events),
            }
        )
    steps_supported = all(item["supported"] for item in step_coverage)
    fully_supported = (
        reason_has_grounding
        and not unsupported_claims
        and steps_supported
    )
    return {
        "reason_supported": reason_has_grounding and not unsupported_claims,
        "reason_has_grounding": reason_has_grounding,
        "unsupported_claims": unsupported_claims,
        "step_coverage": step_coverage,
        "all_sampled_text_supported": fully_supported,
        "basis": "emitted_executor_events_and_observed_live_state",
    }


def _finish_evaluation(
    *,
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
    events: list[dict[str, Any]],
    world: _World,
) -> BranchpointEvaluation:
    satisfied, missing = _satisfaction_from_events(
        scenario=scenario,
        events=events,
        world=world,
    )
    passed = not missing
    events.append(
        _event(
            kind="terminal_verdict",
            source=FIXED_CONTINUATION_REVISION,
            passed=passed,
            satisfied_obligations=satisfied,
            missing_obligations=missing,
            budget_remaining=world.budget_remaining,
        )
    )
    terminal_event = events[-1]
    reward = 1.0 if terminal_event["passed"] is True else 0.0
    coverage = _executor_evidence_coverage(
        scenario=scenario,
        action=action,
        events=events,
    )
    evidence = {
        "required_obligations": list(scenario.required_obligations),
        "satisfied_obligations": list(terminal_event["satisfied_obligations"]),
        "missing_obligations": list(terminal_event["missing_obligations"]),
        "terminal_passed": terminal_event["passed"],
        "terminal_event_index": len(events) - 1,
        "event_count": len(events),
        "final_artifact_status": world.artifact_status,
        "final_material_revision": world.material_revision,
        "verified_revision": world.verified_revision,
        "budget_remaining": world.budget_remaining,
        "executor_evidence_coverage": coverage,
    }
    if reward == 1.0 and not coverage["all_sampled_text_supported"]:
        evidence["exclusion_reason"] = "sampled_text_not_executor_grounded"
        return BranchpointEvaluation(
            disposition="unmodeled",
            reward=None,
            outcome="unmodeled:sampled_text_not_executor_grounded",
            events=tuple(events),
            evidence=evidence,
        )
    return BranchpointEvaluation(
        disposition="eligible",
        reward=reward,
        outcome=(
            "fixed_continuation_verified"
            if reward == 1.0
            else "fixed_continuation_failed"
        ),
        events=tuple(events),
        evidence=evidence,
    )


def _unmodeled(
    *,
    scenario: SyntheticBranchpointScenario,
    reason: str,
) -> BranchpointEvaluation:
    events, world = _initial_events(scenario)
    events.append(
        _event(
            kind="simulation_unmodeled",
            source=FIXED_CONTINUATION_REVISION,
            reason=reason,
        )
    )
    return BranchpointEvaluation(
        disposition="unmodeled",
        reward=None,
        outcome=f"unmodeled:{reason}",
        events=tuple(events),
        evidence={
            "required_obligations": list(scenario.required_obligations),
            "event_count": len(events),
            "budget_remaining": world.budget_remaining,
            "exclusion_reason": reason,
        },
    )


def _dependency_closure(
    steps: tuple[ControlStep, ...],
    index: int,
) -> set[int]:
    closure: set[int] = set()
    pending = list(steps[index].access)
    while pending:
        dependency = pending.pop()
        if dependency in closure:
            continue
        closure.add(dependency)
        pending.extend(steps[dependency].access)
    return closure


def _evaluate_replan(
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
) -> BranchpointEvaluation:
    classified: list[OperationKind] = []
    for index, step in enumerate(action.steps):
        operation = _classify_replan_step(scenario, step)
        if operation is None:
            return _unmodeled(
                scenario=scenario,
                reason=f"ambiguous_replan_step:{index}",
            )
        unsupported_clauses = _unsupported_replan_clauses(
            scenario=scenario,
            step=step,
            operation=operation,
        )
        if unsupported_clauses:
            clause_indexes = ",".join(str(item) for item in unsupported_clauses)
            return _unmodeled(
                scenario=scenario,
                reason=(
                    f"unsupported_replan_step_clauses:{index}:"
                    f"{clause_indexes}"
                ),
            )
        if not _replan_step_target_grounded(
            scenario=scenario,
            step=step,
            operation=operation,
        ):
            return _unmodeled(
                scenario=scenario,
                reason=f"ungrounded_replan_step_target:{index}",
            )
        classified.append(operation)

    events, world = _initial_events(scenario)
    latest_material_step: int | None = None
    initial_material_ready = scenario.initial_artifact_status in {
        "ready",
        "defective",
    }
    for index, (step, operation) in enumerate(
        zip(action.steps, classified, strict=True)
    ):
        material_visible = True
        if operation == "verify" and latest_material_step is not None:
            material_visible = (
                latest_material_step in _dependency_closure(action.steps, index)
            )
        elif operation == "verify" and not initial_material_ready:
            material_visible = False
        completed = _execute_operation(
            scenario=scenario,
            operation=operation,
            worker_id=step.worker_id,
            source=f"sampled_replan_step:{index}",
            world=world,
            events=events,
            material_visible=material_visible,
        )
        if not completed:
            break
        if operation in {"produce", "repair"}:
            latest_material_step = index
            initial_material_ready = True
    return _finish_evaluation(
        scenario=scenario,
        action=action,
        events=events,
        world=world,
    )


def evaluate_synthetic_branchpoint_action(
    scenario: SyntheticBranchpointScenario,
    action: ControlAction,
) -> BranchpointEvaluation:
    """Execute one legal action and derive reward from emitted terminal events."""
    scenario.validate()
    try:
        validate_control_action(action, scenario.state)
    except ControlContractError as exc:
        raise SyntheticBranchpointError(
            "evaluate_synthetic_branchpoint_action requires a parsed legal action"
        ) from exc

    if action.action == "replan":
        return _evaluate_replan(scenario, action)

    events, world = _initial_events(scenario)
    if action.action == "complete":
        events.append(
            _event(
                kind="completion_attempt",
                source="sampled_action",
                evidence_only=True,
            )
        )
        return _finish_evaluation(
            scenario=scenario,
            action=action,
            events=events,
            world=world,
        )

    _execute_fixed_continuation(
        scenario=scenario,
        action=action,
        world=world,
        events=events,
    )
    return _finish_evaluation(
        scenario=scenario,
        action=action,
        events=events,
        world=world,
    )


def _protocol_evaluation(
    *,
    outcome: str,
) -> BranchpointEvaluation:
    return BranchpointEvaluation(
        disposition="protocol_only",
        reward=None,
        outcome=outcome,
        events=(),
        evidence={
            "exclusion_reason": outcome,
            "semantic_reward_assigned": False,
        },
    )


async def sample_synthetic_branchpoint(
    controller: BranchpointController,
    scenario: SyntheticBranchpointScenario,
    *,
    policy: BranchpointPolicyAttestation,
) -> SyntheticBranchpointSample:
    """Sample exactly one action with a fresh controller and one unique seed."""
    scenario.validate()
    policy.validate()
    if (
        controller.supplies_topology is not True
        or controller.capability_refs is not True
    ):
        raise SyntheticBranchpointError(
            "branchpoint training requires anonymous capability topology actions"
        )
    controller.reset_traces()
    action: ControlAction | None = None
    try:
        action = await controller.decide(scenario.state)
    except ControlContractError as exc:
        if len(controller.decision_traces) != 1:
            # Prompt sizing, token-evidence, and local-service contract failures
            # are infrastructure/configuration failures, not policy examples.
            raise
        trace = controller.decision_traces[0]
        _validate_exact_branchpoint_trace(
            trace,
            sampling_seed=policy.sampling_seed,
        )
        if trace.get("finish_reason") == "length":
            evaluation = _protocol_evaluation(
                outcome="protocol_only:length_truncated",
            )
        else:
            evaluation = _protocol_evaluation(
                outcome=f"protocol_only:{type(exc).__name__}",
            )
    else:
        if len(controller.decision_traces) != 1:
            raise SyntheticBranchpointError(
                "a branchpoint must produce exactly one policy trace"
            )
        trace = controller.decision_traces[0]
        _validate_exact_branchpoint_trace(
            trace,
            sampling_seed=policy.sampling_seed,
        )
        if trace.get("finish_reason") == "length":
            # A syntactically complete prefix is still an incomplete policy
            # sample.  Never serialize it as an executable action.
            action = None
            evaluation = _protocol_evaluation(
                outcome="protocol_only:length_truncated",
            )
        else:
            try:
                validate_control_action(action, scenario.state)
            except ControlContractError:
                evaluation = _protocol_evaluation(
                    outcome="protocol_only:invalid_control_action",
                )
            else:
                evaluation = evaluate_synthetic_branchpoint_action(
                    scenario,
                    action,
                )

    return SyntheticBranchpointSample(
        scenario_id=scenario.scenario_id,
        motif=scenario.motif,
        action=action,
        disposition=evaluation.disposition,
        reward=evaluation.reward,
        outcome=evaluation.outcome,
        events=evaluation.events,
        evidence=evaluation.evidence,
        trace=trace,
        policy=policy,
    )


def _validate_exact_branchpoint_trace(
    trace: dict[str, Any],
    *,
    sampling_seed: int,
) -> None:
    if not isinstance(trace, dict):
        raise SyntheticBranchpointError("controller trace is not an object")
    prompt_ids = trace.get("prompt_token_ids")
    completion_ids = trace.get("completion_token_ids")
    logprobs = trace.get("completion_logprobs")
    messages = trace.get("messages")
    response = trace.get("response")

    def valid_ids(values: Any) -> bool:
        return (
            isinstance(values, list)
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
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
        or not has_full_vocabulary_behavior_likelihood_contract(
            trace.get("behavior_likelihood_contract")
        )
    ):
        raise SyntheticBranchpointError(
            "controller trace lacks exact full-vocabulary behavior evidence"
        )
    if len(prompt_ids) + len(completion_ids) > MAX_BRANCHPOINT_SEQUENCE_TOKENS:
        raise SyntheticBranchpointError(
            "controller trace exceeds the proven optimizer window"
        )


@dataclass(frozen=True)
class _Context:
    artifact: str
    path: str
    check: str
    defect: str


_CONTEXTS = (
    _Context(
        "JSON dependency inventory",
        "/work/output/dependencies.json",
        "schema and duplicate-entry checks",
        "duplicate package records",
    ),
    _Context(
        "command-line conversion utility",
        "/work/output/convert_records",
        "round-trip conversion suite",
        "corrupted quoted values",
    ),
    _Context(
        "service configuration bundle",
        "/work/output/service-config",
        "offline syntax and smoke checks",
        "omitted environment key",
    ),
    _Context(
        "tabular reconciliation report",
        "/work/output/reconciliation.csv",
        "row-total and key-consistency checks",
        "dropped unmatched records",
    ),
    _Context(
        "documentation link index",
        "/work/output/link-index.json",
        "reference and anchor checks",
        "incorrect relative anchors",
    ),
    _Context(
        "archive extraction helper",
        "/work/output/safe-extract",
        "path-safety and fixture checks",
        "unsafe nested target path",
    ),
    _Context(
        "database migration script",
        "/work/output/migrate.sql",
        "idempotence and rollback checks",
        "duplicate rows on a second execution",
    ),
    _Context(
        "structured incident timeline",
        "/work/output/timeline.json",
        "ordering and source-consistency checks",
        "events sorted in the wrong timezone",
    ),
)

_DEFAULT_PROFILE_CAPABILITIES = (
    ("reasoner", "verifier", "debugger"),
    ("scientist", "planner", "aggregator"),
    ("mathematician", "coder", "reasoner"),
    ("drafter", "implementer", "fast_pass"),
)


def _validated_profiles(
    profile_capabilities: tuple[tuple[str, ...], ...],
) -> tuple[WorkerProfile, ...]:
    if not isinstance(profile_capabilities, tuple) or not profile_capabilities:
        raise SyntheticBranchpointError(
            "profile_capabilities must be a non-empty tuple"
        )
    workers: list[WorkerProfile] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw_tags in enumerate(profile_capabilities):
        if not isinstance(raw_tags, tuple) or not raw_tags:
            raise SyntheticBranchpointError(
                f"profile_capabilities[{index}] must be non-empty"
            )
        tags = tuple(
            sorted({" ".join(tag.casefold().split()) for tag in raw_tags})
        )
        if (
            len(tags) != len(raw_tags)
            or any(not tag for tag in tags)
            or tags in seen
        ):
            raise SyntheticBranchpointError(
                "anonymous capability profiles must be non-empty and unique"
            )
        seen.add(tags)
        workers.append(
            WorkerProfile(
                worker_id=index,
                capability_tags=tags,
                tool_tags=("filesystem", "terminal", "test_runner"),
            )
        )
    available = {tag for worker in workers for tag in worker.capability_tags}
    if not available & _IMPLEMENT_CAPABILITIES:
        raise SyntheticBranchpointError(
            "anonymous pool has no implementation capability"
        )
    if not available & _VERIFY_CAPABILITIES:
        raise SyntheticBranchpointError(
            "anonymous pool has no independent verification capability"
        )
    return tuple(workers)


def _worker_with(
    workers: tuple[WorkerProfile, ...],
    capabilities: frozenset[str],
    *,
    exclude: frozenset[int] = frozenset(),
) -> int:
    for worker in workers:
        if (
            worker.worker_id not in exclude
            and capabilities & set(worker.capability_tags)
        ):
            return worker.worker_id
    raise SyntheticBranchpointError(
        "anonymous pool lacks a distinct required capability profile"
    )


def _budget(used: int, limit: int) -> ControlBudget:
    return ControlBudget(
        paid_calls_used=used,
        paid_call_limit=limit,
        elapsed_s=float(used * 29),
        wall_time_limit_s=7_200.0,
    )


def _state(
    *,
    task: str,
    workers: tuple[WorkerProfile, ...],
    positions: tuple[ControlPosition, ...],
    active_position_id: int,
    calls_used: int,
    call_limit: int,
    memory: tuple[dict[str, Any], ...],
) -> LiveControlState:
    return LiveControlState(
        original_task=task,
        workers=workers,
        workflow_id=10_000 + active_position_id,
        positions=positions,
        active_position_id=active_position_id,
        terminal_status="ready",
        terminal_observation="The shared terminal is stable.",
        shared_memory=memory,
        budget=_budget(calls_used, call_limit),
    )


def _position(
    *,
    position_id: int,
    worker_id: int,
    subtask: str,
    status: Literal["active", "pending"],
    access: tuple[int, ...] = (),
    completion_requested: bool = False,
    report: str = "",
    artifact: str | None = None,
    verification_failed: str | None = None,
) -> ControlPosition:
    progress: dict[str, Any] | None = None
    if status == "active":
        progress = {
            "completion_requested": completion_requested,
            "worker_report": report,
            "material_progress": {
                "latest_turn_changed_material_state": (
                    "stalled" not in report.casefold()
                )
            },
        }
        if verification_failed is not None:
            progress["verification"] = {
                "passed": False,
                "failure": verification_failed,
            }
    artifacts = (
        ({"path": artifact, "state": "observed"},)
        if artifact is not None
        else ()
    )
    return ControlPosition(
        position_id=position_id,
        worker_id=worker_id,
        subtask=subtask,
        access=access,
        status=status,
        progress=progress,
        artifacts=artifacts,
    )


def _memory(context: _Context, variant: int) -> tuple[dict[str, Any], ...]:
    rows = (
        {
            "artifact_constraint": (
                f"Preserve the required output at {context.path}."
            )
        },
        {
            "verification_constraint": (
                f"The {context.check} must be independently rerun."
            )
        },
        {
            "environment_constraint": (
                "All required inputs are already in the shared workspace."
            )
        },
    )
    return (rows[variant % len(rows)],)


def _counterbalanced_pending_ids(
    *,
    base: int,
    variant: int,
) -> tuple[int, int]:
    lower, higher = base + 2, base + 5
    return (lower, higher) if variant % 2 == 0 else (higher, lower)


def _counterbalanced_positions(
    active: ControlPosition,
    first: ControlPosition,
    second: ControlPosition,
    *,
    variant: int,
) -> tuple[ControlPosition, ...]:
    if (variant // 2) % 2 == 0:
        return (active, first, second)
    return (active, second, first)


def _counterbalanced_fixed_steps(
    first: BranchpointFixedStep,
    second: BranchpointFixedStep,
    *,
    variant: int,
    prefix: tuple[BranchpointFixedStep, ...] = (),
) -> tuple[BranchpointFixedStep, ...]:
    ordered = (
        (first, second)
        if (variant // 4) % 2 == 0
        else (second, first)
    )
    return (*prefix, *ordered)


def _build_scenario(
    *,
    index: int,
    seed: int,
    workers: tuple[WorkerProfile, ...],
) -> SyntheticBranchpointScenario:
    motif_index = index % 6
    context = _CONTEXTS[(seed + index * 3) % len(_CONTEXTS)]
    variant = (seed // len(_CONTEXTS) + index) % 7
    implementer = _worker_with(workers, _IMPLEMENT_CAPABILITIES)
    verifier = _worker_with(
        workers,
        _VERIFY_CAPABILITIES,
        exclude=frozenset({implementer}),
    )
    inspector = _worker_with(
        workers,
        _INSPECT_CAPABILITIES,
        exclude=frozenset({implementer, verifier}),
    )
    base = 100 + index * 10 + seed % 7
    active_id = base
    memory = _memory(context, variant)

    if motif_index == 0:
        recovery_id, verify_id = _counterbalanced_pending_ids(
            base=base,
            variant=variant,
        )
        task = (
            f"Return a ready {context.artifact} at {context.path} with an "
            f"independent passing {context.check} result. Case "
            f"{variant + 1}."
        )
        active = _position(
            position_id=active_id,
            worker_id=inspector,
            subtask=f"Assess the current {context.artifact} candidate.",
            status="active",
            completion_requested=False,
            report=(
                "The assigned investigation stalled without material change. "
                "A candidate exists, but readiness is not established."
            ),
            artifact=context.path,
        )
        recovery = _position(
            position_id=recovery_id,
            worker_id=implementer,
            subtask=f"Finish the {context.artifact} candidate if incomplete.",
            status="pending",
            access=(active_id,),
        )
        verify = _position(
            position_id=verify_id,
            worker_id=verifier,
            subtask=(
                f"Independently establish whether the current candidate "
                f"passes the {context.check}."
            ),
            status="pending",
            access=(active_id,),
        )
        state = _state(
            task=task,
            workers=workers,
            positions=_counterbalanced_positions(
                active,
                recovery,
                verify,
                variant=variant,
            ),
            active_position_id=active_id,
            calls_used=11 + variant,
            call_limit=13 + variant,
            memory=memory,
        )
        return SyntheticBranchpointScenario(
            scenario_id=f"branchpoint-stalled-owner-{seed:08d}-{index:03d}",
            motif="stalled_owner_near_exhaustion",
            evidence_basis=(
                "stalled owners consumed remaining calls without material change",
                "recovering ownership before independent verification was feasible",
            ),
            state=state,
            required_obligations=("produce", "verify"),
            initial_artifact_status="partial",
            initial_material_worker_id=inspector,
            initial_verification_failed=False,
            continuation_budget=2,
            fixed_steps=(
                BranchpointFixedStep(active_id, "stall"),
                BranchpointFixedStep(recovery_id, "produce"),
                BranchpointFixedStep(verify_id, "verify"),
            ),
            artifact_label=context.artifact,
            check_label=context.check,
            defect_label=context.defect,
        )

    if motif_index == 1:
        verify_id, inspect_id = _counterbalanced_pending_ids(
            base=base,
            variant=variant,
        )
        task = (
            f"Return the completed {context.artifact} only after a separate "
            f"profile reruns the {context.check}. Verification case "
            f"{variant + 1}."
        )
        owner = _position(
            position_id=active_id,
            worker_id=implementer,
            subtask=f"Create the {context.artifact}.",
            status="active",
            completion_requested=True,
            report="The artifact is complete; independent verification is pending.",
            artifact=context.path,
        )
        verify = _position(
            position_id=verify_id,
            worker_id=verifier,
            subtask=f"Independently rerun the {context.check}.",
            status="pending",
            access=(active_id,),
        )
        inspect = _position(
            position_id=inspect_id,
            worker_id=inspector,
            subtask=f"Inspect the available {context.artifact} evidence.",
            status="pending",
            access=(active_id,),
        )
        state = _state(
            task=task,
            workers=workers,
            positions=_counterbalanced_positions(
                owner,
                verify,
                inspect,
                variant=variant,
            ),
            active_position_id=active_id,
            calls_used=9 + variant,
            call_limit=10 + variant,
            memory=memory,
        )
        return SyntheticBranchpointScenario(
            scenario_id=f"branchpoint-unverified-{seed:08d}-{index:03d}",
            motif="unverified_completion_pending_verifier",
            evidence_basis=(
                "artifact existence was not independent completion evidence",
                "pending verifier handoff could establish terminal evidence",
            ),
            state=state,
            required_obligations=("verify",),
            initial_artifact_status="ready",
            initial_material_worker_id=implementer,
            initial_verification_failed=False,
            continuation_budget=1,
            fixed_steps=_counterbalanced_fixed_steps(
                BranchpointFixedStep(verify_id, "verify"),
                BranchpointFixedStep(inspect_id, "inspect"),
                variant=variant,
            ),
            artifact_label=context.artifact,
            check_label=context.check,
            defect_label=context.defect,
        )

    if motif_index == 2:
        repair_id, reverify_id = _counterbalanced_pending_ids(
            base=base,
            variant=variant,
        )
        task = (
            f"Return the current {context.artifact} revision at {context.path} "
            f"only if it independently passes the {context.check}. Case "
            f"{variant + 1}."
        )
        failed = _position(
            position_id=active_id,
            worker_id=verifier,
            subtask=f"Verify the {context.artifact}.",
            status="active",
            completion_requested=True,
            report=(
                f"Revision 0 failed the {context.check}: {context.defect}. "
                "A repair attempt was reported but stalled; no material-change "
                "event or post-repair verification evidence was recorded."
            ),
            artifact=context.path,
            verification_failed=context.defect,
        )
        repair = _position(
            position_id=repair_id,
            worker_id=implementer,
            subtask=(
                f"Ensure {context.defect} is corrected in the "
                f"{context.artifact}."
            ),
            status="pending",
            access=(active_id,),
        )
        reverify = _position(
            position_id=reverify_id,
            worker_id=verifier,
            subtask=(
                f"Independently establish the current revision's "
                f"{context.check} result."
            ),
            status="pending",
            access=(active_id,),
        )
        state = _state(
            task=task,
            workers=workers,
            positions=_counterbalanced_positions(
                failed,
                repair,
                reverify,
                variant=variant,
            ),
            active_position_id=active_id,
            calls_used=5 + variant,
            call_limit=7 + variant,
            memory=memory,
        )
        return SyntheticBranchpointScenario(
            scenario_id=f"branchpoint-failed-check-{seed:08d}-{index:03d}",
            motif="failed_verification_repair_and_reverify",
            evidence_basis=(
                "failed verification must prevent completion",
                "repair followed by independent re-verification was required",
            ),
            state=state,
            required_obligations=("repair", "verify"),
            initial_artifact_status="defective",
            initial_material_worker_id=implementer,
            initial_verification_failed=True,
            continuation_budget=2,
            fixed_steps=_counterbalanced_fixed_steps(
                BranchpointFixedStep(repair_id, "repair"),
                BranchpointFixedStep(reverify_id, "verify"),
                variant=variant,
            ),
            artifact_label=context.artifact,
            check_label=context.check,
            defect_label=context.defect,
        )

    if motif_index == 3:
        task = (
            f"With only two execution calls left, create the "
            f"{context.artifact} at {context.path} and independently run the "
            f"{context.check}. Budget case {variant + 1}."
        )
        planning = _position(
            position_id=active_id,
            worker_id=inspector,
            subtask="Inspect inputs before assigning delivery.",
            status="active",
            report="No deliverable exists yet; two execution calls remain.",
        )
        state = _state(
            task=task,
            workers=workers,
            positions=(planning,),
            active_position_id=active_id,
            calls_used=18 + variant,
            call_limit=20 + variant,
            memory=memory,
        )
        return SyntheticBranchpointScenario(
            scenario_id=f"branchpoint-low-budget-{seed:08d}-{index:03d}",
            motif="low_budget_deliverable_first",
            evidence_basis=(
                "inspection-first workflows exhausted the remaining budget",
                "a deliverable-first two-step topology remained feasible",
            ),
            state=state,
            required_obligations=("produce", "verify"),
            initial_artifact_status="absent",
            initial_material_worker_id=None,
            initial_verification_failed=False,
            continuation_budget=2,
            fixed_steps=(
                BranchpointFixedStep(active_id, "inspect"),
            ),
            artifact_label=context.artifact,
            check_label=context.check,
            defect_label=context.defect,
        )

    if motif_index == 5:
        repair_owners = tuple(
            worker.worker_id
            for worker in workers
            if (
                set(worker.capability_tags) & _IMPLEMENT_CAPABILITIES
                and set(worker.capability_tags) & _REPAIR_CAPABILITIES
            )
        )
        if len(repair_owners) < 2:
            raise SyntheticBranchpointError(
                "private-loop branchpoints require two distinct anonymous "
                "repair-capable implementation profiles"
            )
        active_implementer = repair_owners[
            variant % len(repair_owners)
        ]
        loop_verifier = _worker_with(
            workers,
            _VERIFY_CAPABILITIES,
            exclude=frozenset({active_implementer}),
        )
        corrective_reviewer = _worker_with(
            workers,
            _INSPECT_CAPABILITIES,
            exclude=frozenset({active_implementer, loop_verifier}),
        )
        review_id, verify_id = _counterbalanced_pending_ids(
            base=base,
            variant=variant,
        )
        task = (
            f"Return a ready {context.artifact} at {context.path} with an "
            f"independent passing {context.check} result. Private-loop case "
            f"{variant + 1}."
        )
        active = _position(
            position_id=active_id,
            worker_id=active_implementer,
            subtask=(
                f"Repair the {context.artifact} in the private edit and test "
                "loop."
            ),
            status="active",
            completion_requested=False,
            report=(
                f"The latest turn changed material, but {context.defect} "
                "still fails the check and the private repair/test loop "
                "remains unfinished."
            ),
            artifact=context.path,
            verification_failed=context.defect,
        )
        review = _position(
            position_id=review_id,
            worker_id=corrective_reviewer,
            subtask=(
                f"Inspect the failed {context.check} evidence and frame "
                "corrective context."
            ),
            status="pending",
            access=(active_id,),
        )
        verify = _position(
            position_id=verify_id,
            worker_id=loop_verifier,
            subtask=(
                f"Independently establish whether the current candidate "
                f"passes the {context.check}."
            ),
            status="pending",
            access=(active_id,),
        )
        state = _state(
            task=task,
            workers=workers,
            positions=_counterbalanced_positions(
                active,
                review,
                verify,
                variant=variant,
            ),
            active_position_id=active_id,
            calls_used=7 + variant,
            call_limit=10 + variant,
            memory=memory,
        )
        return SyntheticBranchpointScenario(
            scenario_id=(
                f"branchpoint-private-loop-{seed:08d}-{index:03d}"
            ),
            motif="active_private_loop_continue_before_handoff",
            evidence_basis=(
                "an observed train rollout had an unfinished private "
                "repair/test loop with a known defective artifact",
                "that train rollout showed corrective review could not "
                "substitute for owner repair before independent verification",
            ),
            state=state,
            required_obligations=("repair", "verify"),
            initial_artifact_status="defective",
            initial_material_worker_id=active_implementer,
            initial_verification_failed=True,
            continuation_budget=3,
            fixed_steps=(
                BranchpointFixedStep(active_id, "repair"),
                BranchpointFixedStep(review_id, "inspect"),
                BranchpointFixedStep(verify_id, "verify"),
            ),
            artifact_label=context.artifact,
            check_label=context.check,
            defect_label=context.defect,
        )

    finisher_id, verify_id = _counterbalanced_pending_ids(
        base=base,
        variant=variant,
    )
    task = (
        f"Return a ready {context.artifact} at {context.path} with an "
        f"independent passing {context.check} result. Case {variant + 1}."
    )
    owner = _position(
        position_id=active_id,
        worker_id=implementer,
        subtask=f"Develop the {context.artifact} candidate.",
        status="active",
        completion_requested=True,
        report=(
            "The latest turn changed material and emitted a candidate; "
            "readiness is unconfirmed."
        ),
        artifact=context.path,
    )
    finisher = _position(
        position_id=finisher_id,
        worker_id=implementer,
        subtask=f"Finish the {context.artifact} candidate if incomplete.",
        status="pending",
        access=(active_id,),
    )
    verify = _position(
        position_id=verify_id,
        worker_id=verifier,
        subtask=(
            f"Independently establish whether the current candidate passes "
            f"the {context.check}."
        ),
        status="pending",
        access=(active_id,),
    )
    state = _state(
        task=task,
        workers=workers,
        positions=_counterbalanced_positions(
            owner,
            finisher,
            verify,
            variant=variant,
        ),
        active_position_id=active_id,
        calls_used=4 + variant,
        call_limit=6 + variant,
        memory=memory,
    )
    return SyntheticBranchpointScenario(
        scenario_id=f"branchpoint-candidate-finisher-{seed:08d}-{index:03d}",
        motif="candidate_finisher_before_verifier",
        evidence_basis=(
            "candidate emission did not establish deliverable readiness",
            "finishing material before independent verification was feasible",
        ),
        state=state,
        required_obligations=("produce", "verify"),
        initial_artifact_status="partial",
        initial_material_worker_id=implementer,
        initial_verification_failed=False,
        continuation_budget=2,
        fixed_steps=_counterbalanced_fixed_steps(
            BranchpointFixedStep(finisher_id, "produce"),
            BranchpointFixedStep(verify_id, "verify"),
            variant=variant,
        ),
        artifact_label=context.artifact,
        check_label=context.check,
        defect_label=context.defect,
    )


def build_synthetic_branchpoint_curriculum(
    *,
    count: int,
    seed: int,
    profile_capabilities: tuple[tuple[str, ...], ...] = (
        _DEFAULT_PROFILE_CAPABILITIES
    ),
) -> tuple[SyntheticBranchpointScenario, ...]:
    """Build varied deterministic branchpoints against anonymous role priors."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise SyntheticBranchpointError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SyntheticBranchpointError("seed must be an integer")
    workers = _validated_profiles(profile_capabilities)
    scenarios = tuple(
        _build_scenario(index=index, seed=seed, workers=workers)
        for index in range(count)
    )
    for scenario in scenarios:
        scenario.validate()
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise SyntheticBranchpointError("scenario IDs are not unique")
    return scenarios


__all__ = [
    "BRANCHPOINT_CURRICULUM_REVISION",
    "FIXED_CONTINUATION_MODE",
    "FIXED_CONTINUATION_REVISION",
    "BranchpointController",
    "BranchpointEvaluation",
    "BranchpointPolicyAttestation",
    "SyntheticBranchpointError",
    "SyntheticBranchpointSample",
    "SyntheticBranchpointScenario",
    "build_synthetic_branchpoint_curriculum",
    "evaluate_synthetic_branchpoint_action",
    "sample_synthetic_branchpoint",
]
