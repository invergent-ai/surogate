"""Identity-free hindsight-skill contract for Fugu conductor training."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence


SEED_HINDSIGHT_CONTRACT_VERSION = "20260720-fugu-seed-hindsight-v1"
MAX_GUIDANCE_CHARS = 1_600
MIN_GUIDANCE_CHARS = 24
_SCOPES = frozenset({"topology", "continue", "handoff", "replan", "completion"})
_IDENTITY_KEYS = frozenset(
    {
        "api_base",
        "binding_path",
        "model",
        "model_id",
        "model_name",
        "provider",
        "runtime_model",
        "worker_models",
    }
)
_SLOT_REFERENCE = re.compile(r"\b(?:worker(?:_id)?|slot)\s*[:=#-]?\s*\d+\b", re.IGNORECASE)
_LEAKAGE_TERMS = (
    "gold patch",
    "hidden test",
    "oracle answer",
    "oracle solution",
    "reference answer",
    "reference patch",
    "verifier implementation",
    "verifier source",
)


class HindsightContractError(ValueError):
    """Raised when analyzer input or output violates the training contract."""


@dataclass(frozen=True)
class HindsightSkill:
    scope: Literal["topology", "continue", "handoff", "replan", "completion"]
    guidance: str
    evidence_decision_ids: tuple[int, ...]


def _walk_identity_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _IDENTITY_KEYS:
                raise HindsightContractError(f"identity-bearing key is prohibited at {path}.{key_text}")
            _walk_identity_keys(child, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_identity_keys(child, path=f"{path}[{index}]")


def _reject_terms(text: str, *, forbidden_identity_terms: Sequence[str]) -> None:
    folded = text.casefold()
    for term in (*_LEAKAGE_TERMS, *forbidden_identity_terms):
        normalized = term.strip().casefold()
        if normalized and normalized in folded:
            raise HindsightContractError(f"prohibited identity or leakage term: {term!r}")


def validate_analyzer_trajectory(
    trajectory: dict[str, Any],
    *,
    forbidden_identity_terms: Sequence[str] = (),
) -> tuple[int, ...]:
    """Validate an identity-free completed conductor trajectory for analysis."""
    _walk_identity_keys(trajectory)
    rendered = json.dumps(trajectory, ensure_ascii=True, sort_keys=True)
    _reject_terms(rendered, forbidden_identity_terms=forbidden_identity_terms)
    decisions = trajectory.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise HindsightContractError("trajectory must contain at least one conductor decision")
    ids: list[int] = []
    for row in decisions:
        if not isinstance(row, dict):
            raise HindsightContractError("every conductor decision must be an object")
        decision_id = row.get("decision_id")
        if not isinstance(decision_id, int) or isinstance(decision_id, bool) or decision_id <= 0:
            raise HindsightContractError("decision_id must be a positive integer")
        ids.append(decision_id)
    if len(ids) != len(set(ids)):
        raise HindsightContractError("decision_id values must be unique")
    outcome = trajectory.get("outcome")
    if outcome not in {"success", "failure"}:
        raise HindsightContractError("outcome must be success or failure")
    return tuple(ids)


def parse_hindsight_skill(
    text: str,
    *,
    observed_decision_ids: Sequence[int],
    forbidden_identity_terms: Sequence[str] = (),
) -> HindsightSkill:
    """Parse and validate one analyzer-produced reusable conductor skill."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HindsightContractError(f"skill is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "scope",
        "guidance",
        "evidence_decision_ids",
    }:
        raise HindsightContractError("skill must contain exactly scope, guidance, and evidence_decision_ids")
    scope = payload["scope"]
    guidance = payload["guidance"]
    evidence = payload["evidence_decision_ids"]
    if scope not in _SCOPES:
        raise HindsightContractError(f"unsupported skill scope: {scope!r}")
    if not isinstance(guidance, str):
        raise HindsightContractError("guidance must be text")
    guidance = guidance.strip()
    if not MIN_GUIDANCE_CHARS <= len(guidance) <= MAX_GUIDANCE_CHARS:
        raise HindsightContractError(
            f"guidance length must be {MIN_GUIDANCE_CHARS}..{MAX_GUIDANCE_CHARS} characters"
        )
    _reject_terms(guidance, forbidden_identity_terms=forbidden_identity_terms)
    if _SLOT_REFERENCE.search(guidance):
        raise HindsightContractError("guidance may describe capabilities and roles, not fixed worker slots")
    if not isinstance(evidence, list) or not evidence:
        raise HindsightContractError("evidence_decision_ids must be a non-empty list")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in evidence):
        raise HindsightContractError("evidence_decision_ids must contain only integers")
    observed = set(observed_decision_ids)
    if any(item not in observed for item in evidence):
        raise HindsightContractError("skill cites an unobserved conductor decision")
    if len(evidence) != len(set(evidence)):
        raise HindsightContractError("evidence_decision_ids must be unique")
    return HindsightSkill(scope=scope, guidance=guidance, evidence_decision_ids=tuple(evidence))


def render_hindsight_analysis_prompt(
    trajectory: dict[str, Any],
    *,
    forbidden_identity_terms: Sequence[str] = (),
) -> str:
    """Render an already-validated, verifier-safe trajectory for the analyzer role."""
    decision_ids = validate_analyzer_trajectory(
        trajectory,
        forbidden_identity_terms=forbidden_identity_terms,
    )
    schema = {
        "scope": sorted(_SCOPES),
        "guidance": "reusable capability/topology or live-control guidance",
        "evidence_decision_ids": list(decision_ids),
    }
    return (
        "Analyze the completed conductor trajectory using only the observations shown. "
        "Extract one reusable orchestration skill that would improve a future decision. "
        "Describe anonymous capabilities and roles, never provider/model identities or "
        "fixed worker slots. Do not infer hidden tests, oracle solutions, reference patches, "
        "or verifier internals. Cite only observed decision_id values. Return exactly one "
        "JSON object and no prose.\n\n"
        f"OUTPUT CONTRACT:\n{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        f"COMPLETED TRAJECTORY:\n{json.dumps(trajectory, ensure_ascii=True, indent=2, sort_keys=True)}"
    )


def render_training_skill_context(skill: HindsightSkill) -> str:
    """Render privileged training-only context; never included at deployment."""
    return (
        "TRAINING-ONLY HINDSIGHT. Do not quote or mention this context in the action. "
        "Use it only to reconsider the next conductor action:\n"
        f"{json.dumps(asdict(skill), ensure_ascii=True, sort_keys=True)}"
    )
