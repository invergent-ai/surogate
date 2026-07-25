"""Fail-closed corpus and promotion gates for causal coordination learning."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


MIN_TRAIN_PAIRS = 6
MIN_HOLDOUT_PAIRS = 2
MIN_TRAIN_MECHANISMS = 3
MIN_HOLDOUT_MECHANISMS = 2
MIN_REPLAY_TRAIN = 40
MIN_REPLAY_VALIDATION = 8
PERMUTATION_STRATEGY = "cyclic_and_reflected"


class CausalCorpusError(ValueError):
    """Causal evidence is not safe to use for conductor learning."""


@dataclass(frozen=True)
class AnonymousProfile:
    worker_id: int
    capability_tags: tuple[str, ...]
    tool_tags: tuple[str, ...]


@dataclass(frozen=True)
class CausalPreferencePair:
    task_id: str
    mechanism_id: str
    split: Literal["train", "holdout"]
    instruction: str
    rejected_action: dict[str, Any]
    preferred_action: dict[str, Any]
    pool_fingerprint: str
    admission_path: str
    admission_sha256: str


@dataclass(frozen=True)
class CorpusReadiness:
    ready: bool
    reasons: tuple[str, ...]
    train_pairs: int
    holdout_pairs: int
    train_mechanisms: int
    holdout_mechanisms: int
    replay_train: int
    replay_validation: int
    permutation_rows_if_ready: int


@dataclass(frozen=True)
class PromotionMetrics:
    train_pairs: int
    holdout_pairs: int
    replay_guardians: int
    permutation_cases: int
    parent_train_correct: int
    candidate_train_correct: int
    parent_holdout_correct: int
    candidate_holdout_correct: int
    candidate_replay_preserved: int
    candidate_permutation_consistent: int
    candidate_schema_valid: int
    candidate_identity_leaks: int


@dataclass(frozen=True)
class PromotionDecision:
    promotable: bool
    reasons: tuple[str, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CausalCorpusError(f"{label} is missing")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CausalCorpusError(f"{label} line {number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise CausalCorpusError(f"{label} line {number} is not an object")
        rows.append(value)
    return rows


def _validate_profiles(profiles: Sequence[AnonymousProfile]) -> None:
    if len(profiles) < 2:
        raise CausalCorpusError("at least two anonymous profiles are required")
    worker_ids = [profile.worker_id for profile in profiles]
    if sorted(worker_ids) != list(range(len(profiles))):
        raise CausalCorpusError("anonymous worker IDs must be contiguous from zero")
    for profile in profiles:
        if (
            not profile.capability_tags
            or not profile.tool_tags
            or any(not value.strip() for value in profile.capability_tags)
            or any(not value.strip() for value in profile.tool_tags)
        ):
            raise CausalCorpusError("anonymous profiles require capability and tool tags")


def _validate_action(
    action: dict[str, Any], *, worker_ids: set[int], label: str
) -> None:
    if set(action) != {"action", "reason", "steps"}:
        raise CausalCorpusError(f"{label} action schema drift")
    if action["action"] != "replan" or not isinstance(action["reason"], str):
        raise CausalCorpusError(f"{label} must be a reasoned replan")
    steps = action["steps"]
    if not isinstance(steps, list) or not steps:
        raise CausalCorpusError(f"{label} has no workflow steps")
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != {"worker_id", "subtask", "access"}:
            raise CausalCorpusError(f"{label} step schema drift")
        if (
            isinstance(step["worker_id"], bool)
            or step["worker_id"] not in worker_ids
            or not isinstance(step["subtask"], str)
            or not step["subtask"].strip()
            or not isinstance(step["access"], list)
            or any(
                isinstance(parent, bool)
                or not isinstance(parent, int)
                or not 0 <= parent < index
                for parent in step["access"]
            )
            or len(step["access"]) != len(set(step["access"]))
        ):
            raise CausalCorpusError(f"{label} step is invalid")


def _surface_has_forbidden_identity(value: Any, forbidden: set[str]) -> bool:
    surface = json.dumps(value, sort_keys=True, ensure_ascii=True).lower()
    return any(token and token.lower() in surface for token in forbidden)


def _validate_pair(
    pair: CausalPreferencePair,
    *,
    root: Path,
    worker_ids: set[int],
    pool_fingerprint: str,
    forbidden_identities: set[str],
) -> None:
    if (
        not pair.task_id.strip()
        or not pair.mechanism_id.strip()
        or pair.split not in {"train", "holdout"}
        or not pair.instruction.strip()
        or pair.pool_fingerprint != pool_fingerprint
    ):
        raise CausalCorpusError(f"invalid causal pair metadata: {pair.task_id}")
    admission = (root / pair.admission_path).resolve()
    try:
        admission.relative_to(root.resolve())
    except ValueError as exc:
        raise CausalCorpusError("causal admission escapes the project root") from exc
    if not admission.is_file() or sha256(admission) != pair.admission_sha256:
        raise CausalCorpusError(f"causal admission hash drift: {pair.task_id}")
    _validate_action(pair.rejected_action, worker_ids=worker_ids, label="rejected")
    _validate_action(pair.preferred_action, worker_ids=worker_ids, label="preferred")
    if pair.rejected_action == pair.preferred_action:
        raise CausalCorpusError(f"causal actions are identical: {pair.task_id}")
    learned_surface = {
        "instruction": pair.instruction,
        "rejected": pair.rejected_action,
        "preferred": pair.preferred_action,
    }
    if _surface_has_forbidden_identity(learned_surface, forbidden_identities):
        raise CausalCorpusError(f"model identity leaked into pair: {pair.task_id}")


def _permutations(worker_count: int) -> tuple[tuple[int, ...], ...]:
    base = tuple(range(worker_count))
    reflected = tuple(reversed(base))
    result: list[tuple[int, ...]] = []
    for seed in (base, reflected):
        for shift in range(worker_count):
            permutation = seed[shift:] + seed[:shift]
            if permutation not in result:
                result.append(permutation)
    return tuple(result)


def _remap_action(action: dict[str, Any], old_to_new: dict[int, int]) -> dict[str, Any]:
    return {
        "action": action["action"],
        "reason": action["reason"],
        "steps": [
            {
                "worker_id": old_to_new[int(step["worker_id"])],
                "subtask": step["subtask"],
                "access": list(step["access"]),
            }
            for step in action["steps"]
        ],
    }


def _remap_profiles(
    profiles: Sequence[AnonymousProfile], old_to_new: dict[int, int]
) -> list[dict[str, Any]]:
    remapped = [
        {
            "worker_id": old_to_new[profile.worker_id],
            "capability_tags": list(profile.capability_tags),
            "tool_tags": list(profile.tool_tags),
        }
        for profile in profiles
    ]
    return sorted(remapped, key=lambda value: value["worker_id"])


def _replay_rows(
    path: Path, *, label: str, forbidden_identities: set[str]
) -> list[dict[str, Any]]:
    rows = _load_jsonl(path, label=label)
    record_ids: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise CausalCorpusError(f"{label} record identity drift")
        record_ids.add(record_id)
        if _surface_has_forbidden_identity(row.get("messages"), forbidden_identities):
            raise CausalCorpusError(f"model identity leaked into {label}")
    return rows


def _frozen_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise CausalCorpusError("frozen causal task split is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CausalCorpusError("frozen causal task split is invalid JSON") from exc
    if not isinstance(payload, dict) or not str(payload.get("version", "")).startswith(
        "fugu_causal_task_split_"
    ):
        raise CausalCorpusError("frozen causal task split version drift")
    policy = payload.get("policy") or {}
    required_policy = {
        "logical_task_isolation": True,
        "holdout_outcomes_enter_training": False,
        "holdout_prompts_enter_training": False,
        "future_tasks_require_new_split_version": True,
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise CausalCorpusError("frozen causal task split policy drift")
    assignments: dict[str, str] = {}
    for split in ("train", "holdout"):
        rows = payload.get(split)
        if not isinstance(rows, list):
            raise CausalCorpusError("frozen causal task split rows drift")
        for row in rows:
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(task_id, str) or not task_id or task_id in assignments:
                raise CausalCorpusError("frozen causal task assignment drift")
            assignments[task_id] = split
    return assignments


def assess_corpus(
    *,
    root: Path,
    pairs: Sequence[CausalPreferencePair],
    profiles: Sequence[AnonymousProfile],
    pool_fingerprint: str,
    forbidden_identities: set[str],
    frozen_split_path: Path,
    replay_train_path: Path,
    replay_validation_path: Path,
) -> CorpusReadiness:
    """Validate all evidence and report whether training may be materialized."""
    root = root.resolve()
    _validate_profiles(profiles)
    assignments = _frozen_splits(frozen_split_path)
    worker_ids = {profile.worker_id for profile in profiles}
    seen_tasks: set[str] = set()
    seen_admissions: set[tuple[str, str]] = set()
    for pair in pairs:
        _validate_pair(
            pair,
            root=root,
            worker_ids=worker_ids,
            pool_fingerprint=pool_fingerprint,
            forbidden_identities=forbidden_identities,
        )
        if assignments.get(pair.task_id) != pair.split:
            raise CausalCorpusError(
                f"causal pair violates frozen whole-task split: {pair.task_id}"
            )
        admission_identity = (pair.admission_path, pair.admission_sha256)
        if pair.task_id in seen_tasks or admission_identity in seen_admissions:
            raise CausalCorpusError("causal tasks and admissions must be unique")
        seen_tasks.add(pair.task_id)
        seen_admissions.add(admission_identity)

    replay_train = _replay_rows(
        replay_train_path,
        label="replay train",
        forbidden_identities=forbidden_identities,
    )
    replay_validation = _replay_rows(
        replay_validation_path,
        label="replay validation",
        forbidden_identities=forbidden_identities,
    )
    replay_tasks = {
        str(row.get("task_id")) for row in itertools.chain(replay_train, replay_validation)
    }
    if seen_tasks & replay_tasks:
        raise CausalCorpusError("causal and replay task splits overlap")

    train = [pair for pair in pairs if pair.split == "train"]
    holdout = [pair for pair in pairs if pair.split == "holdout"]
    train_mechanisms = len({pair.mechanism_id for pair in train})
    holdout_mechanisms = len({pair.mechanism_id for pair in holdout})
    reasons: list[str] = []
    if len(train) < MIN_TRAIN_PAIRS:
        reasons.append(f"train_pairs:{len(train)}/{MIN_TRAIN_PAIRS}")
    if len(holdout) < MIN_HOLDOUT_PAIRS:
        reasons.append(f"holdout_pairs:{len(holdout)}/{MIN_HOLDOUT_PAIRS}")
    if train_mechanisms < MIN_TRAIN_MECHANISMS:
        reasons.append(
            f"train_mechanisms:{train_mechanisms}/{MIN_TRAIN_MECHANISMS}"
        )
    if holdout_mechanisms < MIN_HOLDOUT_MECHANISMS:
        reasons.append(
            f"holdout_mechanisms:{holdout_mechanisms}/{MIN_HOLDOUT_MECHANISMS}"
        )
    if len(replay_train) < MIN_REPLAY_TRAIN:
        reasons.append(f"replay_train:{len(replay_train)}/{MIN_REPLAY_TRAIN}")
    if len(replay_validation) < MIN_REPLAY_VALIDATION:
        reasons.append(
            f"replay_validation:{len(replay_validation)}/{MIN_REPLAY_VALIDATION}"
        )
    permutation_count = len(_permutations(len(profiles)))
    return CorpusReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        train_pairs=len(train),
        holdout_pairs=len(holdout),
        train_mechanisms=train_mechanisms,
        holdout_mechanisms=holdout_mechanisms,
        replay_train=len(replay_train),
        replay_validation=len(replay_validation),
        permutation_rows_if_ready=len(train) * permutation_count,
    )


def build_pairwise_rows(
    *,
    root: Path,
    pairs: Sequence[CausalPreferencePair],
    profiles: Sequence[AnonymousProfile],
    pool_fingerprint: str,
    forbidden_identities: set[str],
    frozen_split_path: Path,
    replay_train_path: Path,
    replay_validation_path: Path,
) -> list[dict[str, Any]]:
    """Emit preference rows only after the complete causal corpus is ready."""
    readiness = assess_corpus(
        root=root,
        pairs=pairs,
        profiles=profiles,
        pool_fingerprint=pool_fingerprint,
        forbidden_identities=forbidden_identities,
        frozen_split_path=frozen_split_path,
        replay_train_path=replay_train_path,
        replay_validation_path=replay_validation_path,
    )
    if not readiness.ready:
        raise CausalCorpusError(
            "causal training remains locked: " + ", ".join(readiness.reasons)
        )

    result: list[dict[str, Any]] = []
    permutations = _permutations(len(profiles))
    for pair in pairs:
        if pair.split != "train":
            continue
        for index, permutation in enumerate(permutations):
            old_to_new = {old: new for old, new in enumerate(permutation)}
            prompt = {
                "task": pair.instruction,
                "anonymous_worker_profiles": _remap_profiles(profiles, old_to_new),
                "decision": (
                    "Choose a capability-matched agentic workflow. Worker IDs "
                    "only identify the profiles supplied in this request."
                ),
            }
            chosen = _remap_action(pair.preferred_action, old_to_new)
            rejected = _remap_action(pair.rejected_action, old_to_new)
            learned = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            if _surface_has_forbidden_identity(learned, forbidden_identities):
                raise CausalCorpusError("identity leaked during permutation")
            result.append(
                {
                    "record_id": f"causal::{pair.task_id}::perm-{index:02d}",
                    "task_id": pair.task_id,
                    "mechanism_id": pair.mechanism_id,
                    "split": "train",
                    "pool_fingerprint": pool_fingerprint,
                    "permutation_strategy": PERMUTATION_STRATEGY,
                    "old_to_new_worker_id": {
                        str(old): new for old, new in old_to_new.items()
                    },
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "evidence": {
                        "admission_path": pair.admission_path,
                        "admission_sha256": pair.admission_sha256,
                    },
                }
            )
    if len(result) != readiness.permutation_rows_if_ready:
        raise CausalCorpusError("permutation row count drift")
    return result


def assess_promotion(metrics: PromotionMetrics) -> PromotionDecision:
    """Require task generalization, replay preservation, and slot equivariance."""
    reasons: list[str] = []
    if metrics.train_pairs < MIN_TRAIN_PAIRS or metrics.holdout_pairs < MIN_HOLDOUT_PAIRS:
        reasons.append("corpus_minimums")
    if metrics.candidate_train_correct != metrics.train_pairs:
        reasons.append("train_pair_fit")
    if (
        metrics.candidate_holdout_correct != metrics.holdout_pairs
        or metrics.candidate_holdout_correct <= metrics.parent_holdout_correct
    ):
        reasons.append("whole_task_holdout_improvement")
    if metrics.candidate_replay_preserved != metrics.replay_guardians:
        reasons.append("anti_forgetting_replay")
    if metrics.candidate_permutation_consistent != metrics.permutation_cases:
        reasons.append("slot_profile_equivariance")
    expected_schema = (
        metrics.train_pairs + metrics.holdout_pairs + metrics.permutation_cases
    )
    if metrics.candidate_schema_valid != expected_schema:
        reasons.append("action_schema")
    if metrics.candidate_identity_leaks != 0:
        reasons.append("model_identity_leak")
    return PromotionDecision(promotable=not reasons, reasons=tuple(reasons))
