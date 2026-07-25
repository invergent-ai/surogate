"""Build a leakage-safe, model-agnostic corpus from verified agentic outcomes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CORPUS_VERSION = "fugu_outcome_trajectory_corpus_v1"
DEFAULT_SPLIT_SALT = "fugu-outcome-trajectory-corpus-v1"


class OutcomeCorpusError(ValueError):
    """An outcome cannot be represented without violating the corpus contract."""


@dataclass(frozen=True)
class CapabilitySlot:
    worker_id: int
    role_prior: tuple[str, ...]
    context_window_tokens: int | None
    concrete_model: str


@dataclass(frozen=True)
class PoolCapabilities:
    registry_key: str
    accepted_pool_ids: tuple[str, ...]
    slots: tuple[CapabilitySlot, ...]
    source_path: Path


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    sources: tuple[str, ...]
    conflict: bool


@dataclass(frozen=True)
class AuditThresholds:
    generic_train_rows: int
    generic_train_tasks: int
    generic_holdout_rows: int
    generic_holdout_tasks: int
    generic_each_label_per_split: int
    ranker_train_contrast_groups: int
    ranker_holdout_contrast_groups: int
    current_pool_train_rows: int
    current_pool_each_train_label: int
    current_pool_holdout_rows: int
    current_pool_each_holdout_label: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OutcomeCorpusError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OutcomeCorpusError(f"JSON root must be an object: {path}")
    return value


def _as_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _binary_reward(result: Mapping[str, Any]) -> float | None:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    value = rewards.get("reward") if isinstance(rewards, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    reward = float(value)
    return reward if reward in (0.0, 1.0) else None


def _result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    agent_result = result.get("agent_result")
    metadata = agent_result.get("metadata") if isinstance(agent_result, Mapping) else None
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def load_capability_registry(root: Path) -> dict[str, PoolCapabilities]:
    """Load concrete bindings for provenance, exposing only capabilities to rows."""

    definitions = (
        (
            "current_pool_v11",
            root
            / "director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json",
            ("yunwu-sol-gemini-terra-grok-v1",),
        ),
        (
            "local_pool_v1",
            root / "director/manifests/fugu_mechanics/local_pool_v1.json",
            ("fugu-mechanics-local-v1", "mechanics:fugu-mechanics-local-v1"),
        ),
        (
            "local_pool_v2",
            root / "director/manifests/fugu_mechanics/local_pool_v2_27b.json",
            ("fugu-mechanics-local-v2-27b", "mechanics:fugu-mechanics-local-v2-27b"),
        ),
        (
            "local_pool_v3",
            root / "director/manifests/fugu_mechanics/local_pool_v3_ornith35b.json",
            (
                "fugu-mechanics-local-v3-ornith35b",
                "mechanics:fugu-mechanics-local-v3-ornith35b",
            ),
        ),
    )
    registry: dict[str, PoolCapabilities] = {}
    for registry_key, path, accepted_pool_ids in definitions:
        payload = _read_json(path)
        raw_slots = payload.get("slots")
        if not isinstance(raw_slots, list) or not raw_slots:
            raise OutcomeCorpusError(f"pool config has no slots: {path}")
        slots: list[CapabilitySlot] = []
        for raw in raw_slots:
            if not isinstance(raw, Mapping):
                raise OutcomeCorpusError(f"pool slot is not an object: {path}")
            worker_id = raw.get("worker_id")
            roles = raw.get("role_prior")
            concrete_model = raw.get("runtime_model") or raw.get("served_model")
            context = raw.get("context_window_tokens")
            if (
                isinstance(worker_id, bool)
                or not isinstance(worker_id, int)
                or not isinstance(roles, list)
                or not roles
                or not all(isinstance(role, str) and role.strip() for role in roles)
                or not isinstance(concrete_model, str)
                or not concrete_model
                or (
                    context is not None
                    and (isinstance(context, bool) or not isinstance(context, int))
                )
            ):
                raise OutcomeCorpusError(f"invalid pool slot in {path}")
            slots.append(
                CapabilitySlot(
                    worker_id=worker_id,
                    role_prior=tuple(role.strip() for role in roles),
                    context_window_tokens=context,
                    concrete_model=concrete_model,
                )
            )
        slots.sort(key=lambda slot: slot.worker_id)
        if [slot.worker_id for slot in slots] != list(range(len(slots))):
            raise OutcomeCorpusError(f"pool worker IDs must be contiguous: {path}")
        binding = PoolCapabilities(
            registry_key=registry_key,
            accepted_pool_ids=accepted_pool_ids,
            slots=tuple(slots),
            source_path=path,
        )
        for pool_id in accepted_pool_ids:
            if pool_id in registry:
                raise OutcomeCorpusError(f"duplicate pool ID: {pool_id}")
            registry[pool_id] = binding
    return registry


def load_split_registry(root: Path) -> dict[str, SplitAssignment]:
    """Merge historical task splits. Any holdout designation dominates forever."""

    labels: dict[str, list[tuple[str, str]]] = defaultdict(list)

    mechanics_path = root / "scratchpad/fugu_mechanics_admission/split_v2.json"
    mechanics = _read_json(mechanics_path)
    for split, field in (("train", "train"), ("holdout", "heldout")):
        for row in mechanics.get(field) or []:
            if isinstance(row, Mapping) and isinstance(row.get("task_name"), str):
                labels[row["task_name"]].append((split, mechanics_path.as_posix()))

    capability_path = root / "scratchpad/fugu_capability_routing_v1/manifest.json"
    capability = _read_json(capability_path).get("split") or {}
    for split, field in (("train", "train_tasks"), ("holdout", "holdout_tasks")):
        for raw_name in capability.get(field) or []:
            if not isinstance(raw_name, str):
                continue
            name = raw_name.rsplit("__", 1)[-1]
            labels[name].append((split, capability_path.as_posix()))

    causal_path = root / "scratchpad/fugu_causal_corpus_v1/split_v1.json"
    causal = _read_json(causal_path)
    for split in ("train", "holdout"):
        for row in causal.get(split) or []:
            if not isinstance(row, Mapping):
                continue
            name = row.get("task_id") or row.get("task_name")
            if isinstance(name, str):
                labels[name].append((split, causal_path.as_posix()))

    adaptive_path = root / "scratchpad/fugu_adaptive_causal_pool_v2/allocation_frozen.json"
    adaptive = _read_json(adaptive_path)
    for split in ("train", "holdout"):
        for row in adaptive.get(split) or []:
            if not isinstance(row, Mapping):
                continue
            name = row.get("task_id") or row.get("task_name")
            if isinstance(name, str):
                labels[name].append((split, adaptive_path.as_posix()))

    registry: dict[str, SplitAssignment] = {}
    for task_name, rows in labels.items():
        observed = {split for split, _ in rows}
        split = "holdout" if "holdout" in observed else "train"
        registry[task_name] = SplitAssignment(
            split=split,
            sources=tuple(sorted({source for _, source in rows})),
            conflict=len(observed) > 1,
        )
    return registry


def deterministic_split(task_name: str, *, salt: str = DEFAULT_SPLIT_SALT) -> str:
    digest = hashlib.sha256(f"{salt}\0{task_name}".encode("utf-8")).digest()
    return "holdout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def assign_split(
    task_name: str,
    registry: Mapping[str, SplitAssignment],
    *,
    salt: str = DEFAULT_SPLIT_SALT,
) -> SplitAssignment:
    frozen = registry.get(task_name)
    if frozen is not None:
        return frozen
    return SplitAssignment(
        split=deterministic_split(task_name, salt=salt),
        sources=(f"deterministic_sha256_mod5:{salt}",),
        conflict=False,
    )


def load_invalid_result_paths(
    root: Path, *, report_paths: Sequence[Path] | None = None
) -> dict[Path, tuple[str, ...]]:
    invalid: dict[Path, set[str]] = defaultdict(set)
    paths = (
        sorted(report_paths)
        if report_paths is not None
        else sorted(root.glob("scratchpad/**/*pair_report.json"))
    )
    for report_path in paths:
        report = _read_json(report_path)
        for arm in report.get("arms") or []:
            if not isinstance(arm, Mapping) or arm.get("valid") is not False:
                continue
            raw_path = arm.get("result_path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            path = Path(raw_path)
            path = path if path.is_absolute() else root / path
            reasons = arm.get("integrity_errors") or [report.get("status") or "invalid_arm"]
            for reason in reasons:
                invalid[path.resolve()].add(
                    f"invalid_pair_arm:{report_path.relative_to(root).as_posix()}:{reason}"
                )
    return {path: tuple(sorted(reasons)) for path, reasons in invalid.items()}


def _profile_payload(slot: CapabilitySlot) -> dict[str, Any]:
    return {
        "role_prior": list(slot.role_prior),
        "tool_tags": ["filesystem", "terminal", "test_runner"],
        "context_window_tokens": slot.context_window_tokens,
    }


def canonicalize_initial_workflow(
    routes: Sequence[Mapping[str, Any]],
    binding: PoolCapabilities,
    *,
    registered_initial_workflow: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Remove concrete slots while preserving capability/topology semantics."""

    raw_plan: Mapping[str, Any] | None = None
    workflow_ids = sorted(
        {
            route.get("workflow_id")
            for route in routes
            if isinstance(route.get("workflow_id"), int)
            and not isinstance(route.get("workflow_id"), bool)
        }
    )
    if workflow_ids:
        first_id = workflow_ids[0]
        for route in routes:
            if route.get("workflow_id") != first_id:
                continue
            raw = route.get("raw_plan")
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise OutcomeCorpusError("initial raw_plan is invalid JSON") from exc
                if isinstance(parsed, Mapping):
                    raw_plan = parsed
                    break
    if raw_plan is None and isinstance(registered_initial_workflow, Mapping):
        raw_plan = registered_initial_workflow
    if raw_plan is None:
        raise OutcomeCorpusError("initial workflow plan is unavailable")
    raw_steps = raw_plan.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise OutcomeCorpusError("initial workflow has no steps")

    by_worker = {slot.worker_id: slot for slot in binding.slots}
    sorted_profiles = [
        (_profile_payload(slot), slot.worker_id) for slot in binding.slots
    ]
    sorted_profiles.sort(key=lambda pair: stable_json(pair[0]))
    profile_index_by_worker = {
        worker_id: index for index, (_, worker_id) in enumerate(sorted_profiles)
    }
    profiles = [profile for profile, _ in sorted_profiles]

    steps: list[dict[str, Any]] = []
    for position_id, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise OutcomeCorpusError("workflow step is not an object")
        worker_id = raw_step.get("worker_id")
        subtask = raw_step.get("subtask")
        access = raw_step.get("access")
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or worker_id not in by_worker
            or not isinstance(subtask, str)
            or not subtask.strip()
            or not isinstance(access, list)
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < position_id
                for value in access
            )
            or len(set(access)) != len(access)
        ):
            raise OutcomeCorpusError(f"invalid workflow step {position_id}")
        steps.append(
            {
                "position_id": position_id,
                "capability_profile_index": profile_index_by_worker[worker_id],
                "subtask": subtask.strip(),
                "access": access,
            }
        )
    structure = {
        "capability_profiles": profiles,
        "steps": steps,
    }
    topology = {
        "capability_profiles": profiles,
        "steps": [
            {
                "position_id": step["position_id"],
                "capability_profile_index": step["capability_profile_index"],
                "access": step["access"],
            }
            for step in steps
        ],
    }
    return {
        **structure,
        "topology_signature": sha256_bytes(stable_json(topology).encode("ascii")),
        "workflow_signature": sha256_bytes(stable_json(structure).encode("ascii")),
    }


def forbidden_identity_terms(bindings: Iterable[PoolCapabilities]) -> tuple[str, ...]:
    terms = {
        "claude",
        "gemini",
        "gpt",
        "grok",
        "opus",
        "ornith",
        "qwen",
        "terra",
    }
    for binding in bindings:
        for slot in binding.slots:
            terms.add(slot.concrete_model.lower())
    return tuple(sorted(terms, key=lambda value: (-len(value), value)))


def find_identity_leaks(value: Any, forbidden_terms: Sequence[str]) -> tuple[str, ...]:
    surface = stable_json(value).lower()
    leaks: set[str] = set()
    for term in forbidden_terms:
        if not term:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", surface):
            leaks.add(term)
    return tuple(sorted(leaks))


def _task_prompt(result: Mapping[str, Any], root: Path) -> tuple[str, Path]:
    config = result.get("config")
    task = config.get("task") if isinstance(config, Mapping) else None
    raw_path = task.get("path") if isinstance(task, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path:
        task_id = result.get("task_id")
        raw_path = task_id.get("path") if isinstance(task_id, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path:
        raise OutcomeCorpusError("task path is unavailable")
    task_path = Path(raw_path)
    task_path = task_path if task_path.is_absolute() else root / task_path
    instruction = task_path / "instruction.md"
    if not instruction.is_file():
        raise OutcomeCorpusError("task instruction is unavailable")
    text = instruction.read_text(encoding="utf-8").strip()
    if not text:
        raise OutcomeCorpusError("task instruction is empty")
    return text, instruction.resolve()


def _task_family(task_name: str, split_registry: Mapping[str, SplitAssignment]) -> str:
    del split_registry
    if task_name.startswith("fugu-train/"):
        return "fugu-train"
    if task_name.startswith("task_"):
        return "nl2bash"
    return task_name.split("-", 1)[0]


def _operational_invalidity(metadata: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    unavailable = metadata.get("unavailable_worker_models")
    if isinstance(unavailable, Mapping) and unavailable:
        reasons.append("provider_failure")
    numeric_zero_fields = {
        "provider_failure_events": "provider_failure",
        "provider_owner_retries": "provider_retry",
        "provider_replans": "provider_replan",
        "protocol_replans": "protocol_replan",
        "worker_protocol_errors": "worker_protocol_error",
        "paid_worker_call_limit_responses": "call_cap_stop",
        "task_budget_stop_responses": "task_budget_stop",
        "planner_failures": "planner_failure",
        "unrecoverable_planning_failures": "unrecoverable_planning_failure",
        "live_control_failures": "live_control_failure",
        "live_control_replacement_plan_failures": "replacement_plan_failure",
        "workspace_recovery_failures": "workspace_recovery_failure",
        "workspace_cleanup_failures": "workspace_cleanup_failure",
    }
    for field, reason in numeric_zero_fields.items():
        if _as_int(metadata.get(field)) > 0:
            reasons.append(reason)
    if metadata.get("collection_training_eligible") is False:
        reasons.append("explicitly_training_ineligible")
    return sorted(set(reasons))


def _pool_matches_result(
    binding: PoolCapabilities, metadata: Mapping[str, Any]
) -> bool:
    worker_models = metadata.get("worker_models")
    if not isinstance(worker_models, list) or len(worker_models) != len(binding.slots):
        return False
    return all(
        isinstance(observed, str) and observed == slot.concrete_model
        for observed, slot in zip(worker_models, binding.slots, strict=True)
    )


def _route_result_path(route_path: Path) -> Path:
    if route_path.name != "fugu_routes.jsonl" or route_path.parent.name != "agent":
        raise OutcomeCorpusError(f"unexpected route path: {route_path}")
    return route_path.parent.parent / "result.json"


def discover_outcomes(
    *,
    root: Path,
    split_salt: str = DEFAULT_SPLIT_SALT,
    route_paths: Sequence[Path] | None = None,
    invalid_report_paths: Sequence[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return identity-free rows, identity-bearing provenance, and exclusions."""

    pool_registry = load_capability_registry(root)
    split_registry = load_split_registry(root)
    invalid_results = load_invalid_result_paths(
        root, report_paths=invalid_report_paths
    )
    unique_bindings = {binding.registry_key: binding for binding in pool_registry.values()}
    forbidden = forbidden_identity_terms(unique_bindings.values())

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_results: set[Path] = set()

    paths = (
        sorted(route_paths)
        if route_paths is not None
        else sorted(root.glob("scratchpad/**/agent/fugu_routes.jsonl"))
    )
    for route_path in paths:
        result_path = _route_result_path(route_path).resolve()
        if result_path in seen_results:
            continue
        seen_results.add(result_path)
        reasons: list[str] = []
        if not result_path.is_file():
            reasons.append("missing_result")
            exclusions.append(
                {
                    "result_path": result_path.relative_to(root).as_posix(),
                    "route_path": route_path.relative_to(root).as_posix(),
                    "reasons": reasons,
                }
            )
            continue
        try:
            result = _read_json(result_path)
        except OutcomeCorpusError:
            exclusions.append(
                {
                    "result_path": result_path.relative_to(root).as_posix(),
                    "route_path": route_path.relative_to(root).as_posix(),
                    "reasons": ["invalid_result_json"],
                }
            )
            continue

        task_name = result.get("task_name")
        task_name = task_name if isinstance(task_name, str) else ""
        if task_name.startswith("terminal-bench/") or any(
            part.startswith("terminalbench21_") for part in result_path.parts
        ):
            reasons.append("terminalbench_excluded")
        reward = _binary_reward(result)
        if reward is None:
            reasons.append("missing_or_nonbinary_reward")
        if result.get("exception_info") is not None:
            reasons.append("result_exception")
        if result_path in invalid_results:
            reasons.extend(invalid_results[result_path])

        metadata = _result_metadata(result)
        routes = metadata.get("fugu_routes")
        if not isinstance(routes, list) or not routes:
            reasons.append("missing_routes")
        reasons.extend(_operational_invalidity(metadata))

        pool_id = metadata.get("pool_id")
        binding = pool_registry.get(pool_id) if isinstance(pool_id, str) else None
        if binding is None:
            reasons.append("unknown_pool_binding")
        elif not _pool_matches_result(binding, metadata):
            reasons.append("pool_binding_mismatch")

        task_text = ""
        instruction_path: Path | None = None
        if not reasons or set(reasons) <= {"terminalbench_excluded"}:
            try:
                task_text, instruction_path = _task_prompt(result, root)
            except OutcomeCorpusError as exc:
                reasons.append(str(exc))

        workflow: dict[str, Any] | None = None
        if binding is not None and isinstance(routes, list) and routes:
            try:
                workflow = canonicalize_initial_workflow(
                    routes,
                    binding,
                    registered_initial_workflow=(
                        metadata.get("collection_registered_initial_workflow")
                        if isinstance(
                            metadata.get("collection_registered_initial_workflow"),
                            Mapping,
                        )
                        else metadata.get("collection_registered_workflow")
                        if isinstance(metadata.get("collection_registered_workflow"), Mapping)
                        else None
                    ),
                )
            except OutcomeCorpusError as exc:
                reasons.append(f"invalid_initial_workflow:{exc}")

        split_assignment = assign_split(task_name, split_registry, salt=split_salt)
        task_checksum = result.get("task_checksum")
        if not isinstance(task_checksum, str) or not task_checksum:
            reasons.append("missing_task_checksum")

        record_seed = (
            result_path.relative_to(root).as_posix()
            + "\0"
            + sha256_file(result_path)
            + "\0"
            + sha256_file(route_path)
        )
        record_id = sha256_bytes(record_seed.encode("utf-8"))
        task_key = sha256_bytes(task_name.encode("utf-8")) if task_name else ""

        if (
            reward is not None
            and binding is not None
            and workflow is not None
            and task_text
            and isinstance(task_checksum, str)
        ):
            learned_input = {
                "task": task_text,
                "task_family": _task_family(task_name, split_registry),
                "capability_profiles": workflow["capability_profiles"],
                "workflow": {
                    "steps": workflow["steps"],
                    "topology_signature": workflow["topology_signature"],
                    "workflow_signature": workflow["workflow_signature"],
                },
                "runtime_constraints": {
                    "worker_call_limit": _as_int(
                        metadata.get("fair_position_call_budget"), default=120
                    )
                    or 120,
                    "task_wall_time_seconds": metadata.get("terminal_task_budget_s"),
                    "max_agent_turns": metadata.get("max_agent_turns"),
                    "live_control_available": isinstance(
                        metadata.get("live_control_decisions"), list
                    ),
                },
            }
            leaks = find_identity_leaks(learned_input, forbidden)
            if leaks:
                reasons.append("model_identity_leak:" + ",".join(leaks))

        if reasons:
            exclusions.append(
                {
                    "record_id": record_id,
                    "task_name": task_name,
                    "result_path": result_path.relative_to(root).as_posix(),
                    "route_path": route_path.relative_to(root).as_posix(),
                    "reasons": sorted(set(reasons)),
                }
            )
            continue

        assert reward is not None
        assert binding is not None
        assert workflow is not None
        assert instruction_path is not None
        assert isinstance(task_checksum, str)
        learned_row = {
            "version": CORPUS_VERSION,
            "record_id": record_id,
            "task_key": task_key,
            "task_checksum": task_checksum,
            "split": split_assignment.split,
            "input": learned_input,
            "target": {
                "verified_success": int(reward),
                "worker_calls": _as_int(metadata.get("paid_worker_call_attempts")),
            },
        }
        rows.append(learned_row)
        provenance.append(
            {
                "record_id": record_id,
                "task_name": task_name,
                "task_checksum": task_checksum,
                "split": split_assignment.split,
                "split_sources": list(split_assignment.sources),
                "split_source_conflict_resolved_to_holdout": split_assignment.conflict,
                "pool_registry_key": binding.registry_key,
                "pool_id": pool_id,
                "pool_fingerprint": metadata.get("pool_fingerprint"),
                "worker_models": metadata.get("worker_models"),
                "runtime_revision": metadata.get("runtime_revision"),
                "result_path": result_path.relative_to(root).as_posix(),
                "result_sha256": sha256_file(result_path),
                "route_path": route_path.relative_to(root).as_posix(),
                "route_sha256": sha256_file(route_path),
                "instruction_path": instruction_path.relative_to(root).as_posix(),
                "instruction_sha256": sha256_file(instruction_path),
                "binding_path": binding.source_path.relative_to(root).as_posix(),
                "binding_sha256": sha256_file(binding.source_path),
            }
        )

    rows.sort(key=lambda row: row["record_id"])
    provenance.sort(key=lambda row: row["record_id"])
    exclusions.sort(key=lambda row: (row.get("result_path", ""), row.get("record_id", "")))
    _attach_pool_calibration(rows, provenance)
    return rows, provenance, exclusions


def _attach_pool_calibration(
    rows: list[dict[str, Any]], provenance: Sequence[Mapping[str, Any]]
) -> None:
    provenance_by_id = {row["record_id"]: row for row in provenance}
    train_by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            continue
        pool = str(provenance_by_id[row["record_id"]]["pool_registry_key"])
        train_by_pool[pool].append(row)

    for row in rows:
        pool = str(provenance_by_id[row["record_id"]]["pool_registry_key"])
        peers = [
            peer
            for peer in train_by_pool[pool]
            if peer["task_key"] != row["task_key"]
        ]
        successes = sum(peer["target"]["verified_success"] for peer in peers)
        solo = [
            peer
            for peer in peers
            if len(peer["input"]["workflow"]["steps"]) == 1
        ]
        coordinated = [
            peer
            for peer in peers
            if len(peer["input"]["workflow"]["steps"]) > 1
        ]
        row["input"]["pool_calibration"] = {
            "prior_attempts": len(peers),
            "prior_unique_tasks": len({peer["task_key"] for peer in peers}),
            "prior_successes": successes,
            "beta_posterior_success_mean": (successes + 1) / (len(peers) + 2),
            "solo_attempts": len(solo),
            "solo_successes": sum(
                peer["target"]["verified_success"] for peer in solo
            ),
            "coordinated_attempts": len(coordinated),
            "coordinated_successes": sum(
                peer["target"]["verified_success"] for peer in coordinated
            ),
        }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "holdout"):
        selected = [row for row in rows if row["split"] == split]
        labels = Counter(row["target"]["verified_success"] for row in selected)
        result[split] = {
            "rows": len(selected),
            "tasks": len({row["task_key"] for row in selected}),
            "task_checksums": len({row["task_checksum"] for row in selected}),
            "successes": labels[1],
            "failures": labels[0],
            "families": dict(
                sorted(Counter(row["input"]["task_family"] for row in selected).items())
            ),
        }
    return result


def _contrast_groups(
    rows: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    provenance_by_id = {row["record_id"]: row for row in provenance}
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pool = str(provenance_by_id[row["record_id"]]["pool_registry_key"])
        groups[(row["split"], row["task_key"], pool)].append(row)
    summary: dict[str, Any] = {}
    for split in ("train", "holdout"):
        selected = [values for (group_split, _, _), values in groups.items() if group_split == split]
        outcome_variation = [
            values
            for values in selected
            if len({row["target"]["verified_success"] for row in values}) > 1
        ]
        workflow_contrast = [
            values
            for values in outcome_variation
            if len(
                {
                    row["input"]["workflow"]["workflow_signature"]
                    for row in values
                }
            )
            > 1
        ]
        topology_contrast = [
            values
            for values in outcome_variation
            if len(
                {
                    row["input"]["workflow"]["topology_signature"]
                    for row in values
                }
            )
            > 1
        ]
        summary[split] = {
            "task_pool_groups": len(selected),
            "groups_with_outcome_variation": len(outcome_variation),
            "groups_with_workflow_and_outcome_contrast": len(workflow_contrast),
            "groups_with_topology_and_outcome_contrast": len(topology_contrast),
        }
    return summary


def build_audit_report(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    thresholds: AuditThresholds,
    current_pool_registry_key: str = "current_pool_v11",
) -> dict[str, Any]:
    if {row["record_id"] for row in rows} != {
        row["record_id"] for row in provenance
    }:
        raise OutcomeCorpusError("learned rows and provenance do not match")
    learned_surface = [row["input"] for row in rows]
    bindings = {binding.registry_key: binding for binding in load_capability_registry(root).values()}
    leaks = find_identity_leaks(learned_surface, forbidden_identity_terms(bindings.values()))
    train_keys = {row["task_key"] for row in rows if row["split"] == "train"}
    holdout_keys = {row["task_key"] for row in rows if row["split"] == "holdout"}
    counts = _counts(rows)
    contrasts = _contrast_groups(rows, provenance)
    provenance_by_id = {row["record_id"]: row for row in provenance}
    current = [
        row
        for row in rows
        if provenance_by_id[row["record_id"]]["pool_registry_key"]
        == current_pool_registry_key
    ]
    current_counts = _counts(current)
    exclusion_reasons = Counter(
        reason for row in exclusions for reason in row.get("reasons") or []
    )

    generic_checks = {
        "train_rows": counts["train"]["rows"] >= thresholds.generic_train_rows,
        "train_tasks": counts["train"]["tasks"] >= thresholds.generic_train_tasks,
        "holdout_rows": counts["holdout"]["rows"] >= thresholds.generic_holdout_rows,
        "holdout_tasks": counts["holdout"]["tasks"] >= thresholds.generic_holdout_tasks,
        "train_labels": min(
            counts["train"]["successes"], counts["train"]["failures"]
        )
        >= thresholds.generic_each_label_per_split,
        "holdout_labels": min(
            counts["holdout"]["successes"], counts["holdout"]["failures"]
        )
        >= thresholds.generic_each_label_per_split,
        "task_disjoint": not (train_keys & holdout_keys),
        "identity_free": not leaks,
    }
    ranker_checks = {
        **generic_checks,
        "train_workflow_contrasts": contrasts["train"][
            "groups_with_workflow_and_outcome_contrast"
        ]
        >= thresholds.ranker_train_contrast_groups,
        "holdout_workflow_contrasts": contrasts["holdout"][
            "groups_with_workflow_and_outcome_contrast"
        ]
        >= thresholds.ranker_holdout_contrast_groups,
        "current_pool_train_rows": current_counts["train"]["rows"]
        >= thresholds.current_pool_train_rows,
        "current_pool_train_labels": min(
            current_counts["train"]["successes"],
            current_counts["train"]["failures"],
        )
        >= thresholds.current_pool_each_train_label,
        "current_pool_holdout_rows": current_counts["holdout"]["rows"]
        >= thresholds.current_pool_holdout_rows,
        "current_pool_holdout_labels": min(
            current_counts["holdout"]["successes"],
            current_counts["holdout"]["failures"],
        )
        >= thresholds.current_pool_each_holdout_label,
    }
    return {
        "version": CORPUS_VERSION,
        "created_at": utc_now(),
        "objective": (
            "audit historical verified agentic outcomes for a GPS-inspired, "
            "model-agnostic task/workflow value model"
        ),
        "external_calls": 0,
        "paid_calls": 0,
        "terminalbench_rows_in_learning_surface": 0,
        "inventory": {
            "routed_result_candidates": len(rows) + len(exclusions),
            "eligible_rows": len(rows),
            "excluded_rows": len(exclusions),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        },
        "eligible": counts,
        "current_pool_eligible": current_counts,
        "contrast": contrasts,
        "integrity": {
            "task_split_overlap": sorted(train_keys & holdout_keys),
            "model_identity_leaks": list(leaks),
            "learned_rows_match_provenance": True,
            "concrete_model_identities_confined_to_provenance": not leaks,
            "preexisting_holdout_dominance": True,
            "unassigned_split_rule": f"sha256({DEFAULT_SPLIT_SALT}\\0task_name) mod 5",
        },
        "gates": {
            "generic_task_acquisition_pilot": {
                "ready": all(generic_checks.values()),
                "checks": generic_checks,
            },
            "current_pool_workflow_ranker": {
                "ready": all(ranker_checks.values()),
                "checks": ranker_checks,
            },
        },
        "thresholds": thresholds.__dict__,
        "decision_policy": {
            "generic_pilot_only_if_ready": True,
            "product_ranker_training_only_if_current_pool_gate_ready": True,
            "no_optimizer_step_from_this_audit": True,
            "no_paid_rollout_authorized_by_this_audit": True,
            "holdout_rows_enter_training": False,
            "terminalbench_rows_enter_training": False,
            "invalid_or_infrastructure_rows_are_negative_labels": False,
        },
    }
