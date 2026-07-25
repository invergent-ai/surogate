"""Migrate model-neutral control replay between compatible pool bindings."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra.live_control_sft import _action_json, _state_from_row
from ultra.pool_binding import PoolBinding, load_pool_binding


MIGRATION_VERSION = "fugu_role_neutral_replay_migration_v1"


class ReplayMigrationError(ValueError):
    """Replay cannot be rebound without changing its learned semantics."""


@dataclass(frozen=True)
class ReplayMigration:
    rows: tuple[dict[str, Any], ...]
    guardian_rows: tuple[dict[str, Any], ...]
    action_counts: dict[str, int]
    task_count: int
    collection_count: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_role(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReplayMigrationError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ReplayMigrationError(f"{label} must be a non-empty path")
    path = Path(raw)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReplayMigrationError(f"{label} escapes the project root") from exc
    if not path.is_file():
        raise ReplayMigrationError(f"{label} does not exist: {path}")
    return path


def _verified_record_file(
    root: Path, raw: Any, *, label: str
) -> tuple[Path, str]:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_file(root, record["path"], label=label)
    digest = sha256(path)
    if digest != record["sha256"]:
        raise ReplayMigrationError(f"{label} hash drift")
    return path, digest


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayMigrationError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ReplayMigrationError(f"{label} line {line_number} is not an object")
        rows.append(row)
    return rows


def _load_binding(
    root: Path, raw: Any, *, label: str
) -> tuple[PoolBinding, Path]:
    record = _object(
        raw,
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label=label,
    )
    path = _project_file(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise ReplayMigrationError(f"{label} hash drift")
    binding = load_pool_binding(path)
    if (
        binding.pool_id != record["pool_id"]
        or binding.pool_fingerprint != record["pool_fingerprint"]
    ):
        raise ReplayMigrationError(f"{label} identity drift")
    return binding, path


def _validate_slot_mapping(
    raw: Any, *, source: PoolBinding, target: PoolBinding
) -> dict[int, int]:
    if not isinstance(raw, list) or len(raw) != len(source.slots):
        raise ReplayMigrationError("slot mapping must cover every source slot")
    mapping: dict[int, int] = {}
    target_ids: set[int] = set()
    for index, value in enumerate(raw):
        row = _object(
            value,
            fields={"source_worker_id", "target_worker_id", "normalized_role_prior"},
            label=f"slot_mapping[{index}]",
        )
        source_id = row["source_worker_id"]
        target_id = row["target_worker_id"]
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or source_id in mapping
            or target_id in target_ids
            or not 0 <= source_id < len(source.slots)
            or not 0 <= target_id < len(target.slots)
        ):
            raise ReplayMigrationError("slot mapping is not one-to-one")
        source_roles = tuple(
            _normalize_role(role) for role in source.slots[source_id].role_prior
        )
        target_roles = tuple(
            _normalize_role(role) for role in target.slots[target_id].role_prior
        )
        if (
            source_roles != target_roles
            or list(source_roles) != row["normalized_role_prior"]
        ):
            raise ReplayMigrationError("slot mapping changes abstract role semantics")
        mapping[source_id] = target_id
        target_ids.add(target_id)
    if set(mapping) != set(range(len(source.slots))):
        raise ReplayMigrationError("slot mapping omits a source slot")
    return mapping


def _remap_worker_ids(value: Any, mapping: dict[int, int]) -> Any:
    migrated = copy.deepcopy(value)
    for position in migrated["state"]["positions"]:
        position["worker_id"] = mapping[position["worker_id"]]
    action = migrated["action"]
    for step in action.get("steps", []):
        step["worker_id"] = mapping[step["worker_id"]]
    return migrated


def _learned_surface(row: dict[str, Any]) -> str:
    return json.dumps(
        {"state": row.get("state"), "action": row.get("action")},
        sort_keys=True,
        ensure_ascii=True,
    ).lower()


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def migrate_role_neutral_replay(
    manifest_path: Path, *, root: Path
) -> ReplayMigration:
    """Rebind replay only when stable role semantics and artifacts are exact."""
    root = root.resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayMigrationError("replay migration manifest is invalid JSON") from exc
    manifest = _object(
        raw,
        fields={
            "version",
            "migration_id",
            "source_binding",
            "target_binding",
            "source_rows",
            "guardian_labels",
            "converter",
            "slot_mapping",
            "policy",
        },
        label="replay migration manifest",
    )
    if manifest["version"] != MIGRATION_VERSION:
        raise ReplayMigrationError("unsupported replay migration version")
    if not isinstance(manifest["migration_id"], str) or not manifest[
        "migration_id"
    ].strip():
        raise ReplayMigrationError("migration_id must be non-empty")
    source, _ = _load_binding(root, manifest["source_binding"], label="source binding")
    target, _ = _load_binding(root, manifest["target_binding"], label="target binding")
    if source.pool_fingerprint == target.pool_fingerprint:
        raise ReplayMigrationError("source and target pools must differ")

    source_models = {slot.runtime_model for slot in source.slots}
    source_aliases = {slot.model_alias for slot in source.slots}
    source_efforts = {slot.reasoning_effort for slot in source.slots}
    if len(source_models) != 1 or len(source_aliases) != 1 or len(source_efforts) != 1:
        raise ReplayMigrationError(
            "source slots are heterogeneous; labels may encode worker performance"
        )
    mapping = _validate_slot_mapping(
        manifest["slot_mapping"], source=source, target=target
    )

    policy = _object(
        manifest["policy"],
        fields={
            "usage",
            "require_homogeneous_source_workers",
            "permit_performance_claims",
            "forbidden_learned_substrings",
        },
        label="migration policy",
    )
    if (
        policy["usage"] != "anti_forgetting_replay_only"
        or policy["require_homogeneous_source_workers"] is not True
        or policy["permit_performance_claims"] is not False
        or not isinstance(policy["forbidden_learned_substrings"], list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in policy["forbidden_learned_substrings"]
        )
    ):
        raise ReplayMigrationError("migration policy is invalid")

    converter_path, _ = _verified_record_file(
        root, manifest["converter"], label="migration converter"
    )
    if converter_path.resolve() != Path(__file__).resolve():
        raise ReplayMigrationError("migration converter path is not this implementation")
    source_path, source_digest = _verified_record_file(
        root, manifest["source_rows"], label="source replay rows"
    )
    guardian_path, guardian_digest = _verified_record_file(
        root, manifest["guardian_labels"], label="guardian labels"
    )
    source_rows = _read_jsonl(source_path, label="source replay rows")
    guardian_rows = _read_jsonl(guardian_path, label="guardian labels")
    if not source_rows or not guardian_rows:
        raise ReplayMigrationError("source replay or guardian labels are empty")

    guardian_fields = {
        "collection_id",
        "task_name",
        "family",
        "template_id",
        "verifier_reward",
        "completion_claim_correct",
        "worker_calls_are_paid",
        "pool_fingerprint",
        "trajectory_path",
        "route_log_path",
        "result_path",
    }
    guardians: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(guardian_rows):
        if set(row) != guardian_fields:
            raise ReplayMigrationError(f"guardian row {index} schema drift")
        collection_id = row.get("collection_id")
        if not isinstance(collection_id, str) or collection_id in guardians:
            raise ReplayMigrationError("guardian collection IDs must be unique")
        if (
            row.get("pool_fingerprint") != source.pool_fingerprint
            or row.get("worker_calls_are_paid") is not False
            or row.get("completion_claim_correct")
            is not (row.get("verifier_reward") == 1.0)
        ):
            raise ReplayMigrationError(f"guardian row {index} attestation drift")
        guardians[collection_id] = row

    source_fields = {
        "record_id",
        "task_id",
        "pool_fingerprint",
        "terminalbench",
        "label_status",
        "provenance",
        "agentic_evidence",
        "state",
        "action",
    }
    forbidden = {
        item.strip().lower() for item in policy["forbidden_learned_substrings"]
    }
    for binding in (source, target):
        forbidden.add(binding.provider_base.lower())
        for slot in binding.slots:
            forbidden.update(
                {
                    slot.runtime_model.lower(),
                    slot.model_alias.lower(),
                    slot.training_name.lower(),
                }
            )
    migrated_rows: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    verified_collections: set[str] = set()
    for index, row in enumerate(source_rows):
        if set(row) != source_fields:
            raise ReplayMigrationError(f"source row {index} schema drift")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise ReplayMigrationError("source record IDs must be unique")
        record_ids.add(record_id)
        if (
            row.get("pool_fingerprint") != source.pool_fingerprint
            or row.get("terminalbench") is not False
            or row.get("label_status") != "valid_verifier_pass"
        ):
            raise ReplayMigrationError(f"source row {index} provenance drift")
        evidence = row.get("agentic_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("shared_workspace") is not True
            or evidence.get("verifier_audited") is not True
            or isinstance(evidence.get("tool_calls_observed"), bool)
            or not isinstance(evidence.get("tool_calls_observed"), int)
            or evidence.get("tool_calls_observed") < 0
        ):
            raise ReplayMigrationError(f"source row {index} evidence drift")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or set(provenance) != {
            "collection_id",
            "result_sha256",
            "route_log_sha256",
            "trajectory_sha256",
        }:
            raise ReplayMigrationError(f"source row {index} provenance schema drift")
        collection_id = provenance["collection_id"]
        guardian = guardians.get(collection_id)
        if (
            guardian is None
            or guardian.get("verifier_reward") != 1.0
            or guardian.get("completion_claim_correct") is not True
        ):
            raise ReplayMigrationError(
                f"source row {index} lacks a passing completion guardian"
            )
        if collection_id not in verified_collections:
            for path_key, hash_key in (
                ("result_path", "result_sha256"),
                ("route_log_path", "route_log_sha256"),
                ("trajectory_path", "trajectory_sha256"),
            ):
                artifact = Path(guardian[path_key]).resolve()
                try:
                    artifact.relative_to(root)
                except ValueError as exc:
                    raise ReplayMigrationError(
                        f"guardian artifact escapes root: {artifact}"
                    ) from exc
                if not artifact.is_file() or sha256(artifact) != provenance[hash_key]:
                    raise ReplayMigrationError(
                        f"{collection_id} {path_key} artifact drift"
                    )
            result = json.loads(Path(guardian["result_path"]).read_text(encoding="utf-8"))
            if (
                not isinstance(result, dict)
                or _reward(result) != 1.0
                or result.get("exception_info") is not None
            ):
                raise ReplayMigrationError(f"{collection_id} result outcome drift")
            verified_collections.add(collection_id)

        try:
            source_state = _state_from_row(row["state"], source)
            _action_json(row["action"], source_state)
        except Exception as exc:
            raise ReplayMigrationError(
                f"source row {index} violates the control contract: {exc}"
            ) from exc
        migrated = _remap_worker_ids(row, mapping)
        migrated["pool_fingerprint"] = target.pool_fingerprint
        migrated["label_status"] = "audited_role_replay"
        migrated["replay_migrated"] = True
        migrated["source_pool_fingerprint"] = source.pool_fingerprint
        migrated["provenance"].update(
            {
                "replay_source_sha256": source_digest,
                "guardian_source_sha256": guardian_digest,
                "migration_usage": "anti_forgetting_replay_only",
            }
        )
        try:
            target_state = _state_from_row(migrated["state"], target)
            _action_json(migrated["action"], target_state)
        except Exception as exc:
            raise ReplayMigrationError(
                f"migrated row {index} violates the target contract: {exc}"
            ) from exc
        surface = _learned_surface(migrated)
        leaked = sorted(fragment for fragment in forbidden if fragment in surface)
        if leaked:
            raise ReplayMigrationError(
                f"migrated learned surface contains pool identity: {leaked}"
            )
        migrated_rows.append(migrated)

    migrated_guardians = []
    for row in guardian_rows:
        migrated_guardians.append(
            {
                **row,
                "migration_usage": "anti_forgetting_evaluation_only",
                "source_pool_fingerprint": source.pool_fingerprint,
                "target_pool_fingerprint": target.pool_fingerprint,
            }
        )
    counts = Counter(row["action"]["action"] for row in migrated_rows)
    return ReplayMigration(
        rows=tuple(migrated_rows),
        guardian_rows=tuple(migrated_guardians),
        action_counts=dict(sorted(counts.items())),
        task_count=len({row["task_id"] for row in migrated_rows}),
        collection_count=len(verified_collections),
    )
