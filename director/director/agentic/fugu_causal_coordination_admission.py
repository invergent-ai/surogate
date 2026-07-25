"""Fail-closed admission for same-runtime causal coordination evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra.pool_binding import load_pool_binding


ADMISSION_VERSION = "fugu_causal_coordination_admission_v1"
CAMPAIGN_VERSION = "fugu_causal_coordination_v1"
REPORT_VERSION = "fugu_causal_coordination_pair_report_v1"
YUNWU_API_BASE = "https://yunwu.ai/v1"


class CausalCoordinationAdmissionError(ValueError):
    """The frozen pair cannot support trusted coordination evidence."""


@dataclass(frozen=True)
class CausalCoordinationAdmission:
    task_id: str
    mechanism_id: str
    rejected_action: dict[str, Any]
    preferred_action: dict[str, Any]
    solo_calls: int
    coordinated_calls: int
    training_conversion_locked: bool
    evaluation_excluded: bool


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CausalCoordinationAdmissionError(
            f"{label} must contain exactly {sorted(fields)}"
        )
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CausalCoordinationAdmissionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CausalCoordinationAdmissionError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CausalCoordinationAdmissionError(
                f"{label} line {number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise CausalCoordinationAdmissionError(
                f"{label} line {number} must be an object"
            )
        rows.append(value)
    return rows


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise CausalCoordinationAdmissionError(f"{label} path is invalid")
    candidate = Path(raw)
    path = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CausalCoordinationAdmissionError(f"{label} escapes project root") from exc
    if not path.is_file():
        raise CausalCoordinationAdmissionError(f"{label} is missing")
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_file(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise CausalCoordinationAdmissionError(f"{label} hash drift")
    return path


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _validate_campaign_files(campaign: dict[str, Any], root: Path) -> None:
    frozen_code = campaign.get("frozen_code")
    if not isinstance(frozen_code, dict) or not frozen_code:
        raise CausalCoordinationAdmissionError("campaign frozen code is missing")
    for label, record in frozen_code.items():
        _verified_file(root, record, label=f"frozen code {label}")

    frozen_evidence = campaign.get("frozen_evidence")
    if not isinstance(frozen_evidence, dict) or not frozen_evidence:
        raise CausalCoordinationAdmissionError("campaign frozen evidence is missing")
    base_names = sorted(
        key for key in frozen_evidence if not key.endswith("_sha256")
    )
    expected_keys = {
        item for name in base_names for item in (name, f"{name}_sha256")
    }
    if set(frozen_evidence) != expected_keys:
        raise CausalCoordinationAdmissionError("frozen evidence schema drift")
    for name in base_names:
        path = _project_file(root, frozen_evidence[name], label=name)
        if sha256(path) != frozen_evidence[f"{name}_sha256"]:
            raise CausalCoordinationAdmissionError(f"{name} hash drift")


def _validate_result(
    *,
    root: Path,
    ledger: dict[str, Any],
    job: dict[str, Any],
    binding: Any,
    expected_reward: float,
) -> tuple[dict[str, Any], int]:
    for path_key, hash_key, label in (
        ("result_path", "result_sha256", "result"),
        ("route_log_path", "route_log_sha256", "route log"),
        ("trajectory_path", "trajectory_sha256", "trajectory"),
    ):
        path = _project_file(root, ledger.get(path_key), label=label)
        if sha256(path) != ledger.get(hash_key):
            raise CausalCoordinationAdmissionError(f"{label} hash drift")

    result_path = _project_file(root, ledger["result_path"], label="result")
    result = _read_json(result_path, label="result")
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    calls = ledger.get("paid_worker_call_attempts")
    expected_status = "accepted" if expected_reward == 1.0 else "rejected"
    expected_rejection = None if expected_reward == 1.0 else "verifier_nonpass"
    checks = {
        "ledger_identity": ledger.get("collection_id") == job["collection_id"],
        "ledger_status": ledger.get("status") == expected_status,
        "ledger_rejection": ledger.get("rejection_reason") == expected_rejection,
        "ledger_reward": ledger.get("reward") == expected_reward,
        "harbor": ledger.get("harbor_returncode") == 0,
        "result_reward": _reward(result) == expected_reward,
        "exception": result.get("exception_info") is None,
        "provider": ledger.get("worker_provider_base") == YUNWU_API_BASE,
        "models": tuple(ledger.get("worker_models") or ()) == binding.runtime_models,
        "fingerprint": ledger.get("pool_fingerprint") == binding.pool_fingerprint,
        "runtime": ledger.get("runtime_revision")
        == "20260720-r56-anonymous-planning-constraints",
        "metadata_collection": metadata.get("collection_id")
        == job["collection_id"],
        "registered_workflow": metadata.get("collection_registered_workflow")
        == job["action"],
        "not_training_eligible": metadata.get("collection_training_eligible")
        is False,
        "paid": metadata.get("worker_calls_are_paid") is True,
        "metadata_provider": metadata.get("worker_provider_base")
        == YUNWU_API_BASE,
        "provider_set": set(metadata.get("worker_provider_bases") or ())
        == {YUNWU_API_BASE},
        "metadata_models": tuple(metadata.get("worker_models") or ())
        == binding.runtime_models,
        "metadata_fingerprint": metadata.get("pool_fingerprint")
        == binding.pool_fingerprint,
        "call_limit": metadata.get("max_agent_turns") == 120,
        "retry_limit": metadata.get("provider_owner_retry_limit") == 0,
        "retries": metadata.get("provider_owner_retries") == 0,
        "replans": metadata.get("provider_replans") == 0,
        "provider_failures": metadata.get("provider_failure_events") == [],
        "protocol": metadata.get("worker_protocol_errors") == 0,
        "planner": metadata.get("planner_failures") == 0,
        "live_control": metadata.get("live_control_failures") == 0,
        "workspace": metadata.get("blocked_workspace_commands") == 0,
        "steps": metadata.get("completed_workflow_steps")
        == len(job["action"]["steps"]),
        "workflows": metadata.get("completed_workflows") == 1,
        "calls": isinstance(calls, int)
        and not isinstance(calls, bool)
        and 1 <= calls <= 120
        and metadata.get("paid_worker_call_attempts") == calls,
    }
    failed = sorted(label for label, passed in checks.items() if not passed)
    if failed:
        raise CausalCoordinationAdmissionError(
            f"{job['collection_id']} result attestation failed: {failed}"
        )
    return result, calls


def admit_causal_coordination_pair(
    manifest_path: Path, *, root: Path
) -> CausalCoordinationAdmission:
    root = root.resolve()
    manifest = _object(
        _read_json(manifest_path, label="admission manifest"),
        fields={
            "version",
            "admission_id",
            "validator",
            "campaign",
            "ledger",
            "pair_report",
            "pool_binding",
            "verifier_outputs",
            "policy",
        },
        label="admission manifest",
    )
    if manifest["version"] != ADMISSION_VERSION:
        raise CausalCoordinationAdmissionError("unsupported admission version")
    validator = _verified_file(root, manifest["validator"], label="validator")
    if validator.resolve() != Path(__file__).resolve():
        raise CausalCoordinationAdmissionError("validator is not this implementation")

    policy = _object(
        manifest["policy"],
        fields={
            "permanently_exclude_source_task",
            "training_conversion_locked",
            "minimum_train_pairs_before_learning",
            "minimum_whole_task_holdout_pairs_before_learning",
            "permit_expected_lift_claim",
            "permit_model_identity_in_learned_surface",
            "require_single_attempt_per_arm",
            "require_slot_profile_permutation_before_learning",
        },
        label="policy",
    )
    expected_policy = {
        "permanently_exclude_source_task": True,
        "training_conversion_locked": True,
        "minimum_train_pairs_before_learning": 6,
        "minimum_whole_task_holdout_pairs_before_learning": 2,
        "permit_expected_lift_claim": False,
        "permit_model_identity_in_learned_surface": False,
        "require_single_attempt_per_arm": True,
        "require_slot_profile_permutation_before_learning": True,
    }
    if policy != expected_policy:
        raise CausalCoordinationAdmissionError("admission policy drift")

    campaign_path = _verified_file(root, manifest["campaign"], label="campaign")
    ledger_path = _verified_file(root, manifest["ledger"], label="ledger")
    report_path = _verified_file(root, manifest["pair_report"], label="pair report")
    campaign = _read_json(campaign_path, label="campaign")
    report = _read_json(report_path, label="pair report")

    binding_record = _object(
        manifest["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_file(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise CausalCoordinationAdmissionError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.provider_base != YUNWU_API_BASE
        or binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
    ):
        raise CausalCoordinationAdmissionError("pool binding identity drift")

    campaign_checks = {
        "version": campaign.get("version") == CAMPAIGN_VERSION,
        "provider": campaign.get("external_provider") == YUNWU_API_BASE,
        "runtime": campaign.get("runtime_revision")
        == "20260720-r56-anonymous-planning-constraints",
        "binding_path": campaign.get("pool_binding") == binding_record["path"],
        "binding_hash": campaign.get("pool_binding_sha256")
        == binding_record["sha256"],
        "pool_id": campaign.get("pool_id") == binding.pool_id,
        "fingerprint": campaign.get("pool_fingerprint")
        == binding.pool_fingerprint,
        "models": tuple(campaign.get("worker_pool_provenance") or ())
        == binding.runtime_models,
        "call_limit": campaign.get("global_paid_call_ceiling_per_arm") == 120,
        "timeout": campaign.get("worker_timeout_seconds") == 600.0,
        "provider_retries": campaign.get("provider_retries") == 0,
        "task_retries": campaign.get("task_retries") == 0,
        "attempts": campaign.get("attempts_per_arm") == 1,
        "arms": campaign.get("maximum_arms") == 2,
        "conversion": campaign.get("training_conversion_locked") is True,
        "train_gate": campaign.get("minimum_train_pairs_before_learning") == 6,
        "holdout_gate": campaign.get(
            "minimum_whole_task_holdout_pairs_before_learning"
        )
        == 2,
        "exclusion": campaign.get("task_permanently_evaluation_excluded") is True,
        "model_identity": (campaign.get("learned_surface") or {}).get(
            "model_identity_count"
        )
        == 0,
        "preferred": (campaign.get("learned_surface") or {}).get(
            "globally_preferred_worker"
        )
        is False,
        "fallback": (campaign.get("learned_surface") or {}).get(
            "default_or_fallback_worker"
        )
        is False,
    }
    failed = sorted(label for label, passed in campaign_checks.items() if not passed)
    if failed:
        raise CausalCoordinationAdmissionError(
            f"campaign policy or binding drift: {failed}"
        )
    _validate_campaign_files(campaign, root)

    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise CausalCoordinationAdmissionError("campaign must contain two arms")
    solo_job, coordinated_job = jobs
    shared_keys = {
        "task_id",
        "task_name",
        "task_dir",
        "task_tree_sha256",
        "instruction_sha256",
        "source_policy",
        "oracle_result_path",
        "oracle_result_sha256",
        "oracle_reward",
        "verifier_audited",
    }
    if any(solo_job.get(key) != coordinated_job.get(key) for key in shared_keys):
        raise CausalCoordinationAdmissionError("arm task evidence differs")
    task_dir = Path(str(solo_job.get("task_dir") or "")).resolve()
    if (
        not task_dir.is_dir()
        or tree_sha256(task_dir) != solo_job.get("task_tree_sha256")
        or sha256(task_dir / "instruction.md") != solo_job.get("instruction_sha256")
    ):
        raise CausalCoordinationAdmissionError("task artifact drift")
    solo_action = solo_job.get("action")
    coordinated_action = coordinated_job.get("action")
    if (
        not isinstance(solo_action, dict)
        or not isinstance(coordinated_action, dict)
        or solo_action.get("action") != "replan"
        or coordinated_action.get("action") != "replan"
        or len(solo_action.get("steps") or ()) != 1
        or len(coordinated_action.get("steps") or ()) != 2
        or solo_action["steps"][0] != coordinated_action["steps"][0]
        or coordinated_action["steps"][1].get("access") != [0]
        or coordinated_action["steps"][1].get("worker_id")
        != (campaign.get("causal_intervention") or {})
        .get("only_added_position", {})
        .get("anonymous_worker_id")
    ):
        raise CausalCoordinationAdmissionError("causal topology intervention drift")
    learned_surface = json.dumps(
        {"rejected": solo_action, "preferred": coordinated_action},
        sort_keys=True,
        ensure_ascii=True,
    ).lower()
    forbidden = {
        binding.provider_base.lower(),
        *[slot.runtime_model.lower() for slot in binding.slots],
        *[slot.model_alias.lower() for slot in binding.slots],
        *[slot.training_name.lower() for slot in binding.slots],
    }
    if any(value in learned_surface for value in forbidden):
        raise CausalCoordinationAdmissionError("model identity leaked into learned pair")

    rows = _read_jsonl(ledger_path, label="ledger")
    if len(rows) != 2:
        raise CausalCoordinationAdmissionError("ledger must contain two arms")
    by_collection = {row.get("collection_id"): row for row in rows}
    if set(by_collection) != {job["collection_id"] for job in jobs}:
        raise CausalCoordinationAdmissionError("ledger arm identity drift")
    solo_result, solo_calls = _validate_result(
        root=root,
        ledger=by_collection[solo_job["collection_id"]],
        job=solo_job,
        binding=binding,
        expected_reward=0.0,
    )
    coordinated_result, coordinated_calls = _validate_result(
        root=root,
        ledger=by_collection[coordinated_job["collection_id"]],
        job=coordinated_job,
        binding=binding,
        expected_reward=1.0,
    )
    if solo_result.get("task_checksum") != coordinated_result.get("task_checksum"):
        raise CausalCoordinationAdmissionError("arm task checksums differ")

    report_checks = {
        "version": report.get("version") == REPORT_VERSION,
        "campaign": report.get("campaign_sha256") == sha256(campaign_path),
        "status": report.get("status") == "complete_observed_coordination_lift",
        "valid": report.get("valid_observed_coordination_lift_pair") is True,
        "locked": report.get("training_conversion_locked") is True,
        "provider": report.get("external_provider") == YUNWU_API_BASE,
        "provider_retries": report.get("provider_retries") == 0,
        "task_retries": report.get("task_retries") == 0,
        "arms": len(report.get("arms") or ()) == 2,
    }
    failed = sorted(label for label, passed in report_checks.items() if not passed)
    if failed:
        raise CausalCoordinationAdmissionError(f"pair report drift: {failed}")

    outputs = manifest.get("verifier_outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise CausalCoordinationAdmissionError("two verifier outputs are required")
    expected_markers = {
        solo_job["collection_id"]: "Web server returned HTTP 404",
        coordinated_job["collection_id"]: "1 passed",
    }
    for raw in outputs:
        record = _object(
            raw,
            fields={"collection_id", "path", "sha256", "required_marker"},
            label="verifier output",
        )
        collection_id = record["collection_id"]
        if record["required_marker"] != expected_markers.get(collection_id):
            raise CausalCoordinationAdmissionError("verifier outcome marker drift")
        path = _project_file(root, record["path"], label="verifier output")
        if sha256(path) != record["sha256"]:
            raise CausalCoordinationAdmissionError("verifier output hash drift")
        if record["required_marker"] not in path.read_text(encoding="utf-8"):
            raise CausalCoordinationAdmissionError("verifier outcome is missing")

    exclusion_path = _project_file(
        root,
        campaign["frozen_evidence"]["evaluation_exclusion_rows"],
        label="evaluation exclusions",
    )
    excluded = [
        row
        for row in _read_jsonl(exclusion_path, label="evaluation exclusions")
        if row.get("task_id") == solo_job["task_id"]
    ]
    if (
        len(excluded) != 1
        or excluded[0].get("evaluation_excluded") is not True
        or (excluded[0].get("provenance") or {}).get(
            "permanently_evaluation_excluded"
        )
        is not True
    ):
        raise CausalCoordinationAdmissionError("source task is not permanently excluded")

    return CausalCoordinationAdmission(
        task_id=solo_job["task_id"],
        mechanism_id="anonymous_independent_final_state_auditor",
        rejected_action=solo_action,
        preferred_action=coordinated_action,
        solo_calls=solo_calls,
        coordinated_calls=coordinated_calls,
        training_conversion_locked=True,
        evaluation_excluded=True,
    )
