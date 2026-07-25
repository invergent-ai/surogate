"""Fail-closed admission of verified branch-point recovery decisions."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra.live_control import parse_control_action
from ultra.live_control_trajectory import convert_successful_harbor_trajectory
from ultra.pool_binding import load_pool_binding


ADMISSION_VERSION = "fugu_branchpoint_recovery_admission_v1"
ACCEPTED_CAMPAIGN_VERSIONS = {
    "fugu_branchpoint_recovery_v1",
    "fugu_branchpoint_recovery_v2",
}
YUNWU_API_BASE = "https://yunwu.ai/v1"


class RecoveryAdmissionError(ValueError):
    """Recovery evidence cannot support trusted conductor labels."""


@dataclass(frozen=True)
class RecoveryAdmission:
    rows: tuple[dict[str, Any], ...]
    recovery_count: int
    action_counts: dict[str, int]
    evaluation_excluded_task_ids: tuple[str, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise RecoveryAdmissionError(
            f"{label} must contain exactly {sorted(fields)}"
        )
    return value


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RecoveryAdmissionError(f"{label} must be a non-empty path")
    path = Path(raw)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RecoveryAdmissionError(f"{label} escapes the project root") from exc
    if not path.is_file():
        raise RecoveryAdmissionError(f"{label} does not exist: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecoveryAdmissionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RecoveryAdmissionError(f"{label} must be an object")
    return value


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
            raise RecoveryAdmissionError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise RecoveryAdmissionError(
                f"{label} line {line_number} must be an object"
            )
        rows.append(row)
    return rows


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _verified_file(
    root: Path,
    record: dict[str, Any],
    *,
    path_key: str,
    hash_key: str,
    label: str,
) -> Path:
    path = _project_file(root, record.get(path_key), label=label)
    if sha256(path) != record.get(hash_key):
        raise RecoveryAdmissionError(f"{label} hash drift")
    return path


def _validate_frozen_code(campaign: dict[str, Any], root: Path) -> None:
    frozen = campaign.get("frozen_code")
    if not isinstance(frozen, dict) or not frozen:
        raise RecoveryAdmissionError("campaign has no frozen code evidence")
    path_keys = sorted(key for key in frozen if not key.endswith("_sha256"))
    if not path_keys or set(frozen) != {
        item for key in path_keys for item in (key, f"{key}_sha256")
    }:
        raise RecoveryAdmissionError("campaign frozen code schema is invalid")
    for key in path_keys:
        path = _project_file(root, frozen[key], label=f"frozen code {key}")
        if sha256(path) != frozen[f"{key}_sha256"]:
            raise RecoveryAdmissionError(f"frozen campaign code drift: {key}")


def _validate_campaign(
    *,
    campaign_path: Path,
    campaign_sha256: str,
    ledger_path: Path,
    ledger_sha256: str,
    collection_id: str,
    mechanism_id: str,
    sanitized_workflow: dict[str, Any],
    expected_post_start_rows: int,
    binding: Any,
    root: Path,
) -> tuple[list[dict[str, Any]], str]:
    if sha256(campaign_path) != campaign_sha256:
        raise RecoveryAdmissionError(f"{mechanism_id} campaign hash drift")
    if sha256(ledger_path) != ledger_sha256:
        raise RecoveryAdmissionError(f"{mechanism_id} ledger hash drift")
    campaign = _read_json(campaign_path, label=f"{mechanism_id} campaign")
    if campaign.get("version") not in ACCEPTED_CAMPAIGN_VERSIONS:
        raise RecoveryAdmissionError(f"{mechanism_id} campaign version is unsupported")
    if (
        campaign.get("external_provider") != YUNWU_API_BASE
        or campaign.get("global_paid_call_ceiling") != 120
        or campaign.get("worker_timeout_seconds") != 600.0
        or campaign.get("provider_retries") != 0
        or campaign.get("task_attempts") != 1
        or campaign.get("task_retries") != 0
        or campaign.get("maximum_jobs") != 1
        or campaign.get("product_candidate") is not False
        or campaign.get("training_eligible") is not False
        or campaign.get("minimum_verified_recovery_wins_for_conversion") != 2
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} campaign policy drift")
    if (
        campaign.get("pool_id") != binding.pool_id
        or campaign.get("pool_fingerprint") != binding.pool_fingerprint
        or campaign.get("pool_binding_sha256")
        != sha256(root / campaign.get("pool_binding", ""))
        or tuple(campaign.get("worker_pool") or ()) != binding.runtime_models
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} pool binding drift")
    _validate_frozen_code(campaign, root)

    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
        raise RecoveryAdmissionError(f"{mechanism_id} must freeze exactly one job")
    job = jobs[0]
    if (
        job.get("collection_id") != collection_id
        or job.get("source_policy") != "recovery_evidence_only"
        or job.get("verifier_audited") is not True
        or job.get("oracle_reward") != 1.0
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} job registration drift")
    task_dir = Path(str(job.get("task_dir") or "")).resolve()
    if (
        not task_dir.is_dir()
        or tree_sha256(task_dir) != job.get("task_tree_sha256")
        or sha256(task_dir / "instruction.md") != job.get("instruction_sha256")
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} task artifact drift")

    source_record = campaign.get("source_failure")
    oracle_record = campaign.get("oracle")
    if not isinstance(source_record, dict) or not isinstance(oracle_record, dict):
        raise RecoveryAdmissionError(f"{mechanism_id} lacks source/oracle evidence")
    source_result_path = _verified_file(
        root,
        source_record,
        path_key="result",
        hash_key="result_sha256",
        label=f"{mechanism_id} source result",
    )
    for path_key in ("route_log", "trajectory", "verifier_output"):
        _verified_file(
            root,
            source_record,
            path_key=path_key,
            hash_key=f"{path_key}_sha256",
            label=f"{mechanism_id} source {path_key}",
        )
    oracle_result_path = _verified_file(
        root,
        oracle_record,
        path_key="result",
        hash_key="result_sha256",
        label=f"{mechanism_id} oracle result",
    )
    source_result = _read_json(source_result_path, label="source result")
    oracle_result = _read_json(oracle_result_path, label="oracle result")
    source_metadata = (source_result.get("agent_result") or {}).get("metadata") or {}
    if (
        source_record.get("reward") != 0.0
        or _reward(source_result) != 0.0
        or source_result.get("exception_info") is not None
        or source_metadata.get("worker_provider_base") != YUNWU_API_BASE
        or tuple(source_metadata.get("worker_models") or ()) != binding.runtime_models
        or oracle_record.get("reward") != 1.0
        or _reward(oracle_result) != 1.0
        or oracle_result.get("exception_info") is not None
        or source_result.get("task_checksum") != oracle_result.get("task_checksum")
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} source/oracle outcome drift")

    ledger_rows = _read_jsonl(ledger_path, label=f"{mechanism_id} ledger")
    if len(ledger_rows) != 1 or ledger_rows[0].get("collection_id") != collection_id:
        raise RecoveryAdmissionError(f"{mechanism_id} ledger identity drift")
    ledger = ledger_rows[0]
    if (
        ledger.get("status") != "accepted"
        or ledger.get("reward") != 1.0
        or ledger.get("harbor_returncode") != 0
        or ledger.get("rejection_reason") is not None
        or ledger.get("worker_provider_base") != YUNWU_API_BASE
        or tuple(ledger.get("worker_models") or ()) != binding.runtime_models
        or ledger.get("pool_fingerprint") != binding.pool_fingerprint
        or ledger.get("runtime_revision") != campaign.get("runtime_revision")
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} accepted-result drift")
    result_path = _verified_file(
        root,
        ledger,
        path_key="result_path",
        hash_key="result_sha256",
        label=f"{mechanism_id} recovery result",
    )
    route_log_path = _verified_file(
        root,
        ledger,
        path_key="route_log_path",
        hash_key="route_log_sha256",
        label=f"{mechanism_id} recovery routes",
    )
    trajectory_path = _verified_file(
        root,
        ledger,
        path_key="trajectory_path",
        hash_key="trajectory_sha256",
        label=f"{mechanism_id} recovery trajectory",
    )
    result = _read_json(result_path, label="recovery result")
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    call_count = ledger.get("paid_worker_call_attempts")
    if (
        _reward(result) != 1.0
        or result.get("exception_info") is not None
        or metadata.get("collection_id") != collection_id
        or metadata.get("collection_registered_workflow") != job.get("action")
        or metadata.get("collection_training_eligible") is not False
        or metadata.get("worker_calls_are_paid") is not True
        or metadata.get("worker_provider_base") != YUNWU_API_BASE
        or set(metadata.get("worker_provider_bases") or ()) != {YUNWU_API_BASE}
        or tuple(metadata.get("worker_models") or ()) != binding.runtime_models
        or metadata.get("pool_fingerprint") != binding.pool_fingerprint
        or metadata.get("max_agent_turns") != 120
        or metadata.get("provider_owner_retry_limit") != 0
        or metadata.get("provider_owner_retries") != 0
        or metadata.get("provider_replans") != 0
        or metadata.get("provider_failure_events") != []
        or metadata.get("live_control_failures") != 0
        or metadata.get("completed_workflow_steps") != len(job["action"]["steps"])
        or metadata.get("completed_workflows") != 1
        or metadata.get("workspace_snapshot_ready") is not True
        or isinstance(call_count, bool)
        or not isinstance(call_count, int)
        or not 1 <= call_count <= 120
        or metadata.get("paid_worker_call_attempts") != call_count
    ):
        raise RecoveryAdmissionError(f"{mechanism_id} runtime attestation failed")

    source_plan = parse_control_action(json.dumps(job.get("action"), ensure_ascii=True))
    clean_plan = parse_control_action(json.dumps(sanitized_workflow, ensure_ascii=True))
    if (
        source_plan.action != "replan"
        or clean_plan.action != "replan"
        or len(source_plan.steps) != len(clean_plan.steps)
        or any(
            source.worker_id != clean.worker_id or source.access != clean.access
            for source, clean in zip(source_plan.steps, clean_plan.steps, strict=True)
        )
    ):
        raise RecoveryAdmissionError(
            f"{mechanism_id} sanitized workflow changes topology"
        )

    original_task = (task_dir / "instruction.md").read_text(encoding="utf-8")
    converted = convert_successful_harbor_trajectory(
        collection_id=collection_id,
        task_id=str(job.get("task_id")),
        original_task=original_task,
        pool_fingerprint=binding.pool_fingerprint,
        registered_action=job["action"],
        result_path=result_path,
        route_log_path=route_log_path,
        trajectory_path=trajectory_path,
        paid_call_limit=120,
    )
    if (
        not converted
        or converted[0]["action"].get("action") != "replan"
        or len(converted) - 1 != expected_post_start_rows
        or any(row["action"].get("action") == "replan" for row in converted[1:])
    ):
        raise RecoveryAdmissionError(
            f"{mechanism_id} post-start boundary count drift"
        )

    admitted: list[dict[str, Any]] = []
    for row in converted[1:]:
        clean = copy.deepcopy(row)
        for position, clean_step in zip(
            clean["state"]["positions"], clean_plan.steps, strict=True
        ):
            position["subtask"] = clean_step.subtask
        clean["terminalbench"] = True
        clean["evaluation_excluded"] = True
        clean["benchmark_source"] = "terminalbench21_recovery_training"
        clean["label_status"] = "verified_recovery_boundary"
        clean["provenance"].update(
            {
                "mechanism_id": mechanism_id,
                "source_failure_result_sha256": source_record["result_sha256"],
                "oracle_result_sha256": oracle_record["result_sha256"],
                "campaign_sha256": campaign_sha256,
                "ledger_sha256": ledger_sha256,
                "initial_replan_excluded_for_hindsight": True,
            }
        )
        admitted.append(clean)
    return admitted, str(job.get("task_id"))


def _learned_surface(row: dict[str, Any]) -> str:
    return json.dumps(
        {"state": row.get("state"), "action": row.get("action")},
        sort_keys=True,
        ensure_ascii=True,
    ).lower()


def convert_recovery_admission(
    admission_path: Path, *, root: Path
) -> RecoveryAdmission:
    """Validate frozen recoveries and return non-hindsight control boundaries."""
    root = root.resolve()
    admission_path = admission_path.resolve()
    manifest = _read_json(admission_path, label="recovery admission")
    manifest = _object(
        manifest,
        fields={
            "version",
            "admission_id",
            "pool_binding",
            "converter",
            "minimum_verified_recoveries",
            "leakage_policy",
            "recoveries",
        },
        label="recovery admission",
    )
    if manifest["version"] != ADMISSION_VERSION:
        raise RecoveryAdmissionError("unsupported recovery admission version")
    if not isinstance(manifest["admission_id"], str) or not manifest[
        "admission_id"
    ].strip():
        raise RecoveryAdmissionError("admission_id must be non-empty")

    binding_record = _object(
        manifest["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool_binding",
    )
    binding_path = _project_file(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise RecoveryAdmissionError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
        or binding.provider_base != YUNWU_API_BASE
    ):
        raise RecoveryAdmissionError("pool binding identity drift")

    converter = _object(
        manifest["converter"],
        fields={"path", "sha256"},
        label="converter",
    )
    converter_path = _project_file(root, converter["path"], label="converter")
    if sha256(converter_path) != converter["sha256"]:
        raise RecoveryAdmissionError("recovery converter hash drift")

    policy = _object(
        manifest["leakage_policy"],
        fields={
            "exclude_initial_replan",
            "mark_terminalbench_derived",
            "permanently_exclude_from_evaluation",
            "forbidden_learned_substrings",
        },
        label="leakage_policy",
    )
    if (
        policy["exclude_initial_replan"] is not True
        or policy["mark_terminalbench_derived"] is not True
        or policy["permanently_exclude_from_evaluation"] is not True
        or not isinstance(policy["forbidden_learned_substrings"], list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in policy["forbidden_learned_substrings"]
        )
    ):
        raise RecoveryAdmissionError("recovery leakage policy is invalid")

    minimum = manifest["minimum_verified_recoveries"]
    recoveries = manifest["recoveries"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 2
        or not isinstance(recoveries, list)
        or len(recoveries) < minimum
    ):
        raise RecoveryAdmissionError("recovery outcome-lift gate is not satisfied")

    rows: list[dict[str, Any]] = []
    task_ids: list[str] = []
    mechanisms: set[str] = set()
    collections: set[str] = set()
    for index, raw in enumerate(recoveries):
        recovery = _object(
            raw,
            fields={
                "mechanism_id",
                "campaign",
                "ledger",
                "collection_id",
                "expected_post_start_rows",
                "sanitized_workflow",
            },
            label=f"recoveries[{index}]",
        )
        mechanism_id = recovery["mechanism_id"]
        collection_id = recovery["collection_id"]
        if (
            not isinstance(mechanism_id, str)
            or not mechanism_id.strip()
            or mechanism_id in mechanisms
            or not isinstance(collection_id, str)
            or not collection_id.strip()
            or collection_id in collections
        ):
            raise RecoveryAdmissionError(
                "recoveries must have distinct mechanism and collection identities"
            )
        mechanisms.add(mechanism_id)
        collections.add(collection_id)
        expected_rows = recovery["expected_post_start_rows"]
        if (
            isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or expected_rows < 1
        ):
            raise RecoveryAdmissionError("expected post-start rows must be positive")
        campaign_record = _object(
            recovery["campaign"],
            fields={"path", "sha256"},
            label=f"recoveries[{index}].campaign",
        )
        ledger_record = _object(
            recovery["ledger"],
            fields={"path", "sha256"},
            label=f"recoveries[{index}].ledger",
        )
        campaign_path = _project_file(
            root, campaign_record["path"], label=f"{mechanism_id} campaign"
        )
        ledger_path = _project_file(
            root, ledger_record["path"], label=f"{mechanism_id} ledger"
        )
        converted, task_id = _validate_campaign(
            campaign_path=campaign_path,
            campaign_sha256=campaign_record["sha256"],
            ledger_path=ledger_path,
            ledger_sha256=ledger_record["sha256"],
            collection_id=collection_id,
            mechanism_id=mechanism_id,
            sanitized_workflow=recovery["sanitized_workflow"],
            expected_post_start_rows=expected_rows,
            binding=binding,
            root=root,
        )
        rows.extend(converted)
        task_ids.append(task_id)

    if len(set(task_ids)) != len(task_ids):
        raise RecoveryAdmissionError("recovery tasks must be distinct")
    forbidden = {
        item.strip().lower() for item in policy["forbidden_learned_substrings"]
    }
    forbidden.update(model.lower() for model in binding.runtime_models)
    for row in rows:
        surface = _learned_surface(row)
        leaked = sorted(fragment for fragment in forbidden if fragment in surface)
        if leaked:
            raise RecoveryAdmissionError(
                f"learned recovery surface contains forbidden text: {leaked}"
            )
        if row.get("terminalbench") is not True or row.get(
            "evaluation_excluded"
        ) is not True:
            raise RecoveryAdmissionError("benchmark provenance was not preserved")

    counts = Counter(row["action"]["action"] for row in rows)
    if counts.get("replan", 0) != 0 or counts.get("handoff", 0) < minimum:
        raise RecoveryAdmissionError("admitted boundaries do not prove live handoff")
    return RecoveryAdmission(
        rows=tuple(rows),
        recovery_count=len(recoveries),
        action_counts=dict(sorted(counts.items())),
        evaluation_excluded_task_ids=tuple(sorted(task_ids)),
    )
