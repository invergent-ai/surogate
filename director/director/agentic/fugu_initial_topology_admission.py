"""Admit pool-neutral initial topology labels from exact outcome evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from director.agentic.fugu_branchpoint_analog_admission import admit_analog_band
from director.agentic.fugu_recovery_admission import convert_recovery_admission
from ultra.live_control import parse_control_action
from ultra.live_control_trajectory import convert_successful_harbor_trajectory
from ultra.pool_binding import load_pool_binding


ADMISSION_VERSION = "fugu_initial_topology_admission_v1"
YUNWU_API_BASE = "https://yunwu.ai/v1"


class InitialTopologyAdmissionError(ValueError):
    """Outcome evidence cannot support a trusted initial topology label."""


@dataclass(frozen=True)
class InitialTopologyAdmission:
    rows: tuple[dict[str, Any], ...]
    action_counts: dict[str, int]
    correction_count: int
    solo_sufficient_count: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InitialTopologyAdmissionError(
            f"{label} must contain exactly {sorted(fields)}"
        )
    return value


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise InitialTopologyAdmissionError(f"{label} must be a non-empty path")
    path = Path(raw)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise InitialTopologyAdmissionError(f"{label} escapes project root") from exc
    if not path.is_file():
        raise InitialTopologyAdmissionError(f"{label} is missing")
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_file(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise InitialTopologyAdmissionError(f"{label} hash drift")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InitialTopologyAdmissionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise InitialTopologyAdmissionError(f"{label} must be an object")
    return value


def _read_one_jsonl(path: Path, *, label: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise InitialTopologyAdmissionError(f"{label} must contain one row")
    return rows[0]


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _learned_surface(row: dict[str, Any]) -> str:
    return json.dumps(
        {"state": row.get("state"), "action": row.get("action")},
        sort_keys=True,
        ensure_ascii=True,
    ).lower()


def _initial_recovery_rows(
    *, root: Path, recovery_admission_path: Path, pool_fingerprint: str
) -> list[dict[str, Any]]:
    validated = convert_recovery_admission(recovery_admission_path, root=root)
    manifest = _read_json(recovery_admission_path, label="recovery admission")
    rows: list[dict[str, Any]] = []
    for recovery in manifest["recoveries"]:
        campaign_path = _project_file(
            root, recovery["campaign"]["path"], label="recovery campaign"
        )
        ledger_path = _project_file(
            root, recovery["ledger"]["path"], label="recovery ledger"
        )
        campaign = _read_json(campaign_path, label="recovery campaign")
        ledger = _read_one_jsonl(ledger_path, label="recovery ledger")
        job = campaign["jobs"][0]
        converted = convert_successful_harbor_trajectory(
            collection_id=recovery["collection_id"],
            task_id=job["task_id"],
            original_task=(Path(job["task_dir"]) / "instruction.md").read_text(
                encoding="utf-8"
            ),
            pool_fingerprint=pool_fingerprint,
            registered_action=job["action"],
            result_path=root / ledger["result_path"],
            route_log_path=root / ledger["route_log_path"],
            trajectory_path=root / ledger["trajectory_path"],
            paid_call_limit=120,
        )
        if not converted or converted[0]["state"].get("workflow_id") is not None:
            raise InitialTopologyAdmissionError("recovery lacks an initial boundary")
        source_action = parse_control_action(json.dumps(job["action"]))
        clean_action = parse_control_action(json.dumps(recovery["sanitized_workflow"]))
        if (
            source_action.action != "replan"
            or clean_action.action != "replan"
            or len(source_action.steps) != len(clean_action.steps)
            or any(
                source.worker_id != clean.worker_id or source.access != clean.access
                for source, clean in zip(
                    source_action.steps, clean_action.steps, strict=True
                )
            )
        ):
            raise InitialTopologyAdmissionError(
                "sanitized correction changes the successful topology"
            )
        row = copy.deepcopy(converted[0])
        row["record_id"] += "__outcome_selected_correction"
        row["action"] = copy.deepcopy(recovery["sanitized_workflow"])
        row["terminalbench"] = True
        row["evaluation_excluded"] = True
        row["benchmark_source"] = "terminalbench21_recovery_training"
        row["label_status"] = "outcome_selected_topology_correction"
        row["provenance"].update(
            {
                "mechanism_id": recovery["mechanism_id"],
                "recovery_admission_sha256": sha256(recovery_admission_path),
                "campaign_sha256": recovery["campaign"]["sha256"],
                "ledger_sha256": recovery["ledger"]["sha256"],
                "outcome_selected": True,
                "causal_superiority_claim": False,
                "hindsight_text_removed": True,
                "permanently_evaluation_excluded": True,
            }
        )
        rows.append(row)
    if len(rows) != validated.recovery_count:
        raise InitialTopologyAdmissionError("recovery correction count drift")
    return rows


def _solo_sufficient_row(
    *,
    root: Path,
    analog_admission_path: Path,
    campaign_path: Path,
    ledger_path: Path,
    pool_fingerprint: str,
) -> dict[str, Any]:
    analog = admit_analog_band(analog_admission_path, root=root)
    admitted = next(
        (task for task in analog.tasks if task.task_id.endswith("durable-release-hook")),
        None,
    )
    if admitted is None:
        raise InitialTopologyAdmissionError("durable analog is not admitted")
    campaign = _read_json(campaign_path, label="analog solo campaign")
    ledger = _read_one_jsonl(ledger_path, label="analog solo ledger")
    job = campaign["jobs"][0]
    result_path = root / str(ledger.get("result_path") or "")
    route_path = root / str(ledger.get("route_log_path") or "")
    trajectory_path = root / str(ledger.get("trajectory_path") or "")
    for path, expected, label in (
        (result_path, ledger.get("result_sha256"), "analog result"),
        (route_path, ledger.get("route_log_sha256"), "analog routes"),
        (trajectory_path, ledger.get("trajectory_sha256"), "analog trajectory"),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise InitialTopologyAdmissionError(f"{label} hash drift")
    result = _read_json(result_path, label="analog result")
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    if (
        campaign.get("version") != "fugu_branchpoint_analog_probe_v1"
        or campaign.get("training_eligible") is not False
        or campaign.get("terminalbench_tasks") != 0
        or campaign.get("external_provider") != YUNWU_API_BASE
        or campaign.get("global_paid_call_ceiling") != 120
        or campaign.get("worker_timeout_seconds") != 600.0
        or campaign.get("provider_retries") != 0
        or campaign.get("task_attempts") != 1
        or campaign.get("task_retries") != 0
        or campaign.get("maximum_jobs") != 1
        or ledger.get("status") != "accepted"
        or ledger.get("reward") != 1.0
        or ledger.get("rejection_reason") is not None
        or ledger.get("paid_worker_call_attempts") != 6
        or _reward(result) != 1.0
        or result.get("exception_info") is not None
        or metadata.get("worker_provider_base") != YUNWU_API_BASE
        or metadata.get("provider_owner_retries") != 0
        or metadata.get("provider_owner_retry_limit") != 0
        or metadata.get("completed_workflow_steps") != 1
        or metadata.get("completed_workflows") != 1
        or result.get("task_checksum") != admitted.task_checksum
    ):
        raise InitialTopologyAdmissionError("analog solo outcome drift")
    converted = convert_successful_harbor_trajectory(
        collection_id=ledger["collection_id"],
        task_id=job["task_id"],
        original_task=(Path(job["task_dir"]) / "instruction.md").read_text(
            encoding="utf-8"
        ),
        pool_fingerprint=pool_fingerprint,
        registered_action=job["action"],
        result_path=result_path,
        route_log_path=route_path,
        trajectory_path=trajectory_path,
        paid_call_limit=120,
    )
    if (
        not converted
        or converted[0]["state"].get("workflow_id") is not None
        or converted[0]["action"] != job["action"]
        or len(job["action"].get("steps", [])) != 1
    ):
        raise InitialTopologyAdmissionError("analog solo initial decision drift")
    row = copy.deepcopy(converted[0])
    row["record_id"] += "__verified_solo_sufficient"
    row["terminalbench"] = False
    row["evaluation_excluded"] = True
    row["benchmark_source"] = "train_only_branchpoint_analog"
    row["label_status"] = "verified_solo_sufficient_initial"
    row["provenance"].update(
        {
            "mechanism_id": job["mechanism_id"],
            "analog_admission_sha256": sha256(analog_admission_path),
            "campaign_sha256": sha256(campaign_path),
            "ledger_sha256": sha256(ledger_path),
            "outcome_selected": False,
            "causal_superiority_claim": False,
            "verified_solo_sufficient": True,
            "evaluation_excluded": True,
        }
    )
    return row


def admit_initial_topology_labels(
    manifest_path: Path, *, root: Path
) -> InitialTopologyAdmission:
    root = root.resolve()
    manifest = _object(
        _read_json(manifest_path, label="initial topology manifest"),
        fields={
            "version",
            "admission_id",
            "validator",
            "pool_binding",
            "recovery_admission",
            "analog_admission",
            "analog_solo_campaign",
            "analog_solo_ledger",
            "policy",
        },
        label="initial topology manifest",
    )
    if manifest["version"] != ADMISSION_VERSION:
        raise InitialTopologyAdmissionError("unsupported admission version")
    validator = _verified_file(root, manifest["validator"], label="validator")
    if validator.resolve() != Path(__file__).resolve():
        raise InitialTopologyAdmissionError("validator is not this implementation")
    binding_record = _object(
        manifest["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_file(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise InitialTopologyAdmissionError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
        or binding.provider_base != YUNWU_API_BASE
    ):
        raise InitialTopologyAdmissionError("pool binding identity drift")
    policy = _object(
        manifest["policy"],
        fields={
            "permit_outcome_selected_corrections",
            "permit_causal_superiority_claims",
            "remove_hindsight_text",
            "permanently_exclude_source_tasks_from_evaluation",
            "required_corrections",
            "required_solo_sufficient",
        },
        label="policy",
    )
    if policy != {
        "permit_outcome_selected_corrections": True,
        "permit_causal_superiority_claims": False,
        "remove_hindsight_text": True,
        "permanently_exclude_source_tasks_from_evaluation": True,
        "required_corrections": 2,
        "required_solo_sufficient": 1,
    }:
        raise InitialTopologyAdmissionError("initial topology policy drift")
    recovery_admission = _verified_file(
        root, manifest["recovery_admission"], label="recovery admission"
    )
    analog_admission = _verified_file(
        root, manifest["analog_admission"], label="analog admission"
    )
    analog_campaign = _verified_file(
        root, manifest["analog_solo_campaign"], label="analog solo campaign"
    )
    analog_ledger = _verified_file(
        root, manifest["analog_solo_ledger"], label="analog solo ledger"
    )
    corrections = _initial_recovery_rows(
        root=root,
        recovery_admission_path=recovery_admission,
        pool_fingerprint=binding.pool_fingerprint,
    )
    solo = _solo_sufficient_row(
        root=root,
        analog_admission_path=analog_admission,
        campaign_path=analog_campaign,
        ledger_path=analog_ledger,
        pool_fingerprint=binding.pool_fingerprint,
    )
    rows = [*corrections, solo]
    forbidden = {
        binding.provider_base.lower(),
        *[slot.runtime_model.lower() for slot in binding.slots],
        *[slot.model_alias.lower() for slot in binding.slots],
        "a prior current-pool attempt",
        "external verifier",
        "http 404",
        "expected pov-ray 2.2 file",
    }
    for row in rows:
        if row.get("pool_fingerprint") != binding.pool_fingerprint:
            raise InitialTopologyAdmissionError("row pool fingerprint drift")
        if row.get("evaluation_excluded") is not True:
            raise InitialTopologyAdmissionError("source task is not eval-excluded")
        leaked = sorted(token for token in forbidden if token in _learned_surface(row))
        if leaked:
            raise InitialTopologyAdmissionError(
                f"initial topology learned surface leaks identity/evidence: {leaked}"
            )
    statuses = Counter(row["label_status"] for row in rows)
    if (
        statuses["outcome_selected_topology_correction"]
        != policy["required_corrections"]
        or statuses["verified_solo_sufficient_initial"]
        != policy["required_solo_sufficient"]
    ):
        raise InitialTopologyAdmissionError("required topology label balance drift")
    return InitialTopologyAdmission(
        rows=tuple(rows),
        action_counts=dict(Counter(row["action"]["action"] for row in rows)),
        correction_count=statuses["outcome_selected_topology_correction"],
        solo_sufficient_count=statuses["verified_solo_sufficient_initial"],
    )
