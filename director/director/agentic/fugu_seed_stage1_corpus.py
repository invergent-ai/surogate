"""Fail-closed inventory for model-agnostic SEED Stage-1 trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultra.pool_binding import load_pool_binding
from ultra.seed_hindsight import validate_analyzer_trajectory


INVENTORY_VERSION = "fugu_seed_stage1_inventory_v1"
CANDIDATE_VERSION = "fugu_seed_stage1_analyzer_candidate_v1"
REPORT_VERSION = "fugu_seed_stage1_inventory_report_v1"


class SeedStage1CorpusError(ValueError):
    """The source evidence cannot support a trusted Stage-1 inventory."""


@dataclass(frozen=True)
class SeedStage1Inventory:
    candidates: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SeedStage1CorpusError(
            f"{label} must contain exactly {sorted(fields)}"
        )
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SeedStage1CorpusError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SeedStage1CorpusError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SeedStage1CorpusError(
                f"{label} line {number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise SeedStage1CorpusError(f"{label} line {number} must be an object")
        rows.append(value)
    return rows


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SeedStage1CorpusError(f"{label} path is invalid")
    candidate = Path(raw)
    path = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SeedStage1CorpusError(f"{label} escapes project root") from exc
    if not path.is_file():
        raise SeedStage1CorpusError(f"{label} is missing")
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_file(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise SeedStage1CorpusError(f"{label} hash drift")
    return path


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mutable_frozen_code_drift(campaign: dict[str, Any], *, root: Path) -> bool:
    frozen = campaign.get("frozen_code")
    if not isinstance(frozen, dict):
        return True
    for key, raw in frozen.items():
        if key.endswith("_sha256"):
            continue
        expected = frozen.get(f"{key}_sha256")
        try:
            path = _project_file(root, raw, label=f"historical frozen code {key}")
        except SeedStage1CorpusError:
            return True
        if not isinstance(expected, str) or sha256(path) != expected:
            return True
    return False


def _validated_initial_artifacts(
    *,
    root: Path,
    binding: Any,
    admission_path: Path,
    report_path: Path,
    rows_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    admission = _read_json(admission_path, label="initial topology admission")
    report = _read_json(report_path, label="initial topology report")
    rows = _read_jsonl(rows_path, label="initial topology rows")
    if (
        report.get("status") != "READY"
        or report.get("manifest")
        != {"path": str(admission_path.relative_to(root)), "sha256": sha256(admission_path)}
        or report.get("rows")
        != {"path": str(rows_path.relative_to(root)), "sha256": sha256(rows_path)}
        or report.get("row_count") != 3
        or report.get("action_counts") != {"replan": 3}
        or report.get("outcome_selected_corrections") != 2
        or report.get("verified_solo_sufficient") != 1
        or len(rows) != 3
    ):
        raise SeedStage1CorpusError("initial topology conversion report drift")
    binding_record = admission.get("pool_binding") or {}
    if (
        binding_record.get("sha256")
        != sha256(_project_file(root, binding_record.get("path"), label="source binding"))
        or binding_record.get("pool_fingerprint") != binding.pool_fingerprint
        or binding_record.get("pool_id") != binding.pool_id
    ):
        raise SeedStage1CorpusError("initial topology source binding drift")
    _verified_file(root, admission.get("validator"), label="historical validator")

    analog_admission_path = _verified_file(
        root, admission.get("analog_admission"), label="analog task admission"
    )
    analog_admission = _read_json(analog_admission_path, label="analog task admission")
    analog_campaign_path = _verified_file(
        root, admission.get("analog_solo_campaign"), label="analog solo campaign"
    )
    analog_ledger_path = _verified_file(
        root, admission.get("analog_solo_ledger"), label="analog solo ledger"
    )
    campaign = _read_json(analog_campaign_path, label="analog solo campaign")
    ledger_rows = _read_jsonl(analog_ledger_path, label="analog solo ledger")
    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or len(ledger_rows) != 1:
        raise SeedStage1CorpusError("analog solo source cardinality drift")
    job = jobs[0]
    ledger = ledger_rows[0]
    task_entries = analog_admission.get("tasks")
    admitted_task = next(
        (
            item
            for item in task_entries or []
            if item.get("task_id") == "fugu-train/durable-release-hook"
        ),
        None,
    )
    campaign_checks = {
        "version": campaign.get("version") == "fugu_branchpoint_analog_probe_v1",
        "not_direct_training": campaign.get("training_eligible") is False,
        "not_terminalbench": campaign.get("terminalbench_tasks") == 0,
        "provider": campaign.get("external_provider") == binding.provider_base,
        "fingerprint": campaign.get("pool_fingerprint") == binding.pool_fingerprint,
        "call_ceiling": campaign.get("global_paid_call_ceiling") == 120,
        "timeout": campaign.get("worker_timeout_seconds") == 600.0,
        "provider_retries": campaign.get("provider_retries") == 0,
        "task_retries": campaign.get("task_retries") == 0,
        "task_attempts": campaign.get("task_attempts") == 1,
        "maximum_jobs": campaign.get("maximum_jobs") == 1,
        "admitted_task": isinstance(admitted_task, dict),
    }
    failed = sorted(key for key, passed in campaign_checks.items() if not passed)
    if failed:
        raise SeedStage1CorpusError(f"analog solo campaign drift: {failed}")

    paths: dict[str, Path] = {}
    for path_key, hash_key, label in (
        ("result_path", "result_sha256", "analog result"),
        ("route_log_path", "route_log_sha256", "analog route log"),
        ("trajectory_path", "trajectory_sha256", "analog trajectory"),
    ):
        path = _project_file(root, ledger.get(path_key), label=label)
        if sha256(path) != ledger.get(hash_key):
            raise SeedStage1CorpusError(f"{label} hash drift")
        paths[path_key] = path
    result = _read_json(paths["result_path"], label="analog result")
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    result_checks = {
        "collection": ledger.get("collection_id") == job.get("collection_id"),
        "status": ledger.get("status") == "accepted",
        "reward": ledger.get("reward") == 1.0 and _reward(result) == 1.0,
        "rejection": ledger.get("rejection_reason") is None,
        "harbor": ledger.get("harbor_returncode") == 0,
        "exception": result.get("exception_info") is None,
        "task_checksum": result.get("task_checksum") == admitted_task.get("task_checksum"),
        "provider": ledger.get("worker_provider_base") == binding.provider_base,
        "models": tuple(ledger.get("worker_models") or ()) == binding.runtime_models,
        "fingerprint": ledger.get("pool_fingerprint") == binding.pool_fingerprint,
        "metadata_provider": metadata.get("worker_provider_base") == binding.provider_base,
        "metadata_models": tuple(metadata.get("worker_models") or ())
        == binding.runtime_models,
        "metadata_fingerprint": metadata.get("pool_fingerprint")
        == binding.pool_fingerprint,
        "registered_action": metadata.get("collection_registered_workflow")
        == job.get("action"),
        "calls": metadata.get("paid_worker_call_attempts")
        == ledger.get("paid_worker_call_attempts")
        == 6,
        "provider_retry_limit": metadata.get("provider_owner_retry_limit") == 0,
        "provider_retries": metadata.get("provider_owner_retries") == 0,
        "provider_failures": metadata.get("provider_failure_events") == [],
        "protocol": metadata.get("worker_protocol_errors") == 0,
        "planner": metadata.get("planner_failures") == 0,
        "live_control": metadata.get("live_control_failures") == 0,
        "workspace": metadata.get("blocked_workspace_commands") == 0,
        "steps": metadata.get("completed_workflow_steps") == 1,
        "workflows": metadata.get("completed_workflows") == 1,
    }
    failed = sorted(key for key, passed in result_checks.items() if not passed)
    if failed:
        raise SeedStage1CorpusError(f"analog solo result drift: {failed}")

    eligible_rows = [row for row in rows if row.get("terminalbench") is False]
    if len(eligible_rows) != 1:
        raise SeedStage1CorpusError("initial topology non-benchmark row count drift")
    source_row = eligible_rows[0]
    provenance = source_row.get("provenance") or {}
    task_dir = _project_file(
        root,
        str(Path(str(job.get("task_dir"))) / "instruction.md"),
        label="analog instruction",
    )
    if (
        source_row.get("action") != job.get("action")
        or ((source_row.get("state") or {}).get("original_task"))
        != task_dir.read_text(encoding="utf-8")
        or provenance.get("result_sha256") != ledger.get("result_sha256")
        or provenance.get("route_log_sha256") != ledger.get("route_log_sha256")
        or provenance.get("trajectory_sha256") != ledger.get("trajectory_sha256")
        or provenance.get("campaign_sha256") != sha256(analog_campaign_path)
        or provenance.get("ledger_sha256") != sha256(analog_ledger_path)
    ):
        raise SeedStage1CorpusError("analyzer source row provenance drift")
    return rows, ledger_rows, _mutable_frozen_code_drift(campaign, root=root)


def _validated_causal_artifacts(
    *,
    root: Path,
    binding: Any,
    admission_path: Path,
    report_path: Path,
) -> tuple[list[dict[str, Any]], bool]:
    admission = _read_json(admission_path, label="causal coordination admission")
    admission_report = _read_json(report_path, label="causal admission report")
    policy = admission.get("policy") or {}
    if (
        admission_report.get("status") != "ADMITTED_OBSERVED_COORDINATION_LIFT"
        or admission_report.get("manifest")
        != {"path": str(admission_path.relative_to(root)), "sha256": sha256(admission_path)}
        or admission_report.get("training_conversion_locked") is not True
        or admission_report.get("evaluation_excluded") is not True
        or policy.get("training_conversion_locked") is not True
        or policy.get("minimum_train_pairs_before_learning") != 6
        or policy.get("require_slot_profile_permutation_before_learning") is not True
    ):
        raise SeedStage1CorpusError("causal admission policy or report drift")
    binding_record = admission.get("pool_binding") or {}
    binding_path = _project_file(root, binding_record.get("path"), label="causal binding")
    if (
        sha256(binding_path) != binding_record.get("sha256")
        or binding_record.get("pool_fingerprint") != binding.pool_fingerprint
        or binding_record.get("pool_id") != binding.pool_id
    ):
        raise SeedStage1CorpusError("causal source binding drift")
    campaign_path = _verified_file(root, admission.get("campaign"), label="causal campaign")
    pair_report_path = _verified_file(
        root, admission.get("pair_report"), label="causal pair report"
    )
    ledger_path = _verified_file(root, admission.get("ledger"), label="causal ledger")
    campaign = _read_json(campaign_path, label="causal campaign")
    pair_report = _read_json(pair_report_path, label="causal pair report")
    ledger = _read_jsonl(ledger_path, label="causal ledger")
    if (
        pair_report.get("status") != "complete_observed_coordination_lift"
        or pair_report.get("valid_observed_coordination_lift_pair") is not True
        or pair_report.get("training_conversion_locked") is not True
        or pair_report.get("external_provider") != binding.provider_base
        or len(ledger) != 2
    ):
        raise SeedStage1CorpusError("causal pair report drift")
    for row in ledger:
        if (
            not str(row.get("task_id") or "").startswith("terminalbench21__")
            or row.get("worker_provider_base") != binding.provider_base
            or tuple(row.get("worker_models") or ()) != binding.runtime_models
            or row.get("pool_fingerprint") != binding.pool_fingerprint
            or row.get("runtime_revision")
            != "20260720-r56-anonymous-planning-constraints"
        ):
            raise SeedStage1CorpusError("causal ledger identity or provenance drift")
        for path_key, hash_key, label in (
            ("result_path", "result_sha256", "causal result"),
            ("route_log_path", "route_log_sha256", "causal route log"),
            ("trajectory_path", "trajectory_sha256", "causal trajectory"),
        ):
            path = _project_file(root, row.get(path_key), label=label)
            if sha256(path) != row.get(hash_key):
                raise SeedStage1CorpusError(f"{label} hash drift")
        result = _read_json(
            _project_file(root, row.get("result_path"), label="causal result"),
            label="causal result",
        )
        if result.get("exception_info") is not None or _reward(result) != row.get("reward"):
            raise SeedStage1CorpusError("causal result outcome drift")
    return ledger, _mutable_frozen_code_drift(campaign, root=root)


def _forbidden_identity_terms(binding: Any) -> tuple[str, ...]:
    values = {binding.provider_base}
    for slot in binding.slots:
        values.update(
            {
                slot.runtime_model,
                slot.model_alias,
                slot.training_name,
            }
        )
    return tuple(sorted(value for value in values if value))


def _profile_permutation(binding: Any, rotation: int) -> tuple[list[dict[str, Any]], dict[int, int]]:
    slots = list(binding.slots)
    rotated = slots[rotation:] + slots[:rotation]
    worker_to_profile = {
        slot.worker_id: profile_id for profile_id, slot in enumerate(rotated)
    }
    profiles = [
        {
            "capability_profile_id": profile_id,
            "role_prior": list(slot.role_prior),
            "tool_tags": ["filesystem", "terminal", "test_runner"],
        }
        for profile_id, slot in enumerate(rotated)
    ]
    return profiles, worker_to_profile


def _sanitized_action(
    action: dict[str, Any], *, worker_to_profile: dict[int, int]
) -> dict[str, Any]:
    if action.get("action") != "replan" or not isinstance(action.get("steps"), list):
        raise SeedStage1CorpusError("Stage-1 source must have an initial replan action")
    steps: list[dict[str, Any]] = []
    for position_id, source in enumerate(action["steps"]):
        if not isinstance(source, dict):
            raise SeedStage1CorpusError("initial action step must be an object")
        worker_id = source.get("worker_id")
        if worker_id not in worker_to_profile:
            raise SeedStage1CorpusError("initial action references an unknown profile")
        access = source.get("access")
        if not isinstance(access, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in access
        ):
            raise SeedStage1CorpusError("initial action access list is invalid")
        subtask = source.get("subtask")
        if not isinstance(subtask, str) or not subtask.strip():
            raise SeedStage1CorpusError("initial action subtask is invalid")
        steps.append(
            {
                "position_id": position_id,
                "capability_profile_id": worker_to_profile[worker_id],
                "subtask": subtask.strip(),
                "access": access,
            }
        )
    reason = action.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise SeedStage1CorpusError("initial action reason is invalid")
    return {"action": "replan", "reason": reason.strip(), "steps": steps}


def _route_event(source: Any) -> str:
    events = {
        "conductor_workflow": "position_start",
        "workflow_step_start": "position_start",
        "workflow_agent_continuation": "position_continue",
    }
    if source not in events:
        raise SeedStage1CorpusError(f"unsupported route source: {source!r}")
    return events[source]


def _sanitized_observations(
    route_rows: list[dict[str, Any]],
    *,
    action: dict[str, Any],
    worker_to_profile: dict[int, int],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for event_id, route in enumerate(route_rows, 1):
        worker_id = route.get("worker_id")
        if worker_id not in worker_to_profile:
            raise SeedStage1CorpusError("route row references an unknown profile")
        step_index = route.get("workflow_step_index")
        if (
            not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or not 1 <= step_index <= len(action["steps"])
        ):
            raise SeedStage1CorpusError("route row workflow position is invalid")
        position_id = step_index - 1
        source_step = action["steps"][position_id]
        if (
            route.get("subtask") != source_step.get("subtask")
            or route.get("workflow_access") != source_step.get("access")
            or worker_id != source_step.get("worker_id")
        ):
            raise SeedStage1CorpusError("route row does not match the registered topology")
        progress = route.get("reported_progress")
        if not isinstance(progress, dict):
            raise SeedStage1CorpusError("route row is missing reported progress")
        phase = progress.get("phase")
        evidence = progress.get("evidence")
        if not isinstance(phase, str) or not isinstance(evidence, str):
            raise SeedStage1CorpusError("route progress must contain text phase and evidence")
        artifacts = route.get("reported_artifacts")
        if not isinstance(artifacts, list):
            raise SeedStage1CorpusError("route artifacts must be a list")
        observations.append(
            {
                "event_id": event_id,
                "decision_id": 1,
                "event": _route_event(route.get("route_source")),
                "position_id": position_id,
                "capability_profile_id": worker_to_profile[worker_id],
                "access": copy.deepcopy(source_step["access"]),
                "subtask": source_step["subtask"],
                "private_turn": route.get("agent_private_turn"),
                "phase": phase.strip(),
                "evidence": evidence.strip(),
                "reported_artifacts": copy.deepcopy(artifacts),
                "terminal_ready": route.get("terminal_ready") is True,
            }
        )
    if not observations:
        raise SeedStage1CorpusError("completed trajectory has no route observations")
    return observations


def _candidate_variants(
    *,
    binding: Any,
    source_row: dict[str, Any],
    route_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task = ((source_row.get("state") or {}).get("original_task"))
    action = source_row.get("action")
    if not isinstance(task, str) or not task.strip() or not isinstance(action, dict):
        raise SeedStage1CorpusError("source row lacks task or action")
    independent_source_id = _stable_id(
        {
            "record_id": source_row.get("record_id"),
            "result_sha256": (source_row.get("provenance") or {}).get("result_sha256"),
            "route_log_sha256": (source_row.get("provenance") or {}).get(
                "route_log_sha256"
            ),
        }
    )
    forbidden = _forbidden_identity_terms(binding)
    candidates: list[dict[str, Any]] = []
    profile_count = len(binding.slots)
    seen_mappings: set[tuple[int, ...]] = set()
    for rotation in range(profile_count):
        profiles, worker_to_profile = _profile_permutation(binding, rotation)
        mapping = tuple(worker_to_profile[index] for index in range(profile_count))
        seen_mappings.add(mapping)
        clean_action = _sanitized_action(action, worker_to_profile=worker_to_profile)
        trajectory = {
            "task": task.strip(),
            "available_capability_profiles": profiles,
            "decisions": [
                {
                    "decision_id": 1,
                    "decision_origin": "registered_initial_topology_observed_to_completion",
                    **clean_action,
                }
            ],
            "observations": _sanitized_observations(
                route_rows,
                action=action,
                worker_to_profile=worker_to_profile,
            ),
            "outcome": "success",
        }
        validate_analyzer_trajectory(
            trajectory,
            forbidden_identity_terms=forbidden,
        )
        candidate = {
            "version": CANDIDATE_VERSION,
            "candidate_id": _stable_id(
                {
                    "independent_source_id": independent_source_id,
                    "profile_permutation_index": rotation,
                    "trajectory": trajectory,
                }
            ),
            "independent_source_id": independent_source_id,
            "split": "train",
            "analysis_only": True,
            "action_label_eligible": False,
            "profile_permutation_index": rotation,
            "trajectory": trajectory,
        }
        rendered = json.dumps(candidate, ensure_ascii=True, sort_keys=True).casefold()
        leaked = sorted(term for term in forbidden if term.casefold() in rendered)
        if leaked:
            raise SeedStage1CorpusError(
                f"analyzer candidate leaks pool identity: {leaked}"
            )
        candidates.append(candidate)
    if len(seen_mappings) != profile_count:
        raise SeedStage1CorpusError("profile permutation coverage is incomplete")
    return candidates


def build_seed_stage1_inventory(
    manifest_path: Path, *, root: Path
) -> SeedStage1Inventory:
    root = root.resolve()
    manifest = _object(
        _read_json(manifest_path, label="Stage-1 inventory manifest"),
        fields={"version", "validator", "pool_binding", "sources", "policy"},
        label="Stage-1 inventory manifest",
    )
    if manifest["version"] != INVENTORY_VERSION:
        raise SeedStage1CorpusError("unsupported Stage-1 inventory version")
    validator = _verified_file(root, manifest["validator"], label="validator")
    if validator.resolve() != Path(__file__).resolve():
        raise SeedStage1CorpusError("validator is not this implementation")

    binding_record = _object(
        manifest["pool_binding"],
        fields={"path", "sha256", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_file(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise SeedStage1CorpusError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if binding.pool_fingerprint != binding_record["pool_fingerprint"]:
        raise SeedStage1CorpusError("pool binding fingerprint drift")

    sources = _object(
        manifest["sources"],
        fields={
            "initial_topology_admission",
            "initial_topology_report",
            "initial_topology_rows",
            "causal_coordination_admission",
            "causal_coordination_report",
        },
        label="sources",
    )
    initial_path = _verified_file(
        root, sources["initial_topology_admission"], label="initial topology admission"
    )
    initial_report_path = _verified_file(
        root, sources["initial_topology_report"], label="initial topology report"
    )
    initial_rows_path = _verified_file(
        root, sources["initial_topology_rows"], label="initial topology rows"
    )
    causal_path = _verified_file(
        root,
        sources["causal_coordination_admission"],
        label="causal coordination admission",
    )
    causal_report_path = _verified_file(
        root,
        sources["causal_coordination_report"],
        label="causal coordination report",
    )

    policy = _object(
        manifest["policy"],
        fields={
            "exclude_terminalbench",
            "exclude_holdout",
            "exclude_oracle",
            "minimum_causal_train_pairs",
            "minimum_clean_train_trajectories",
            "minimum_distinct_tasks",
            "minimum_failures",
            "minimum_profile_permutations_per_source",
            "minimum_successes",
            "require_current_pool",
        },
        label="policy",
    )
    expected_policy = {
        "exclude_terminalbench": True,
        "exclude_holdout": True,
        "exclude_oracle": True,
        "minimum_causal_train_pairs": 6,
        "minimum_clean_train_trajectories": 12,
        "minimum_distinct_tasks": 6,
        "minimum_failures": 3,
        "minimum_profile_permutations_per_source": len(binding.slots),
        "minimum_successes": 3,
        "require_current_pool": True,
    }
    if policy != expected_policy:
        raise SeedStage1CorpusError("Stage-1 admission policy drift")

    initial_rows, analog_ledger, initial_runtime_drift = _validated_initial_artifacts(
        root=root,
        binding=binding,
        admission_path=initial_path,
        report_path=initial_report_path,
        rows_path=initial_rows_path,
    )
    causal_ledger, causal_runtime_drift = _validated_causal_artifacts(
        root=root,
        binding=binding,
        admission_path=causal_path,
        report_path=causal_report_path,
    )

    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    independent_sources: list[dict[str, str]] = []
    for row in initial_rows:
        record_id = str(row.get("record_id") or "")
        task_id = str(row.get("task_id") or "")
        if row.get("terminalbench") is True:
            exclusions.append(
                {
                    "source": record_id,
                    "task_id": task_id,
                    "reason": "terminalbench_derived",
                }
            )
            continue
        if row.get("benchmark_source") != "train_only_branchpoint_analog":
            raise SeedStage1CorpusError("non-benchmark source is not train-only")
        collection_id = (row.get("provenance") or {}).get("collection_id")
        matches = [item for item in analog_ledger if item.get("collection_id") == collection_id]
        if len(matches) != 1:
            raise SeedStage1CorpusError("analog source ledger match is not unique")
        ledger = matches[0]
        route_path = _project_file(root, ledger.get("route_log_path"), label="route log")
        if sha256(route_path) != ledger.get("route_log_sha256"):
            raise SeedStage1CorpusError("route log hash drift")
        variants = _candidate_variants(
            binding=binding,
            source_row=row,
            route_rows=_read_jsonl(route_path, label="route log"),
        )
        candidates.extend(variants)
        independent_sources.append(
            {
                "independent_source_id": variants[0]["independent_source_id"],
                "task_id": task_id,
                "outcome": "success",
            }
        )

    for row in causal_ledger:
        task_id = str(row.get("task_id") or "")
        if not task_id.startswith("terminalbench21__"):
            raise SeedStage1CorpusError("causal pair task provenance drift")
        exclusions.append(
            {
                "source": str(row.get("collection_id") or ""),
                "task_id": task_id,
                "reason": "terminalbench_derived_and_conversion_locked",
            }
        )

    independent_count = len(independent_sources)
    successes = sum(row["outcome"] == "success" for row in independent_sources)
    failures = sum(row["outcome"] == "failure" for row in independent_sources)
    distinct_tasks = len({row["task_id"] for row in independent_sources})
    permutation_counts: dict[str, int] = {}
    for candidate in candidates:
        source_id = candidate["independent_source_id"]
        permutation_counts[source_id] = permutation_counts.get(source_id, 0) + 1
    permutation_gate = bool(permutation_counts) and all(
        count >= policy["minimum_profile_permutations_per_source"]
        for count in permutation_counts.values()
    )
    gates = {
        "causal_train_pairs": 0 >= policy["minimum_causal_train_pairs"],
        "clean_train_trajectories": independent_count
        >= policy["minimum_clean_train_trajectories"],
        "distinct_tasks": distinct_tasks >= policy["minimum_distinct_tasks"],
        "failures": failures >= policy["minimum_failures"],
        "identity_free_candidates": all(
            validate_analyzer_trajectory(
                candidate["trajectory"],
                forbidden_identity_terms=_forbidden_identity_terms(binding),
            )
            for candidate in candidates
        ),
        "profile_permutation_coverage": permutation_gate,
        "successes": successes >= policy["minimum_successes"],
    }
    training_authorized = bool(gates) and all(gates.values())
    report = {
        "version": REPORT_VERSION,
        "verdict": (
            "STAGE1_CORPUS_READY"
            if training_authorized
            else "INSUFFICIENT_ADMITTED_TRAJECTORIES_NO_TRAINING"
        ),
        "training_authorized": training_authorized,
        "pool_fingerprint": binding.pool_fingerprint,
        "policy": copy.deepcopy(policy),
        "counts": {
            "authoritative_completed_trajectories_reviewed": len(initial_rows)
            + len(causal_ledger),
            "eligible_independent_train_trajectories": independent_count,
            "analyzer_candidate_variants": len(candidates),
            "excluded_trajectories": len(exclusions),
            "distinct_tasks": distinct_tasks,
            "successes": successes,
            "failures": failures,
            "eligible_causal_train_pairs": 0,
        },
        "gates": gates,
        "exclusions": exclusions,
        "source_validation": {
            "basis": "immutable_conversion_rows_plus_result_route_trajectory_hash_chain",
            "historical_validator_reexecuted": False,
            "mutable_runtime_source_drift_detected": initial_runtime_drift
            or causal_runtime_drift,
            "candidate_source_artifacts_verified": True,
        },
        "candidate_sha256": _stable_id(candidates),
        "external_calls_made": 0,
        "paid_calls_made": 0,
        "optimizer_steps": 0,
        "candidate_checkpoint_created": False,
    }
    return SeedStage1Inventory(candidates=tuple(candidates), report=report)


def write_seed_stage1_inventory(
    inventory: SeedStage1Inventory, *, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        for row in inventory.candidates
    )
    (output_dir / "analyzer_candidates.jsonl").write_text(rows, encoding="utf-8")
    (output_dir / "inventory_report.json").write_text(
        json.dumps(inventory.report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
