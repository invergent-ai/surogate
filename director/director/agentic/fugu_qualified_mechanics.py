"""Fail-closed admission and conversion for pool-specific agentic trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from director.agentic.fugu_mechanics_runner import registered_templates
from director.agentic.fugu_mechanics_terminal import (
    load_mechanics_pool,
    mechanics_pool_fingerprint,
)
from director.agentic.prepared_index_test_protection import split_environment_setup
from ultra.live_control_trajectory import convert_successful_harbor_trajectory


CAMPAIGN_VERSION = "fugu_qualified_mechanics_campaign_v1"


class QualifiedMechanicsError(ValueError):
    """A campaign or trajectory cannot support trusted conductor training."""


@dataclass(frozen=True)
class QualifiedConversion:
    source_rows: tuple[dict[str, Any], ...]
    guardian_rows: tuple[dict[str, Any], ...]
    matched_pairs: int
    coordination_wins: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QualifiedMechanicsError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise QualifiedMechanicsError(f"{label} must be a non-empty path")
    path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise QualifiedMechanicsError(f"{label} escapes the project root") from exc
    if not path.is_file():
        raise QualifiedMechanicsError(f"{label} does not exist: {path}")
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QualifiedMechanicsError(
                f"{path} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise QualifiedMechanicsError(f"{path} line {line_number} is not an object")
        rows.append(row)
    return rows


def _load_campaign(path: Path, root: Path) -> tuple[dict[str, Any], Any, Path, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    campaign = _object(
        raw,
        fields={
            "version",
            "campaign_id",
            "pool_manifest",
            "admission_manifest",
            "runtime_revision",
            "runner_revision",
            "ledgers",
            "matched_pairs",
            "minimum_coordination_wins",
        },
        label="campaign",
    )
    if campaign["version"] != CAMPAIGN_VERSION:
        raise QualifiedMechanicsError("unsupported qualified campaign version")
    for key in ("campaign_id", "runtime_revision", "runner_revision"):
        if not isinstance(campaign[key], str) or not campaign[key].strip():
            raise QualifiedMechanicsError(f"campaign.{key} must be non-empty text")

    pool_record = _object(
        campaign["pool_manifest"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="campaign.pool_manifest",
    )
    pool_path = _project_file(root, pool_record["path"], label="pool manifest")
    if sha256(pool_path) != pool_record["sha256"]:
        raise QualifiedMechanicsError("pool manifest hash drift")
    pool = load_mechanics_pool(pool_path)
    if pool.pool_id != pool_record["pool_id"]:
        raise QualifiedMechanicsError("pool ID differs from frozen campaign")
    if mechanics_pool_fingerprint(pool) != pool_record["pool_fingerprint"]:
        raise QualifiedMechanicsError("pool fingerprint differs from frozen campaign")

    admission_record = _object(
        campaign["admission_manifest"],
        fields={"path", "sha256"},
        label="campaign.admission_manifest",
    )
    admission_path = _project_file(
        root, admission_record["path"], label="admission manifest"
    )
    if sha256(admission_path) != admission_record["sha256"]:
        raise QualifiedMechanicsError("admission manifest hash drift")
    return campaign, pool, pool_path, admission_path


def _admitted_tasks(admission_path: Path, root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(admission_path.read_text(encoding="utf-8"))
    raw_tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(raw_tasks, list):
        raise QualifiedMechanicsError("admission manifest has no tasks list")
    admitted: dict[str, dict[str, Any]] = {}
    for row in raw_tasks:
        if not isinstance(row, dict) or not isinstance(row.get("task_name"), str):
            raise QualifiedMechanicsError("admission manifest contains an invalid task")
        if row.get("status") != "admitted" or row.get("split") != "train":
            continue
        name = row["task_name"]
        if name in admitted:
            raise QualifiedMechanicsError(f"admission manifest duplicates {name}")
        oracle = row.get("oracle")
        baseline = row.get("unchanged_sanitized_baseline")
        if not isinstance(oracle, dict) or oracle.get("reward") != 1.0:
            raise QualifiedMechanicsError(f"{name} lacks a passing oracle")
        if (
            not isinstance(baseline, dict)
            or baseline.get("reward") != 0.0
            or baseline.get("provider_calls") != 0
            or baseline.get("paid_calls") != 0
            or baseline.get("git_commit_count") != 1
            or baseline.get("git_remotes") != []
        ):
            raise QualifiedMechanicsError(
                f"{name} lacks a clean failing sanitized baseline"
            )
        task_dir = Path(str(row.get("task_dir", "")))
        task_dir = (root / task_dir).resolve() if not task_dir.is_absolute() else task_dir.resolve()
        try:
            task_dir.relative_to(root.resolve())
        except ValueError as exc:
            raise QualifiedMechanicsError(f"{name} task path escapes project root") from exc
        if not task_dir.is_dir() or tree_sha256(task_dir) != row.get("task_tree_sha256"):
            raise QualifiedMechanicsError(f"{name} task tree drift")
        admitted[name] = {**row, "resolved_task_dir": task_dir}
    if not admitted:
        raise QualifiedMechanicsError("admission manifest has no train tasks")
    return admitted


def _validate_result(
    row: dict[str, Any],
    *,
    root: Path,
    campaign: dict[str, Any],
    pool: Any,
    admitted: dict[str, dict[str, Any]],
    templates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    collection_id = row.get("collection_id")
    task_name = row.get("task_name")
    template_id = row.get("template_id")
    if not all(isinstance(value, str) and value for value in (collection_id, task_name, template_id)):
        raise QualifiedMechanicsError("ledger row lacks collection/task/template identity")
    if task_name not in admitted:
        raise QualifiedMechanicsError(f"{collection_id} is not a train-admitted task")
    if template_id not in templates:
        raise QualifiedMechanicsError(f"{collection_id} uses an unregistered template")
    if row.get("runner_revision") != campaign["runner_revision"]:
        raise QualifiedMechanicsError(f"{collection_id} runner revision drift")
    if row.get("status") != "graded" or row.get("harbor_returncode") != 0:
        raise QualifiedMechanicsError(f"{collection_id} is not a successful Harbor grade")
    reward = row.get("reward")
    if reward not in {0, 0.0, 1, 1.0} or isinstance(reward, bool):
        raise QualifiedMechanicsError(f"{collection_id} has an invalid reward")
    if row.get("exception_type") is not None:
        raise QualifiedMechanicsError(f"{collection_id} has a Harbor exception")
    if row.get("worker_calls_are_paid") is not False:
        raise QualifiedMechanicsError(f"{collection_id} is not an unpaid local run")
    if row.get("mechanics_pool_id") != pool.pool_id:
        raise QualifiedMechanicsError(f"{collection_id} pool ID drift")
    if row.get("live_control_failures") != 0 or row.get("protected_test_restores") != 0:
        raise QualifiedMechanicsError(f"{collection_id} violated runtime safety gates")
    expected_dir = admitted[task_name]["resolved_task_dir"]
    result_task_dir = Path(str(row.get("task_dir", "")))
    if not result_task_dir.is_absolute():
        result_task_dir = root / result_task_dir
    if result_task_dir.resolve() != expected_dir:
        raise QualifiedMechanicsError(f"{collection_id} task path drift")

    paths = {
        key: _project_file(root, row.get(key), label=f"{collection_id}.{key}")
        for key in ("result_path", "route_log_path", "trajectory_path")
    }
    result = json.loads(paths["result_path"].read_text(encoding="utf-8"))
    result_reward = (
        ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    )
    if (
        isinstance(result_reward, bool)
        or not isinstance(result_reward, (int, float))
        or float(result_reward) != float(reward)
    ):
        raise QualifiedMechanicsError(f"{collection_id} ledger/result reward mismatch")
    metadata = ((result.get("agent_result") or {}).get("metadata")) or {}
    action = templates[template_id]
    sanitation = metadata.get("prepared_git_history_sanitization") or {}
    workspace_root = metadata.get("workspace_root")
    expected_fingerprint = mechanics_pool_fingerprint(pool)
    failure_events = metadata.get("provider_failure_events")
    if not isinstance(failure_events, list) or any(
        not isinstance(event, dict) for event in failure_events
    ):
        raise QualifiedMechanicsError(f"{collection_id} failure evidence is invalid")
    protocol_only_failure = bool(failure_events) and float(reward) == 0.0 and all(
        event.get("failure_kind") == "owner_call_failed_without_retry"
        and event.get("error_type") == "RuntimeError"
        and "produced no executable content" in str(event.get("error", ""))
        for event in failure_events
    )
    if failure_events and not protocol_only_failure:
        raise QualifiedMechanicsError(
            f"{collection_id} contains a non-protocol provider failure"
        )
    if (
        metadata.get("collection_id") != collection_id
        or metadata.get("collection_fixed_workflow") != action
        or metadata.get("mechanics_pool_id") != pool.pool_id
        or metadata.get("pool_fingerprint") != expected_fingerprint
        or metadata.get("worker_calls_are_paid") is not False
        or metadata.get("collection_is_product_candidate") is not False
        or metadata.get("runtime_revision") != campaign["runtime_revision"]
        or metadata.get("live_control_failures") != 0
        or metadata.get("provider_owner_retries") != 0
        or metadata.get("protected_test_restores") != []
        or metadata.get("workspace_snapshot_ready") is not True
        or metadata.get("workspace_recoveries") != 0
        or metadata.get("workspace_recovery_failures") != 0
        or metadata.get("workspace_cleanup_failures") != 0
        or sanitation.get("commit_count") != 1
        or sanitation.get("remotes") != []
        or sanitation.get("repo") != workspace_root
        or metadata.get("protected_test_repo") != workspace_root
        or (metadata.get("workspace_snapshot_summary") or {}).get("workspace_root")
        != workspace_root
    ):
        raise QualifiedMechanicsError(f"{collection_id} result attestation failed")
    attempts = metadata.get("paid_worker_call_attempts")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
        or attempts > 120
        or attempts != row.get("worker_call_attempts")
    ):
        raise QualifiedMechanicsError(f"{collection_id} call-count attestation failed")
    if tuple(metadata.get("worker_models") or ()) != tuple(
        slot.served_model for slot in pool.slots
    ):
        raise QualifiedMechanicsError(f"{collection_id} worker model attestation failed")
    completed = metadata.get("completed_workflow_steps")
    if completed != row.get("completed_workflow_steps"):
        raise QualifiedMechanicsError(
            f"{collection_id} completed-step attestation failed"
        )
    if reward == 1.0 and completed != len(action["steps"]):
        raise QualifiedMechanicsError(
            f"{collection_id} passed without completing its registered topology"
        )
    return {
        **row,
        **paths,
        "task_dir": expected_dir,
        "result": result,
        "metadata": metadata,
        "reward": float(reward),
        "training_eligible": not failure_events and completed == len(action["steps"]),
    }


def convert_qualified_campaign(campaign_path: Path, *, root: Path) -> QualifiedConversion:
    root = root.resolve()
    campaign, pool, _, admission_path = _load_campaign(campaign_path.resolve(), root)
    admitted = _admitted_tasks(admission_path, root)
    templates = registered_templates()

    ledgers = campaign["ledgers"]
    if not isinstance(ledgers, list) or not ledgers:
        raise QualifiedMechanicsError("campaign must freeze at least one ledger")
    raw_results: list[dict[str, Any]] = []
    for index, raw_ledger in enumerate(ledgers):
        ledger = _object(
            raw_ledger,
            fields={"path", "sha256"},
            label=f"campaign.ledgers[{index}]",
        )
        ledger_path = _project_file(root, ledger["path"], label="campaign ledger")
        if sha256(ledger_path) != ledger["sha256"]:
            raise QualifiedMechanicsError(f"campaign ledger hash drift: {ledger_path}")
        raw_results.extend(_rows(ledger_path))
    if not raw_results:
        raise QualifiedMechanicsError("campaign ledgers are empty")
    validated: dict[str, dict[str, Any]] = {}
    for row in raw_results:
        collection_id = row.get("collection_id")
        if collection_id in validated:
            raise QualifiedMechanicsError(f"campaign duplicates {collection_id}")
        checked = _validate_result(
            row,
            root=root,
            campaign=campaign,
            pool=pool,
            admitted=admitted,
            templates=templates,
        )
        validated[str(collection_id)] = checked

    pairs = campaign["matched_pairs"]
    minimum_wins = campaign["minimum_coordination_wins"]
    if (
        not isinstance(pairs, list)
        or not pairs
        or isinstance(minimum_wins, bool)
        or not isinstance(minimum_wins, int)
        or minimum_wins < 1
    ):
        raise QualifiedMechanicsError("campaign has an invalid matched-lift gate")
    coordination_wins = 0
    seen_pair_tasks: set[str] = set()
    for index, raw_pair in enumerate(pairs):
        pair = _object(
            raw_pair,
            fields={"task_name", "solo_collection_id", "coordination_collection_id"},
            label=f"campaign.matched_pairs[{index}]",
        )
        task_name = pair["task_name"]
        if not isinstance(task_name, str) or task_name in seen_pair_tasks:
            raise QualifiedMechanicsError("matched pairs must name unique tasks")
        seen_pair_tasks.add(task_name)
        try:
            solo = validated[pair["solo_collection_id"]]
            coordination = validated[pair["coordination_collection_id"]]
        except (KeyError, TypeError) as exc:
            raise QualifiedMechanicsError("matched pair references an unknown result") from exc
        if (
            solo["task_name"] != task_name
            or coordination["task_name"] != task_name
            or not solo["template_id"].startswith("solo_w")
            or coordination["template_id"].startswith("solo_w")
        ):
            raise QualifiedMechanicsError("matched pair roles or task identity are invalid")
        if coordination["reward"] > solo["reward"]:
            coordination_wins += 1
    if coordination_wins < minimum_wins:
        raise QualifiedMechanicsError(
            f"coordination-lift gate failed: {coordination_wins} wins, "
            f"minimum {minimum_wins}"
        )

    source_rows: list[dict[str, Any]] = []
    guardian_rows: list[dict[str, Any]] = []
    fingerprint = mechanics_pool_fingerprint(pool)
    for collection_id in sorted(validated):
        row = validated[collection_id]
        if row["training_eligible"]:
            guardian_rows.append(
                {
                    "collection_id": collection_id,
                    "task_name": row["task_name"],
                    "family": row["family"],
                    "template_id": row["template_id"],
                    "verifier_reward": row["reward"],
                    "completion_claim_correct": row["reward"] == 1.0,
                    "worker_calls_are_paid": False,
                    "pool_fingerprint": fingerprint,
                    "runtime_revision": campaign["runtime_revision"],
                    "runner_revision": campaign["runner_revision"],
                    "result_sha256": sha256(row["result_path"]),
                    "route_log_sha256": sha256(row["route_log_path"]),
                    "trajectory_sha256": sha256(row["trajectory_path"]),
                }
            )
        if row["reward"] != 1.0:
            continue
        _, task_instruction = split_environment_setup(
            (Path(row["task_dir"]) / "instruction.md").read_text(encoding="utf-8")
        )
        source_rows.extend(
            convert_successful_harbor_trajectory(
                collection_id=collection_id,
                task_id=f"qualified_mechanics__{row['task_name']}",
                original_task=task_instruction,
                pool_fingerprint=fingerprint,
                registered_action=templates[row["template_id"]],
                result_path=row["result_path"],
                route_log_path=row["route_log_path"],
                trajectory_path=row["trajectory_path"],
                paid_call_limit=120,
            )
        )
    return QualifiedConversion(
        source_rows=tuple(source_rows),
        guardian_rows=tuple(guardian_rows),
        matched_pairs=len(pairs),
        coordination_wins=coordination_wins,
    )
