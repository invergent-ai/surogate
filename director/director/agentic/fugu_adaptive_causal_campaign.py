"""Fail-closed contracts for adaptive initial-intervention causal campaigns."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ultra.pool_binding import load_pool_binding


ALLOCATION_VERSION = "fugu_adaptive_causal_pool_v1"
SPEC_VERSION = "fugu_adaptive_causal_campaign_spec_v1"
YUNWU_API_BASE = "https://yunwu.ai/v1"
CAMPAIGN_ID = re.compile(r"^fugu_adaptive_causal_[a-z0-9_]+_v[0-9]+$")


class AdaptiveCausalCampaignError(ValueError):
    """The frozen evidence cannot support an adaptive causal attempt."""


@dataclass(frozen=True)
class AdmittedAdaptiveTask:
    task_id: str
    split: str
    mechanism_id: str
    task_dir: Path
    task_tree_sha256: str
    instruction_sha256: str
    task_checksum: str
    oracle_result: Path
    baseline_result: Path


@dataclass(frozen=True)
class AdmittedAdaptiveCampaign:
    campaign_id: str
    task: AdmittedAdaptiveTask
    allocation: Path
    pool_binding: Path
    pool_id: str
    pool_fingerprint: str
    runtime_revision: str
    collection_revision: str
    worker_timeout_seconds: float
    paid_call_ceiling_per_arm: int
    solo_action: dict[str, Any]
    coordinated_action: dict[str, Any]
    added_positions: tuple[dict[str, Any], ...]
    frozen_code: tuple[tuple[str, Path], ...]


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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdaptiveCausalCampaignError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AdaptiveCausalCampaignError(f"{label} must be an object")
    return value


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AdaptiveCausalCampaignError(
            f"{label} must contain exactly {sorted(fields)}"
        )
    return value


def _project_path(root: Path, raw: Any, *, label: str, file: bool = True) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AdaptiveCausalCampaignError(f"{label} path is invalid")
    candidate = Path(raw)
    path = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AdaptiveCausalCampaignError(f"{label} escapes the project root") from exc
    exists = path.is_file() if file else path.is_dir()
    if not exists:
        raise AdaptiveCausalCampaignError(f"{label} is missing")
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_path(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise AdaptiveCausalCampaignError(f"{label} hash drift")
    return path


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _zero_usage(result: dict[str, Any]) -> bool:
    agent_result = result.get("agent_result") or {}
    return all(
        agent_result.get(field) is None
        for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")
    )


def admit_allocation(path: Path, *, root: Path) -> tuple[tuple[AdmittedAdaptiveTask, ...], Path]:
    root = root.resolve()
    path = path.resolve()
    allocation = _object(
        _read_json(path, label="adaptive allocation"),
        fields={
            "version",
            "created_at",
            "objective",
            "policy",
            "inputs",
            "preserved_previous_holdouts",
            "closed_tasks",
            "counts",
            "train",
            "holdout",
        },
        label="adaptive allocation",
    )
    if allocation["version"] != ALLOCATION_VERSION:
        raise AdaptiveCausalCampaignError("unsupported adaptive allocation version")
    policy = _object(
        allocation["policy"],
        fields={
            "allocation_fixed_before_current_pool_outcome",
            "current_pool_paid_outcomes_before_freeze",
            "terminalbench_derived",
            "benchmark_evaluation",
            "training_conversion_locked",
            "holdout_prompts_enter_training",
            "holdout_outcomes_enter_training",
            "previous_holdouts_remain_holdout",
            "closed_tasks_remain_closed",
            "local_mechanics_outcomes_are_not_current_pool_causal_outcomes",
            "future_tasks_require_new_allocation_version",
        },
        label="allocation policy",
    )
    if policy != {
        "allocation_fixed_before_current_pool_outcome": True,
        "current_pool_paid_outcomes_before_freeze": 0,
        "terminalbench_derived": False,
        "benchmark_evaluation": False,
        "training_conversion_locked": True,
        "holdout_prompts_enter_training": False,
        "holdout_outcomes_enter_training": False,
        "previous_holdouts_remain_holdout": True,
        "closed_tasks_remain_closed": True,
        "local_mechanics_outcomes_are_not_current_pool_causal_outcomes": True,
        "future_tasks_require_new_allocation_version": True,
    }:
        raise AdaptiveCausalCampaignError("adaptive allocation policy drift")

    inputs = _object(
        allocation["inputs"],
        fields={"oracle_ledger", "previous_split", "pool_binding"},
        label="allocation inputs",
    )
    _verified_file(root, inputs["oracle_ledger"], label="oracle ledger")
    previous_path = _verified_file(root, inputs["previous_split"], label="previous split")
    _verified_file(root, inputs["pool_binding"], label="allocation pool binding")
    previous = _read_json(previous_path, label="previous split")
    prior_holdouts = {
        row.get("task_id")
        for row in previous.get("holdout", [])
        if isinstance(row, dict)
    }
    if prior_holdouts != set(allocation["preserved_previous_holdouts"]):
        raise AdaptiveCausalCampaignError("previous holdout preservation drift")
    if set(allocation["closed_tasks"]) != {
        "atomic-quota-journal",
        "release-activation-rollback",
    }:
        raise AdaptiveCausalCampaignError("closed task inventory drift")

    admitted: list[AdmittedAdaptiveTask] = []
    seen_ids: set[str] = set()
    seen_checksums: set[str] = set()
    seen_mechanisms: dict[str, str] = {}
    for split in ("train", "holdout"):
        rows = allocation[split]
        if not isinstance(rows, list):
            raise AdaptiveCausalCampaignError(f"allocation {split} rows drift")
        for index, raw in enumerate(rows):
            row = _object(
                raw,
                fields={
                    "task_id",
                    "split",
                    "mechanism_id",
                    "task_dir",
                    "task_tree_sha256",
                    "instruction_sha256",
                    "task_checksum",
                    "oracle",
                    "unchanged_baseline",
                },
                label=f"allocation {split}[{index}]",
            )
            task_id = row["task_id"]
            mechanism = row["mechanism_id"]
            checksum = row["task_checksum"]
            if (
                not isinstance(task_id, str)
                or not task_id.startswith("swesmith-")
                or task_id in seen_ids
                or row["split"] != split
                or not isinstance(mechanism, str)
                or not mechanism
                or mechanism in seen_mechanisms
                or not isinstance(checksum, str)
                or checksum in seen_checksums
            ):
                raise AdaptiveCausalCampaignError("allocation task identity or split drift")
            task_dir = _project_path(root, row["task_dir"], label=task_id, file=False)
            instruction = task_dir / "instruction.md"
            if (
                tree_sha256(task_dir) != row["task_tree_sha256"]
                or not instruction.is_file()
                or sha256(instruction) != row["instruction_sha256"]
            ):
                raise AdaptiveCausalCampaignError(f"{task_id} task tree drift")

            oracle_record = _object(
                row["oracle"],
                fields={"agent", "reward", "result"},
                label=f"{task_id} oracle",
            )
            baseline_record = _object(
                row["unchanged_baseline"],
                fields={
                    "agent",
                    "reward",
                    "job_result",
                    "trial_result",
                    "model_calls",
                    "external_calls",
                    "paid_calls",
                    "retries",
                },
                label=f"{task_id} baseline",
            )
            oracle_path = _verified_file(root, oracle_record["result"], label=f"{task_id} oracle")
            baseline_job_path = _verified_file(
                root, baseline_record["job_result"], label=f"{task_id} baseline job"
            )
            baseline_path = _verified_file(
                root, baseline_record["trial_result"], label=f"{task_id} baseline trial"
            )
            oracle = _read_json(oracle_path, label=f"{task_id} oracle result")
            baseline_job = _read_json(baseline_job_path, label=f"{task_id} baseline job")
            baseline = _read_json(baseline_path, label=f"{task_id} baseline result")
            stats = baseline_job.get("stats") or {}
            if (
                oracle_record["agent"] != "oracle"
                or oracle_record["reward"] != 1.0
                or (oracle.get("agent_info") or {}).get("name") != "oracle"
                or _reward(oracle) != 1.0
                or oracle.get("exception_info") is not None
                or not _zero_usage(oracle)
                or baseline_record
                != {
                    **baseline_record,
                    "agent": "nop",
                    "reward": 0.0,
                    "model_calls": 0,
                    "external_calls": 0,
                    "paid_calls": 0,
                    "retries": 0,
                }
                or (baseline.get("agent_info") or {}).get("name") != "nop"
                or _reward(baseline) != 0.0
                or baseline.get("exception_info") is not None
                or oracle.get("task_checksum") != checksum
                or baseline.get("task_checksum") != checksum
                or baseline_job.get("n_total_trials") != 1
                or stats.get("n_completed_trials") != 1
                or stats.get("n_errored_trials") != 0
                or stats.get("n_retries") != 0
                or any(
                    stats.get(field) is not None
                    for field in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")
                )
            ):
                raise AdaptiveCausalCampaignError(f"{task_id} zero-call evidence drift")
            seen_ids.add(task_id)
            seen_checksums.add(checksum)
            seen_mechanisms[mechanism] = split
            admitted.append(
                AdmittedAdaptiveTask(
                    task_id=task_id,
                    split=split,
                    mechanism_id=mechanism,
                    task_dir=task_dir,
                    task_tree_sha256=row["task_tree_sha256"],
                    instruction_sha256=row["instruction_sha256"],
                    task_checksum=checksum,
                    oracle_result=oracle_path,
                    baseline_result=baseline_path,
                )
            )
    counts = allocation["counts"]
    if (
        not isinstance(counts, dict)
        or counts != {"tasks": 10, "train": 7, "holdout": 3}
        or len(admitted) != 10
    ):
        raise AdaptiveCausalCampaignError("adaptive allocation counts drift")
    return tuple(admitted), path


def _validate_action(action: Any, *, worker_ids: set[int], label: str) -> dict[str, Any]:
    value = _object(action, fields={"action", "reason", "steps"}, label=label)
    if value["action"] != "replan" or not isinstance(value["reason"], str):
        raise AdaptiveCausalCampaignError(f"{label} must be a reasoned replan")
    steps = value["steps"]
    if not isinstance(steps, list) or not steps:
        raise AdaptiveCausalCampaignError(f"{label} requires workflow steps")
    for index, raw_step in enumerate(steps):
        step = _object(
            raw_step,
            fields={"worker_id", "subtask", "access"},
            label=f"{label} step {index}",
        )
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
            raise AdaptiveCausalCampaignError(f"{label} step {index} is invalid")
    return value


def admit_campaign(
    spec_path: Path,
    *,
    root: Path,
    expected_code: Mapping[str, Path] | None = None,
) -> AdmittedAdaptiveCampaign:
    root = root.resolve()
    spec = _object(
        _read_json(spec_path, label="adaptive causal spec"),
        fields={
            "version",
            "campaign_id",
            "purpose",
            "allocation",
            "task",
            "pool_binding",
            "actions",
            "policy",
            "provenance",
            "frozen_code",
        },
        label="adaptive causal spec",
    )
    if spec["version"] != SPEC_VERSION:
        raise AdaptiveCausalCampaignError("unsupported adaptive causal spec version")
    campaign_id = spec["campaign_id"]
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID.fullmatch(campaign_id):
        raise AdaptiveCausalCampaignError("invalid adaptive campaign ID")
    if not isinstance(spec["purpose"], str) or not spec["purpose"].strip():
        raise AdaptiveCausalCampaignError("adaptive campaign purpose is missing")

    allocation_path = _verified_file(root, spec["allocation"], label="allocation")
    tasks, _ = admit_allocation(allocation_path, root=root)
    task_record = _object(
        spec["task"],
        fields={"task_id", "split", "mechanism_id"},
        label="campaign task",
    )
    matches = [task for task in tasks if task.task_id == task_record["task_id"]]
    if (
        len(matches) != 1
        or matches[0].split != task_record["split"]
        or matches[0].mechanism_id != task_record["mechanism_id"]
        or matches[0].split != "train"
    ):
        raise AdaptiveCausalCampaignError("campaign task is not a unique train allocation")
    task = matches[0]

    binding_record = _object(
        spec["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_path(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise AdaptiveCausalCampaignError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.provider_base != YUNWU_API_BASE
        or binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
    ):
        raise AdaptiveCausalCampaignError("pool identity or provider drift")

    actions = _object(spec["actions"], fields={"solo", "coordinated"}, label="actions")
    worker_ids = {slot.worker_id for slot in binding.slots}
    solo = _validate_action(actions["solo"], worker_ids=worker_ids, label="solo")
    coordinated = _validate_action(
        actions["coordinated"], worker_ids=worker_ids, label="coordinated"
    )
    solo_steps = solo["steps"]
    coordinated_steps = coordinated["steps"]
    if (
        len(coordinated_steps) <= len(solo_steps)
        or coordinated_steps[: len(solo_steps)] != solo_steps
    ):
        raise AdaptiveCausalCampaignError(
            "coordinated seed must preserve the exact solo prefix and add positions"
        )
    for index, step in enumerate(
        coordinated_steps[len(solo_steps) :], start=len(solo_steps)
    ):
        if not step["access"] or any(parent >= index for parent in step["access"]):
            raise AdaptiveCausalCampaignError("added positions require prior-position access")
    surface = json.dumps(actions, sort_keys=True, ensure_ascii=True).lower()
    forbidden = {
        binding.provider_base.lower(),
        *[slot.runtime_model.lower() for slot in binding.slots],
        *[slot.model_alias.lower() for slot in binding.slots],
        *[slot.training_name.lower() for slot in binding.slots],
    }
    if any(value in surface for value in forbidden):
        raise AdaptiveCausalCampaignError("model identity leaked into learned actions")

    policy = _object(
        spec["policy"],
        fields={
            "external_provider",
            "runtime_revision",
            "collection_revision",
            "worker_timeout_seconds",
            "paid_call_ceiling_per_arm",
            "provider_retries",
            "task_retries",
            "attempts_per_arm",
            "maximum_arms",
            "solo_first",
            "solo_pass_stops",
            "coordinated_requires_clean_solo_failure",
            "invalid_outcome_stops",
            "initial_intervention_only",
            "dynamic_downstream_workflows_allowed",
            "same_product_conductor_after_initial_intervention",
            "automatic_irreversible_invalidity_stop",
            "training_conversion_locked",
        },
        label="campaign policy",
    )
    runtime_revision = policy.pop("runtime_revision")
    collection_revision = policy.pop("collection_revision")
    if not isinstance(runtime_revision, str) or not isinstance(collection_revision, str):
        raise AdaptiveCausalCampaignError("campaign revision drift")
    if policy != {
        "external_provider": YUNWU_API_BASE,
        "worker_timeout_seconds": 600.0,
        "paid_call_ceiling_per_arm": 120,
        "provider_retries": 0,
        "task_retries": 0,
        "attempts_per_arm": 1,
        "maximum_arms": 2,
        "solo_first": True,
        "solo_pass_stops": True,
        "coordinated_requires_clean_solo_failure": True,
        "invalid_outcome_stops": True,
        "initial_intervention_only": True,
        "dynamic_downstream_workflows_allowed": True,
        "same_product_conductor_after_initial_intervention": True,
        "automatic_irreversible_invalidity_stop": True,
        "training_conversion_locked": True,
    }:
        raise AdaptiveCausalCampaignError("adaptive causal campaign policy drift")

    provenance = _object(
        spec["provenance"],
        fields={
            "terminalbench_derived",
            "benchmark_evaluation",
            "allocation_fixed_before_current_pool_outcome",
            "supersedes",
        },
        label="campaign provenance",
    )
    if (
        provenance["terminalbench_derived"] is not False
        or provenance["benchmark_evaluation"] is not False
        or provenance["allocation_fixed_before_current_pool_outcome"] is not True
    ):
        raise AdaptiveCausalCampaignError("adaptive campaign provenance drift")
    if provenance["supersedes"] is not None:
        _verified_file(root, provenance["supersedes"], label="superseded campaign")

    code_paths = {
        key: value.resolve()
        for key, value in (
            expected_code
            or {
                "campaign_validator": Path(__file__),
                "campaign_runner": root / "scratchpad/run_fugu_adaptive_causal_campaign.py",
                "collection_agent": root / "director/director/agentic/fugu_adaptive_causal_collection.py",
                "product_runtime": root / "director/director/agentic/fugu_ultra_terminal.py",
                "generic_job_runner": root / "scratchpad/run_fugu_live_control_training_v2.py",
                "trajectory_converter": root / "ultra/ultra/live_control_trajectory.py",
                "pool_binding_implementation": root / "ultra/ultra/pool_binding.py",
            }
        ).items()
    }
    frozen_raw = spec["frozen_code"]
    if not isinstance(frozen_raw, dict) or set(frozen_raw) != set(code_paths):
        raise AdaptiveCausalCampaignError("frozen code inventory drift")
    frozen: list[tuple[str, Path]] = []
    for label, expected_path in code_paths.items():
        path = _verified_file(root, frozen_raw[label], label=f"code {label}")
        if path != expected_path:
            raise AdaptiveCausalCampaignError(f"code {label} path drift")
        frozen.append((label, path))

    return AdmittedAdaptiveCampaign(
        campaign_id=campaign_id,
        task=task,
        allocation=allocation_path,
        pool_binding=binding_path,
        pool_id=binding.pool_id,
        pool_fingerprint=binding.pool_fingerprint,
        runtime_revision=runtime_revision,
        collection_revision=collection_revision,
        worker_timeout_seconds=float(policy["worker_timeout_seconds"]),
        paid_call_ceiling_per_arm=int(policy["paid_call_ceiling_per_arm"]),
        solo_action=solo,
        coordinated_action=coordinated,
        added_positions=tuple(coordinated_steps[len(solo_steps) :]),
        frozen_code=tuple(frozen),
    )
