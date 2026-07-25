"""Fail-closed specification for reusable causal coordination campaigns."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from director.agentic.fugu_branchpoint_analog_admission import admit_analog_band
from ultra.pool_binding import load_pool_binding


SPEC_VERSION = "fugu_causal_campaign_spec_v1"
YUNWU_API_BASE = "https://yunwu.ai/v1"
CAMPAIGN_ID = re.compile(r"^fugu_causal_[a-z0-9_]+_v[0-9]+$")


class CausalCampaignSpecError(ValueError):
    """A campaign specification cannot support a trusted causal attempt."""


@dataclass(frozen=True)
class AdmittedCausalCampaignSpec:
    campaign_id: str
    task_id: str
    mechanism_id: str
    split: str
    task_dir: Path
    task_tree_sha256: str
    instruction_sha256: str
    task_checksum: str
    docker_image: str
    docker_image_id: str
    analog_admission: Path
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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CausalCampaignSpecError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CausalCampaignSpecError(f"{label} must be an object")
    return value


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CausalCampaignSpecError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _project_path(root: Path, raw: Any, *, label: str, file: bool = True) -> Path:
    if not isinstance(raw, str) or not raw:
        raise CausalCampaignSpecError(f"{label} path is invalid")
    candidate = Path(raw)
    path = (
        (root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CausalCampaignSpecError(f"{label} escapes the project root") from exc
    exists = path.is_file() if file else path.is_dir()
    if not exists:
        raise CausalCampaignSpecError(f"{label} is missing")
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_path(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise CausalCampaignSpecError(f"{label} hash drift")
    return path


def _validate_action(
    action: Any, *, worker_ids: set[int], label: str
) -> dict[str, Any]:
    value = _object(action, fields={"action", "reason", "steps"}, label=label)
    if value["action"] != "replan" or not isinstance(value["reason"], str):
        raise CausalCampaignSpecError(f"{label} must be a reasoned replan")
    steps = value["steps"]
    if not isinstance(steps, list) or not steps:
        raise CausalCampaignSpecError(f"{label} requires workflow steps")
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
            raise CausalCampaignSpecError(f"{label} step {index} is invalid")
    return value


def _split_assignments(path: Path) -> dict[str, str]:
    payload = _read_json(path, label="causal split")
    if not str(payload.get("version", "")).startswith("fugu_causal_task_split_"):
        raise CausalCampaignSpecError("causal split version drift")
    policy = payload.get("policy") or {}
    if (
        policy.get("logical_task_isolation") is not True
        or policy.get("holdout_outcomes_enter_training") is not False
        or policy.get("holdout_prompts_enter_training") is not False
        or policy.get("future_tasks_require_new_split_version") is not True
    ):
        raise CausalCampaignSpecError("causal split policy drift")
    assignments: dict[str, str] = {}
    for split in ("train", "holdout"):
        rows = payload.get(split)
        if not isinstance(rows, list):
            raise CausalCampaignSpecError("causal split rows drift")
        for row in rows:
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(task_id, str) or not task_id or task_id in assignments:
                raise CausalCampaignSpecError("causal split assignment drift")
            assignments[task_id] = split
    return assignments


def admit_causal_campaign_spec(
    spec_path: Path,
    *,
    root: Path,
    observed_image_ids: Mapping[str, str] | None = None,
) -> AdmittedCausalCampaignSpec:
    """Validate one immutable solo-first conditional campaign specification."""
    root = root.resolve()
    spec = _object(
        _read_json(spec_path, label="causal campaign spec"),
        fields={
            "version",
            "campaign_id",
            "purpose",
            "task",
            "pool_binding",
            "split",
            "actions",
            "policy",
            "provenance",
            "frozen_code",
        },
        label="causal campaign spec",
    )
    if spec["version"] != SPEC_VERSION:
        raise CausalCampaignSpecError("unsupported causal campaign spec version")
    campaign_id = spec["campaign_id"]
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID.fullmatch(campaign_id):
        raise CausalCampaignSpecError("invalid causal campaign ID")
    if not isinstance(spec["purpose"], str) or not spec["purpose"].strip():
        raise CausalCampaignSpecError("causal campaign purpose is missing")

    binding_record = _object(
        spec["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_path(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise CausalCampaignSpecError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.provider_base != YUNWU_API_BASE
        or binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
    ):
        raise CausalCampaignSpecError("pool binding identity or provider drift")

    split_path = _verified_file(root, spec["split"], label="causal split")
    assignments = _split_assignments(split_path)
    task_record = _object(
        spec["task"],
        fields={"task_id", "mechanism_id", "split", "analog_admission"},
        label="task",
    )
    task_id = task_record["task_id"]
    mechanism_id = task_record["mechanism_id"]
    split = task_record["split"]
    if (
        not isinstance(task_id, str)
        or not task_id.startswith("fugu-train/")
        or not isinstance(mechanism_id, str)
        or not mechanism_id
        or split not in {"train", "holdout"}
        or assignments.get(task_id) != split
    ):
        raise CausalCampaignSpecError("task identity, mechanism, or split drift")

    admission_path = _verified_file(
        root, task_record["analog_admission"], label="analog admission"
    )
    analog = admit_analog_band(
        admission_path,
        root=root,
        observed_image_ids=observed_image_ids,
    )
    matches = [task for task in analog.tasks if task.task_id == task_id]
    if len(matches) != 1 or matches[0].mechanism_id != mechanism_id:
        raise CausalCampaignSpecError("task is not uniquely admitted for its mechanism")
    task = matches[0]

    actions = _object(
        spec["actions"],
        fields={"solo", "coordinated"},
        label="actions",
    )
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
        raise CausalCampaignSpecError(
            "coordinated action must preserve the exact solo prefix and add positions"
        )
    for index, step in enumerate(
        coordinated_steps[len(solo_steps) :], start=len(solo_steps)
    ):
        if not step["access"] or any(parent >= index for parent in step["access"]):
            raise CausalCampaignSpecError(
                "added positions require prior-position access"
            )

    surface = json.dumps(actions, sort_keys=True, ensure_ascii=True).lower()
    forbidden = {
        binding.provider_base.lower(),
        *[slot.runtime_model.lower() for slot in binding.slots],
        *[slot.model_alias.lower() for slot in binding.slots],
        *[slot.training_name.lower() for slot in binding.slots],
    }
    if any(value in surface for value in forbidden):
        raise CausalCampaignSpecError("model identity leaked into learned actions")

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
            "training_conversion_locked",
        },
        label="policy",
    )
    runtime_revision = policy["runtime_revision"]
    collection_revision = policy["collection_revision"]
    if (
        not isinstance(runtime_revision, str)
        or not runtime_revision.strip()
        or not isinstance(collection_revision, str)
        or not collection_revision.strip()
    ):
        raise CausalCampaignSpecError("causal campaign revision drift")
    fixed_policy = dict(policy)
    del fixed_policy["runtime_revision"]
    del fixed_policy["collection_revision"]
    if fixed_policy != {
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
        "training_conversion_locked": True,
    }:
        raise CausalCampaignSpecError("causal campaign policy drift")

    provenance = _object(
        spec["provenance"],
        fields={
            "terminalbench_derived",
            "benchmark_evaluation",
            "allocation_fixed_before_paid_outcome",
            "supersedes",
        },
        label="provenance",
    )
    if (
        provenance["terminalbench_derived"] is not False
        or provenance["benchmark_evaluation"] is not False
        or provenance["allocation_fixed_before_paid_outcome"] is not True
    ):
        raise CausalCampaignSpecError("causal campaign provenance drift")
    supersedes = provenance["supersedes"]
    if supersedes is not None:
        _verified_file(root, supersedes, label="superseded launch")

    expected_code = {
        "campaign_validator": Path(__file__).resolve(),
        "campaign_runner": (root / "scratchpad/run_fugu_causal_campaign.py").resolve(),
        "generic_job_runner": (
            root / "scratchpad/run_fugu_live_control_training_v2.py"
        ).resolve(),
        "collection_agent": (
            root / "director/director/agentic/fugu_branchpoint_collection.py"
        ).resolve(),
        "product_runtime": (
            root / "director/director/agentic/fugu_ultra_terminal.py"
        ).resolve(),
        "trajectory_converter": (
            root / "ultra/ultra/live_control_trajectory.py"
        ).resolve(),
        "analog_validator": (
            root / "director/director/agentic/fugu_branchpoint_analog_admission.py"
        ).resolve(),
        "pool_binding_implementation": (root / "ultra/ultra/pool_binding.py").resolve(),
    }
    frozen_code_raw = spec["frozen_code"]
    if not isinstance(frozen_code_raw, dict) or set(frozen_code_raw) != set(
        expected_code
    ):
        raise CausalCampaignSpecError("frozen code inventory drift")
    frozen_code: list[tuple[str, Path]] = []
    for label, expected_path in expected_code.items():
        path = _verified_file(root, frozen_code_raw[label], label=f"code {label}")
        if path != expected_path:
            raise CausalCampaignSpecError(f"code {label} path drift")
        frozen_code.append((label, path))

    raw_admission = _read_json(admission_path, label="analog admission")
    raw_task = next(row for row in raw_admission["tasks"] if row["task_id"] == task_id)
    return AdmittedCausalCampaignSpec(
        campaign_id=campaign_id,
        task_id=task_id,
        mechanism_id=mechanism_id,
        split=split,
        task_dir=task.task_dir,
        task_tree_sha256=raw_task["task_tree_sha256"],
        instruction_sha256=raw_task["instruction_sha256"],
        task_checksum=task.task_checksum,
        docker_image=task.docker_image,
        docker_image_id=task.docker_image_id,
        analog_admission=admission_path,
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
        frozen_code=tuple(frozen_code),
    )
