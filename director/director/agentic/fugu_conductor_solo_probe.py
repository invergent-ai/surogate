"""Fail-closed admission for conductor-versus-solo probe campaigns.

A probe campaign measures, on one frozen verifier-ready train task, whether
the accepted product conductor's initial topology (with its full adaptive
downstream control) outperforms a strongest-capability solo seed under the
identical runtime, harness, pool, and live control. The solo arm runs first;
the product arm runs only after a clean solo failure. A valid pair is one
observed same-task, same-environment conductor-over-solo lift; it does not
estimate expected lift and does not authorize training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from director.agentic.fugu_adaptive_causal_campaign import (
    YUNWU_API_BASE,
    AdaptiveCausalCampaignError,
    AdmittedAdaptiveCampaign,
    AdmittedAdaptiveTask,
    _object,
    _project_path,
    _read_json,
    _reward,
    _validate_action,
    _verified_file,
    _zero_usage,
    sha256,
    tree_sha256,
)
from director.agentic.fugu_adaptive_causal_campaign_v2 import (
    _executed_test_count,
    _rows_by_name,
    _valid_sha256,
)
from ultra.pool_binding import load_pool_binding


ALLOCATION_VERSION = "fugu_adaptive_causal_pool_v3"
SPEC_VERSION = "fugu_conductor_solo_probe_spec_v1"
PROBE_CAMPAIGN_ID = re.compile(r"^fugu_conductor_solo_probe_[a-z0-9_]+_v[0-9]+$")


def admit_allocation_v3(
    path: Path, *, root: Path
) -> tuple[tuple[AdmittedAdaptiveTask, ...], Path]:
    root = root.resolve()
    path = path.resolve()
    allocation = _object(
        _read_json(path, label="adaptive v3 allocation"),
        fields={
            "version",
            "created_at",
            "objective",
            "policy",
            "inputs",
            "consumed_v2_train_tasks",
            "excluded_tasks",
            "counts",
            "train",
            "holdout",
        },
        label="adaptive v3 allocation",
    )
    if allocation["version"] != ALLOCATION_VERSION:
        raise AdaptiveCausalCampaignError("unsupported adaptive v3 allocation")
    policy = _object(
        allocation["policy"],
        fields={
            "new_task_paid_outcomes_before_freeze",
            "inherited_v2_assignments_preserved",
            "consumed_v2_train_tasks_closed",
            "terminalbench_derived",
            "benchmark_evaluation",
            "training_conversion_locked",
            "holdout_prompts_enter_training",
            "holdout_outcomes_enter_training",
            "previous_holdouts_remain_holdout",
            "closed_tasks_remain_closed",
            "verifier_requires_real_test_collection",
            "environment_identity",
            "future_tasks_require_new_allocation_version",
        },
        label="adaptive v3 policy",
    )
    if policy != {
        "new_task_paid_outcomes_before_freeze": 0,
        "inherited_v2_assignments_preserved": True,
        "consumed_v2_train_tasks_closed": True,
        "terminalbench_derived": False,
        "benchmark_evaluation": False,
        "training_conversion_locked": True,
        "holdout_prompts_enter_training": False,
        "holdout_outcomes_enter_training": False,
        "previous_holdouts_remain_holdout": True,
        "closed_tasks_remain_closed": True,
        "verifier_requires_real_test_collection": True,
        "environment_identity": "harbor_task_digest_plus_rootfs_layers",
        "future_tasks_require_new_allocation_version": True,
    }:
        raise AdaptiveCausalCampaignError("adaptive v3 policy drift")

    inputs = _object(
        allocation["inputs"],
        fields={
            "previous_adaptive_allocation",
            "pool_binding",
            "verifier_repair_manifest",
            "verifier_readiness_manifest",
            "candidate_difficulty_ranking",
        },
        label="adaptive v3 inputs",
    )
    verified_inputs = {
        label: _verified_file(root, record, label=label)
        for label, record in inputs.items()
    }
    previous = _read_json(
        verified_inputs["previous_adaptive_allocation"], label="previous allocation"
    )
    previous_ids = {
        row.get("task_id")
        for split in ("train", "holdout")
        for row in previous.get(split, [])
        if isinstance(row, dict)
    }

    consumed = allocation["consumed_v2_train_tasks"]
    if not isinstance(consumed, dict) or set(consumed) != {
        "swesmith-03234",
        "swesmith-23289",
        "swesmith-07937",
    }:
        raise AdaptiveCausalCampaignError("adaptive v3 consumed-task inventory drift")
    for task_id, record in consumed.items():
        _verified_file(root, record, label=f"consumed {task_id}")

    excluded = allocation["excluded_tasks"]
    if not isinstance(excluded, list) or len(excluded) != 1:
        raise AdaptiveCausalCampaignError("adaptive v3 exclusion inventory drift")
    excluded_row = _object(
        excluded[0], fields={"task_id", "reason", "result", "job"}, label="excluded task"
    )
    excluded_result_path = _verified_file(root, excluded_row["result"], label="excluded result")
    _verified_file(root, excluded_row["job"], label="excluded job")
    excluded_result = _read_json(excluded_result_path, label="excluded result")
    if (
        excluded_row["task_id"] != "swesmith-05355"
        or excluded_row["reason"] != "unchanged_repaired_baseline_reward_1"
        or _reward(excluded_result) != 1.0
        or excluded_result.get("exception_info") is not None
    ):
        raise AdaptiveCausalCampaignError("unchanged-pass exclusion drift")

    repair_rows = _rows_by_name(
        _read_json(verified_inputs["verifier_repair_manifest"], label="repair manifest"),
        label="repair manifest",
    )
    readiness_rows = _rows_by_name(
        _read_json(
            verified_inputs["verifier_readiness_manifest"], label="readiness manifest"
        ),
        label="readiness manifest",
    )

    admitted: list[AdmittedAdaptiveTask] = []
    seen_ids: set[str] = set()
    seen_checksums: set[str] = set()
    for split in ("train", "holdout"):
        raw_rows = allocation[split]
        if not isinstance(raw_rows, list):
            raise AdaptiveCausalCampaignError(f"adaptive v3 {split} rows are invalid")
        for index, raw in enumerate(raw_rows):
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
                    "verifier_repair",
                    "environment",
                    "oracle",
                    "unchanged_baseline",
                },
                label=f"adaptive v3 {split}[{index}]",
            )
            task_id = row["task_id"]
            checksum = row["task_checksum"]
            if (
                not isinstance(task_id, str)
                or not task_id.startswith("swesmith-")
                or task_id in seen_ids
                or task_id in previous_ids
                or row["split"] != split
                or not isinstance(row["mechanism_id"], str)
                or not row["mechanism_id"]
                or not _valid_sha256(checksum)
                or checksum in seen_checksums
            ):
                raise AdaptiveCausalCampaignError("adaptive v3 identity or split drift")
            task_dir = _project_path(root, row["task_dir"], label=task_id, file=False)
            instruction = task_dir / "instruction.md"
            patch = task_dir / "tests/test_patch.diff"
            if (
                tree_sha256(task_dir) != row["task_tree_sha256"]
                or not instruction.is_file()
                or sha256(instruction) != row["instruction_sha256"]
                or not patch.is_file()
            ):
                raise AdaptiveCausalCampaignError(f"{task_id} repaired task tree drift")

            repair = _object(
                row["verifier_repair"],
                fields={
                    "patch_sha256",
                    "challenge_commit",
                    "fixed_commit",
                    "configured_test_files",
                    "patched_test_files",
                },
                label=f"{task_id} verifier repair",
            )
            repair_source = repair_rows.get(task_id) or {}
            readiness = readiness_rows.get(task_id) or {}
            if (
                sha256(patch) != repair["patch_sha256"]
                or repair["patch_sha256"] != repair_source.get("verifier_patch_sha256")
                or repair["challenge_commit"] != repair_source.get("challenge_commit")
                or repair["fixed_commit"] != repair_source.get("fixed_commit")
                or repair["configured_test_files"]
                != repair_source.get("configured_test_files")
                or repair["patched_test_files"] != repair_source.get("patched_test_files")
                or readiness.get("ready") is not True
                or readiness.get("missing_test_files") != []
                or readiness.get("commit") != repair["challenge_commit"]
            ):
                raise AdaptiveCausalCampaignError(f"{task_id} verifier repair drift")

            environment = _object(
                row["environment"],
                fields={
                    "harbor_task_digest",
                    "rootfs_layers_sha256",
                    "rootfs_layer_count",
                },
                label=f"{task_id} environment",
            )
            if (
                not _valid_sha256(environment["harbor_task_digest"], prefix=True)
                or not _valid_sha256(environment["rootfs_layers_sha256"])
                or isinstance(environment["rootfs_layer_count"], bool)
                or not isinstance(environment["rootfs_layer_count"], int)
                or environment["rootfs_layer_count"] <= 0
            ):
                raise AdaptiveCausalCampaignError(f"{task_id} environment identity drift")

            oracle_record = _object(
                row["oracle"],
                fields={"agent", "reward", "result", "verifier_stdout", "environment_image_id"},
                label=f"{task_id} oracle",
            )
            baseline_record = _object(
                row["unchanged_baseline"],
                fields={
                    "agent",
                    "reward",
                    "result",
                    "verifier_stdout",
                    "environment_image_id",
                    "model_calls",
                    "external_calls",
                    "paid_calls",
                    "retries",
                },
                label=f"{task_id} baseline",
            )
            oracle_path = _verified_file(root, oracle_record["result"], label=f"{task_id} oracle")
            baseline_path = _verified_file(
                root, baseline_record["result"], label=f"{task_id} baseline"
            )
            oracle_stdout_path = _verified_file(
                root, oracle_record["verifier_stdout"], label=f"{task_id} oracle stdout"
            )
            baseline_stdout_path = _verified_file(
                root, baseline_record["verifier_stdout"], label=f"{task_id} baseline stdout"
            )
            oracle = _read_json(oracle_path, label=f"{task_id} oracle result")
            baseline = _read_json(baseline_path, label=f"{task_id} baseline result")
            metadata = (baseline.get("agent_result") or {}).get("metadata") or {}
            oracle_stdout = oracle_stdout_path.read_text(encoding="utf-8", errors="replace")
            baseline_stdout = baseline_stdout_path.read_text(
                encoding="utf-8", errors="replace"
            )
            if (
                oracle_record["agent"] != "oracle"
                or oracle_record["reward"] != 1.0
                or not _valid_sha256(oracle_record["environment_image_id"], prefix=True)
                or (oracle.get("agent_info") or {}).get("name") != "oracle"
                or _reward(oracle) != 1.0
                or oracle.get("exception_info") is not None
                or not _zero_usage(oracle)
                or baseline_record
                != {
                    **baseline_record,
                    "agent": "fugu-sanitized-workspace-snapshot-preflight",
                    "reward": 0.0,
                    "model_calls": 0,
                    "external_calls": 0,
                    "paid_calls": 0,
                    "retries": 0,
                }
                or not _valid_sha256(baseline_record["environment_image_id"], prefix=True)
                or (baseline.get("agent_info") or {}).get("name")
                != "fugu-sanitized-workspace-snapshot-preflight"
                or _reward(baseline) != 0.0
                or baseline.get("exception_info") is not None
                or not _zero_usage(baseline)
                or metadata.get("paid_worker_call_attempts") != 0
                or metadata.get("prepared_repository_setup_executed") is not True
                or oracle.get("task_checksum") != checksum
                or baseline.get("task_checksum") != checksum
                or _executed_test_count(oracle_stdout) <= 1
                or _executed_test_count(baseline_stdout) <= 1
                or any(
                    marker in oracle_stdout or marker in baseline_stdout
                    for marker in ("file or directory not found", "no tests ran")
                )
            ):
                raise AdaptiveCausalCampaignError(f"{task_id} verifier evidence drift")
            seen_ids.add(task_id)
            seen_checksums.add(checksum)
            admitted.append(
                AdmittedAdaptiveTask(
                    task_id=task_id,
                    split=split,
                    mechanism_id=row["mechanism_id"],
                    task_dir=task_dir,
                    task_tree_sha256=row["task_tree_sha256"],
                    instruction_sha256=row["instruction_sha256"],
                    task_checksum=checksum,
                    oracle_result=oracle_path,
                    baseline_result=baseline_path,
                )
            )
    if allocation["counts"] != {"tasks": 21, "train": 18, "holdout": 3}:
        raise AdaptiveCausalCampaignError("adaptive v3 allocation counts drift")
    if len(admitted) != 21:
        raise AdaptiveCausalCampaignError("adaptive v3 admitted task count drift")
    return tuple(admitted), path


def admit_probe_campaign(
    spec_path: Path,
    *,
    root: Path,
    expected_code: dict[str, Path] | None = None,
) -> AdmittedAdaptiveCampaign:
    root = root.resolve()
    spec = _object(
        _read_json(spec_path, label="conductor solo probe spec"),
        fields={
            "version",
            "campaign_id",
            "purpose",
            "allocation",
            "task",
            "pool_binding",
            "accepted_planner",
            "actions",
            "policy",
            "provenance",
            "frozen_code",
        },
        label="conductor solo probe spec",
    )
    if spec["version"] != SPEC_VERSION:
        raise AdaptiveCausalCampaignError("unsupported conductor solo probe spec")
    campaign_id = spec["campaign_id"]
    if not isinstance(campaign_id, str) or not PROBE_CAMPAIGN_ID.fullmatch(campaign_id):
        raise AdaptiveCausalCampaignError("invalid probe campaign ID")
    if not isinstance(spec["purpose"], str) or not spec["purpose"].strip():
        raise AdaptiveCausalCampaignError("probe purpose is missing")

    allocation_path = _verified_file(root, spec["allocation"], label="allocation")
    tasks, _ = admit_allocation_v3(allocation_path, root=root)
    task_record = _object(
        spec["task"], fields={"task_id", "split", "mechanism_id"}, label="probe task"
    )
    matches = [task for task in tasks if task.task_id == task_record["task_id"]]
    if (
        len(matches) != 1
        or matches[0].split != task_record["split"]
        or matches[0].mechanism_id != task_record["mechanism_id"]
        or matches[0].split != "train"
    ):
        raise AdaptiveCausalCampaignError("probe task is not a unique v3 train task")
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

    planner_record = _object(
        spec["accepted_planner"],
        fields={"adapter", "initial_observation", "raw_plan", "steps"},
        label="accepted planner",
    )
    if planner_record["adapter"] != "output/fugu_ultra_planner_composite_v11_s20":
        raise AdaptiveCausalCampaignError("probe planner adapter drift")

    actions = _object(spec["actions"], fields={"solo", "product"}, label="actions")
    worker_ids = {slot.worker_id for slot in binding.slots}
    solo = _validate_action(actions["solo"], worker_ids=worker_ids, label="solo")
    product = _validate_action(actions["product"], worker_ids=worker_ids, label="product")
    if len(solo["steps"]) != 1:
        raise AdaptiveCausalCampaignError("solo seed must contain exactly one position")
    if len(product["steps"]) < 2:
        raise AdaptiveCausalCampaignError("product seed must be a multi-step workflow")
    if product["steps"] != planner_record["steps"]:
        raise AdaptiveCausalCampaignError(
            "product seed must equal the accepted planner's frozen workflow"
        )
    surface = json.dumps(actions, sort_keys=True, ensure_ascii=True).lower()
    forbidden = {
        binding.provider_base.lower(),
        *[slot.runtime_model.lower() for slot in binding.slots],
        *[slot.model_alias.lower() for slot in binding.slots],
        *[slot.training_name.lower() for slot in binding.slots],
    }
    if any(value in surface for value in forbidden):
        raise AdaptiveCausalCampaignError("model identity leaked into probe actions")

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
            "product_requires_clean_solo_failure",
            "invalid_outcome_stops",
            "initial_intervention_only",
            "dynamic_downstream_workflows_allowed",
            "same_product_conductor_after_initial_intervention",
            "automatic_irreversible_invalidity_stop",
            "training_conversion_locked",
        },
        label="probe policy",
    )
    runtime_revision = policy.pop("runtime_revision")
    collection_revision = policy.pop("collection_revision")
    if not isinstance(runtime_revision, str) or not isinstance(collection_revision, str):
        raise AdaptiveCausalCampaignError("probe revision drift")
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
        "product_requires_clean_solo_failure": True,
        "invalid_outcome_stops": True,
        "initial_intervention_only": True,
        "dynamic_downstream_workflows_allowed": True,
        "same_product_conductor_after_initial_intervention": True,
        "automatic_irreversible_invalidity_stop": True,
        "training_conversion_locked": True,
    }:
        raise AdaptiveCausalCampaignError("conductor solo probe policy drift")

    provenance = _object(
        spec["provenance"],
        fields={
            "terminalbench_derived",
            "benchmark_evaluation",
            "task_paid_outcomes_before_freeze",
            "probe_objective",
            "initial_topology_source",
            "solo_seed_rationale",
            "planner_external_calls",
            "planner_paid_calls",
        },
        label="probe provenance",
    )
    if (
        provenance["terminalbench_derived"] is not False
        or provenance["benchmark_evaluation"] is not False
        or provenance["task_paid_outcomes_before_freeze"] != 0
        or provenance["initial_topology_source"]
        != "accepted_product_planner_single_deterministic_call"
        or provenance["planner_external_calls"] != 0
        or provenance["planner_paid_calls"] != 0
        or not isinstance(provenance["probe_objective"], str)
        or not isinstance(provenance["solo_seed_rationale"], str)
    ):
        raise AdaptiveCausalCampaignError("conductor solo probe provenance drift")

    code_paths = {
        key: value.resolve()
        for key, value in (
            expected_code
            or {
                "probe_validator": Path(__file__),
                "campaign_runner": root / "scratchpad/run_fugu_conductor_solo_probe_v1.py",
                "collection_agent": root
                / "director/director/agentic/fugu_adaptive_causal_collection.py",
                "product_runtime": root
                / "director/director/agentic/fugu_ultra_terminal.py",
                "generic_job_runner": root
                / "scratchpad/run_fugu_live_control_training_v2.py",
                "trajectory_converter": root / "ultra/ultra/live_control_trajectory.py",
                "pool_binding_implementation": root / "ultra/ultra/pool_binding.py",
            }
        ).items()
    }
    frozen_raw = spec["frozen_code"]
    if not isinstance(frozen_raw, dict) or set(frozen_raw) != set(code_paths):
        raise AdaptiveCausalCampaignError("probe code inventory drift")
    frozen: list[tuple[str, Path]] = []
    for label, expected_path in code_paths.items():
        code_path = _verified_file(root, frozen_raw[label], label=f"code {label}")
        if code_path != expected_path:
            raise AdaptiveCausalCampaignError(f"code {label} path drift")
        frozen.append((label, code_path))

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
        coordinated_action=product,
        added_positions=(),
        frozen_code=tuple(frozen),
    )
