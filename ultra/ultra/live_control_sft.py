"""Build a pool-bound live-control continuation SFT dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .live_control import (
    ControlBudget,
    ControlPosition,
    LiveControlState,
    WorkerProfile,
    build_control_action_messages,
    capability_reference_map,
    parse_capability_control_action,
    parse_control_action,
    render_control_prompt,
    serialize_capability_control_action,
    validate_control_action,
)
from .pool_binding import PoolBinding, load_pool_binding, verify_checkpoint_artifacts


VALID_LABEL_STATUSES = {
    "audited_role_replay",
    "audited_early_completion",
    "human_audited",
    "valid_verifier_pass",
    "verified_recovery_boundary",
    "outcome_selected_topology_correction",
    "verified_solo_sufficient_initial",
}
REQUIRED_ACTION_MINIMUMS = {
    "replan": 12,
    "continue": 64,
    "handoff": 24,
    "complete": 12,
}
MIN_UNIQUE_TASKS = 12
MIN_TOTAL_ROWS = sum(REQUIRED_ACTION_MINIMUMS.values())


class LiveControlDatasetError(ValueError):
    """A training row violates the live-control data contract."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LiveControlDatasetError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise LiveControlDatasetError(f"row {line_number} must be an object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _profiles(binding: PoolBinding) -> tuple[WorkerProfile, ...]:
    return tuple(
        WorkerProfile(
            worker_id=slot.worker_id,
            capability_tags=slot.role_prior,
            tool_tags=("terminal", "filesystem", "test_runner"),
        )
        for slot in binding.slots
    )


def _state_from_row(raw: Any, binding: PoolBinding) -> LiveControlState:
    if not isinstance(raw, dict):
        raise LiveControlDatasetError("state must be an object")
    required = {
        "original_task",
        "workflow_id",
        "positions",
        "active_position_id",
        "terminal_status",
        "terminal_observation",
        "shared_memory",
        "budget",
    }
    allowed = required | {"unavailable_worker_ids"}
    if not required.issubset(raw) or not set(raw).issubset(allowed):
        raise LiveControlDatasetError(
            f"state fields must contain {sorted(required)} and only optional "
            "unavailable_worker_ids"
        )
    raw_positions = raw.get("positions")
    if not isinstance(raw_positions, list):
        raise LiveControlDatasetError("state.positions must be a list")
    positions: list[ControlPosition] = []
    for index, position in enumerate(raw_positions):
        if not isinstance(position, dict) or set(position) != {
            "position_id",
            "worker_id",
            "subtask",
            "access",
            "status",
            "progress",
            "artifacts",
        }:
            raise LiveControlDatasetError(
                f"state.positions[{index}] has an invalid schema"
            )
        access = position.get("access")
        artifacts = position.get("artifacts")
        if not isinstance(access, list) or not isinstance(artifacts, list):
            raise LiveControlDatasetError(
                f"state.positions[{index}] access and artifacts must be lists"
            )
        positions.append(
            ControlPosition(
                position_id=position.get("position_id"),
                worker_id=position.get("worker_id"),
                subtask=position.get("subtask"),
                access=tuple(access),
                status=position.get("status"),
                progress=position.get("progress"),
                artifacts=tuple(artifacts),
            )
        )
    budget = raw.get("budget")
    if not isinstance(budget, dict) or set(budget) != {
        "paid_calls_used",
        "paid_call_limit",
        "elapsed_s",
        "wall_time_limit_s",
    }:
        raise LiveControlDatasetError("state.budget has an invalid schema")
    shared_memory = raw.get("shared_memory")
    if not isinstance(shared_memory, list):
        raise LiveControlDatasetError("state.shared_memory must be a list")
    unavailable_worker_ids = raw.get("unavailable_worker_ids", [])
    if not isinstance(unavailable_worker_ids, list):
        raise LiveControlDatasetError("state.unavailable_worker_ids must be a list")
    return LiveControlState(
        original_task=raw.get("original_task"),
        workers=_profiles(binding),
        workflow_id=raw.get("workflow_id"),
        positions=tuple(positions),
        active_position_id=raw.get("active_position_id"),
        terminal_status=raw.get("terminal_status"),
        terminal_observation=raw.get("terminal_observation"),
        shared_memory=tuple(shared_memory),
        budget=ControlBudget(**budget),
        unavailable_worker_ids=tuple(unavailable_worker_ids),
    )


def _action_json(
    raw: Any,
    state: LiveControlState,
    *,
    capability_refs: bool,
) -> tuple[str, str]:
    if not isinstance(raw, dict):
        raise LiveControlDatasetError("action must be an object")
    content = json.dumps(raw, sort_keys=True, ensure_ascii=True)
    action = parse_control_action(content)
    validate_control_action(action, state)
    if capability_refs:
        references = capability_reference_map(state.workers)
        content = serialize_capability_control_action(action, references)
        typed_action = parse_capability_control_action(content, references)
        validate_control_action(typed_action, state)
        if typed_action != action:
            raise LiveControlDatasetError(
                "typed capability action did not round-trip to the source action"
            )
    return action.action, content


def _validate_provenance(row: dict[str, Any], binding: PoolBinding) -> None:
    if row.get("pool_fingerprint") != binding.pool_fingerprint:
        raise LiveControlDatasetError(
            "row pool_fingerprint does not match the checkpoint binding"
        )
    terminalbench = row.get("terminalbench")
    if terminalbench not in {False, True} or not isinstance(terminalbench, bool):
        raise LiveControlDatasetError("terminalbench provenance must be boolean")
    if terminalbench and (
        row.get("evaluation_excluded") is not True
        or row.get("benchmark_source") != "terminalbench21_recovery_training"
    ):
        raise LiveControlDatasetError(
            "TerminalBench rows are evaluation-only unless permanently "
            "evaluation-excluded as recovery training"
        )
    if row.get("label_status") not in VALID_LABEL_STATUSES:
        raise LiveControlDatasetError(
            "row does not have an accepted reward/action label"
        )
    if row.get("label_status") == "audited_role_replay":
        source_fingerprint = row.get("source_pool_fingerprint")
        provenance = row.get("provenance")
        if (
            row.get("replay_migrated") is not True
            or not isinstance(source_fingerprint, str)
            or len(source_fingerprint) != 64
            or source_fingerprint == binding.pool_fingerprint
            or not isinstance(provenance, dict)
            or provenance.get("migration_usage")
            != "anti_forgetting_replay_only"
        ):
            raise LiveControlDatasetError(
                "role replay lacks an audited cross-pool migration"
            )
    evidence = row.get("agentic_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "tool_calls_observed",
        "shared_workspace",
        "verifier_audited",
    }:
        raise LiveControlDatasetError("agentic_evidence has an invalid schema")
    tool_calls = evidence.get("tool_calls_observed")
    if (
        isinstance(tool_calls, bool)
        or not isinstance(tool_calls, int)
        or tool_calls < 0
    ):
        raise LiveControlDatasetError(
            "tool_calls_observed must be a non-negative integer"
        )
    if evidence.get("shared_workspace") is not True:
        raise LiveControlDatasetError("training rows must come from a shared workspace")
    if evidence.get("verifier_audited") is not True:
        raise LiveControlDatasetError("training row verifier must be audited")


def _write_sft_config(
    path: Path,
    *,
    binding: PoolBinding,
    repo_root: Path,
    base_model_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    adapter_path = (repo_root / binding.checkpoint.adapter_path).resolve()
    text = f"""model: "{base_model_path.resolve()}"
adapter_path: "{adapter_path}"
output_dir: "{output_dir.resolve()}"

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
sample_packing: true
sequence_len: 8192
max_steps: 50
save_steps: 10
save_total_limit: 3
logging_steps: 1

learning_rate: 2e-5
lr_scheduler_type: cosine
warmup_ratio: 0.05
max_grad_norm: 1.0
weight_decay: 0.01
optimizer: adamw

recipe: bf16
recompute: true
cpu_training: true
template: qwen3_nothinking
loss_scale: default

lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
resume_from_checkpoint: false
report_to: [surogate]

datasets:
  - path: "{dataset_path.resolve()}"
    type: conversation
    messages_field: messages
"""
    path.write_text(text, encoding="utf-8")


def build_live_control_sft(
    *,
    source_jsonl: Path,
    pool_binding_path: Path,
    repo_root: Path,
    base_model_path: Path,
    out_dir: Path,
    capability_refs: bool = False,
) -> dict[str, Any]:
    binding = load_pool_binding(pool_binding_path)
    verify_checkpoint_artifacts(binding, repo_root=repo_root)
    source_rows = _read_jsonl(source_jsonl)
    output_rows: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    tasks: set[str] = set()
    tool_calls_observed = 0
    evaluation_excluded_tasks: set[str] = set()
    for index, row in enumerate(source_rows):
        try:
            _validate_provenance(row, binding)
            state = _state_from_row(row.get("state"), binding)
            action_name, action_content = _action_json(
                row.get("action"),
                state,
                capability_refs=capability_refs,
            )
        except Exception as exc:
            raise LiveControlDatasetError(f"source row {index}: {exc}") from exc
        task_id = row.get("task_id")
        record_id = row.get("record_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise LiveControlDatasetError(
                f"source row {index}: task_id must be non-empty"
            )
        if not isinstance(record_id, str) or not record_id.strip():
            raise LiveControlDatasetError(
                f"source row {index}: record_id must be non-empty"
            )
        actions[action_name] += 1
        tasks.add(task_id)
        if row.get("terminalbench") is True:
            evaluation_excluded_tasks.add(task_id)
        tool_calls_observed += row["agentic_evidence"]["tool_calls_observed"]
        if capability_refs:
            messages, _, _ = build_control_action_messages(
                state,
                capability_refs=True,
            )
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the live conductor for a multi-step, tool-using "
                        "agentic task. Return one valid control-action JSON object."
                    ),
                },
                {"role": "user", "content": render_control_prompt(state)},
            ]
        output_rows.append(
            {
                "record_id": record_id,
                "task_id": task_id,
                "pool_id": binding.pool_id,
                "pool_fingerprint": binding.pool_fingerprint,
                "action": action_name,
                "control_protocol": (
                    "unified_capability_action_v2"
                    if capability_refs
                    else "unified_runtime_worker_action_v1"
                ),
                "messages": [
                    *messages,
                    {"role": "assistant", "content": action_content},
                ],
            }
        )

    required_action_minimums = dict(REQUIRED_ACTION_MINIMUMS)
    registered_transitions = sum(
        max(0, len(row.get("action", {}).get("steps", [])) - 1)
        for row in source_rows
        if row.get("action", {}).get("action") == "replan"
    )
    required_action_minimums["handoff"] = min(
        required_action_minimums["handoff"], registered_transitions
    )
    action_deficits = {
        action: max(0, minimum - actions[action])
        for action, minimum in required_action_minimums.items()
    }
    gates = {
        "minimum_total_rows": len(output_rows) >= MIN_TOTAL_ROWS,
        "minimum_unique_tasks": len(tasks) >= MIN_UNIQUE_TASKS,
        "required_action_coverage": not any(action_deficits.values()),
        "contains_tool_calls": tool_calls_observed > 0,
    }
    ready = all(gates.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "live_control_sft.jsonl"
    config_path = out_dir / "train_live_control_continue.yaml"
    report_path = out_dir / "live_control_sft_report.json"
    _write_jsonl(dataset_path, output_rows)
    _write_sft_config(
        config_path,
        binding=binding,
        repo_root=repo_root,
        base_model_path=base_model_path,
        dataset_path=dataset_path,
        output_dir=repo_root
        / (
            "output/fugu_ultra_live_control_capability_sft"
            if capability_refs
            else "output/fugu_ultra_live_control_sft"
        ),
    )
    report = {
        "version": (
            "fugu_live_control_capability_sft_v2"
            if capability_refs
            else "fugu_live_control_sft_v1"
        ),
        "control_protocol": (
            "unified_capability_action_v2"
            if capability_refs
            else "unified_runtime_worker_action_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "dataset_jsonl": str(dataset_path),
        "sft_config": str(config_path),
        "pool_id": binding.pool_id,
        "pool_fingerprint": binding.pool_fingerprint,
        "parent_adapter": binding.checkpoint.adapter_path,
        "external_calls_made": 0,
        "counts": {
            "rows": len(output_rows),
            "unique_tasks": len(tasks),
            "tool_calls_observed": tool_calls_observed,
            "actions": dict(sorted(actions.items())),
            "evaluation_excluded_tasks": len(evaluation_excluded_tasks),
        },
        "evaluation_excluded_task_ids": sorted(evaluation_excluded_tasks),
        "action_deficits": action_deficits,
        "required_action_minimums": required_action_minimums,
        "gates": gates,
        "ready_for_continued_training": ready,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--pool-binding", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--capability-refs",
        action="store_true",
        help="emit anonymous typed capability actions instead of runtime worker IDs",
    )
    args = parser.parse_args(argv)
    report = build_live_control_sft(
        source_jsonl=args.source_jsonl,
        pool_binding_path=args.pool_binding,
        repo_root=args.repo_root,
        base_model_path=args.base_model_path,
        out_dir=args.out_dir,
        capability_refs=args.capability_refs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
