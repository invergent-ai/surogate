"""Plan train-allowed repo trace collection for state-branch data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .scaffold_canary import load_taskspecs
from .scaffold_tournament import canonical_arms, canonical_workers
from .schemas import TaskSpec

TRACE_CAPTURE_VERSION = "fugu_ultra_trace_capture_plan_v1"

DEFAULT_TRACE_ARMS = (
    "solo__opencode_kimi_builder",
    "solo__codex_gpt_coding_agent",
    "solo__claude_code_opus_debugger",
)

REQUIRED_TRACE_ARTIFACTS = (
    "repo_state",
    "workspace_snapshot_ref",
    "final_patch_ref",
    "public_test_log_ref",
    "hidden_grade_ref",
    "command_log_ref",
)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe[:180] or "item"


def _stable_key(seed: int, task: TaskSpec) -> str:
    payload = f"{TRACE_CAPTURE_VERSION}:{seed}:{task.task_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _has_repo_payload(task: TaskSpec) -> bool:
    for asset in task.input.assets:
        if isinstance(asset, dict) and isinstance(asset.get("opencode_instance"), dict):
            inst = asset["opencode_instance"]
            return bool(inst.get("image_name") and inst.get("problem_statement"))
    return False


def _eligible_repo_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    return [
        task
        for task in tasks
        if task.source.name == "generated_repo_tasks"
        and task.source.policy == "train_allowed"
        and task.splitting.split == "grpo_train"
        and task.capability == "agentic_coding"
        and _has_repo_payload(task)
    ]


def build_trace_capture_plan(
    *,
    manifest_dir: Path,
    tasks_jsonl: Path,
    out_json: Path,
    jobs_out: Path,
    task_limit: int = 8,
    seed: int = 0,
    arms: tuple[str, ...] = DEFAULT_TRACE_ARMS,
) -> dict[str, Any]:
    """Write a deterministic trace-capture manifest and job list.

    This does not execute models. It defines the first train-allowed trace batch
    that should be run with artifact capture enabled before branch TaskSpecs are
    admitted to GRPO.
    """

    manifest_dir = manifest_dir.resolve()
    tasks_jsonl = tasks_jsonl.resolve()
    out_json = out_json.resolve()
    jobs_out = jobs_out.resolve()

    if task_limit <= 0:
        raise ValueError("task_limit must be positive")

    available_arms = {arm.name: arm for arm in canonical_arms(canonical_workers())}
    missing_arms = [arm for arm in arms if arm not in available_arms]
    if missing_arms:
        raise ValueError(f"unknown trace capture arms: {missing_arms}")

    tasks = load_taskspecs(tasks_jsonl)
    eligible = sorted(_eligible_repo_tasks(tasks), key=lambda task: _stable_key(seed, task))
    selected = eligible[:task_limit]
    if len(selected) < task_limit:
        raise ValueError(
            f"trace capture needs {task_limit} eligible generated repo tasks, found {len(selected)}"
        )

    jobs: list[dict[str, Any]] = []
    for task in selected:
        for arm_name in arms:
            arm = available_arms[arm_name]
            job_id = f"tracecap__{_safe_id(arm_name)}__{_safe_id(task.task_id)}"
            jobs.append(
                {
                    "job_id": job_id,
                    "task_id": task.task_id,
                    "source_name": task.source.name,
                    "split": task.splitting.split,
                    "task_jsonl": str(tasks_jsonl),
                    "arm": arm_name,
                    "worker_names": list(arm.worker_names),
                    "budget": "short",
                    "required_artifacts": list(REQUIRED_TRACE_ARTIFACTS),
                    "rollout_out": str(manifest_dir / "trace_capture" / "rollouts" / f"{job_id}.json"),
                    "agent_trace_out": str(manifest_dir / "trace_capture" / "agent_traces" / f"{job_id}.json"),
                    "artifact_dir": str(manifest_dir / "trace_capture" / "artifacts" / job_id),
                    "branch_states_to_extract": [
                        "after_initial_repo_inspection",
                        "after_first_patch",
                        "after_first_test_or_grade_feedback",
                    ],
                    "acceptance_gate": {
                        "source_must_be_train_allowed": True,
                        "split_must_be_grpo_train": True,
                        "must_have_repo_state": True,
                        "must_have_final_patch_ref": True,
                        "must_have_execution_feedback": True,
                    },
                }
            )

    jobs_out.parent.mkdir(parents=True, exist_ok=True)
    with jobs_out.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")

    plan = {
        "version": TRACE_CAPTURE_VERSION,
        "manifest_dir": str(manifest_dir),
        "tasks_jsonl": str(tasks_jsonl),
        "jobs_jsonl": str(jobs_out),
        "task_limit": task_limit,
        "seed": seed,
        "eligible_task_count": len(eligible),
        "selected_task_count": len(selected),
        "job_count": len(jobs),
        "arms": list(arms),
        "required_artifacts": list(REQUIRED_TRACE_ARTIFACTS),
        "selected_tasks": [task.task_id for task in selected],
        "selection_policy": [
            "Only train_allowed grpo_train generated_repo_tasks are eligible.",
            "Deep SWE and diagnostic canaries are excluded.",
            "Each selected task is run through OpenCode, Codex, and Claude Code solo scaffolds.",
            "Rollouts are not branch-train-ready unless repo state, patch ref, and execution feedback artifacts are captured.",
        ],
        "next_command_template": (
            "ultra scaffold-canary --tasks-jsonl {task_jsonl} --task-id {task_id} "
            "--arm {arm} --budget short --out {rollout_out} "
            "--artifact-dir {artifact_dir} --agent-trace-out {agent_trace_out}"
        ),
        "live_calls": False,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan
