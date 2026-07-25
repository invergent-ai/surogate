"""Bulk local-worker collection over admitted mechanics tasks.

Runs preregistered topologies through the product runtime with the local
mechanics pool. There is no fail-fast: every graded outcome, pass or fail,
is recorded as data (difficulty statistics, completion-guardian labels, and
reward-1.0 rows for audited conversion).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from director.agentic.fugu_mechanics_terminal import (
    COLLECTION_ID_ENV,
    POOL_ENV,
    WORKFLOW_ENV,
)

RUNNER_REVISION = "20260719-mechanics-runner-v2-frozen-band"
IMMUTABLE_TESTS_PREAMBLE = (
    "Benchmark-owned existing test files are immutable inputs: inspect and run "
    "them, but do not edit, delete, rename, or replace them. "
)
LEGACY_V1_IMMUTABLE_TESTS_PREAMBLE = (
    "Benchmark-owned existing test files are immutable inputs: inspect and run "
    "them, but do not edit, delete, rename, or replace them. Implement the "
    "requested behavior in production code. "
)

_WRITE_LOCK = threading.Lock()


def registered_templates() -> dict[str, dict[str, Any]]:
    """Preregistered mechanics topologies over stable local slots 0-3."""

    def step(worker_id: int, subtask: str, access: list[int]) -> dict[str, Any]:
        return {
            "worker_id": worker_id,
            "subtask": IMMUTABLE_TESTS_PREAMBLE + subtask,
            "access": access,
        }

    solo_subtask = (
        "Complete the entire requested task yourself in the shared workspace. "
        "Investigate the repository and original request, implement the "
        "concrete repair, run the relevant tests, and mark this position "
        "complete only after the result is locally verified."
    )
    templates: dict[str, dict[str, Any]] = {}
    for worker_id in range(4):
        templates[f"solo_w{worker_id}"] = {
            "action": "replan",
            "reason": "Single-position baseline for difficulty statistics.",
            "steps": [step(worker_id, solo_subtask, [])],
        }
    templates["diagnose_build_verify"] = {
        "action": "replan",
        "reason": "Use independent diagnosis around a dedicated implementation role.",
        "steps": [
            step(
                0,
                "Investigate the repository and original request. Reproduce the "
                "issue, identify the exact behavioral gap and relevant tests, and "
                "report a concrete repair plan. Do not modify production code in "
                "this position; mark the assigned diagnosis complete once its "
                "evidence and plan are ready, without treating it as overall "
                "completion.",
                [],
            ),
            step(
                2,
                "Implement the complete repair in the shared workspace using the "
                "permitted diagnosis. Inspect all relevant code yourself, run "
                "focused tests, and mark this position complete only after the "
                "implementation is concrete and locally checked.",
                [0],
            ),
            step(
                0,
                "Return as final adversarial verifier. Re-evaluate the current "
                "code rather than assuming the builder is correct, run "
                "verifier-relevant tests, repair any residual issue, and mark the "
                "overall task complete only when all requested behavior is "
                "satisfied.",
                [0, 1],
            ),
        ],
    }
    templates["draft_debug"] = {
        "action": "replan",
        "reason": "Fast draft handed to an adversarial debugger.",
        "steps": [
            step(
                3,
                "Produce a fast, concrete first implementation of the requested "
                "behavior in the shared workspace. Prefer working code over "
                "exhaustive analysis, run the most relevant test, and request "
                "completion when the draft is in place.",
                [],
            ),
            step(
                0,
                "Take over the drafted workspace as debugger and finisher. "
                "Re-derive the expected behavior from the original request, find "
                "and fix defects in the draft, run the relevant tests, and mark "
                "the overall task complete only when the behavior is verified.",
                [0],
            ),
        ],
    }
    return templates


def legacy_registered_templates_v1() -> dict[str, dict[str, Any]]:
    """Return the exact registrations used by the frozen pool-v1 corpus."""

    templates = json.loads(json.dumps(registered_templates()))
    for action in templates.values():
        for step in action["steps"]:
            subtask = str(step["subtask"])
            if not subtask.startswith(IMMUTABLE_TESTS_PREAMBLE):
                raise RuntimeError("current mechanics template lost its immutable preamble")
            step["subtask"] = (
                LEGACY_V1_IMMUTABLE_TESTS_PREAMBLE
                + subtask.removeprefix(IMMUTABLE_TESTS_PREAMBLE)
            )
    templates["diagnose_build_verify"]["steps"][0]["subtask"] = (
        LEGACY_V1_IMMUTABLE_TESTS_PREAMBLE
        + "Investigate the repository and original request. Reproduce the issue, "
        "identify the exact behavioral gap and relevant tests, and report a concrete "
        "repair plan without treating this diagnosis as overall completion."
    )
    return templates


def unique_admitted_tasks(
    admitted_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve the first admitted artifact for each logical task."""

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in admitted_rows:
        task_name = str(row["task_name"])
        if task_name in seen:
            continue
        seen.add(task_name)
        unique.append(row)
    return unique


def enforce_frozen_swesmith_band(
    admitted_rows: list[dict[str, Any]],
    band: dict[str, Any],
) -> list[dict[str, Any]]:
    """Exclude repaired tasks absent from the frozen discriminative band."""

    if band.get("version") != "swesmith_sanitized_band_v1":
        raise ValueError("unsupported frozen SWE-Smith band version")
    raw_tasks = band.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("frozen SWE-Smith band tasks must be a list")
    admitted_band: dict[str, dict[str, Any]] = {}
    for row in raw_tasks:
        if not isinstance(row, dict) or not isinstance(row.get("task_name"), str):
            raise ValueError("frozen SWE-Smith band contains an invalid task row")
        name = row["task_name"]
        if name in admitted_band:
            raise ValueError(f"frozen SWE-Smith band duplicates {name}")
        if row.get("status") == "admitted":
            admitted_band[name] = row

    filtered: list[dict[str, Any]] = []
    for row in admitted_rows:
        if row.get("family") != "swesmith_repaired":
            filtered.append(row)
            continue
        name = str(row.get("task_name", ""))
        frozen = admitted_band.get(name)
        if frozen is None:
            continue
        if Path(str(row.get("task_dir", ""))).resolve() != Path(
            str(frozen.get("task_dir", ""))
        ).resolve():
            raise ValueError(f"frozen SWE-Smith task path mismatch for {name}")
        if row.get("task_tree_sha256") != frozen.get("task_tree_sha256"):
            raise ValueError(f"frozen SWE-Smith task tree mismatch for {name}")
        filtered.append(row)
    return filtered


@dataclass(frozen=True)
class MechanicsRunnerConfig:
    repo_root: Path
    pool_path: Path
    runs_root: Path
    results_path: Path
    harbor_bin: Path
    agent_import_path: str = (
        "director.agentic.fugu_mechanics_terminal:FuguMechanicsCollectionAgent"
    )
    concurrency: int = 4
    per_job_timeout_s: float = 2400.0


def job_list(
    admitted_rows: list[dict[str, Any]],
    template_ids: list[str],
) -> list[dict[str, Any]]:
    templates = registered_templates()
    unknown = set(template_ids) - set(templates)
    if unknown:
        raise ValueError(f"unknown template ids: {sorted(unknown)}")
    jobs = []
    for row in unique_admitted_tasks(admitted_rows):
        for template_id in template_ids:
            jobs.append(
                {
                    "collection_id": f"{row['task_name']}__{template_id}",
                    "task_name": row["task_name"],
                    "task_dir": row["task_dir"],
                    "family": row["family"],
                    "template_id": template_id,
                    "action": templates[template_id],
                }
            )
    return jobs


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["collection_id"]] = row
    return results


def _append_result(path: Path, row: dict[str, Any]) -> None:
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _decision_counts(route_log: Path) -> dict[str, int]:
    counts = {"routes": 0}
    if route_log.is_file():
        for line in route_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts["routes"] += 1
    return counts


def run_job(job: dict[str, Any], config: MechanicsRunnerConfig) -> dict[str, Any]:
    job_name = f"mech-{job['collection_id']}-{int(time.time())}"
    env = os.environ.copy()
    env.pop("YUNWU_API_KEY", None)
    env["PYTHONPATH"] = (
        f"{config.repo_root / 'director'}:{config.repo_root / 'ultra'}:{config.repo_root}"
    )
    env[POOL_ENV] = str(config.pool_path)
    env[WORKFLOW_ENV] = json.dumps(job["action"], ensure_ascii=True)
    env[COLLECTION_ID_ENV] = job["collection_id"]
    command = [
        str(config.harbor_bin),
        "run",
        "-p",
        job["task_dir"],
        "--agent-import-path",
        config.agent_import_path,
        "-m",
        f"fugu-mechanics/{job['template_id']}",
        "-l",
        "1",
        "-n",
        "1",
        "-o",
        str(config.runs_root),
        "--job-name",
        job_name,
        "-q",
        "-y",
    ]
    row: dict[str, Any] = {
        "runner_revision": RUNNER_REVISION,
        "collection_id": job["collection_id"],
        "task_name": job["task_name"],
        "task_dir": job["task_dir"],
        "family": job["family"],
        "template_id": job["template_id"],
        "job_name": job_name,
        "worker_calls_are_paid": False,
    }
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=config.repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.per_job_timeout_s,
        )
        row["harbor_returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        row.update({"status": "runner_timeout", "reward": None})
        _append_result(config.results_path, row)
        return row
    row["elapsed_s"] = round(time.monotonic() - started, 1)
    trials = sorted((config.runs_root / job_name).glob("*/result.json"))
    if len(trials) != 1:
        row.update({"status": f"result_count_{len(trials)}", "reward": None})
        _append_result(config.results_path, row)
        return row
    result_path = trials[0]
    trial_dir = result_path.parent
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rewards = ((payload.get("verifier_result") or {}).get("rewards")) or {}
    reward = rewards.get("reward")
    metadata = ((payload.get("agent_result") or {}).get("metadata")) or {}
    exception = (payload.get("exception_info") or {}).get("exception_type")
    row.update(
        {
            "result_path": str(result_path),
            "route_log_path": str(trial_dir / "agent" / "fugu_routes.jsonl"),
            "trajectory_path": str(trial_dir / "agent" / "trajectory.json"),
            "reward": float(reward)
            if isinstance(reward, (int, float)) and not isinstance(reward, bool)
            else None,
            "exception_type": exception,
            "worker_call_attempts": metadata.get("paid_worker_call_attempts"),
            "completed_workflow_steps": metadata.get("completed_workflow_steps"),
            "live_control_failures": metadata.get("live_control_failures"),
            "protected_test_restores": len(
                metadata.get("protected_test_restores") or []
            ),
            "mechanics_pool_id": metadata.get("mechanics_pool_id"),
            **_decision_counts(trial_dir / "agent" / "fugu_routes.jsonl"),
        }
    )
    if row["reward"] is not None:
        row["status"] = "graded"
    else:
        row["status"] = f"ungraded:{exception}"
    _append_result(config.results_path, row)
    return row


def run_jobs(
    jobs: list[dict[str, Any]],
    config: MechanicsRunnerConfig,
    *,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    existing = _load_results(config.results_path)
    pending = [job for job in jobs if job["collection_id"] not in existing]
    if progress:
        print(
            f"jobs={len(jobs)} done={len(existing)} pending={len(pending)}",
            flush=True,
        )
    completed = 0
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for row in pool.map(lambda job: run_job(job, config), pending):
            completed += 1
            if progress:
                print(
                    f"[{completed}/{len(pending)}] {row['collection_id']} "
                    f"status={row['status']} reward={row.get('reward')}",
                    flush=True,
                )
    return _load_results(config.results_path)
