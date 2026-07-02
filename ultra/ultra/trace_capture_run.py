"""Execute trace-capture jobs with resume-friendly output handling."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .providers import load_dotenv
from .scaffold_canary import rollout_to_json, run_canary, load_taskspecs, select_task
from .scaffold_tournament import canonical_workers
from .trace_export import write_agent_trace


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _job_done(job: dict[str, Any]) -> bool:
    rollout = Path(str(job["rollout_out"]))
    trace = Path(str(job["agent_trace_out"]))
    artifact_dir = Path(str(job["artifact_dir"]))
    return rollout.exists() and trace.exists() and artifact_dir.exists()


def _select_jobs(
    jobs: list[dict[str, Any]],
    *,
    job_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    selected = [job for job in jobs if job_ids is None or str(job.get("job_id")) in job_ids]
    if resume:
        selected = [job for job in selected if not _job_done(job)]
    if limit is not None:
        selected = selected[:limit]
    return selected


async def run_trace_capture_jobs(
    *,
    jobs_jsonl: Path,
    report_out: Path,
    limit: int | None = None,
    job_ids: set[str] | None = None,
    resume: bool = True,
    parallel: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    reasoning: str | None = "high",
    dotenv: Path | None = None,
) -> dict[str, Any]:
    loaded_env = load_dotenv(dotenv)
    jobs = _read_jobs(jobs_jsonl)
    requested = [job for job in jobs if job_ids is None or str(job.get("job_id")) in job_ids]
    skipped_existing = sum(1 for job in requested if resume and _job_done(job))
    selected = _select_jobs(jobs, job_ids=job_ids, limit=limit, resume=resume)
    workers = canonical_workers()
    worker_models = {worker.worker_id: worker.model for worker in workers}

    counts: Counter[str] = Counter()
    parallel = max(1, parallel)

    async def run_one(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        job_id = str(job["job_id"])
        rollout_out = Path(str(job["rollout_out"]))
        trace_out = Path(str(job["agent_trace_out"]))
        artifact_dir = Path(str(job["artifact_dir"]))
        try:
            record = await run_canary(
                tasks_jsonl=Path(str(job["task_jsonl"])),
                task_id=str(job["task_id"]),
                arm_name=str(job["arm"]),
                rollout_id=job_id,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                budget=job.get("budget") or "short",
                artifact_dir=artifact_dir,
            )
            rollout_out.parent.mkdir(parents=True, exist_ok=True)
            rollout_out.write_text(rollout_to_json(record) + "\n")
            task = select_task(load_taskspecs(Path(str(job["task_jsonl"]))), str(job["task_id"]))
            trace = write_agent_trace(record, task, worker_models=worker_models, out=trace_out)
            status = "ok"
            return status, (
                {
                    "job_id": job_id,
                    "status": status,
                    "task_id": record.task_id,
                    "arm": job["arm"],
                    "reward": record.reward,
                    "grade_success": record.grade.success if record.grade else None,
                    "rollout_out": str(rollout_out),
                    "agent_trace_out": str(trace_out),
                    "artifact_dir": str(artifact_dir),
                    "trace_event_count": len(trace.events),
                    "trace_has_patch": bool(trace.artifacts.final_patch_ref),
                    "trace_has_workspace": bool(trace.artifacts.workspace_snapshot_ref),
                    "trace_has_grade": bool(trace.artifacts.hidden_grade_ref),
                }
            )
        except Exception as exc:  # noqa: BLE001 - live providers/CLIs fail heterogeneously
            error_out = rollout_out.with_suffix(".error.json")
            payload = {
                "job_id": job_id,
                "status": "error",
                "task_id": job.get("task_id"),
                "arm": job.get("arm"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(error_out, payload)
            return "error", {**payload, "error_out": str(error_out)}

    semaphore = asyncio.Semaphore(parallel)

    async def guarded(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            return await run_one(job)

    rows: list[dict[str, Any]] = []
    if selected:
        results = await asyncio.gather(*(guarded(job) for job in selected))
        for status, row in results:
            counts[status] += 1
            rows.append(row)

    report = {
        "version": "fugu_ultra_trace_capture_run_v1",
        "jobs_jsonl": str(jobs_jsonl.resolve()),
        "report_out": str(report_out.resolve()),
        "total_jobs": len(jobs),
        "selected_jobs": len(selected),
        "skipped_existing": skipped_existing,
        "parallel": parallel,
        "loaded_env_keys": loaded_env,
        "counts": dict(counts),
        "rows": rows,
        "live_calls": bool(selected),
    }
    _write_json(report_out, report)
    return report
