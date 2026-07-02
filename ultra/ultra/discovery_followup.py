"""Plan targeted follow-up discovery jobs from completed rollout evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .workflow_pool_selection import load_completed_rows

VERSION = "fugu_ultra_discovery_followup_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _reward(row: dict[str, Any]) -> float | None:
    value = row.get("reward")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if _reward(row) is not None and row.get("valid_for_training") is not False]
    if not scored:
        return None
    return max(scored, key=lambda row: (_reward(row) or 0.0, bool(row.get("success")), str(row.get("arm"))))


def _split_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {part.strip() for part in value.split(",") if part.strip()}
    return out or None


def _task_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("tournament_task_id") or row.get("source_task_id") or row.get("job_id"))
        groups[key].append(row)
    return groups


def _evidence(group: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = sorted({_reward(row) for row in group if _reward(row) is not None})
    singles = [row for row in group if row.get("stage") == "single_scaffold"]
    roles = [row for row in group if row.get("stage") == "role_workflow"]
    best_single = _best(singles)
    best_role = _best(roles)
    role_delta = None
    if best_single is not None and best_role is not None:
        role_delta = (_reward(best_role) or 0.0) - (_reward(best_single) or 0.0)
    return {
        "completed_rollouts": len(group),
        "trainable_rollouts": sum(1 for row in group if _reward(row) is not None),
        "reward_values": rewards,
        "reward_variance": len(rewards) > 1,
        "best_single_reward": _reward(best_single) if best_single else None,
        "best_role_reward": _reward(best_role) if best_role else None,
        "role_delta": role_delta,
        "grader_quarantines": sum(1 for row in group if row.get("outcome_class") == "grader_crash_quarantine"),
    }


def _matches(job: dict[str, Any], filters: dict[str, set[str] | None]) -> bool:
    for key, allowed in filters.items():
        if allowed is not None and str(job.get(key)) not in allowed:
            return False
    return True


def _candidate_jobs(
    *,
    jobs: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    filters: dict[str, set[str] | None],
    mode: str,
    stages: set[str] | None,
) -> list[dict[str, Any]]:
    completed_job_ids = {str(row.get("job_id")) for row in completed_rows}
    evidence_by_task = _task_groups(completed_rows)
    jobs_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if not _matches(job, filters):
            continue
        if stages is not None and str(job.get("stage")) not in stages:
            continue
        jobs_by_task[str(job.get("tournament_task_id") or job.get("source_task_id"))].append(job)

    candidates: list[dict[str, Any]] = []
    for task_id, task_jobs in jobs_by_task.items():
        evidence = _evidence(evidence_by_task.get(task_id, []))
        missing = [job for job in task_jobs if str(job.get("job_id")) not in completed_job_ids]
        if not missing:
            continue

        reasons: list[str] = []
        priority = 50
        if mode in {"targeted", "complete-variance"} and evidence["reward_variance"]:
            reasons.append("complete_reward_variance_task")
            priority = min(priority, 0)
        if mode in {"targeted", "role-followup"} and evidence["best_single_reward"] is not None:
            if float(evidence["best_single_reward"]) < 1.0:
                reasons.append("best_single_not_solved")
                priority = min(priority, 1)
        if mode in {"targeted", "single-prefilter"} and evidence["trainable_rollouts"] == 0:
            reasons.append("fresh_single_prefilter")
            priority = min(priority, 2)
        if mode == "all-missing":
            reasons.append("all_missing_jobs")
            priority = min(priority, 3)

        if not reasons:
            continue

        for job in missing:
            job_stage = str(job.get("stage"))
            if mode == "single-prefilter" and str(job.get("stage")) != "single_scaffold":
                continue
            if mode == "role-followup" and str(job.get("stage")) != "role_workflow":
                continue
            if mode == "targeted":
                if priority == 2 and job_stage != "single_scaffold":
                    continue
                if priority == 1 and not evidence["reward_variance"] and job_stage != "role_workflow":
                    continue
            candidates.append(
                {
                    "priority": priority,
                    "reason": sorted(set(reasons)),
                    "evidence": evidence,
                    "job": job,
                }
            )

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        evidence = item["evidence"]
        job = item["job"]
        return (
            int(item["priority"]),
            -int(evidence["trainable_rollouts"]),
            str(job.get("source")),
            str(job.get("tournament_task_id")),
            str(job.get("stage")),
            str(job.get("job_id")),
        )

    return sorted(candidates, key=sort_key)


def build_discovery_followup_plan(
    *,
    manifest_dir: Path,
    jobs_jsonl: Path | None = None,
    out_json: Path | None = None,
    jobs_out: Path | None = None,
    sources: set[str] | None = None,
    lanes: set[str] | None = None,
    arm_domains: set[str] | None = None,
    stages: set[str] | None = None,
    mode: str = "targeted",
    max_jobs: int = 32,
    max_task_groups: int = 12,
) -> dict[str, Any]:
    jobs_path = jobs_jsonl or manifest_dir / "scaffold_tournament_jobs.jsonl"
    jobs = _read_jsonl(jobs_path)
    completed_rows = load_completed_rows(manifest_dir)
    filters = {
        "source": sources,
        "lane": lanes,
        "arm_domain": arm_domains,
    }
    candidates = _candidate_jobs(
        jobs=jobs,
        completed_rows=completed_rows,
        filters=filters,
        mode=mode,
        stages=stages,
    )

    selected: list[dict[str, Any]] = []
    selected_task_groups: set[str] = set()
    for candidate in candidates:
        job = candidate["job"]
        task_id = str(job.get("tournament_task_id") or job.get("source_task_id"))
        if task_id not in selected_task_groups and len(selected_task_groups) >= max_task_groups:
            continue
        selected_task_groups.add(task_id)
        selected.append(candidate)
        if len(selected) >= max_jobs:
            break

    selected_jobs = [item["job"] for item in selected]
    reason_counts = Counter(reason for item in selected for reason in item["reason"])
    stage_counts = Counter(str(job.get("stage")) for job in selected_jobs)
    source_counts = Counter(str(job.get("source")) for job in selected_jobs)
    lane_counts = Counter(str(job.get("lane")) for job in selected_jobs)
    job_ids = [str(job.get("job_id")) for job in selected_jobs]
    job_id_args = " ".join(f"--job-id {job_id}" for job_id in job_ids)
    report = {
        "version": VERSION,
        "purpose": "Plan targeted fixed-workflow discovery followups from completed rollout evidence.",
        "manifest_dir": str(manifest_dir.resolve()),
        "jobs_jsonl": str(jobs_path.resolve()),
        "mode": mode,
        "max_jobs": max_jobs,
        "max_task_groups": max_task_groups,
        "completed_rollouts": len(completed_rows),
        "candidate_jobs": len(candidates),
        "selected_jobs": len(selected_jobs),
        "selected_task_groups": len(selected_task_groups),
        "reason_counts": dict(sorted(reason_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "lane_counts": dict(sorted(lane_counts.items())),
        "job_ids": job_ids,
        "job_id_args": job_id_args,
        "run_command_template": (
            f"../.venv/bin/python -m ultra.cli scaffold-discovery-run --jobs-jsonl {jobs_path.resolve()} "
            "<out-dir/report/provider/options> "
            f"{job_id_args}"
        ).strip(),
        "selected": [
            {
                "job_id": str(item["job"].get("job_id")),
                "tournament_task_id": item["job"].get("tournament_task_id"),
                "source": item["job"].get("source"),
                "lane": item["job"].get("lane"),
                "arm": item["job"].get("arm"),
                "stage": item["job"].get("stage"),
                "priority": item["priority"],
                "reason": item["reason"],
                "evidence": item["evidence"],
            }
            for item in selected
        ],
    }
    if out_json is not None:
        _write_json(out_json, report)
    if jobs_out is not None:
        _write_jsonl(jobs_out, selected_jobs)
    return report


def split_csv(value: str | None) -> set[str] | None:
    return _split_csv(value)
