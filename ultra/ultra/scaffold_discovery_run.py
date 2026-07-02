"""Run and analyze fixed-workflow discovery jobs.

The scaffold tournament manifest is deliberately larger than a canary. This module
executes controlled shards from ``scaffold_tournament_jobs.jsonl`` and records one
RolloutRecord per job. It defaults to dry-run mode so the full 1,600-job surface is
never launched by accident.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

from .config import PoolConfig, WorkerSpec
from .docker_janitor import cleanup_stale_docker_networks
from .executor import execute_workflow
from .failure_taxonomy import apply_outcome_class
from .providers import logical_name
from .providers import provider as provider_config
from .providers import required_key_envs
from .scaffold_canary import load_taskspecs, rollout_to_json, select_arm, select_task
from .scaffold_tournament import TASKTROVE_UNIT_CODE_SOURCES, canonical_workers, worker_harness_map
from .schemas import RolloutRecord, StepBudget, Workflow
from .workers import FakeProvider, Sampling, WorkerPool
from .workers.factory import build_pool as build_live_pool

RUN_VERSION = "fugu_ultra_scaffold_discovery_run_v1"
ANALYSIS_VERSION = "fugu_ultra_scaffold_discovery_analysis_v1"


def _normalized_lane(job: dict[str, Any]) -> Any:
    if str(job.get("source") or "") in TASKTROVE_UNIT_CODE_SOURCES:
        return "unit_and_scientific_code"
    return job.get("lane")


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


def _load_dotenv(path: Path) -> list[str]:
    if not path.exists():
        return []
    loaded: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_cache() -> dict[Path, Any]:
    return {}


def _load_task(path: Path, task_id: str, cache: dict[Path, Any]) -> Any:
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = load_taskspecs(resolved)
    return select_task(cache[resolved], task_id)


def _split_csv(values: str | None) -> set[str] | None:
    if not values:
        return None
    out = {part.strip() for part in values.split(",") if part.strip()}
    return out or None


def _reasoning(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "off", "false"}:
        return None
    return value


def _job_paths(out_dir: Path, job_id: str) -> dict[str, Path]:
    return {
        "rollout": out_dir / "rollouts" / f"{job_id}.json",
        "artifact_dir": out_dir / "artifacts" / job_id,
        "error": out_dir / "errors" / f"{job_id}.json",
    }


def _job_done(out_dir: Path, job: dict[str, Any]) -> bool:
    return _job_paths(out_dir, str(job["job_id"]))["rollout"].exists()


def _select_jobs(
    jobs: list[dict[str, Any]],
    *,
    out_dir: Path,
    job_ids: set[str] | None = None,
    lanes: set[str] | None = None,
    arms: set[str] | None = None,
    stages: set[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    selected = []
    skipped_existing = 0
    for job in jobs:
        if job_ids is not None and str(job.get("job_id")) not in job_ids:
            continue
        if lanes is not None and str(job.get("lane")) not in lanes:
            continue
        if arms is not None and str(job.get("arm")) not in arms:
            continue
        if stages is not None and str(job.get("stage")) not in stages:
            continue
        if resume and _job_done(out_dir, job):
            skipped_existing += 1
            continue
        selected.append(job)
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped_existing


def _worker_specs() -> list[WorkerSpec]:
    return [WorkerSpec(worker_id=worker.name, model=worker.model) for worker in canonical_workers()]


def _fake_pool() -> WorkerPool:
    return WorkerPool(_worker_specs(), FakeProvider())


def _live_pool(
    *,
    provider_name: str | None,
    cache_dir: Path,
    max_concurrency: int,
    requests_per_minute: float | None,
    timeout_s: float,
    max_retries: int,
) -> WorkerPool:
    split_provider_routing = provider_name is None
    if split_provider_routing:
        missing = [key_env for key_env in required_key_envs([w.model for w in _worker_specs()]) if not os.environ.get(key_env)]
        if missing:
            raise RuntimeError(
                f"{', '.join(missing)} missing; put required provider keys in the environment or .env before live discovery"
            )
        base_url = PoolConfig().base_url
        key_env = PoolConfig().api_key_env
    else:
        cfg = provider_config(provider_name)
        key_env = str(cfg.get("key_env") or "")
        if key_env and not os.environ.get(key_env):
            raise RuntimeError(f"{key_env} is not set; put it in the environment or .env before live discovery")
        base_url = str(cfg["base_url"])
    return build_live_pool(
        _worker_specs(),
        PoolConfig(
            base_url=base_url,
            api_key_env=key_env,
            split_provider_routing=split_provider_routing,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            timeout_s=timeout_s,
            cache_dir=str(cache_dir),
            budget_usd=None,
            prompt_caching=True,
        ),
    )


def _job_harnesses(job: dict[str, Any]) -> dict[str, str]:
    mapping = job.get("worker_harness_map")
    if isinstance(mapping, dict) and mapping:
        return {str(key): str(value) for key, value in mapping.items()}
    workers = canonical_workers()
    task_harness = str(job.get("task_harness") or "")
    return {
        name: worker_harness_map(workers, task_harness=task_harness)[name]
        for name in job.get("worker_names", [])
    }


def _selected_worker_models(jobs: list[dict[str, Any]]) -> set[str]:
    worker_by_name = {worker.name: worker.model for worker in canonical_workers()}
    return {
        worker_by_name[name]
        for job in jobs
        for name in (job.get("worker_names") or [])
        if name in worker_by_name
    }


def _selected_uses_terminal_sandbox(jobs: list[dict[str, Any]]) -> bool:
    for job in jobs:
        if str(job.get("task_harness") or "") == "terminal_sandbox":
            return True
        if "terminal_sandbox" in set(_job_harnesses(job).values()):
            return True
    return False


def _validate_provider_override(provider_name: str | None, selected_jobs: list[dict[str, Any]]) -> None:
    if provider_name != "openrouter":
        return
    if any(logical_name(model) == "gpt" for model in _selected_worker_models(selected_jobs)):
        raise ValueError("GPT workers must not be routed through OpenRouter")


def _job_allows_training(job: dict[str, Any]) -> bool:
    source_policy = str(job.get("source_policy") or "train_allowed")
    task_split = str(job.get("task_split") or "grpo_train")
    return source_policy == "train_allowed" and task_split == "grpo_train"


def _apply_job_policy(record: RolloutRecord, job: dict[str, Any]) -> RolloutRecord:
    if _job_allows_training(job):
        return record
    return record.model_copy(update={"valid_for_training": False})


async def _run_one(
    job: dict[str, Any],
    *,
    out_dir: Path,
    pool: WorkerPool,
    sampling: Sampling,
    budget: StepBudget | None,
    task_cache: dict[Path, Any],
) -> tuple[str, dict[str, Any]]:
    job_id = str(job["job_id"])
    paths = _job_paths(out_dir, job_id)
    try:
        task = _load_task(Path(str(job["task_jsonl"])), str(job["source_task_id"]), task_cache)
        arm = select_arm(str(job["arm"]))
        workflow = arm.workflow
        if budget is not None:
            workflow = Workflow(steps=[step.model_copy(update={"budget": budget}) for step in workflow.steps])
        workers = canonical_workers()
        record = await execute_workflow(
            task,
            workflow,
            pool,
            sampling,
            job_id,
            worker_ids=[worker.name for worker in workers],
            worker_harnesses=_job_harnesses(job),
            artifact_dir=paths["artifact_dir"],
        )
        record = _apply_job_policy(record, job)
        paths["rollout"].parent.mkdir(parents=True, exist_ok=True)
        paths["rollout"].write_text(rollout_to_json(record) + "\n")
        row = _summary_row(job, record, paths["rollout"], paths["artifact_dir"])
        return "ok", row
    except Exception as exc:  # noqa: BLE001 - live providers and CLIs fail heterogeneously
        payload = {
            "job_id": job_id,
            "status": "error",
            "lane": job.get("lane"),
            "arm": job.get("arm"),
            "source_task_id": job.get("source_task_id"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(paths["error"], payload)
        return "error", {**payload, "error_out": str(paths["error"])}


async def _run_one_with_timeout(
    job: dict[str, Any],
    *,
    out_dir: Path,
    pool: WorkerPool,
    sampling: Sampling,
    budget: StepBudget | None,
    task_cache: dict[Path, Any],
    job_timeout_s: float | None,
) -> tuple[str, dict[str, Any]]:
    if job_timeout_s is None or job_timeout_s <= 0:
        return await _run_one(
            job,
            out_dir=out_dir,
            pool=pool,
            sampling=sampling,
            budget=budget,
            task_cache=task_cache,
        )
    try:
        return await asyncio.wait_for(
            _run_one(
                job,
                out_dir=out_dir,
                pool=pool,
                sampling=sampling,
                budget=budget,
                task_cache=task_cache,
            ),
            timeout=job_timeout_s,
        )
    except asyncio.TimeoutError:
        job_id = str(job["job_id"])
        paths = _job_paths(out_dir, job_id)
        payload = {
            "job_id": job_id,
            "status": "timeout",
            "lane": job.get("lane"),
            "arm": job.get("arm"),
            "source_task_id": job.get("source_task_id"),
            "job_timeout_s": job_timeout_s,
        }
        _write_json(paths["error"], payload)
        return "timeout", {**payload, "error_out": str(paths["error"])}


def _summary_row(job: dict[str, Any], record: RolloutRecord, rollout_out: Path, artifact_dir: Path) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "status": "ok",
        "lane": job.get("lane"),
        "arm": job.get("arm"),
        "stage": job.get("stage"),
        "source": job.get("source"),
        "source_task_id": job.get("source_task_id"),
        "source_policy": job.get("source_policy"),
        "task_split": job.get("task_split"),
        "worker_names": job.get("worker_names", []),
        "worker_harness_map": job.get("worker_harness_map", {}),
        "task_id": record.task_id,
        "reward": record.reward,
        "outcome_class": record.outcome_class,
        "valid_for_training": record.valid_for_training,
        "grade_success": record.grade.success if record.grade else None,
        "grade_score": record.grade.score if record.grade else None,
        "failure_class": record.failure_class,
        "step_terminations": [step.termination for step in record.execution.steps],
        "rollout_out": str(rollout_out),
        "artifact_dir": str(artifact_dir),
    }


async def run_scaffold_discovery_jobs(
    *,
    jobs_jsonl: Path,
    out_dir: Path,
    report_out: Path,
    dry_run: bool = True,
    live: bool = False,
    fake: bool = False,
    limit: int | None = None,
    job_ids: set[str] | None = None,
    lanes: set[str] | None = None,
    arms: set[str] | None = None,
    stages: set[str] | None = None,
    resume: bool = True,
    parallel: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    reasoning: str | None = "high",
    budget: StepBudget | None = None,
    provider_name: str | None = None,
    dotenv: Path | None = None,
    max_concurrency: int = 4,
    requests_per_minute: float | None = None,
    timeout_s: float = 300.0,
    job_timeout_s: float | None = None,
    max_retries: int = 4,
    docker_network_janitor: bool = False,
    docker_network_janitor_dry_run: bool = False,
) -> dict[str, Any]:
    if live and fake:
        raise ValueError("choose only one of live or fake")
    if live:
        dry_run = False
    if fake:
        dry_run = False

    loaded_env = _load_dotenv(dotenv or _repo_root() / ".env")
    jobs = _read_jsonl(jobs_jsonl)
    selected, skipped_existing = _select_jobs(
        jobs,
        out_dir=out_dir,
        job_ids=job_ids,
        lanes=lanes,
        arms=arms,
        stages=stages,
        limit=limit,
        resume=resume,
    )
    _validate_provider_override(provider_name, selected)
    janitor_report: dict[str, Any] | None = None
    if docker_network_janitor and _selected_uses_terminal_sandbox(selected):
        janitor_report = cleanup_stale_docker_networks(
            dry_run=dry_run or docker_network_janitor_dry_run,
        )

    if dry_run:
        report = {
            "version": RUN_VERSION,
            "mode": "dry_run",
            "jobs_jsonl": str(jobs_jsonl.resolve()),
            "out_dir": str(out_dir.resolve()),
            "report_out": str(report_out.resolve()),
            "total_jobs": len(jobs),
            "selected_jobs": len(selected),
            "skipped_existing": skipped_existing,
            "filters": {
                "job_ids": sorted(job_ids) if job_ids else None,
                "lanes": sorted(lanes) if lanes else None,
                "arms": sorted(arms) if arms else None,
                "stages": sorted(stages) if stages else None,
                "limit": limit,
                "resume": resume,
            },
            "selected_sample": selected[:20],
            "loaded_env_keys": loaded_env,
            "docker_network_janitor": janitor_report,
            "live_calls": False,
        }
        _write_json(report_out, report)
        return report

    pool = (
        _live_pool(
            provider_name=provider_name,
            cache_dir=out_dir / "completion_cache",
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        if live
        else _fake_pool()
    )
    sampling = Sampling(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        reasoning_effort=_reasoning(reasoning),
    )
    task_cache = _task_cache()
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, parallel))

    async def guarded(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            return await _run_one_with_timeout(
                job,
                out_dir=out_dir,
                pool=pool,
                sampling=sampling,
                budget=budget,
                task_cache=task_cache,
                job_timeout_s=job_timeout_s,
            )

    if selected:
        results = await asyncio.gather(*(guarded(job) for job in selected))
        for status, row in results:
            counts[status] += 1
            rows.append(row)

    report = {
        "version": RUN_VERSION,
        "mode": "live" if live else "fake",
        "jobs_jsonl": str(jobs_jsonl.resolve()),
        "out_dir": str(out_dir.resolve()),
        "report_out": str(report_out.resolve()),
        "total_jobs": len(jobs),
        "selected_jobs": len(selected),
        "skipped_existing": skipped_existing,
        "parallel": parallel,
        "job_timeout_s": job_timeout_s,
        "counts": dict(counts),
        "rows": rows,
        "loaded_env_keys": loaded_env,
        "docker_network_janitor": janitor_report,
        "live_calls": live and bool(selected),
    }
    _write_json(report_out, report)
    return report


def _load_rollout(path: Path) -> RolloutRecord | None:
    try:
        return apply_outcome_class(RolloutRecord.model_validate(json.loads(path.read_text())))
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze_scaffold_discovery(
    *,
    jobs_jsonl: Path,
    out_dir: Path,
    report_out: Path | None = None,
) -> dict[str, Any]:
    jobs = {str(job["job_id"]): job for job in _read_jsonl(jobs_jsonl)}
    rows = []
    for path in sorted((out_dir / "rollouts").glob("*.json")):
        record = _load_rollout(path)
        if record is None:
            continue
        job = jobs.get(record.rollout_id, {})
        record = _apply_job_policy(record, job)
        rows.append(
            {
                "job_id": record.rollout_id,
                "lane": _normalized_lane(job),
                "arm": job.get("arm"),
                "stage": job.get("stage"),
                "tournament_task_id": job.get("tournament_task_id"),
                "source_policy": job.get("source_policy"),
                "task_split": job.get("task_split"),
                "reward": record.reward,
                "success": record.grade.success if record.grade else None,
                "valid_for_training": record.valid_for_training,
                "outcome_class": record.outcome_class,
            }
        )

    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lane[str(row.get("lane"))].append(row)
        by_arm[str(row.get("arm"))].append(row)
        if row.get("tournament_task_id"):
            by_task[str(row["tournament_task_id"])].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        rewards = [float(row["reward"]) for row in group if row.get("reward") is not None]
        return {
            "rollouts": len(group),
            "trainable": sum(1 for row in group if row.get("valid_for_training")),
            "successes": sum(1 for row in group if row.get("success") is True),
            "mean_reward": _mean(rewards),
        }

    task_groups_with_multiple = [
        group for group in by_task.values() if len([row for row in group if row.get("reward") is not None]) >= 2
    ]
    varying_groups = []
    for group in task_groups_with_multiple:
        rewards = {row.get("reward") for row in group if row.get("reward") is not None}
        if len(rewards) > 1:
            varying_groups.append(group)

    lane_stage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        lane_stage[str(row.get("lane"))][str(row.get("stage"))] += 1

    paired = []
    for task_id, group in sorted(by_task.items()):
        singles = [row for row in group if row.get("stage") == "single_scaffold" and row.get("reward") is not None]
        roles = [row for row in group if row.get("stage") == "role_workflow" and row.get("reward") is not None]
        if not singles or not roles:
            continue
        best_single_reward = max(float(row["reward"]) for row in singles)
        best_role_reward = max(float(row["reward"]) for row in roles)
        best_single_success = any(row.get("success") is True for row in singles)
        best_role_success = any(row.get("success") is True for row in roles)
        paired.append(
            {
                "tournament_task_id": task_id,
                "lane": group[0].get("lane"),
                "single_rollouts": len(singles),
                "role_rollouts": len(roles),
                "best_single_reward": best_single_reward,
                "best_role_reward": best_role_reward,
                "reward_delta": best_role_reward - best_single_reward,
                "best_single_success": best_single_success,
                "best_role_success": best_role_success,
                "role_success_delta": int(best_role_success) - int(best_single_success),
            }
        )

    reward_deltas = [float(row["reward_delta"]) for row in paired]
    role_success_deltas = [int(row["role_success_delta"]) for row in paired]

    report = {
        "version": ANALYSIS_VERSION,
        "jobs_jsonl": str(jobs_jsonl.resolve()),
        "out_dir": str(out_dir.resolve()),
        "rollouts": len(rows),
        "by_lane": {lane: summarize(group) for lane, group in sorted(by_lane.items())},
        "by_arm": {arm: summarize(group) for arm, group in sorted(by_arm.items())},
        "lane_stage_counts": {lane: dict(counter) for lane, counter in sorted(lane_stage.items())},
        "task_groups": len(by_task),
        "task_groups_with_multiple_rewards": len(task_groups_with_multiple),
        "task_groups_with_reward_variance": len(varying_groups),
        "task_reward_variance_rate": (
            len(varying_groups) / len(task_groups_with_multiple) if task_groups_with_multiple else None
        ),
        "paired_single_vs_role": {
            "task_groups": len(paired),
            "role_beats_best_single_reward": sum(1 for delta in reward_deltas if delta > 0),
            "role_matches_best_single_reward": sum(1 for delta in reward_deltas if delta == 0),
            "role_loses_to_best_single_reward": sum(1 for delta in reward_deltas if delta < 0),
            "mean_reward_delta": _mean(reward_deltas),
            "role_improves_success": sum(1 for delta in role_success_deltas if delta > 0),
            "role_matches_success": sum(1 for delta in role_success_deltas if delta == 0),
            "role_loses_success": sum(1 for delta in role_success_deltas if delta < 0),
            "examples": paired[:20],
        },
        "go_no_go_hint": (
            "pending_rollouts"
            if not rows
            else "needs_more_groups"
            if len(task_groups_with_multiple) == 0
            else "variance_present"
            if varying_groups
            else "no_variance_observed"
        ),
    }
    if report_out is not None:
        _write_json(report_out, report)
    return report
