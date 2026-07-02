"""Plan repeatable TaskTrove prefilter shards for GRPO pilot expansion."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .scaffold_tournament import (
    DIRECT_LANE,
    LONG_CONTEXT_LANE,
    REPO_LANE,
    TOOL_LANE,
    UNIT_CODE_LANE,
    TASKTROVE_UNIT_CODE_SOURCES,
    write_concrete_manifest,
    write_readiness,
)
from .sources.harbor import materialize_tasktrove_parquet
from .workflow_pool_selection import load_completed_rows

VERSION = "fugu_ultra_tasktrove_prefilter_batch_v1"
DEFAULT_SELECTION = "ranked"

TASKTROVE_SHARDS = {
    "tasktrove_inferredbugs": "tasktrove_harbor/inferredbugs_train_taskspecs.jsonl",
    "tasktrove_pymethods2test": "tasktrove_harbor/pymethods2test_train_taskspecs.jsonl",
}
TASKTROVE_SOURCE_PREFIX = "tasktrove_"
TASKTROVE_FOLDER_TO_SOURCE = {
    "DCAgent2__nl2bash-tasks-cleaned-oracle": "tasktrove_nl2bash",
    "DCAgent__code-contests-noblock": "tasktrove_code_contests",
    "DCAgent__exp_rpt_multifile": "tasktrove_multifile_composition",
    "DCAgent__inferredbugs-sandboxes-verifier": "tasktrove_inferredbugs",
    "DCAgent__r2egym-patched-full-oracle": "tasktrove_r2egym",
    "SankalpKJ__swesmith-oracle-filtered": "tasktrove_swesmith",
    "laion__exp_rpt_scaffold-v2": "tasktrove_repo_scaffold",
    "laion__swegym-tasks-patched-validated-v5": "tasktrove_swegym",
}
TASKTROVE_SOURCE_TO_FOLDER = {source: folder for folder, source in TASKTROVE_FOLDER_TO_SOURCE.items()}
DEFAULT_AGENTTROVE_EXACT_COUNTS = {
    "tasktrove_r2egym": 20,
    "tasktrove_swesmith": 20,
    "tasktrove_nl2bash": 8,
}

DISCOVERY_LANES = (REPO_LANE, UNIT_CODE_LANE, DIRECT_LANE, TOOL_LANE, LONG_CONTEXT_LANE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def _is_tasktrove_source(source: str) -> bool:
    return source.startswith(TASKTROVE_SOURCE_PREFIX)


def _source_name(row: dict[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        return str(source.get("name") or "")
    return ""


def _lane_for_tasktrove_row(row: dict[str, Any]) -> str:
    source = _source_name(row)
    if source in TASKTROVE_UNIT_CODE_SOURCES:
        return UNIT_CODE_LANE
    return REPO_LANE


def _task_mix_for_tasktrove_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    lane_counts = Counter(_lane_for_tasktrove_row(row) for row in rows)
    return {lane: int(lane_counts.get(lane, 0)) for lane in DISCOVERY_LANES}


def _split(row: dict[str, Any]) -> str:
    splitting = row.get("splitting")
    if isinstance(splitting, dict):
        return str(splitting.get("split") or "")
    return ""


def _policy(row: dict[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        return str(source.get("policy") or "")
    return ""


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("source_task_id") or "")


def _agenttrove_exact_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("tasktrove_folder") or ""), str(row.get("tasktrove_path") or "")


def _model_count(row: dict[str, Any]) -> int:
    models = row.get("models")
    return len(models) if isinstance(models, dict) else 0


def _success_rate(row: dict[str, Any]) -> float | None:
    value = row.get("success_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prompt_text(row: dict[str, Any]) -> str:
    messages = ((row.get("input") or {}).get("messages") or [])
    return "\n".join(str(message.get("content") or "") for message in messages if isinstance(message, dict))


def _numeric_suffix(row: dict[str, Any]) -> int | None:
    match = re.search(r"-(\d+)$", _task_id(row))
    return int(match.group(1)) if match else None


def _stable_jitter(task_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _complexity_features(row: dict[str, Any]) -> dict[str, Any]:
    text = _prompt_text(row)
    lower = text.lower()
    return {
        "prompt_chars": len(text),
        "prompt_lines": text.count("\n") + 1 if text else 0,
        "code_fences": text.count("```") // 2,
        "constraint_terms": sum(
            lower.count(term)
            for term in (
                "constraint",
                "must",
                "should",
                "edge",
                "bug",
                "test",
                "input",
                "output",
                "function",
                "class",
                "file",
            )
        ),
        "numeric_suffix": _numeric_suffix(row),
    }


def _complexity_score(row: dict[str, Any], *, seed: int) -> float:
    features = _complexity_features(row)
    task_id = _task_id(row)
    return (
        min(float(features["prompt_chars"]) / 2000.0, 60.0)
        + min(float(features["prompt_lines"]) / 40.0, 20.0)
        + float(features["code_fences"]) * 2.0
        + min(float(features["constraint_terms"]) / 4.0, 12.0)
        + _stable_jitter(task_id, seed) * 0.01
    )


def _select_rows(
    available: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    selection: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count <= 0:
        return [], []
    if selection == "sequential":
        selected = available[:count]
        diagnostics = [
            {
                "task_id": _task_id(row),
                "score": None,
                "features": _complexity_features(row),
            }
            for row in selected
        ]
        return selected, diagnostics
    if selection != "ranked":
        raise ValueError("selection must be 'ranked' or 'sequential'")

    ranked = sorted(
        available,
        key=lambda row: (_complexity_score(row, seed=seed), _stable_jitter(_task_id(row), seed)),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_nums: list[int] = []
    min_gap = 3 if len(available) >= count * 3 else 1
    for row in ranked:
        num = _numeric_suffix(row)
        if num is not None and any(abs(num - other) < min_gap for other in selected_nums):
            continue
        selected.append(row)
        if num is not None:
            selected_nums.append(num)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_ids = {_task_id(row) for row in selected}
        for row in ranked:
            if _task_id(row) in selected_ids:
                continue
            selected.append(row)
            if len(selected) == count:
                break

    diagnostics = [
        {
            "task_id": _task_id(row),
            "score": round(_complexity_score(row, seed=seed), 4),
            "features": _complexity_features(row),
        }
        for row in selected
    ]
    return selected, diagnostics


def discover_tasktrove_task_files(manifest_dir: Path) -> dict[str, list[Path]]:
    """Find locally materialized train-allowed TaskTrove TaskSpec files by source."""

    source_paths: dict[str, set[Path]] = defaultdict(set)
    roots = [manifest_dir / "tasktrove_harbor"]
    for root in roots:
        if not root.exists():
            continue
        patterns = ["*taskspecs.jsonl", "**/*taskspecs.jsonl"]
        for pattern in patterns:
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                for row in _read_jsonl(path):
                    source = _source_name(row)
                    if not _is_tasktrove_source(source):
                        continue
                    if _policy(row) != "train_allowed" or _split(row) != "grpo_train":
                        continue
                    source_paths[source].add(path)
                    break

    for source, rel in TASKTROVE_SHARDS.items():
        path = manifest_dir / rel
        if path.exists():
            source_paths[source].add(path)
    return {source: sorted(paths) for source, paths in sorted(source_paths.items())}


def collect_seen_tasktrove_ids(manifest_dir: Path) -> dict[str, set[str]]:
    """Collect TaskTrove task IDs already used by completed evidence or prefilter shards."""

    known_sources = set(discover_tasktrove_task_files(manifest_dir)) | set(TASKTROVE_SHARDS)
    seen: dict[str, set[str]] = {source: set() for source in known_sources}
    for row in load_completed_rows(manifest_dir):
        source = str(row.get("source") or "")
        if _is_tasktrove_source(source) and row.get("source_task_id"):
            seen.setdefault(source, set())
            seen[source].add(str(row["source_task_id"]))

    seed_manifest = manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl"
    for row in _read_jsonl(seed_manifest):
        source = str(row.get("source") or "")
        if _is_tasktrove_source(source) and row.get("source_task_id"):
            seen.setdefault(source, set())
            seen[source].add(str(row["source_task_id"]))

    for path in sorted(manifest_dir.glob("tasktrove_prefilter*/taskspecs.jsonl")):
        for row in _read_jsonl(path):
            source = _source_name(row)
            if _is_tasktrove_source(source):
                seen.setdefault(source, set())
                seen[source].add(_task_id(row))
    return seen


def _available_rows(manifest_dir: Path, source: str, seen: set[str], files_by_source: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows = []
    dedupe: set[str] = set()
    for path in files_by_source.get(source, []):
        for row in _read_jsonl(path):
            if _source_name(row) != source:
                continue
            if _policy(row) != "train_allowed" or _split(row) != "grpo_train":
                continue
            task_id = _task_id(row)
            if task_id in seen or task_id in dedupe:
                continue
            dedupe.add(task_id)
            rows.append(row)
    return rows


def build_tasktrove_reservoir_report(
    *,
    manifest_dir: Path,
    report_out: Path | None = None,
) -> dict[str, Any]:
    """Summarize local TaskTrove reservoir coverage before live discovery."""

    files_by_source = discover_tasktrove_task_files(manifest_dir)
    seen = collect_seen_tasktrove_ids(manifest_dir)
    sources: dict[str, Any] = {}
    total_rows = 0
    total_available = 0
    total_seen = 0
    for source, files in sorted(files_by_source.items()):
        rows = []
        row_ids: set[str] = set()
        for path in files:
            for row in _read_jsonl(path):
                if _source_name(row) != source:
                    continue
                if _policy(row) != "train_allowed" or _split(row) != "grpo_train":
                    continue
                row_ids.add(_task_id(row))
                rows.append(row)
        source_seen = seen.get(source, set())
        available = [row for row in rows if _task_id(row) not in source_seen]
        ranked, diagnostics = _select_rows(available, count=min(25, len(available)), seed=0, selection=DEFAULT_SELECTION)
        total_rows += len(row_ids)
        total_available += len({_task_id(row) for row in available})
        total_seen += len(source_seen & row_ids)
        sources[source] = {
            "task_files": [str(path) for path in files],
            "total_train_allowed": len(row_ids),
            "seen": len(source_seen & row_ids),
            "available": len({_task_id(row) for row in available}),
            "top_ranked_preview_task_ids": [_task_id(row) for row in ranked],
            "top_ranked_preview_diagnostics": diagnostics,
        }

    report = {
        "version": "fugu_ultra_tasktrove_reservoir_v1",
        "manifest_dir": str(manifest_dir.resolve()),
        "source_count": len(sources),
        "total_train_allowed_tasks": total_rows,
        "total_seen_tasks": total_seen,
        "total_available_tasks": total_available,
        "sources": sources,
        "live_calls": False,
    }
    if report_out is not None:
        _write_json(report_out, report)
    return report


def build_tasktrove_prefilter_batch(
    *,
    manifest_dir: Path,
    out_dir: Path,
    inferredbugs_count: int = 6,
    pymethods_count: int = 6,
    source_counts: dict[str, int] | None = None,
    seed: int = 0,
    selection: str = DEFAULT_SELECTION,
) -> dict[str, Any]:
    """Write a fresh TaskTrove prefilter shard and matching scaffold jobs."""

    if inferredbugs_count < 0 or pymethods_count < 0:
        raise ValueError("Task counts must be non-negative")
    if source_counts is not None and any(count < 0 for count in source_counts.values()):
        raise ValueError("Task counts must be non-negative")

    out_dir.mkdir(parents=True, exist_ok=True)
    files_by_source = discover_tasktrove_task_files(manifest_dir)
    seen = collect_seen_tasktrove_ids(manifest_dir)
    requested = (
        dict(source_counts)
        if source_counts is not None
        else {
            "tasktrove_inferredbugs": inferredbugs_count,
            "tasktrove_pymethods2test": pymethods_count,
        }
    )

    selected: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}
    for source, count in requested.items():
        if not _is_tasktrove_source(source):
            raise ValueError(f"unsupported TaskTrove source name: {source}")
        available = _available_rows(manifest_dir, source, seen.get(source, set()), files_by_source)
        take = min(count, len(available))
        source_selected, selected_diagnostics = _select_rows(
            available,
            count=take,
            seed=seed,
            selection=selection,
        )
        selected.extend(source_selected)
        source_report[source] = {
            "requested": count,
            "selected": take,
            "deficit": count - take,
            "seen_count": len(seen.get(source, set())),
            "available_after_seen": len(available),
            "task_files": [str(path) for path in files_by_source.get(source, [])],
            "selected_task_ids": [_task_id(row) for row in source_selected],
            "selected_task_diagnostics": selected_diagnostics,
        }

    taskspecs_out = out_dir / "taskspecs.jsonl"
    empty_branch_out = out_dir / "empty_branch_tasks.jsonl"
    manifest_out = out_dir / "scaffold_tournament_manifest.json"
    jobs_out = out_dir / "scaffold_tournament_jobs.jsonl"
    readiness_out = out_dir / "readiness.json"
    selection_report_out = out_dir / "selection_report.json"

    _write_jsonl(taskspecs_out, selected)
    _write_jsonl(empty_branch_out, [])
    manifest = write_concrete_manifest(
        manifest_dir,
        manifest_out,
        jobs_out,
        task_mix=_task_mix_for_tasktrove_rows(selected),
        seed=seed,
        tasks_jsonl=taskspecs_out,
        branch_tasks_jsonl=empty_branch_out,
    )
    readiness = write_readiness(manifest_out, readiness_out)

    task_counts = Counter(_source_name(row) for row in selected)
    report = {
        "version": VERSION,
        "manifest_dir": str(manifest_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "requested": requested,
        "selection_policy": {
            "mode": selection,
            "seed": seed,
            "goal": "prefer higher-complexity and ID-diverse verifier-backed tasks before live prefiltering",
        },
        "known_source_count": len(files_by_source),
        "selected_tasks": len(selected),
        "selected_source_counts": dict(sorted(task_counts.items())),
        "sources": source_report,
        "taskspecs_jsonl": str(taskspecs_out),
        "empty_branch_tasks_jsonl": str(empty_branch_out),
        "manifest_json": str(manifest_out),
        "jobs_jsonl": str(jobs_out),
        "readiness_json": str(readiness_out),
        "job_count": int(manifest.get("job_count") or 0),
        "worker_call_count": int(manifest.get("worker_call_count") or 0),
        "ready_jobs": int((readiness.get("jobs_by_status") or {}).get("ready") or 0),
        "live_calls": False,
    }
    _write_json(selection_report_out, report)
    return report


def _normalize_agenttrove_source_counts(source_counts: dict[str, int] | None) -> dict[str, int]:
    requested = dict(source_counts or DEFAULT_AGENTTROVE_EXACT_COUNTS)
    normalized: dict[str, int] = {}
    for raw_source, count in requested.items():
        source = raw_source.strip()
        if count < 0:
            raise ValueError("Task counts must be non-negative")
        folder = TASKTROVE_SOURCE_TO_FOLDER.get(source, source)
        if folder not in TASKTROVE_FOLDER_TO_SOURCE:
            known = ", ".join(sorted([*TASKTROVE_SOURCE_TO_FOLDER, *TASKTROVE_FOLDER_TO_SOURCE]))
            raise ValueError(f"unsupported TaskTrove exact-match source {raw_source!r}; known sources: {known}")
        normalized[folder] = normalized.get(folder, 0) + count
    return normalized


def _seen_agenttrove_exact_keys(manifest_dir: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    disagreement_dir = manifest_dir / "agenttrove_disagreement"
    for path in sorted(disagreement_dir.glob("exact_match_selection_*/*.jsonl")):
        for row in _read_jsonl(path):
            key = _agenttrove_exact_key(row)
            if all(key):
                seen.add(key)

    for path in sorted(manifest_dir.glob("tasktrove_harbor/**/report.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        parquet_path = Path(str(report.get("parquet_path") or ""))
        folder = parquet_path.parent.name
        if folder not in TASKTROVE_FOLDER_TO_SOURCE:
            continue
        extraction = report.get("extraction") or {}
        for task in extraction.get("extracted_tasks") or []:
            task_path = str(task.get("path") or "")
            if task_path:
                seen.add((folder, task_path))
    return seen


def _agenttrove_exact_score(row: dict[str, Any], *, seed: int) -> float:
    attempts = int(row.get("attempts") or 0)
    success_rate = _success_rate(row)
    balance = 0.0 if success_rate is None else max(0.0, 1.0 - abs(success_rate - 0.5) * 2.0)
    teacher_count = int(row.get("teacher_count") or 0)
    model_count = _model_count(row)
    task_path = str(row.get("tasktrove_path") or row.get("task_id") or "")
    return (
        balance * 30.0
        + min(attempts, 16) * 0.75
        + min(teacher_count, 6) * 4.0
        + min(model_count, 6) * 2.0
        + _stable_jitter(task_path, seed) * 0.01
    )


def select_agenttrove_exact_matches(
    *,
    exact_matches_jsonl: Path,
    manifest_dir: Path,
    out_dir: Path,
    source_counts: dict[str, int] | None = None,
    seed: int = 0,
    min_attempts: int = 4,
    min_teacher_count: int = 3,
    min_model_count: int = 2,
    min_success_rate: float = 0.25,
    max_success_rate: float = 0.75,
) -> dict[str, Any]:
    """Select exact TaskTrove matches using cross-model AgentTrove disagreement priors."""

    requested = _normalize_agenttrove_source_counts(source_counts)
    if min_success_rate > max_success_rate:
        raise ValueError("min_success_rate must be <= max_success_rate")

    out_dir.mkdir(parents=True, exist_ok=True)
    seen = _seen_agenttrove_exact_keys(manifest_dir)
    rows = _read_jsonl(exact_matches_jsonl)
    selected: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}

    for folder, count in requested.items():
        source_name = TASKTROVE_FOLDER_TO_SOURCE[folder]
        candidates = []
        for row in rows:
            key = _agenttrove_exact_key(row)
            if key[0] != folder or not key[1] or key in seen:
                continue
            success_rate = _success_rate(row)
            if success_rate is None or not (min_success_rate <= success_rate <= max_success_rate):
                continue
            if int(row.get("attempts") or 0) < min_attempts:
                continue
            if int(row.get("teacher_count") or 0) < min_teacher_count:
                continue
            if _model_count(row) < min_model_count:
                continue
            candidates.append(row)

        ranked = sorted(
            candidates,
            key=lambda row: (_agenttrove_exact_score(row, seed=seed), str(row.get("tasktrove_path") or "")),
            reverse=True,
        )
        source_selected = ranked[:count]
        selected.extend(source_selected)
        source_report[source_name] = {
            "folder": folder,
            "requested": count,
            "selected": len(source_selected),
            "deficit": count - len(source_selected),
            "eligible_after_seen": len(candidates),
            "selected_tasktrove_paths": [str(row.get("tasktrove_path") or "") for row in source_selected],
            "selected_candidate_ids": [str(row.get("candidate_id") or "") for row in source_selected],
            "selected_task_diagnostics": [
                {
                    "tasktrove_path": str(row.get("tasktrove_path") or ""),
                    "attempts": int(row.get("attempts") or 0),
                    "success_rate": _success_rate(row),
                    "teacher_count": int(row.get("teacher_count") or 0),
                    "model_count": _model_count(row),
                    "score": round(_agenttrove_exact_score(row, seed=seed), 4),
                }
                for row in source_selected
            ],
        }
        if source_selected:
            _write_jsonl(out_dir / f"{folder}.jsonl", source_selected)

    report = {
        "version": "fugu_ultra_agenttrove_exact_selection_v1",
        "manifest_dir": str(manifest_dir.resolve()),
        "exact_matches_jsonl": str(exact_matches_jsonl),
        "out_dir": str(out_dir.resolve()),
        "requested": {TASKTROVE_FOLDER_TO_SOURCE[folder]: count for folder, count in requested.items()},
        "selection_policy": {
            "seed": seed,
            "min_attempts": min_attempts,
            "min_teacher_count": min_teacher_count,
            "min_model_count": min_model_count,
            "min_success_rate": min_success_rate,
            "max_success_rate": max_success_rate,
            "goal": "prefer exact TaskTrove tasks with mixed self-reported outcomes across multiple historical teacher/model identities",
        },
        "rows_read": len(rows),
        "seen_exact_keys": len(seen),
        "selected_tasks": len(selected),
        "selected_source_counts": dict(sorted(Counter(TASKTROVE_FOLDER_TO_SOURCE[_agenttrove_exact_key(row)[0]] for row in selected).items())),
        "sources": source_report,
        "live_calls": False,
    }
    _write_json(out_dir / "selection_report.json", report)
    return report


def build_agenttrove_exact_prefilter_batch(
    *,
    exact_matches_jsonl: Path,
    tasktrove_root: Path,
    manifest_dir: Path,
    out_dir: Path,
    source_counts: dict[str, int] | None = None,
    seed: int = 0,
    min_attempts: int = 4,
    min_teacher_count: int = 3,
    min_model_count: int = 2,
    min_success_rate: float = 0.25,
    max_success_rate: float = 0.75,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select, materialize, and plan a TaskTrove shard from AgentTrove exact-match priors."""

    selection = select_agenttrove_exact_matches(
        exact_matches_jsonl=exact_matches_jsonl,
        manifest_dir=manifest_dir,
        out_dir=out_dir / "selection",
        source_counts=source_counts,
        seed=seed,
        min_attempts=min_attempts,
        min_teacher_count=min_teacher_count,
        min_model_count=min_model_count,
        min_success_rate=min_success_rate,
        max_success_rate=max_success_rate,
    )

    materialized_rows: list[dict[str, Any]] = []
    materialization_reports: dict[str, Any] = {}
    for source_name, source_info in sorted((selection.get("sources") or {}).items()):
        paths = source_info.get("selected_tasktrove_paths") or []
        if not paths:
            continue
        folder = str(source_info["folder"])
        parquet_path = tasktrove_root / folder / "tasks.parquet"
        if not parquet_path.exists():
            materialization_reports[source_name] = {
                "folder": folder,
                "selected": len(paths),
                "materialized": 0,
                "error": f"missing parquet: {parquet_path}",
            }
            continue

        source_out = out_dir / "materialized" / source_name
        taskspecs_out = source_out / "taskspecs.jsonl"
        report_out = source_out / "report.json"
        report = materialize_tasktrove_parquet(
            parquet_path=parquet_path,
            extract_dir=source_out / "extracted",
            out_jsonl=taskspecs_out,
            report_path=report_out,
            source_name=source_name,
            source_version="v3",
            policy="train_allowed",
            split="grpo_train",
            include_paths=set(str(path) for path in paths),
            overwrite=overwrite,
        )
        materialization_reports[source_name] = report
        materialized_rows.extend(_read_jsonl(taskspecs_out))

    taskspecs_out = out_dir / "taskspecs.jsonl"
    empty_branch_out = out_dir / "empty_branch_tasks.jsonl"
    manifest_out = out_dir / "scaffold_tournament_manifest.json"
    jobs_out = out_dir / "scaffold_tournament_jobs.jsonl"
    readiness_out = out_dir / "readiness.json"

    _write_jsonl(taskspecs_out, materialized_rows)
    _write_jsonl(empty_branch_out, [])
    manifest = write_concrete_manifest(
        manifest_dir,
        manifest_out,
        jobs_out,
        task_mix=_task_mix_for_tasktrove_rows(materialized_rows),
        seed=seed,
        tasks_jsonl=taskspecs_out,
        branch_tasks_jsonl=empty_branch_out,
    )
    readiness = write_readiness(manifest_out, readiness_out)

    report = {
        **selection,
        "version": "fugu_ultra_agenttrove_exact_prefilter_batch_v1",
        "tasktrove_root": str(tasktrove_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "taskspecs_jsonl": str(taskspecs_out),
        "empty_branch_tasks_jsonl": str(empty_branch_out),
        "manifest_json": str(manifest_out),
        "jobs_jsonl": str(jobs_out),
        "readiness_json": str(readiness_out),
        "materialized_tasks": len(materialized_rows),
        "materialized_source_counts": dict(sorted(Counter(_source_name(row) for row in materialized_rows).items())),
        "materialization_reports": materialization_reports,
        "job_count": int(manifest.get("job_count") or 0),
        "worker_call_count": int(manifest.get("worker_call_count") or 0),
        "ready_jobs": int((readiness.get("jobs_by_status") or {}).get("ready") or 0),
        "live_calls": False,
    }
    _write_json(out_dir / "batch_report.json", report)
    return report
