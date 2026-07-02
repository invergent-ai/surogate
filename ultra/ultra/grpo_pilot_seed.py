"""Build a first GRPO pilot seed from observed workflow disagreement."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .schemas import TaskSpec
from .workflow_pool_selection import load_completed_rows

VERSION = "fugu_ultra_grpo_pilot_seed_v1"

LANE_GROUP_SIZES = {
    "repo_open_repo_terminal": 4,
    "trace_state_branches": 4,
    "unit_and_scientific_code": 8,
    "math_science_knowledge": 8,
    "tool_dialogue": 4,
    "long_context_memory_planning": 4,
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _task_key(row: dict[str, Any]) -> str:
    if row.get("tournament_task_id"):
        return str(row["tournament_task_id"])
    task_jsonl = row.get("task_jsonl") or ""
    source_task_id = row.get("source_task_id") or row.get("tournament_task_id") or row.get("task_id")
    return f"{task_jsonl}::{source_task_id}"


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("reward") is not None and row.get("valid_for_training") is not False]
    if not scored:
        return None
    return max(scored, key=lambda row: (float(row["reward"]), bool(row.get("success")), str(row.get("arm"))))


def _success(row: dict[str, Any] | None) -> bool | None:
    if row is None:
        return None
    return row.get("reward") is not None and float(row["reward"]) >= 1.0


def _repo_root_from_manifest(manifest_dir: Path) -> Path:
    if len(manifest_dir.parents) >= 3:
        return manifest_dir.parents[2]
    return manifest_dir.parent


def _resolve_task_jsonl(path: Path, manifest_dir: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        path,
        manifest_dir / path,
        _repo_root_from_manifest(manifest_dir) / path,
    ]
    raw = str(path)
    while raw.startswith("../"):
        raw = raw[3:]
        candidates.append(_repo_root_from_manifest(manifest_dir) / raw)
    marker = "director/manifests/"
    if marker in str(path):
        suffix = str(path)[str(path).index(marker) :]
        candidates.append(_repo_root_from_manifest(manifest_dir) / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _load_task(
    path: Path,
    task_id: str,
    cache: dict[Path, dict[str, dict[str, Any]]],
    *,
    manifest_dir: Path,
) -> dict[str, Any] | None:
    path = _resolve_task_jsonl(path, manifest_dir)
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = {}
        for row in _read_jsonl(resolved):
            spec = TaskSpec.model_validate(row)
            cache[resolved][spec.task_id] = spec.model_dump(mode="json")
    return cache[resolved].get(task_id)


def _group_candidate(key: str, group: list[dict[str, Any]]) -> dict[str, Any] | None:
    rewards = [float(row["reward"]) for row in group if row.get("reward") is not None]
    if not rewards:
        return None
    reward_values = sorted(set(rewards))
    best_all = _best(group)
    best_single = _best([row for row in group if row.get("stage") == "single_scaffold"])
    best_role = _best([row for row in group if row.get("stage") == "role_workflow"])
    role_delta = None
    if best_single is not None and best_role is not None:
        role_delta = float(best_role["reward"]) - float(best_single["reward"])

    reasons = []
    if len(reward_values) > 1:
        reasons.append("reward_variance")
    if role_delta is not None and role_delta > 0:
        reasons.append("role_beats_best_single")
    if best_single is not None and best_all is not None and float(best_all["reward"]) > float(best_single["reward"]):
        reasons.append("workflow_oracle_headroom")
    if not reasons:
        return None

    first = group[0]
    return {
        "pilot_task_id": key,
        "tournament_task_id": first.get("tournament_task_id"),
        "lane": first.get("lane"),
        "arm_domain": first.get("arm_domain"),
        "source": first.get("source"),
        "source_task_id": first.get("source_task_id"),
        "task_jsonl": first.get("task_jsonl"),
        "task_harness": first.get("task_harness"),
        "selection_reasons": sorted(set(reasons)),
        "reward_values": reward_values,
        "rollouts_observed": len(group),
        "successful_arms": sorted({str(row.get("arm")) for row in group if _success(row) is True}),
        "best_single": {
            "arm": best_single.get("arm") if best_single else None,
            "reward": best_single.get("reward") if best_single else None,
            "success": _success(best_single),
        },
        "best_role": {
            "arm": best_role.get("arm") if best_role else None,
            "reward": best_role.get("reward") if best_role else None,
            "success": _success(best_role),
        },
        "workflow_oracle": {
            "arm": best_all.get("arm") if best_all else None,
            "reward": best_all.get("reward") if best_all else None,
            "success": _success(best_all),
        },
        "recommended_group_size": LANE_GROUP_SIZES.get(str(first.get("lane")), 4),
    }


def build_grpo_pilot_seed(
    *,
    manifest_dir: Path,
    out_jsonl: Path,
    report_out: Path | None = None,
    task_jsonl_out: Path | None = None,
) -> dict[str, Any]:
    rows = load_completed_rows(manifest_dir)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task_jsonl") and row.get("source_task_id"):
            groups[_task_key(row)].append(row)

    candidate_selected = []
    for key, group in sorted(groups.items()):
        candidate = _group_candidate(key, group)
        if candidate is not None:
            candidate_selected.append(candidate)

    task_rows = []
    missing_tasks = []
    cache: dict[Path, dict[str, dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    if task_jsonl_out is not None:
        seen_task_ids = set()
        for row in candidate_selected:
            path = Path(str(row["task_jsonl"]))
            task_id = str(row["source_task_id"])
            task = _load_task(path, task_id, cache, manifest_dir=manifest_dir)
            if task is None:
                missing_tasks.append({"task_jsonl": str(path), "source_task_id": task_id})
                continue
            selected.append(row)
            if task["task_id"] in seen_task_ids:
                continue
            seen_task_ids.add(task["task_id"])
            task_rows.append(task)
        task_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with task_jsonl_out.open("w") as f:
            for task in task_rows:
                f.write(json.dumps(task, sort_keys=True) + "\n")
    else:
        selected = candidate_selected

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    reason_counts = Counter(reason for row in selected for reason in row["selection_reasons"])
    lane_counts = Counter(str(row["lane"]) for row in selected)
    source_counts = Counter(str(row["source"]) for row in selected)
    report = {
        "version": VERSION,
        "status": "seed_not_full_pilot",
        "purpose": "First GRPO pilot seed from tasks with observed workflow disagreement/headroom.",
        "manifest_dir": str(manifest_dir.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "task_jsonl_out": str(task_jsonl_out.resolve()) if task_jsonl_out else None,
        "candidate_tasks_before_materialization": len(candidate_selected),
        "selected_tasks": len(selected),
        "materialized_tasks": len(task_rows),
        "excluded_missing_tasks": missing_tasks,
        "reason_counts": dict(sorted(reason_counts.items())),
        "lane_counts": dict(sorted(lane_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "group_size_by_lane": LANE_GROUP_SIZES,
        "notes": [
            "This is a seed set, not the final 300-500 task pilot.",
            "Only tasks with observed reward variance or workflow/headroom evidence are included.",
            "Deep SWE remains excluded.",
        ],
    }
    if report_out is not None:
        _write_json(report_out, report)
    return report
