"""Freeze the first GRPO pilot seed into a reproducible training manifest."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import TaskSpec

VERSION = "fugu_ultra_grpo_pilot_freeze_v1"
DEFAULT_LANE_TARGETS = {
    "repo_open_repo_terminal": 80,
    "trace_state_branches": 20,
    "unit_and_scientific_code": 55,
    "math_science_knowledge": 55,
    "tool_dialogue": 60,
    "long_context_memory_planning": 30,
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _task_id_hash(task_ids: list[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest()


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _manifest_record(
    *,
    manifest_name: str,
    path: Path,
    row_count: int,
    row_schema: str,
    task_ids: list[str],
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "manifest_name": manifest_name,
        "path": str(path.resolve()),
        "row_count": row_count,
        "sha256": _sha256_file(path),
        "row_schema": row_schema,
        "task_id_sha256": _task_id_hash(task_ids),
        "created_at_utc": created_at_utc,
    }


def _task_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["task_id"]) for row in rows if row.get("task_id")}


def _contamination_groups(rows: list[dict[str, Any]]) -> set[str]:
    groups: set[str] = set()
    for row in rows:
        splitting = row.get("splitting")
        if isinstance(splitting, dict) and splitting.get("contamination_group"):
            groups.add(str(splitting["contamination_group"]))
    return groups


def _load_eval_rows(manifest_dir: Path, name: str) -> list[dict[str, Any]]:
    return _read_jsonl(manifest_dir / "frozen_manifests" / f"{name}.jsonl")


def _lane_deficits_from_gap(gap_plan: dict[str, Any] | None) -> dict[str, int] | None:
    if gap_plan is None:
        return None
    deficits = gap_plan.get("lane_deficits")
    if not isinstance(deficits, dict):
        return None
    return {str(k): int(v) for k, v in deficits.items()}


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fugu-Ultra GRPO Pilot Freeze",
        "",
        f"Version: {report['version']}",
        f"Created: {report['created_at_utc']}",
        f"Status: {report['status']}",
        f"Freeze complete: {report['freeze_complete']}",
        "",
        "## Counts",
        f"- Tasks: {report['task_count']}",
        f"- Seed rows: {report['seed_row_count']}",
        f"- Lane counts: {report['lane_counts']}",
        f"- Source counts: {report['source_counts']}",
        f"- Harness counts: {report['harness_counts']}",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Manifests"])
    for manifest in report["manifests"]:
        lines.extend(
            [
                f"### {manifest['manifest_name']}",
                f"- Path: {manifest['path']}",
                f"- Rows: {manifest['row_count']}",
                f"- SHA-256: {manifest['sha256']}",
                f"- Task IDs: {manifest['task_id_sha256']}",
                "",
            ]
        )
    return "\n".join(lines)


def build_grpo_pilot_freeze(
    *,
    manifest_dir: Path,
    seed_jsonl: Path,
    tasks_jsonl: Path,
    out_dir: Path,
    report_out: Path | None = None,
    md_out: Path | None = None,
    gap_plan_json: Path | None = None,
    target_task_count: int = 300,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    seed_jsonl = seed_jsonl.resolve()
    tasks_jsonl = tasks_jsonl.resolve()
    out_dir = out_dir.resolve()
    created_at_utc = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    seed_rows = sorted(_read_jsonl(seed_jsonl), key=lambda row: str(row.get("pilot_task_id") or row.get("task_id")))
    task_rows_raw = _read_jsonl(tasks_jsonl)
    tasks = sorted((TaskSpec.model_validate(row) for row in task_rows_raw), key=lambda spec: spec.task_id)
    task_rows = [task.model_dump(mode="json") for task in tasks]

    task_ids = [task.task_id for task in tasks]
    duplicate_task_count = len(task_ids) - len(set(task_ids))
    seed_source_task_ids = {str(row.get("source_task_id")) for row in seed_rows if row.get("source_task_id")}
    missing_seed_task_ids = sorted(seed_source_task_ids - set(task_ids))
    extra_task_ids = sorted(set(task_ids) - seed_source_task_ids)

    lane_counts = Counter(str(row.get("lane")) for row in seed_rows)
    source_counts = Counter(task.source.name for task in tasks)
    harness_counts = Counter(task.environment.harness for task in tasks)
    split_counts = Counter(task.splitting.split for task in tasks)
    policy_counts = Counter(task.source.policy for task in tasks)
    reason_counts = Counter(reason for row in seed_rows for reason in row.get("selection_reasons", []))

    frozen_online = _load_eval_rows(manifest_dir, "online_validation")
    frozen_final = _load_eval_rows(manifest_dir, "final_eval")
    frozen_deep_swe = _load_eval_rows(manifest_dir, "deep_swe_target_eval")
    train_task_ids = set(task_ids)
    train_contamination = _contamination_groups(task_rows)
    eval_contamination = (
        _contamination_groups(frozen_online)
        | _contamination_groups(frozen_final)
        | _contamination_groups(frozen_deep_swe)
    )
    overlap_ids = {
        "online_validation_task_id_overlap": len(train_task_ids & _task_ids(frozen_online)),
        "final_eval_task_id_overlap": len(train_task_ids & _task_ids(frozen_final)),
        "deep_swe_task_id_overlap": len(train_task_ids & _task_ids(frozen_deep_swe)),
        "eval_contamination_group_overlap": len(train_contamination & eval_contamination),
    }

    gap_plan = _read_json(gap_plan_json.resolve()) if gap_plan_json else _read_json(manifest_dir / "grpo_pilot_seed" / "gap_plan.json")
    lane_deficits = _lane_deficits_from_gap(gap_plan)
    target_lanes = dict(gap_plan.get("target_lane_counts", DEFAULT_LANE_TARGETS)) if gap_plan else DEFAULT_LANE_TARGETS
    lane_target_checks = {
        lane: {
            "target": int(target),
            "actual": int(lane_counts.get(lane, 0)),
            "deficit": max(int(target) - int(lane_counts.get(lane, 0)), 0),
        }
        for lane, target in sorted(target_lanes.items())
    }

    frozen_tasks_path = out_dir / "taskspecs.jsonl"
    frozen_seed_path = out_dir / "seed_manifest.jsonl"
    _write_jsonl(frozen_tasks_path, task_rows)
    _write_jsonl(frozen_seed_path, seed_rows)

    manifests = [
        _manifest_record(
            manifest_name="grpo_pilot_tasks",
            path=frozen_tasks_path,
            row_count=len(task_rows),
            row_schema="TaskSpec v2",
            task_ids=task_ids,
            created_at_utc=created_at_utc,
        ),
        _manifest_record(
            manifest_name="grpo_pilot_seed_evidence",
            path=frozen_seed_path,
            row_count=len(seed_rows),
            row_schema="grpo_pilot_seed_v1",
            task_ids=[str(row.get("source_task_id")) for row in seed_rows if row.get("source_task_id")],
            created_at_utc=created_at_utc,
        ),
    ]

    checks = {
        "task_count_at_least_target": len(task_rows) >= target_task_count,
        "seed_task_count_matches_tasks": len(seed_rows) == len(task_rows),
        "duplicate_task_count": duplicate_task_count,
        "missing_seed_task_count": len(missing_seed_task_ids),
        "extra_task_count": len(extra_task_ids),
        "all_tasks_grpo_train": set(split_counts) == {"grpo_train"},
        "all_sources_train_allowed": set(policy_counts) == {"train_allowed"},
        "all_lane_deficits_zero": all(value == 0 for value in (lane_deficits or {}).values()) if lane_deficits is not None else None,
        **overlap_ids,
    }
    freeze_complete = (
        checks["task_count_at_least_target"]
        and checks["seed_task_count_matches_tasks"]
        and checks["duplicate_task_count"] == 0
        and checks["missing_seed_task_count"] == 0
        and checks["extra_task_count"] == 0
        and checks["all_tasks_grpo_train"]
        and checks["all_sources_train_allowed"]
        and checks["online_validation_task_id_overlap"] == 0
        and checks["final_eval_task_id_overlap"] == 0
        and checks["deep_swe_task_id_overlap"] == 0
        and checks["eval_contamination_group_overlap"] == 0
        and checks["all_lane_deficits_zero"] is not False
    )

    report = {
        "version": VERSION,
        "status": "grpo_pilot_training_manifest_frozen" if freeze_complete else "grpo_pilot_training_manifest_not_ready",
        "purpose": "Frozen first GRPO pilot training manifest from reward-varying/headroom tasks.",
        "manifest_dir": str(manifest_dir),
        "seed_jsonl": str(seed_jsonl),
        "source_tasks_jsonl": str(tasks_jsonl),
        "out_dir": str(out_dir),
        "created_at_utc": created_at_utc,
        "target_task_count": target_task_count,
        "task_count": len(task_rows),
        "seed_row_count": len(seed_rows),
        "lane_counts": _counter_json(lane_counts),
        "source_counts": _counter_json(source_counts),
        "harness_counts": _counter_json(harness_counts),
        "split_counts": _counter_json(split_counts),
        "policy_counts": _counter_json(policy_counts),
        "reason_counts": _counter_json(reason_counts),
        "lane_target_checks": lane_target_checks,
        "lane_deficits_from_gap_plan": lane_deficits,
        "checks": checks,
        "missing_seed_task_ids": missing_seed_task_ids[:100],
        "extra_task_ids": extra_task_ids[:100],
        "manifests": manifests,
        "freeze_complete": bool(freeze_complete),
        "live_calls": False,
        "notes": [
            "Deep SWE remains excluded from training.",
            "TaskCraft remains excluded until source freeze and deterministic grading audits pass.",
            "Use this artifact for the first GRPO pilot; do not resample from the 1,000-row candidate pool without rerunning the freeze.",
        ],
    }
    if report_out is not None:
        _write_json(report_out, report)
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_render_markdown(report) + "\n")
    return report
