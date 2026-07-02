"""Build the first GRPO pilot config from frozen tasks and selected workers."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import TaskSpec

VERSION = "fugu_ultra_grpo_pilot_config_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _render_markdown(config: dict[str, Any]) -> str:
    lines = [
        "# Fugu-Ultra GRPO Pilot Config",
        "",
        f"Version: {config['version']}",
        f"Status: {config['status']}",
        f"Ready: {config['ready_for_pilot']}",
        f"Tasks: {config['task_count']}",
        f"Max workflow steps: {config['workflow_policy']['max_workflow_steps']}",
        "",
        "## Worker Pool",
    ]
    lines.extend(f"- `{worker}`" for worker in config["worker_pool_names"])
    lines.extend(["", "## Lane Masks"])
    for lane, workers in config["lane_worker_masks"].items():
        lines.append(f"- `{lane}`: " + ", ".join(f"`{worker}`" for worker in workers))
    lines.extend(["", "## Checks"])
    for key, value in config["checks"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def build_grpo_pilot_config(
    *,
    freeze_report_json: Path,
    pool_report_json: Path,
    out_json: Path,
    md_out: Path | None = None,
    max_workflow_steps: int = 3,
) -> dict[str, Any]:
    freeze_report_json = freeze_report_json.resolve()
    pool_report_json = pool_report_json.resolve()
    freeze = _read_json(freeze_report_json)
    pool = _read_json(pool_report_json)
    task_manifest = next(item for item in freeze["manifests"] if item["manifest_name"] == "grpo_pilot_tasks")
    seed_manifest = next(item for item in freeze["manifests"] if item["manifest_name"] == "grpo_pilot_seed_evidence")
    task_path = Path(task_manifest["path"])
    seed_path = Path(seed_manifest["path"])
    tasks = [TaskSpec.model_validate(row) for row in _read_jsonl(task_path)]
    seed_rows = _read_jsonl(seed_path)

    task_ids_by_lane: dict[str, list[str]] = defaultdict(list)
    for row in seed_rows:
        if row.get("lane") and row.get("source_task_id"):
            task_ids_by_lane[str(row["lane"])].append(str(row["source_task_id"]))
    task_lanes = set(task_ids_by_lane)
    selected_workers = list(pool["recommendations"]["recommended_mvp_grpo_workers"])
    worker_set = set(selected_workers)
    lane_masks = {
        str(lane): list(workers)
        for lane, workers in sorted(pool["recommendations"]["lane_worker_masks"].items())
        if str(lane) in task_lanes
    }
    missing_lane_masks = sorted(task_lanes - set(lane_masks))
    masked_unknown_workers = sorted(
        {worker for workers in lane_masks.values() for worker in workers if worker not in worker_set}
    )
    task_worker_pool = {
        name: pool["workers"][name]["identity"]
        for name in selected_workers
        if name in pool.get("workers", {})
    }
    group_sizes = {}
    for lane, rows in sorted(
        ((lane, [row for row in seed_rows if row.get("lane") == lane]) for lane in task_lanes),
        key=lambda item: item[0],
    ):
        counts = Counter(int(row.get("recommended_group_size") or 4) for row in rows)
        group_sizes[lane] = counts.most_common(1)[0][0] if counts else 4

    checks = {
        "freeze_complete": bool(freeze.get("freeze_complete")),
        "pool_selected_for_pilot": pool["recommendations"].get("selection_status")
        == "mvp_grpo_pool_selected_for_pilot_not_final_ultra_claim",
        "task_manifest_hash_matches": _sha256_file(task_path) == task_manifest["sha256"],
        "seed_manifest_hash_matches": _sha256_file(seed_path) == seed_manifest["sha256"],
        "all_training_lanes_have_masks": not missing_lane_masks,
        "all_masked_workers_selected": not masked_unknown_workers,
        "all_workers_have_identity": len(task_worker_pool) == len(selected_workers),
    }
    ready = all(checks.values())

    config = {
        "version": VERSION,
        "status": "ready_for_first_grpo_pilot" if ready else "not_ready_for_first_grpo_pilot",
        "ready_for_pilot": ready,
        "freeze_report": str(freeze_report_json),
        "pool_selection_report": str(pool_report_json),
        "task_manifest": task_manifest,
        "seed_evidence_manifest": seed_manifest,
        "task_count": len(tasks),
        "lane_counts": _counter_json(Counter(row.get("lane") for row in seed_rows)),
        "task_ids_by_lane": {lane: sorted(ids) for lane, ids in sorted(task_ids_by_lane.items())},
        "worker_pool_names": selected_workers,
        "worker_pool": task_worker_pool,
        "challenger_workers_not_in_action_space": sorted(
            set(pool["recommendations"].get("challenger_workers", {})) - worker_set
        ),
        "lane_worker_masks": lane_masks,
        "group_size_by_lane": group_sizes,
        "workflow_policy": {
            "max_workflow_steps": max_workflow_steps,
            "reward_mapping": {
                "invalid_workflow_trainable": 0.0,
                "valid_incorrect_trainable": 0.5,
                "budget_exhausted_trainable": 0.5,
                "valid_correct_trainable": 1.0,
            },
            "notes": [
                "Start at three workflow steps; move toward five only after stability.",
                "Use lane masks during early GRPO to reduce invalid or irrelevant worker choices.",
            ],
        },
        "provider_policy": {
            "yunwu_only_workers": [
                worker
                for worker, identity in task_worker_pool.items()
                if str(identity.get("model", "")).startswith("gpt")
                or "opus" in str(identity.get("model", ""))
                or "gemini" in str(identity.get("model", ""))
            ],
            "openrouter_workers": [
                worker
                for worker, identity in task_worker_pool.items()
                if worker not in {
                    name
                    for name, ident in task_worker_pool.items()
                    if str(ident.get("model", "")).startswith("gpt")
                    or "opus" in str(ident.get("model", ""))
                    or "gemini" in str(ident.get("model", ""))
                }
            ],
            "gpt_never_openrouter": True,
        },
        "checks": checks,
        "missing_lane_masks": missing_lane_masks,
        "masked_unknown_workers": masked_unknown_workers,
    }
    _write_json(out_json, config)
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_render_markdown(config))
    return config
