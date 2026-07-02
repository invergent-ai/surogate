"""Select existing-bank tasks using precomputed label/frontier disagreement priors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from typing import Any

from .scaffold_tournament import write_concrete_manifest, write_readiness

VERSION = "fugu_ultra_label_prior_selection_v1"

LANE_DOMAINS = {
    "unit_and_scientific_code": {"code"},
    "math_science_knowledge": {"math", "science", "general"},
}


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


def _base_task_id(task_id: Any) -> str:
    text = str(task_id)
    if text.startswith("existing_bank__"):
        return text[len("existing_bank__") :]
    return text


def _float_list(values: list[Any]) -> list[float]:
    return [float(value) for value in values]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _seen_existing_bank_task_ids(manifest_dir: Path, out_dir: Path) -> set[str]:
    seen: set[str] = set()
    seed_paths = [
        manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl",
        manifest_dir / "grpo_pilot_seed" / "taskspecs.jsonl",
    ]
    for path in seed_paths:
        for row in _read_jsonl(path):
            for key in ("source_task_id", "task_id"):
                if key in row:
                    seen.add(_base_task_id(row[key]))

    for pattern in ("label_prior*", "direct_unit_expansion*"):
        for root in manifest_dir.glob(pattern):
            if root.resolve() == out_dir.resolve():
                continue
            for row in _read_jsonl(root / "taskspecs.jsonl"):
                if row.get("source", {}).get("name") == "existing_bank":
                    seen.add(_base_task_id(row.get("task_id")))
            for row in _read_jsonl(root / "selected_candidates.jsonl"):
                if "task_id" in row:
                    seen.add(_base_task_id(row["task_id"]))
    return seen


def _score_candidate(row: dict[str, Any], *, lane: str, seed: int) -> float:
    open_avg = float(row["open_avg"])
    open_spread = float(row["open_spread"])
    frontier_spread = float(row.get("frontier_spread") or 0.0)
    frontier_avg = row.get("frontier_avg")

    # Boundary tasks are best for the current seed: not all-solve, not all-fail,
    # and with strong worker spread. Code gets a lower target because prior
    # commercial screens saturated on rows that open workers found difficult.
    target = 0.35 if lane == "unit_and_scientific_code" else 0.5
    mid_score = max(0.0, 1.0 - abs(open_avg - target) / target)
    saturation_penalty = 1.0 if open_avg in {0.0, 1.0} else 0.0
    frontier_bonus = 0.0
    if frontier_avg is not None:
        frontier_mid = max(0.0, 1.0 - abs(float(frontier_avg) - 0.5) / 0.5)
        frontier_bonus = 25.0 * frontier_spread + 10.0 * frontier_mid

    noise = random.Random(f"{seed}:{row['task_id']}").random() / 1000.0
    return 100.0 * open_spread + 40.0 * mid_score + frontier_bonus - 50.0 * saturation_penalty + noise


def _balanced_select(candidates: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    if len(candidates) <= count:
        return candidates
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row.get("source") or "unknown")].append(row)
    for rows in by_source.values():
        rows.sort(key=lambda row: (-float(row["score"]), row["task_id"]))

    selected: list[dict[str, Any]] = []
    labels = sorted(by_source)
    while len(selected) < count:
        progressed = False
        for label in labels:
            rows = by_source[label]
            if not rows:
                continue
            selected.append(rows.pop(0))
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return sorted(selected, key=lambda row: (-float(row["score"]), row["task_id"]))


def build_label_prior_shard(
    *,
    manifest_dir: Path,
    out_dir: Path,
    lane: str = "unit_and_scientific_code",
    count: int = 32,
    seed: int = 0,
    require_frontier_overlap: bool = False,
) -> dict[str, Any]:
    if lane not in LANE_DOMAINS:
        raise ValueError(f"unsupported lane for label-prior selection: {lane}")

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = manifest_dir / "labels_n4_tau0.1.jsonl"
    frontier_path = manifest_dir / "pool_matrix_frontier.jsonl"
    taskspec_path = manifest_dir / "data_mix" / "existing_bank_taskspecs.jsonl"

    labels = {_base_task_id(row["task_id"]): row for row in _read_jsonl(labels_path)}
    frontier = {_base_task_id(row["task_id"]): row for row in _read_jsonl(frontier_path)}
    seen = _seen_existing_bank_task_ids(manifest_dir, out_dir)

    candidates: list[dict[str, Any]] = []
    domains = LANE_DOMAINS[lane]
    for task in _read_jsonl(taskspec_path):
        if task.get("source", {}).get("name") != "existing_bank":
            continue
        if task.get("source", {}).get("policy") != "train_allowed":
            continue
        if task.get("splitting", {}).get("split") != "grpo_train":
            continue
        domain = task.get("metadata", {}).get("domain")
        if domain not in domains:
            continue
        base_id = _base_task_id(task["task_id"])
        if base_id in seen:
            continue
        label = labels.get(base_id)
        if label is None:
            continue
        frontier_row = frontier.get(base_id)
        if require_frontier_overlap and frontier_row is None:
            continue

        open_r_bar = _float_list(label.get("r_bar") or [])
        frontier_r_bar = _float_list(frontier_row.get("r_bar") or []) if frontier_row else None
        candidate = {
            "task_id": base_id,
            "lane": lane,
            "domain": domain,
            "source": label.get("source"),
            "grader": label.get("grader"),
            "open_worker_ids": label.get("worker_ids") or [],
            "open_r_bar": open_r_bar,
            "open_avg": round(_mean(open_r_bar), 6),
            "open_spread": round(_spread(open_r_bar), 6),
            "frontier_worker_ids": frontier_row.get("worker_ids") if frontier_row else None,
            "frontier_r_bar": frontier_r_bar,
            "frontier_avg": round(_mean(frontier_r_bar), 6) if frontier_r_bar is not None else None,
            "frontier_spread": round(_spread(frontier_r_bar), 6) if frontier_r_bar is not None else None,
            "_task": task,
        }
        candidate["score"] = round(_score_candidate(candidate, lane=lane, seed=seed), 6)
        candidates.append(candidate)

    candidates.sort(key=lambda row: (-float(row["score"]), row["task_id"]))
    selected = _balanced_select(candidates, count=count)

    taskspecs: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for row in selected:
        task = json.loads(json.dumps(row["_task"]))
        task.setdefault("metadata", {})
        tags = set(task["metadata"].get("tags") or [])
        tags.update(["label_prior_selection", out_dir.name, "labels_n4_tau0.1"])
        if row.get("frontier_r_bar") is not None:
            tags.add("frontier_matrix_overlap")
        task["metadata"]["tags"] = sorted(tags)
        task["metadata"]["label_prior"] = {
            "source_file": labels_path.name,
            "worker_ids": row["open_worker_ids"],
            "r_bar": row["open_r_bar"],
            "open_avg": row["open_avg"],
            "open_spread": row["open_spread"],
            "selection_score": row["score"],
        }
        if row.get("frontier_r_bar") is not None:
            task["metadata"]["frontier_matrix_prior"] = {
                "source_file": frontier_path.name,
                "worker_ids": row["frontier_worker_ids"],
                "r_bar": row["frontier_r_bar"],
                "frontier_avg": row["frontier_avg"],
                "frontier_spread": row["frontier_spread"],
            }
        taskspecs.append(task)
        selected_rows.append({key: value for key, value in row.items() if not key.startswith("_")})

    taskspecs_out = out_dir / "taskspecs.jsonl"
    selected_out = out_dir / "selected_candidates.jsonl"
    empty_branch_out = out_dir / "empty_branch_tasks.jsonl"
    manifest_out = out_dir / "scaffold_tournament_manifest.json"
    jobs_out = out_dir / "scaffold_tournament_jobs.jsonl"
    readiness_out = out_dir / "scaffold_tournament_readiness.json"
    _write_jsonl(taskspecs_out, taskspecs)
    _write_jsonl(selected_out, selected_rows)
    _write_jsonl(empty_branch_out, [])

    manifest = write_concrete_manifest(
        manifest_dir,
        manifest_out,
        jobs_out,
        task_mix={lane: len(taskspecs)},
        seed=seed,
        tasks_jsonl=taskspecs_out,
        branch_tasks_jsonl=empty_branch_out,
    )
    readiness = write_readiness(manifest_out, readiness_out)

    report = {
        "version": VERSION,
        "purpose": "Select fresh existing-bank tasks from precomputed label/frontier disagreement priors.",
        "lane": lane,
        "requested": count,
        "selected": len(taskspecs),
        "eligible_candidates": len(candidates),
        "excluded_seen_existing_bank_tasks": len(seen),
        "require_frontier_overlap": require_frontier_overlap,
        "selected_domain_counts": dict(Counter(row["domain"] for row in selected_rows)),
        "selected_source_counts": dict(Counter(str(row.get("source")) for row in selected_rows)),
        "frontier_overlap_selected": sum(1 for row in selected_rows if row.get("frontier_r_bar") is not None),
        "score_policy": {
            "main_signal": "open-worker reward spread from labels_n4_tau0.1",
            "medium_difficulty_target": 0.35 if lane == "unit_and_scientific_code" else 0.5,
            "frontier_matrix_bonus": True,
            "source_balancing": True,
        },
        "output_files": {
            "taskspecs": str(taskspecs_out),
            "selected_candidates": str(selected_out),
            "manifest": str(manifest_out),
            "jobs": str(jobs_out),
            "readiness": str(readiness_out),
        },
        "used_files": [str(labels_path), str(frontier_path), str(taskspec_path)],
        "job_count": manifest["job_count"],
        "ready_jobs": readiness["jobs_by_status"].get("ready", 0),
        "live_calls": False,
    }
    _write_json(out_dir / "selection_report.json", report)
    return report


def default_manifest_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "director" / "manifests" / "fugu_clean_v1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a label/frontier-prior existing-bank shard.")
    parser.add_argument("--manifest-dir", type=Path, default=default_manifest_dir())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lane", choices=sorted(LANE_DOMAINS), default="unit_and_scientific_code")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-frontier-overlap", action="store_true")
    args = parser.parse_args(argv)

    report = build_label_prior_shard(
        manifest_dir=args.manifest_dir,
        out_dir=args.out_dir,
        lane=args.lane,
        count=args.count,
        seed=args.seed,
        require_frontier_overlap=args.require_frontier_overlap,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
