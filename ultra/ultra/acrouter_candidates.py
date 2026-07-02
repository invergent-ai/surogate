"""Utilities for mining Agent-as-a-Router / CodeRouterBench disagreement tasks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
from typing import Any


TRUTHY = {"1", "1.0", "true", "True", "yes", "YES"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score(value: Any) -> int:
    return 1 if str(value).strip() in TRUTHY else 0


def _docker_images() -> set[str]:
    proc = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()
    return {line.strip().lower() for line in proc.stdout.splitlines() if line.strip()}


def _swebench_image_for_instance(instance_id: str, image_prefix: str) -> str:
    return f"{image_prefix}.{instance_id.replace('__', '_1776_')}:latest".lower()


def _load_swebench_verified_index() -> dict[str, dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    index: dict[str, dict[str, Any]] = {}
    safe_fields = {
        "repo",
        "instance_id",
        "base_commit",
        "problem_statement",
        "hints_text",
        "created_at",
        "version",
        "environment_setup_commit",
        "difficulty",
    }
    for row in dataset:
        index[row["instance_id"]] = {key: row.get(key) for key in safe_fields if key in row}
    return index


def build_ood176_reconstruction_queue(
    candidates_jsonl: Path,
    out_jsonl: Path,
    report_out: Path,
    *,
    md_out: Path | None = None,
    ready_swebench_out: Path | None = None,
    ready_swebench_report_out: Path | None = None,
    load_swebench_verified: bool = False,
    detect_docker_images: bool = False,
    image_prefix: str = "swebench/sweb.eval.x86_64",
    swebench_index: dict[str, dict[str, Any]] | None = None,
    docker_images: set[str] | None = None,
) -> dict[str, Any]:
    """Rank ACRouter OOD176 candidates by reconstruction usefulness.

    This queue does not emit trainable TaskSpecs. It identifies which
    disagreement tasks can be safely reconstructed next and records why each one
    is not yet GRPO-ready.
    """

    candidates = _read_jsonl(candidates_jsonl)
    if swebench_index is None:
        swebench_index = _load_swebench_verified_index() if load_swebench_verified else {}
    if docker_images is None:
        docker_images = _docker_images() if detect_docker_images else set()
    docker_images = {image.lower() for image in docker_images}

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        bench = candidate.get("bench")
        original_task_id = str(candidate.get("original_task_id") or "")
        prompt = candidate.get("prompt") or ""
        model_count = int(candidate.get("model_count") or 0)
        success_count = int(candidate.get("success_count") or 0)
        center_distance = abs(success_count - (model_count / 2.0)) if model_count else 0.0
        disagreement_balance = 1.0 - (center_distance / (model_count / 2.0)) if model_count else 0.0

        status = "needs_original_environment"
        permitted_use = "reconstruction_candidate"
        blockers: list[str] = []
        next_step = "Recover original executable environment and grader."
        dataset_row = swebench_index.get(original_task_id)
        image = None
        image_available = False

        if bench == "old112":
            permitted_use = "pool_validation_only"
            image = _swebench_image_for_instance(original_task_id, image_prefix)
            image_available = image in docker_images if docker_images else False
            has_dataset = dataset_row is not None
            if has_dataset and image_available:
                status = "ready_for_swebench_adapter_validation"
                next_step = (
                    "Use the SWE-bench Verified dataset row and local eval image to "
                    "smoke a SWE-bench grading adapter; keep held out from GRPO train."
                )
            elif has_dataset:
                status = "needs_swebench_eval_image"
                blockers.append("missing_local_swebench_eval_image")
                next_step = "Build or pull the matching SWE-bench eval image."
            else:
                status = "needs_swebench_dataset_row"
                blockers.append("missing_swebench_verified_dataset_row")
                next_step = "Load the SWE-bench Verified row for this instance id."
        elif bench == "featurebench":
            permitted_use = "pool_validation_candidate"
            status = "needs_featurebench_harness"
            blockers.extend(["missing_featurebench_base_workspace", "missing_executable_featurebench_grader"])
            next_step = "Recover FeatureBench repo snapshot and test harness before TaskSpec materialization."
        elif bench == "longcli":
            permitted_use = "pool_validation_candidate"
            status = "needs_longcli_harness"
            blockers.extend(["missing_longcli_base_environment", "missing_executable_longcli_grader"])
            next_step = "Recover LongCLI container/environment and tests before TaskSpec materialization."

        if status == "ready_for_swebench_adapter_validation":
            readiness_rank = 0
        elif status.startswith("needs_swebench"):
            readiness_rank = 1
        elif status == "needs_longcli_harness":
            readiness_rank = 2
        else:
            readiness_rank = 3

        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "task_id": candidate.get("task_id"),
                "bench": bench,
                "original_task_id": original_task_id,
                "source_dataset": candidate.get("source_dataset"),
                "dimension": candidate.get("dimension"),
                "success_count": success_count,
                "model_count": model_count,
                "disagreement_balance": round(disagreement_balance, 4),
                "prompt_len": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None,
                "reconstruction_status": status,
                "permitted_use": permitted_use,
                "grpo_ready": False,
                "task_spec_ready": False,
                "blockers": blockers,
                "next_step": next_step,
                "swebench": {
                    "dataset_row_available": dataset_row is not None,
                    "repo": dataset_row.get("repo") if dataset_row else None,
                    "base_commit": dataset_row.get("base_commit") if dataset_row else None,
                    "difficulty": dataset_row.get("difficulty") if dataset_row else None,
                    "docker_image": image,
                    "docker_image_available": image_available,
                }
                if bench == "old112"
                else None,
                "rank_key": {
                    "readiness_rank": readiness_rank,
                    "negative_disagreement_balance": -round(disagreement_balance, 4),
                    "success_center_distance": center_distance,
                },
            }
        )

    rows.sort(
        key=lambda row: (
            row["rank_key"]["readiness_rank"],
            row["rank_key"]["negative_disagreement_balance"],
            row["prompt_len"],
            row["task_id"] or "",
        )
    )
    _write_jsonl(out_jsonl, rows)
    ready_swebench = [
        row for row in rows if row["reconstruction_status"] == "ready_for_swebench_adapter_validation"
    ]
    ready_report: dict[str, Any] | None = None
    if ready_swebench_out is not None:
        _write_jsonl(ready_swebench_out, ready_swebench)
        ready_report = {
            "version": "acrouter_swebench_ready_subset_v1",
            "source_queue": str(out_jsonl),
            "out_jsonl": str(ready_swebench_out),
            "rows": len(ready_swebench),
            "instance_ids": [row["original_task_id"] for row in ready_swebench],
            "policy": "pool_validation_only",
            "next_step": (
                "Implement/smoke swebench_verified grading adapter against one ready "
                "instance before any live model rollouts."
            ),
            "grpo_ready": 0,
        }
        if ready_swebench_report_out is not None:
            ready_swebench_report_out.parent.mkdir(parents=True, exist_ok=True)
            ready_swebench_report_out.write_text(
                json.dumps(ready_report, indent=2, ensure_ascii=False) + "\n"
            )

    status_counts = Counter(row["reconstruction_status"] for row in rows)
    bench_counts = Counter(row["bench"] for row in rows)
    report: dict[str, Any] = {
        "version": "acrouter_reconstruction_queue_v1",
        "candidates_jsonl": str(candidates_jsonl),
        "out_jsonl": str(out_jsonl),
        "candidate_count": len(rows),
        "status_counts": dict(status_counts),
        "bench_counts": dict(bench_counts),
        "swebench_verified_index_loaded": bool(swebench_index),
        "docker_images_detected": bool(docker_images),
        "ready_for_swebench_adapter_validation": status_counts.get(
            "ready_for_swebench_adapter_validation", 0
        ),
        "grpo_ready": 0,
        "ready_swebench_output": str(ready_swebench_out) if ready_swebench_out else None,
        "ready_swebench_report": ready_report,
        "top_candidates": rows[:10],
        "verdict": (
            "Use the ready SWE-bench rows to validate a held-out SWE-bench adapter; "
            "do not add ACRouter OOD176 tasks to GRPO training until an explicit "
            "train-allowed source is reconstructed."
        ),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if md_out is not None:
        lines = [
            "# ACRouter Reconstruction Queue",
            "",
            f"Candidates: {len(rows)}",
            "",
            "## Status Counts",
        ]
        for status, count in status_counts.most_common():
            lines.append(f"- {status}: {count}")
        lines.extend(
            [
                "",
                "## Decision",
                "- Ready SWE-bench candidates are held-out pool-validation adapter targets, not GRPO train rows.",
                "- FeatureBench and LongCLI need original executable environments before materialization.",
                "- Raw ACRouter outcomes remain disagreement evidence, not Fugu reward labels.",
            ]
        )
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text("\n".join(lines) + "\n")
    return report


def extract_ood176_disagreement_candidates(
    coderouterbench_dir: Path,
    out_jsonl: Path,
    report_out: Path,
    *,
    index_out: Path | None = None,
    md_out: Path | None = None,
) -> dict[str, Any]:
    """Extract partial-solve OOD176 tasks as reconstruction candidates.

    The output is deliberately not a TaskSpec manifest. CodeRouterBench provides
    prompts and model outcomes, but Fugu-Ultra still needs the original
    executable environment and grader before these can become GRPO tasks.
    """

    tasks_path = coderouterbench_dir / "ood176_tasks.jsonl"
    results_path = coderouterbench_dir / "ood176_results_long.csv"
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    tasks = {row["task_id"]: row for row in _read_jsonl(tasks_path)}
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    with results_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["task_id"]].append(row)

    candidates: list[dict[str, Any]] = []
    all_solve = 0
    all_fail = 0
    missing_task_meta = 0

    for task_id, result_rows in sorted(by_task.items()):
        meta = tasks.get(task_id)
        if meta is None:
            missing_task_meta += 1
            meta = {"task_id": task_id}

        outcomes: dict[str, int] = {}
        apply_ok: dict[str, int] = {}
        graded: dict[str, int] = {}
        for row in result_rows:
            model = row["model"]
            outcomes[model] = _score(row.get("resolved"))
            apply_ok[model] = _score(row.get("apply_ok"))
            graded[model] = _score(row.get("graded"))

        success_count = sum(outcomes.values())
        model_count = len(outcomes)
        if success_count == 0:
            all_fail += 1
            continue
        if success_count == model_count:
            all_solve += 1
            continue

        candidates.append(
            {
                "candidate_id": f"acrouter_ood176_partial__{task_id}",
                "source": "agent-as-a-router/CodeRouterBench",
                "source_file": str(tasks_path),
                "task_id": task_id,
                "source_split": meta.get("source_split"),
                "bench": meta.get("bench"),
                "original_task_id": meta.get("original_task_id"),
                "source_dataset": meta.get("source_dataset"),
                "dimension": meta.get("dimension"),
                "language": meta.get("language"),
                "difficulty": meta.get("difficulty"),
                "prompt": meta.get("prompt"),
                "model_outcomes": outcomes,
                "model_apply_ok": apply_ok,
                "model_graded": graded,
                "success_count": success_count,
                "model_count": model_count,
                "disagreement_type": "partial_solve",
                "grpo_ready": False,
                "usable_now_for": [
                    "pool_validation_candidate",
                    "disagreement_mining",
                    "router_baseline",
                ],
                "requires_reconstruction": [
                    "repo_base_commit_or_environment",
                    "executable_grader",
                    "workspace_snapshot_or_task_harness",
                ],
                "notes": (
                    "Prompt and model outcome matrix are present; reconstruct from the "
                    "original benchmark/source before any GRPO reward use."
                ),
            }
        )

    _write_jsonl(out_jsonl, candidates)
    if index_out is not None:
        _write_jsonl(index_out, [{k: v for k, v in row.items() if k != "prompt"} for row in candidates])

    bench_counts = Counter(
        (row.get("bench"), row.get("source_dataset"), row.get("dimension")) for row in candidates
    )
    success_histogram = Counter(row["success_count"] for row in candidates)
    report: dict[str, Any] = {
        "source": str(coderouterbench_dir),
        "tasks_file": str(tasks_path),
        "results_file": str(results_path),
        "ood_tasks": len(tasks),
        "ood_result_task_groups": len(by_task),
        "partial_solve_disagreement_candidates": len(candidates),
        "all_solve_tasks": all_solve,
        "all_fail_tasks": all_fail,
        "missing_task_metadata": missing_task_meta,
        "bench_counts": [
            {
                "bench": key[0],
                "source_dataset": key[1],
                "dimension": key[2],
                "count": count,
            }
            for key, count in bench_counts.most_common()
        ],
        "success_count_histogram": dict(sorted(success_histogram.items())),
        "outputs": {
            "with_prompts": str(out_jsonl),
            "compact_index": str(index_out) if index_out else None,
        },
        "verdict": (
            "Use as disagreement-mined reconstruction candidates and "
            "pool-validation/router baselines; do not use as GRPO training tasks "
            "until original environments and executable graders are reconstructed."
        ),
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if md_out is not None:
        lines = [
            "# ACRouter OOD176 Disagreement Candidates",
            "",
            f"Partial-solve candidates: {len(candidates)} / {len(tasks)}",
            f"All-solve: {all_solve}",
            f"All-fail: {all_fail}",
            "",
            "## Bench Mix",
        ]
        for row in report["bench_counts"]:
            lines.append(
                f"- {row['count']}: bench={row['bench']}, "
                f"source={row['source_dataset']}, dimension={row['dimension']}"
            )
        lines.extend(
            [
                "",
                "## Use",
                "- Use now for pool-validation candidate selection, disagreement mining, and router baselines.",
                "- Reconstruct original repo/environment/grader before GRPO reward use.",
                "- Do not treat ACRouter model outcomes as training rewards for Fugu-Ultra.",
            ]
        )
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text("\n".join(lines) + "\n")

    return report
