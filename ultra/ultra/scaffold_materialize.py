"""Materialize scaffold-tournament task records into canonical TaskSpecs."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    RepoRef,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def deep_swe_task_to_spec(task: dict[str, Any]) -> TaskSpec | None:
    """Convert one local Deep SWE tournament task into an OpenCode TaskSpec.

    This supplies the repo/image/problem payload that OpenCode needs. The local Deep
    SWE verifier is intentionally recorded as a grader command/ref; connecting that
    grader to the OpenCode harness is a separate execution step.
    """

    task_dir = Path(str(task.get("task_dir", "")))
    if not task_dir.exists():
        return None
    task_toml = task_dir / "task.toml"
    instruction = task_dir / "instruction.md"
    tests_dir = task_dir / "tests"
    if not task_toml.exists() or not instruction.exists() or not tests_dir.exists():
        return None

    meta = tomllib.loads(task_toml.read_text())
    metadata = meta.get("metadata", {})
    environment = meta.get("environment", {})
    verifier = meta.get("verifier", {})
    task_id = str(metadata.get("task_id") or task.get("source_task_id") or task_dir.name)
    repo_url = str(metadata.get("repository_url") or "")
    base_commit = str(metadata.get("base_commit_hash") or "")
    image = str(environment.get("docker_image") or "")
    problem = instruction.read_text()
    if not repo_url or not base_commit or not image or not problem.strip():
        return None

    opencode_instance = {
        "image_name": image,
        "instance_id": "",
        "problem_statement": problem,
        "testbed": "/app",
        "activate": "",
        "task_id": task_id,
        "task_dir": str(task_dir),
        "tests_dir": str(tests_dir),
        "test_command": "bash /tests/test.sh",
        "grader": "deep_swe_v1.1",
        "config_ref": str(tests_dir / "config.json"),
        "test_patch_ref": str(tests_dir / "test.patch"),
    }

    return TaskSpec(
        task_id=f"deep_swe_local__{task_id}",
        capability="agentic_coding",
        source=SourceRef(
            name="deep_swe_local",
            version=str(meta.get("schema_version") or "local"),
            policy="final_eval_only",
            url_or_ref=str(task_dir),
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": problem}],
            assets=[{"opencode_instance": opencode_instance}],
            repo=RepoRef(url=repo_url, base_commit=base_commit),
        ),
        environment=EnvironmentSpec(
            harness="opencode",
            image=image,
            cpu_limit=int((environment.get("cpus") or 2)),
            memory_mb=int((environment.get("memory_mb") or 8192)),
            disk_mb=int((environment.get("storage_mb") or 20480)),
            wall_time_seconds=int(float(verifier.get("timeout_sec") or 1800)),
        ),
        grader=GraderSpec(
            type="deep_swe_hidden_tests",
            command=["bash", "/tests/test.sh"],
            success_threshold=1.0,
        ),
        splitting=SplittingSpec(
            group_id=repo_url,
            split="final_eval",
            contamination_group=repo_url,
        ),
        metadata=TaskMetadata(
            domain="software_engineering",
            subdomain=str(metadata.get("language") or "unknown"),
            tags=[*[str(tag) for tag in task.get("selection_tags", [])], "materialized", "deep_swe_local"],
            requires_tools=True,
            estimated_worker_calls=3,
        ),
    )


def materialize_repo_tasks(manifest_path: Path, out_jsonl: Path, report_path: Path | None = None) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    specs: list[TaskSpec] = []
    unresolved = []
    for task in manifest.get("tasks", []):
        if task.get("domain") != "coding_repo":
            continue
        if task.get("source") == "deep_swe_local":
            spec = deep_swe_task_to_spec(task)
            if spec is None:
                unresolved.append(
                    {
                        "source": task.get("source"),
                        "source_task_id": task.get("source_task_id"),
                        "reason": "deep_swe_task_missing_required_files_or_metadata",
                    }
                )
            else:
                specs.append(spec)
        else:
            unresolved.append(
                {
                    "source": task.get("source"),
                    "source_task_id": task.get("source_task_id"),
                    "reason": "saved_live_coding_payload_not_available",
                }
            )

    rows = [spec.model_dump(mode="json") for spec in specs]
    _write_jsonl(out_jsonl, rows)
    report = {
        "version": "scaffold_repo_taskspec_materialization_v1",
        "manifest_path": str(manifest_path),
        "out_jsonl": str(out_jsonl),
        "materialized": len(specs),
        "unresolved": len(unresolved),
        "unresolved_tasks": unresolved,
        "grader_status": {
            "deep_swe_hidden_tests": "opencode_bridge_implemented_not_live_smoked",
            "note": "TaskSpecs contain local verifier refs; OpenCode can write model.patch and invoke /tests/test.sh; run a no/low-spend Docker canary before paid rollouts.",
        },
        "live_calls": False,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
