"""Harbor task-bundle source adapter.

TaskTrove and related OpenThoughts-Agent datasets materialize as Harbor task
directories: ``instruction.md`` plus ``task.toml``, ``environment/``, and
``tests/``. This adapter records those bundles as Ultra ``terminal_sandbox``
tasks while leaving execution to the Harbor harness.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ..schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)
from .raw import RawRecordAdapter


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _slug(path: Path) -> str:
    return path.name.replace("/", "__").replace(" ", "_")


def _has_verifier(task_dir: Path, meta: dict[str, Any]) -> bool:
    return bool(meta.get("verifier")) or (task_dir / "tests").exists()


def discover_harbor_task_dirs(root: Path, *, verifier_backed_only: bool = True) -> list[Path]:
    """Return Harbor task directories below ``root``.

    A directory is treated as a Harbor task when it contains both
    ``instruction.md`` and ``task.toml``. If ``verifier_backed_only`` is true,
    tasks without a verifier/tests payload are skipped.
    """

    out: list[Path] = []
    for instruction in sorted(root.rglob("instruction.md")):
        task_dir = instruction.parent
        task_toml = task_dir / "task.toml"
        if not task_toml.exists():
            continue
        try:
            meta = tomllib.loads(task_toml.read_text())
        except tomllib.TOMLDecodeError:
            continue
        if verifier_backed_only and not _has_verifier(task_dir, meta):
            continue
        out.append(task_dir)
    return out


def harbor_task_to_spec(
    task_dir: Path,
    *,
    source_name: str = "tasktrove_harbor",
    source_version: str = "v3",
    policy: str = "pool_only",
) -> TaskSpec | None:
    """Convert one Harbor task directory into an Ultra TaskSpec."""

    instruction_path = task_dir / "instruction.md"
    task_toml = task_dir / "task.toml"
    if not instruction_path.exists() or not task_toml.exists():
        return None

    try:
        meta = tomllib.loads(task_toml.read_text())
    except tomllib.TOMLDecodeError:
        return None

    instruction = instruction_path.read_text()
    if not instruction.strip():
        return None

    metadata = meta.get("metadata") or {}
    environment = meta.get("environment") or {}
    verifier = meta.get("verifier") or {}
    agent = meta.get("agent") or {}
    task_id = str(metadata.get("task_id") or _slug(task_dir))
    docker_image = environment.get("docker_image")
    difficulty = metadata.get("difficulty")

    harbor_task = {
        "task_dir": str(task_dir.resolve()),
        "agent": "terminus-2",
        "environment": "docker",
        "task_id": task_id,
        "docker_image": docker_image,
    }

    return TaskSpec(
        task_id=f"{source_name}__{task_id}",
        capability="terminal_agentic",
        source=SourceRef(
            name=source_name,
            version=source_version,
            policy=policy,  # type: ignore[arg-type]
            url_or_ref=str(task_dir.resolve()),
            license=metadata.get("license"),
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": instruction}],
            assets=[{"harbor_task": harbor_task}],
        ),
        environment=EnvironmentSpec(
            harness="terminal_sandbox",
            image=str(docker_image) if docker_image else None,
            cpu_limit=environment.get("cpus"),
            memory_mb=environment.get("memory_mb"),
            disk_mb=environment.get("storage_mb"),
            wall_time_seconds=int(float(agent.get("timeout_sec") or verifier.get("timeout_sec") or 900)),
        ),
        grader=GraderSpec(
            type="harbor_verifier",
            command=["harbor", "jobs", "start"],
            success_threshold=1.0,
        ),
        splitting=SplittingSpec(
            group_id=str(metadata.get("category") or task_dir.parent.name or source_name),
            split="pool_validation",
            contamination_group=str(task_dir.resolve()),
        ),
        metadata=TaskMetadata(
            domain="terminal",
            subdomain=metadata.get("category"),
            difficulty_estimate=None,
            tags=[
                *[str(tag) for tag in metadata.get("tags", [])],
                "harbor",
                "tasktrove",
                *(["verifier_backed"] if _has_verifier(task_dir, meta) else ["no_verifier"]),
                *([str(difficulty)] if difficulty else []),
            ],
            requires_tools=True,
            estimated_worker_calls=2,
        ),
    )


def materialize_harbor_tasks(
    root: Path,
    out_jsonl: Path,
    report_path: Path | None = None,
    *,
    source_name: str = "tasktrove_harbor",
    source_version: str = "v3",
    limit: int | None = None,
    verifier_backed_only: bool = True,
) -> dict[str, Any]:
    task_dirs = discover_harbor_task_dirs(root, verifier_backed_only=verifier_backed_only)
    if limit is not None:
        task_dirs = task_dirs[:limit]

    specs: list[TaskSpec] = []
    skipped = []
    for task_dir in task_dirs:
        spec = harbor_task_to_spec(task_dir, source_name=source_name, source_version=source_version)
        if spec is None:
            skipped.append(str(task_dir))
        else:
            specs.append(spec)

    _write_jsonl(out_jsonl, [spec.model_dump(mode="json") for spec in specs])
    report = {
        "version": "harbor_task_materialization_v1",
        "root": str(root),
        "out_jsonl": str(out_jsonl),
        "materialized": len(specs),
        "skipped": len(skipped),
        "verifier_backed_only": verifier_backed_only,
        "live_calls": False,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


class HarborTaskBundleAdapter(RawRecordAdapter):
    """Raw-record adapter for preselected Harbor task directories."""

    source_name = "tasktrove_harbor"
    capability = "terminal_agentic"
    policy = "pool_only"
    harness = "terminal_sandbox"
    source_type = "harbor_task_bundle"
    version = "v3"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        task_dir = Path(str(raw.get("task_dir") or raw.get("path") or ""))
        if not task_dir.exists():
            return None
        return harbor_task_to_spec(
            task_dir,
            source_name=str(raw.get("source_name") or self.source_name),
            source_version=str(raw.get("source_version") or self.version),
            policy=str(raw.get("policy") or self.policy),
        )
