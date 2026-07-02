"""Harbor task-bundle source adapter.

TaskTrove and related OpenThoughts-Agent datasets materialize as Harbor task
directories: ``instruction.md`` plus ``task.toml``, ``environment/``, and
``tests/``. This adapter records those bundles as Ultra ``terminal_sandbox``
tasks while leaving execution to the Harbor harness.
"""

from __future__ import annotations

import io
import json
import random
import re
import shutil
import tarfile
import tomllib
from pathlib import Path
from pathlib import PurePosixPath
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


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:120] or "task"


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
    split: str = "pool_validation",
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
            split=split,  # type: ignore[arg-type]
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
    policy: str = "pool_only",
    split: str = "pool_validation",
    limit: int | None = None,
    verifier_backed_only: bool = True,
) -> dict[str, Any]:
    task_dirs = discover_harbor_task_dirs(root, verifier_backed_only=verifier_backed_only)
    if limit is not None:
        task_dirs = task_dirs[:limit]

    specs: list[TaskSpec] = []
    skipped = []
    for task_dir in task_dirs:
        spec = harbor_task_to_spec(
            task_dir,
            source_name=source_name,
            source_version=source_version,
            policy=policy,
            split=split,
        )
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
        "policy": policy,
        "split": split,
        "verifier_backed_only": verifier_backed_only,
        "live_calls": False,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _tasktrove_rows_from_parquet(parquet_path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is present in the project venv
        raise RuntimeError("pyarrow is required to read TaskTrove parquet shards") from exc

    table = pq.read_table(parquet_path, columns=["path", "task_binary"])
    return table.to_pylist()


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    name = member.name.replace("\\", "/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe tar member path: {member.name!r}")
    if member.issym() or member.islnk():
        raise ValueError(f"unsafe tar link member: {member.name!r}")
    if not (member.isdir() or member.isfile()):
        raise ValueError(f"unsupported tar member type: {member.name!r}")


def extract_tasktrove_parquet_bundles(
    parquet_path: Path,
    out_dir: Path,
    *,
    include_paths: set[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    seed: int = 0,
    shuffle: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract TaskTrove ``task_binary`` Harbor bundles from one parquet shard.

    The public TaskTrove shards store each Harbor bundle as a gzipped tar payload
    in ``task_binary`` with a source path in ``path``. This function only unpacks
    bundles; model execution and grading remain separate Harbor-harness steps.
    """

    rows = _tasktrove_rows_from_parquet(parquet_path)
    indexed = list(enumerate(rows))
    if include_paths is not None:
        indexed = [(row_index, row) for row_index, row in indexed if str(row.get("path") or "") in include_paths]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indexed)
    if offset:
        indexed = indexed[offset:]
    if limit is not None:
        indexed = indexed[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for row_index, row in indexed:
        source_path = str(row.get("path") or f"row-{row_index:06d}")
        binary = row.get("task_binary")
        if binary is None:
            skipped.append({"row_index": row_index, "path": source_path, "reason": "missing_task_binary"})
            continue
        if isinstance(binary, memoryview):
            payload = binary.tobytes()
        else:
            payload = bytes(binary)

        base_name = _safe_name(Path(source_path).stem or f"row-{row_index:06d}")
        safe_name = base_name
        suffix = 1
        while safe_name in used_names:
            suffix += 1
            safe_name = _safe_name(f"{base_name}_{suffix}")
        used_names.add(safe_name)
        task_root = out_dir / safe_name
        if task_root.exists():
            if not overwrite:
                skipped.append({"row_index": row_index, "path": source_path, "reason": "target_exists"})
                continue
            shutil.rmtree(task_root)
        task_root.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tar:
                members = tar.getmembers()
                for member in members:
                    _validate_tar_member(member)
                tar.extractall(task_root, members=members, filter="data")
        except Exception as exc:
            shutil.rmtree(task_root, ignore_errors=True)
            skipped.append(
                {
                    "row_index": row_index,
                    "path": source_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        task_dirs = [str(path.resolve()) for path in discover_harbor_task_dirs(task_root)]
        extracted.append(
            {
                "row_index": row_index,
                "path": source_path,
                "task_root": str(task_root.resolve()),
                "harbor_task_dirs": task_dirs,
            }
        )

    return {
        "version": "tasktrove_parquet_extract_v1",
        "parquet_path": str(parquet_path),
        "out_dir": str(out_dir),
        "rows_read": len(rows),
        "rows_selected": len(indexed),
        "include_paths_count": len(include_paths) if include_paths is not None else None,
        "extracted": len(extracted),
        "skipped": len(skipped),
        "shuffle": shuffle,
        "seed": seed,
        "offset": offset,
        "live_calls": False,
        "extracted_tasks": extracted,
        "skipped_tasks": skipped,
    }


def materialize_tasktrove_parquet(
    parquet_path: Path,
    extract_dir: Path,
    out_jsonl: Path,
    report_path: Path | None = None,
    *,
    source_name: str = "tasktrove_harbor",
    source_version: str = "v3",
    policy: str = "train_allowed",
    split: str = "grpo_train",
    include_paths: set[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    seed: int = 0,
    shuffle: bool = False,
    overwrite: bool = False,
    verifier_backed_only: bool = True,
) -> dict[str, Any]:
    extraction = extract_tasktrove_parquet_bundles(
        parquet_path,
        extract_dir,
        include_paths=include_paths,
        limit=limit,
        offset=offset,
        seed=seed,
        shuffle=shuffle,
        overwrite=overwrite,
    )
    materialization = materialize_harbor_tasks(
        extract_dir,
        out_jsonl,
        None,
        source_name=source_name,
        source_version=source_version,
        policy=policy,
        split=split,
        verifier_backed_only=verifier_backed_only,
    )
    report = {
        "version": "tasktrove_parquet_materialization_v1",
        "parquet_path": str(parquet_path),
        "extract_dir": str(extract_dir),
        "out_jsonl": str(out_jsonl),
        "source_name": source_name,
        "source_version": source_version,
        "policy": policy,
        "split": split,
        "include_paths_count": len(include_paths) if include_paths is not None else None,
        "verifier_backed_only": verifier_backed_only,
        "live_calls": False,
        "extraction": extraction,
        "materialization": materialization,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def download_tasktrove_parquet(
    *,
    hf_file: str,
    cache_dir: Path | None = None,
    repo_id: str = "open-thoughts/TaskTrove",
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is present in the project venv
        raise RuntimeError("huggingface_hub is required for --hf-file downloads") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=hf_file,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


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
            split=str(raw.get("split") or "pool_validation"),
        )
