"""Hash-lock validation and final-eval manifests before live discovery."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import TaskSpec

FREEZE_VERSION = "fugu_ultra_manifest_freeze_v1"

TASKSPEC_SHARDS = (
    "data_mix/existing_bank_taskspecs.jsonl",
    "generated_repo_tasks/taskspecs.jsonl",
    "tasktrove_harbor/inferredbugs_train_taskspecs.jsonl",
    "tasktrove_harbor/pymethods2test_train_taskspecs.jsonl",
    "tool_dialog_tasks/taskspecs.jsonl",
    "long_context_tasks/taskspecs.jsonl",
    "trace_capture/branch_taskspecs.jsonl",
    "training_repo_canaries/taskspecs.jsonl",
    "scaffold_repo_taskspecs.jsonl",
)

POOL_VALIDATION_EVIDENCE = (
    ("frontier_direct_matrix", "pool_matrix_frontier.jsonl"),
    ("frontier_tau_live", "agentic_frontier_tau4.jsonl"),
    ("frontier_coding_live", "agentic_coding_frontier_direct3.jsonl"),
)


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


def _read_taskspecs(manifest_dir: Path) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    seen: set[str] = set()
    for rel in TASKSPEC_SHARDS:
        path = manifest_dir / rel
        for row in _read_jsonl(path):
            spec = TaskSpec.model_validate(row)
            if spec.task_id in seen:
                continue
            specs.append(spec)
            seen.add(spec.task_id)
    return specs


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


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _manifest_record(
    *,
    manifest_name: str,
    path: Path,
    rows: list[dict[str, Any]],
    row_schema: str,
    created_at_utc: str,
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    harness_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    task_ids: list[str] = []
    for row in rows:
        payload = row.get("payload") if row.get("record_type") == "pool_validation_evidence" else row
        source = payload.get("source")
        if isinstance(source, dict):
            source_counts[str(source.get("name"))] += 1
        elif source:
            source_counts[str(source)] += 1
        elif row.get("origin"):
            source_counts[str(row["origin"])] += 1
        splitting = payload.get("splitting")
        if isinstance(splitting, dict):
            split_counts[str(splitting.get("split"))] += 1
        else:
            split_counts[manifest_name] += 1
        environment = payload.get("environment")
        if isinstance(environment, dict):
            harness_counts[str(environment.get("harness"))] += 1
        domain = payload.get("domain")
        metadata = payload.get("metadata")
        if not domain and isinstance(metadata, dict):
            domain = metadata.get("domain")
        if domain:
            domain_counts[str(domain)] += 1
        task_id = payload.get("task_id") or row.get("task_id")
        if task_id:
            task_ids.append(str(task_id))

    task_id_hash = hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest()
    return {
        "manifest_name": manifest_name,
        "path": str(path),
        "row_count": len(rows),
        "sha256": _sha256_file(path),
        "row_schema": row_schema,
        "source_counts": _counter_json(source_counts),
        "split_counts": _counter_json(split_counts),
        "harness_counts": _counter_json(harness_counts),
        "domain_counts": _counter_json(domain_counts),
        "task_id_sha256": "sha256:" + task_id_hash,
        "created_at_utc": created_at_utc,
    }


def _pool_validation_rows(manifest_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin, rel in POOL_VALIDATION_EVIDENCE:
        path = manifest_dir / rel
        for idx, payload in enumerate(_read_jsonl(path), start=1):
            rows.append(
                {
                    "record_type": "pool_validation_evidence",
                    "origin": origin,
                    "origin_path": str(path),
                    "row_index": idx,
                    "task_id": payload.get("task_id") or payload.get("item_id"),
                    "domain": payload.get("domain"),
                    "payload": payload,
                }
            )
    return sorted(rows, key=lambda r: (str(r["origin"]), str(r["task_id"]), int(r["row_index"])))


def _task_rows(specs: list[TaskSpec], *, split: str | None = None, source_name: str | None = None) -> list[dict[str, Any]]:
    out: list[TaskSpec] = []
    for spec in specs:
        if split is not None and spec.splitting.split != split:
            continue
        if source_name is not None and spec.source.name != source_name:
            continue
        out.append(spec)
    return [spec.model_dump(mode="json") for spec in sorted(out, key=lambda s: s.task_id)]


def render_freeze_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fugu-Ultra Frozen Manifests",
        "",
        f"Version: {report['version']}",
        f"Created: {report['created_at_utc']}",
        f"Manifest dir: {report['manifest_dir']}",
        "",
        "## Manifests",
    ]
    for manifest in report["manifests"]:
        lines.extend(
            [
                f"### {manifest['manifest_name']}",
                f"- Path: {manifest['path']}",
                f"- Rows: {manifest['row_count']}",
                f"- SHA-256: {manifest['sha256']}",
                f"- Schema: {manifest['row_schema']}",
                f"- Sources: {manifest['source_counts']}",
                f"- Splits: {manifest['split_counts']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Checks",
            f"- Online/final overlap: {report['checks']['online_final_overlap_count']}",
            f"- Final/Deep SWE target overlap: {report['checks']['final_deep_swe_overlap_count']}",
            f"- Freeze complete: {report['freeze_complete']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest_freeze(
    *,
    manifest_dir: Path,
    out_dir: Path,
    created_at_utc: str | None = None,
    report_out: Path | None = None,
    md_out: Path | None = None,
) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    out_dir = out_dir.resolve()
    created_at_utc = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    specs = _read_taskspecs(manifest_dir)

    targets = [
        ("online_validation", _task_rows(specs, split="online_validation"), "TaskSpec v2"),
        ("pool_validation", _pool_validation_rows(manifest_dir), "pool_validation_evidence_v1"),
        ("final_eval", _task_rows(specs, split="final_eval"), "TaskSpec v2"),
        ("deep_swe_target_eval", _task_rows(specs, split="final_eval", source_name="deep_swe_local"), "TaskSpec v2"),
    ]

    manifests: list[dict[str, Any]] = []
    task_ids_by_name: dict[str, set[str]] = {}
    for name, rows, row_schema in targets:
        path = out_dir / f"{name}.jsonl"
        _write_jsonl(path, rows)
        manifests.append(
            _manifest_record(
                manifest_name=name,
                path=path,
                rows=rows,
                row_schema=row_schema,
                created_at_utc=created_at_utc,
            )
        )
        ids: set[str] = set()
        for row in rows:
            payload = row.get("payload") if row.get("record_type") == "pool_validation_evidence" else row
            task_id = payload.get("task_id") or row.get("task_id")
            if task_id:
                ids.add(str(task_id))
        task_ids_by_name[name] = ids

    checks = {
        "online_final_overlap_count": len(task_ids_by_name["online_validation"] & task_ids_by_name["final_eval"]),
        "final_deep_swe_overlap_count": len(task_ids_by_name["final_eval"] & task_ids_by_name["deep_swe_target_eval"]),
    }
    required_nonempty = ["online_validation", "pool_validation", "final_eval", "deep_swe_target_eval"]
    freeze_complete = all(next(m for m in manifests if m["manifest_name"] == name)["row_count"] > 0 for name in required_nonempty)
    freeze_complete = freeze_complete and checks["online_final_overlap_count"] == 0

    report = {
        "version": FREEZE_VERSION,
        "manifest_dir": str(manifest_dir),
        "out_dir": str(out_dir),
        "created_at_utc": created_at_utc,
        "manifests": manifests,
        "checks": checks,
        "freeze_complete": freeze_complete,
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_freeze_markdown(report))
    return report
