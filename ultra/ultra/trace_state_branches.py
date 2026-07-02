"""Audit trace exports for state-level branch training candidates."""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import AgentTrace


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _candidate_id(*parts: str) -> str:
    return "branch_" + hashlib.sha256("::".join(parts).encode()).hexdigest()[:20]


def _trace_candidate(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    trace = AgentTrace.model_validate(row)
    event_types = [event.type for event in trace.events]
    has_repo_state = bool(trace.repo.url and trace.repo.base_commit)
    has_patch = bool(trace.artifacts.final_patch_ref)
    has_workspace_snapshot = bool(trace.artifacts.workspace_snapshot_ref)
    has_public_test_log = bool(trace.artifacts.public_test_log_ref)
    has_command_or_error = any(t in {"command", "error"} for t in event_types)
    has_test_result = "test_result" in event_types
    train_ready = (
        has_repo_state
        and has_patch
        and has_workspace_snapshot
        and (has_public_test_log or has_command_or_error or has_test_result)
    )
    missing = []
    if not has_repo_state:
        missing.append("repo_state")
    if not has_patch:
        missing.append("final_patch")
    if not has_workspace_snapshot:
        missing.append("workspace_snapshot")
    if not (has_public_test_log or has_command_or_error or has_test_result):
        missing.append("execution_feedback")
    return {
        "candidate_id": _candidate_id(str(path), trace.trace_id),
        "source_kind": "agent_trace",
        "source_path": str(path),
        "task_id": trace.task_id,
        "origin_harness": trace.origin_harness,
        "worker_model": trace.worker_model,
        "state_type": "trace_checkpoint" if train_ready else "outcome_only_trace",
        "event_counts": dict(Counter(event_types)),
        "available_artifacts": {
            "repo_state": has_repo_state,
            "final_patch": has_patch,
            "workspace_snapshot": has_workspace_snapshot,
            "public_test_log": has_public_test_log,
            "command_or_error": has_command_or_error,
            "test_result": has_test_result,
        },
        "train_ready": train_ready,
        "missing_for_training": missing,
    }


def _rollout_candidate(path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    steps = ((row.get("execution") or {}).get("steps") or [])
    if not steps:
        return None
    patch_text = "\n".join(str(step.get("text") or "") for step in steps if step.get("text"))
    if not patch_text.strip():
        return None
    grade = row.get("grade") or {}
    source_name = str(row.get("source_name") or "")
    forbidden_sources = {"training_repo_canary", "deep_swe_local"}
    train_ready = bool(row.get("valid_for_training")) and source_name not in forbidden_sources
    missing = []
    if source_name in forbidden_sources:
        missing.append("train_allowed_source")
    if not row.get("valid_for_training"):
        missing.append("valid_training_flag")
    return {
        "candidate_id": _candidate_id(str(path), str(row.get("rollout_id") or row.get("task_id"))),
        "source_kind": "rollout_record",
        "source_path": str(path),
        "task_id": row.get("task_id"),
        "origin_harness": row.get("harness"),
        "worker_ids": [step.get("worker_id") for step in steps],
        "state_type": "post_patch_rollout",
        "patch_sha256": _sha256_text(patch_text),
        "patch_chars": len(patch_text),
        "reward": row.get("reward"),
        "grade_success": grade.get("success"),
        "train_ready": train_ready,
        "missing_for_training": missing,
    }


def build_trace_state_branch_report(
    *,
    trace_jsonls: list[Path],
    rollout_jsons: list[Path],
    out_jsonl: Path,
    report_out: Path,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in trace_jsonls:
        for row in _read_json_records(path):
            candidates.append(_trace_candidate(path, row))
    for path in rollout_jsons:
        if not path.exists():
            continue
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        candidate = _rollout_candidate(path, row)
        if candidate is not None:
            candidates.append(candidate)

    _write_jsonl(out_jsonl, candidates)
    report = {
        "version": "trace_state_branch_audit_v1",
        "candidate_jsonl": str(out_jsonl),
        "trace_jsonls": [str(path) for path in trace_jsonls],
        "rollout_jsons": [str(path) for path in rollout_jsons],
        "candidate_count": len(candidates),
        "train_ready_count": sum(1 for c in candidates if c["train_ready"]),
        "by_source_kind": dict(Counter(c["source_kind"] for c in candidates)),
        "by_state_type": dict(Counter(c["state_type"] for c in candidates)),
        "missing_for_training": dict(
            Counter(reason for c in candidates for reason in c.get("missing_for_training", []))
        ),
        "live_calls": False,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(Path(path) for path in matched)
    return paths
