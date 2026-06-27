"""Convert saved OpenCode rollout JSONL rows into canonical AgentTrace records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..schemas import AgentTrace
from .adapters import OpenCodeTraceAdapter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _stable_id(row: dict[str, Any]) -> str:
    key = {
        "task_id": row.get("task_id"),
        "arm": row.get("arm"),
        "workers": row.get("workers", []),
        "steps": row.get("steps", []),
    }
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "trace_opencode_" + hashlib.sha256(blob).hexdigest()[:20]


def _artifact_ref(trace_id: str, filename: str) -> str:
    return f"artifact://{trace_id}/{filename}"


def _worker_model(row: dict[str, Any]) -> str:
    slugs = [str(step.get("slug")) for step in row.get("steps", []) if step.get("slug")]
    if slugs:
        return "+".join(slugs)
    workers = [str(worker) for worker in row.get("workers", [])]
    return "+".join(workers) if workers else str(row.get("arm", "unknown"))


def _reported_cost(row: dict[str, Any]) -> float | None:
    if "cost" not in row:
        return None
    cost = float(row.get("cost") or 0.0)
    return cost if cost > 0.0 else None


def _raw_to_trace_payload(row: dict[str, Any], trace_id: str) -> dict[str, Any]:
    steps = list(row.get("steps", []))
    events = [
        {
            "type": "message",
            "agent_turn": 0,
            "content_ref": _artifact_ref(trace_id, "raw_row.json"),
            "metadata": {
                "arm": row.get("arm"),
                "stage": row.get("stage"),
                "workers": row.get("workers", []),
            },
        }
    ]
    for i, step in enumerate(steps):
        status = step.get("status")
        event_type = "error" if status not in (None, "ok") else "file_edit"
        events.append(
            {
                "type": event_type,
                "agent_turn": i + 1,
                "content_ref": _artifact_ref(trace_id, "steps.json"),
                "metadata": {
                    "worker_id": step.get("worker_id"),
                    "slug": step.get("slug"),
                    "status": status,
                    "diff_len": step.get("diff_len"),
                    "reported_cost_usd": step.get("cost"),
                },
            }
        )
    events.append(
        {
            "type": "test_result",
            "agent_turn": len(steps) + 1,
            "content_ref": _artifact_ref(trace_id, "grade.json"),
            "metadata": {
                "valid": row.get("valid", True),
                "error": row.get("error"),
            },
        }
    )
    return {
        "trace_id": trace_id,
        "harness_version": "opencode-rollout-jsonl-v1",
        "worker_model": _worker_model(row),
        "task_id": row["task_id"],
        "prompt": {
            "user_task": f"OpenCode rollout for {row['task_id']} via {row.get('arm', 'unknown arm')}",
        },
        "events": events,
        "artifacts": {
            "hidden_grade_ref": _artifact_ref(trace_id, "grade.json"),
        },
        "grade": {
            "score": float(row.get("reward", 0.0)),
            "success": float(row.get("reward", 0.0)) >= 1.0,
            "details": {
                "valid": row.get("valid", True),
                "error": row.get("error"),
                "arm": row.get("arm"),
                "stage": row.get("stage"),
                "workers": row.get("workers", []),
            },
        },
        "usage": {
            "cost_usd": _reported_cost(row),
            "wall_time_seconds": float(row.get("elapsed_s", 0.0) or 0.0),
        },
        "privacy": {
            "redacted": True,
            "contains_user_secret": False,
            "license_status": "internal_rollout_review_required",
        },
        "settings": {
            "arm": row.get("arm"),
            "stage": row.get("stage"),
            "workers": row.get("workers", []),
        },
    }


def row_to_trace(row: dict[str, Any], artifact_root: Path) -> AgentTrace:
    trace_id = _stable_id(row)
    trace_dir = artifact_root / trace_id
    _write_json(trace_dir / "raw_row.json", row)
    _write_json(trace_dir / "steps.json", row.get("steps", []))
    _write_json(
        trace_dir / "grade.json",
        {
            "score": float(row.get("reward", 0.0)),
            "success": float(row.get("reward", 0.0)) >= 1.0,
            "valid": row.get("valid", True),
            "error": row.get("error"),
        },
    )
    return OpenCodeTraceAdapter().normalize(_raw_to_trace_payload(row, trace_id))


def convert_rollouts(input_jsonl: Path, out_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = out_dir / "artifacts"
    trace_path = out_dir / "traces.jsonl"
    traces = [row_to_trace(row, artifact_root) for row in rows]
    trace_path.write_text("\n".join(trace.model_dump_json() for trace in traces) + ("\n" if traces else ""))
    report = validate_traces(trace_path, artifact_root=artifact_root)
    report.update(
        {
            "input_jsonl": str(input_jsonl),
            "trace_jsonl": str(trace_path),
            "artifact_root": str(artifact_root),
        }
    )
    _write_json(out_dir / "validation_report.json", report)
    return report


def validate_traces(trace_jsonl: Path, *, artifact_root: Path | None = None) -> dict[str, Any]:
    rows = _read_jsonl(trace_jsonl)
    traces = [AgentTrace(**row) for row in rows]
    by_model = Counter(trace.worker_model for trace in traces)
    by_success = Counter("success" if trace.grade and trace.grade.success else "failure" for trace in traces)
    missing_artifacts = 0
    if artifact_root is not None:
        for trace in traces:
            for event in trace.events:
                if not event.content_ref or not event.content_ref.startswith("artifact://"):
                    continue
                rel = event.content_ref.removeprefix("artifact://")
                if not (artifact_root / rel).exists():
                    missing_artifacts += 1
    return {
        "n_traces": len(traces),
        "origins": dict(Counter(trace.origin_harness for trace in traces)),
        "by_worker_model": dict(by_model),
        "by_success": dict(by_success),
        "missing_event_artifacts": missing_artifacts,
        "valid": missing_artifacts == 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert_rollouts(args.input_jsonl, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
