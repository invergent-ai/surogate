"""Frozen failure taxonomy and reward mapping for Ultra rollouts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import ExecStep, Grade, RolloutRecord

FAILURE_TAXONOMY_VERSION = "fugu_ultra_failure_taxonomy_v1"

FAILURE_TAXONOMY_ENTRIES: dict[str, dict[str, Any]] = {
    "invalid_workflow_trainable": {
        "reward": 0.0,
        "use": "train",
        "retry": False,
        "meaning": "Conductor emitted malformed workflow syntax, bad worker id, bad access list, or too many steps.",
    },
    "valid_incorrect_trainable": {
        "reward": 0.5,
        "use": "train",
        "retry": False,
        "meaning": "Workflow executed and grader emitted an incorrect terminal result.",
    },
    "valid_correct_trainable": {
        "reward": 1.0,
        "use": "train",
        "retry": False,
        "meaning": "Workflow executed and grader emitted a successful terminal result.",
    },
    "budget_exhausted_trainable": {
        "reward": 0.5,
        "use": "train",
        "retry": False,
        "meaning": "Workflow is valid but stopped by model/tool budget or truncation before success.",
    },
    "provider_failure_retry_or_exclude": {
        "reward": None,
        "use": "retry_or_exclude",
        "retry": True,
        "meaning": "Provider/auth/rate-limit/connectivity failure; do not train on first failure.",
    },
    "harness_crash_exclude": {
        "reward": None,
        "use": "exclude",
        "retry": False,
        "meaning": "Harness/CLI/container crashed or could not produce a usable terminal artifact.",
    },
    "grader_crash_quarantine": {
        "reward": None,
        "use": "quarantine",
        "retry": False,
        "meaning": "Task ran but verifier/grader failed to emit a trustworthy reward.",
    },
    "task_setup_failure_quarantine": {
        "reward": None,
        "use": "quarantine",
        "retry": False,
        "meaning": "Task payload, initial state, tests, or bundled context are missing or invalid.",
    },
}

PROVIDER_FAILURE_TERMINATIONS = {
    "missing_provider_key",
    "provider_error",
    "rate_limited",
    "connection_error",
    "api_error",
}
TASK_SETUP_FAILURE_TERMINATIONS = {
    "missing_task_payload",
    "missing_context",
    "missing_tools",
    "initial_patch_failed",
}
HARNESS_CRASH_TERMINATIONS = {
    "missing_cli",
    "missing_harbor_cli",
    "container_start_failed",
    "workspace_export_failed",
    "diff_failed",
    "cli_failed",
    "harbor_failed",
    "not_implemented",
}
BUDGET_EXHAUSTED_TERMINATIONS = {
    "timeout",
    "harbor_timeout",
    "truncated",
    "budget_exhausted",
}
MODEL_INCORRECT_TERMINATIONS = {
    "completed",
    "cli_nonzero_with_patch",
    "max_turns_or_no_tool_call",
}

_GRADE_SETUP_ERRORS = (
    "unsupported",
    "missing an opencode_instance payload",
    "has no final container",
)
_GRADE_CRASH_ERRORS = (
    "no Harbor verifier rewards found",
    "verifier",
    "grader",
)
_GRADE_HARNESS_ERRORS = (
    "harbor environment setup failed",
    "docker compose command failed",
    "failed to create network",
    "all predefined address pools",
)
_PROVIDER_FAILURE_ERRORS = (
    "ratelimit",
    "rate limit",
    "429",
    "apierror",
    "api error",
    "apitimeout",
    "connection error",
    "authentication",
    "api key",
    "provider",
    "upstream",
)


def reward_for_class(outcome_class: str) -> float | None:
    return FAILURE_TAXONOMY_ENTRIES[outcome_class]["reward"]


def use_for_class(outcome_class: str) -> str:
    return str(FAILURE_TAXONOMY_ENTRIES[outcome_class]["use"])


def is_trainable_class(outcome_class: str) -> bool:
    return use_for_class(outcome_class) == "train"


def _step_class(step: ExecStep) -> str | None:
    term = step.termination
    if term in PROVIDER_FAILURE_TERMINATIONS:
        return "provider_failure_retry_or_exclude"
    if term in TASK_SETUP_FAILURE_TERMINATIONS:
        return "task_setup_failure_quarantine"
    if term in HARNESS_CRASH_TERMINATIONS:
        return "harness_crash_exclude"
    if term in BUDGET_EXHAUSTED_TERMINATIONS:
        return "budget_exhausted_trainable"
    return None


def _grade_error_class(grade: Grade | None) -> str | None:
    if grade is None:
        return "harness_crash_exclude"
    details = grade.details or {}
    error = str(details.get("error") or details.get("step_error") or "")
    if not error:
        return None
    lowered = error.lower()
    if any(needle in lowered for needle in _GRADE_SETUP_ERRORS):
        return "task_setup_failure_quarantine"
    if any(needle in lowered for needle in _GRADE_HARNESS_ERRORS):
        return "harness_crash_exclude"
    if any(needle in lowered for needle in _GRADE_CRASH_ERRORS):
        return "grader_crash_quarantine"
    return None


def classify_rollout_outcome(
    *,
    workflow_parse_valid: bool,
    grade: Grade | None,
    exec_steps: list[ExecStep],
    failure_class: str | None = None,
) -> str:
    """Return the frozen taxonomy class for a rollout-like outcome."""

    if not workflow_parse_valid or (failure_class or "").startswith("invalid_workflow"):
        return "invalid_workflow_trainable"
    lowered_failure = (failure_class or "").lower()
    if any(needle in lowered_failure for needle in _PROVIDER_FAILURE_ERRORS):
        return "provider_failure_retry_or_exclude"
    for step in exec_steps:
        step_class = _step_class(step)
        if step_class in {
            "provider_failure_retry_or_exclude",
            "task_setup_failure_quarantine",
            "harness_crash_exclude",
        }:
            return step_class
    for step in exec_steps:
        if _step_class(step) == "budget_exhausted_trainable":
            return "budget_exhausted_trainable"
    grade_class = _grade_error_class(grade)
    if grade_class in {"task_setup_failure_quarantine", "grader_crash_quarantine", "harness_crash_exclude"}:
        return grade_class
    if grade is None:
        return "harness_crash_exclude"
    return "valid_correct_trainable" if grade.success else "valid_incorrect_trainable"


def apply_outcome_class(record: RolloutRecord, *, detail: str | None = None) -> RolloutRecord:
    """Return ``record`` with reward/training fields aligned to the frozen taxonomy."""

    outcome_class = classify_rollout_outcome(
        workflow_parse_valid=record.conductor.workflow_parse_valid,
        grade=record.grade,
        exec_steps=list(record.execution.steps),
        failure_class=record.failure_class,
    )
    reward = reward_for_class(outcome_class)
    valid_for_training = is_trainable_class(outcome_class)
    failure_class = record.failure_class
    if outcome_class not in {
        "valid_correct_trainable",
        "valid_incorrect_trainable",
        "budget_exhausted_trainable",
    }:
        if detail:
            failure_class = f"{outcome_class}: {detail}"
        elif failure_class is None:
            failure_class = outcome_class
        elif not failure_class.startswith(outcome_class):
            failure_class = f"{outcome_class}: {failure_class}"
    return record.model_copy(
        update={
            "outcome_class": outcome_class,
            "reward": reward,
            "valid_for_training": valid_for_training,
            "failure_class": failure_class,
        }
    )


def _taxonomy_hash() -> str:
    payload = {
        "entries": FAILURE_TAXONOMY_ENTRIES,
        "provider_failure_terminations": sorted(PROVIDER_FAILURE_TERMINATIONS),
        "task_setup_failure_terminations": sorted(TASK_SETUP_FAILURE_TERMINATIONS),
        "harness_crash_terminations": sorted(HARNESS_CRASH_TERMINATIONS),
        "budget_exhausted_terminations": sorted(BUDGET_EXHAUSTED_TERMINATIONS),
        "model_incorrect_terminations": sorted(MODEL_INCORRECT_TERMINATIONS),
        "provider_failure_errors": sorted(_PROVIDER_FAILURE_ERRORS),
    }
    data = json.dumps(payload, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_rollout(path: Path) -> RolloutRecord | None:
    try:
        data = json.loads(path.read_text())
        return RolloutRecord.model_validate(data)
    except Exception:
        return None


def _audit_saved_rollouts(manifest_dir: Path) -> dict[str, Any]:
    patterns = [
        "canaries/*.json",
        "trace_capture/rollouts/*.json",
    ]
    counts: Counter[str] = Counter()
    scanned = 0
    accepted = 0
    examples: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(manifest_dir.glob(pattern)):
            scanned += 1
            rollout = _read_rollout(path)
            if rollout is None:
                continue
            accepted += 1
            outcome_class = classify_rollout_outcome(
                workflow_parse_valid=rollout.conductor.workflow_parse_valid,
                grade=rollout.grade,
                exec_steps=list(rollout.execution.steps),
                failure_class=rollout.failure_class,
            )
            counts[outcome_class] += 1
            examples.setdefault(outcome_class, str(path))
    return {
        "patterns": patterns,
        "files_scanned": scanned,
        "rollout_records": accepted,
        "outcome_class_counts": dict(sorted(counts.items())),
        "examples": examples,
    }


def build_failure_taxonomy_report(
    *,
    manifest_dir: Path,
    report_out: Path | None = None,
    md_out: Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    created_at_utc = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "version": FAILURE_TAXONOMY_VERSION,
        "created_at_utc": created_at_utc,
        "manifest_dir": str(manifest_dir),
        "taxonomy_sha256": _taxonomy_hash(),
        "frozen": True,
        "entries": FAILURE_TAXONOMY_ENTRIES,
        "termination_mapping": {
            "provider_failure_retry_or_exclude": sorted(PROVIDER_FAILURE_TERMINATIONS),
            "task_setup_failure_quarantine": sorted(TASK_SETUP_FAILURE_TERMINATIONS),
            "harness_crash_exclude": sorted(HARNESS_CRASH_TERMINATIONS),
            "budget_exhausted_trainable": sorted(BUDGET_EXHAUSTED_TERMINATIONS),
            "model_terminal_but_not_infra": sorted(MODEL_INCORRECT_TERMINATIONS),
        },
        "rollout_field_mapping": {
            "outcome_class": "canonical frozen taxonomy label",
            "reward": "mapped from outcome_class; null means retry/exclude/quarantine, not training reward",
            "valid_for_training": "true only when entry.use == train",
            "failure_class": "diagnostic detail for invalid, retry, exclude, and quarantine classes",
        },
        "saved_rollout_audit": _audit_saved_rollouts(manifest_dir),
        "rules": [
            "Invalid workflows are trainable with reward 0.0.",
            "Valid incorrect workflows are trainable with reward 0.5.",
            "Valid correct workflows are trainable with reward 1.0.",
            "Budget exhaustion is trainable with reward 0.5 only when the workflow/harness state is otherwise valid.",
            "Provider, harness, grader, and task setup failures have reward null and must not become learned negative reward.",
        ],
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_failure_taxonomy_markdown(report))
    return report


def render_failure_taxonomy_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fugu-Ultra Failure Taxonomy",
        "",
        f"Version: {report['version']}",
        f"Created: {report['created_at_utc']}",
        f"Manifest dir: {report['manifest_dir']}",
        f"Frozen: {report['frozen']}",
        f"Taxonomy SHA-256: {report['taxonomy_sha256']}",
        "",
        "## Entries",
    ]
    for name, entry in report["entries"].items():
        lines.extend(
            [
                f"### {name}",
                f"- Reward: {entry['reward']}",
                f"- Use: {entry['use']}",
                f"- Retry: {entry['retry']}",
                f"- Meaning: {entry['meaning']}",
                "",
            ]
        )
    lines.extend(["## Saved Rollout Audit"])
    audit = report["saved_rollout_audit"]
    lines.append(f"- Files scanned: {audit['files_scanned']}")
    lines.append(f"- Rollout records: {audit['rollout_records']}")
    for name, count in audit["outcome_class_counts"].items():
        example = audit["examples"].get(name)
        lines.append(f"- {name}: {count}" + (f" (example: {example})" if example else ""))
    lines.extend(["", "## Rules"])
    lines.extend([f"- {rule}" for rule in report["rules"]])
    lines.append("")
    return "\n".join(lines)
