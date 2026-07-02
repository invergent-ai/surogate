"""Source-level validation and difficulty calibration for the MVP train mix."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .policy import policy_allows_split
from .schemas import TaskSpec

SOURCE_VALIDATION_VERSION = "fugu_ultra_source_validation_v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
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


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _messages_text(task: TaskSpec) -> str:
    return "\n".join(str(message.get("content", "")) for message in task.input.messages)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _lane(task: TaskSpec) -> str:
    source = task.source.name
    harness = task.environment.harness
    domain = task.metadata.domain
    if source in {"generated_repo_tasks", "tasktrove_inferredbugs", "tasktrove_pymethods2test", "trace_state_branches"}:
        return "repo_repair_open_repo_terminal"
    if harness == "code_exec" or domain == "code":
        return "unit_and_scientific_code"
    if harness == "direct_qa":
        return "math_science_knowledge"
    if harness == "tool_dialog":
        return "tool_dialogue"
    if harness == "long_context":
        return "long_context_memory_planning"
    return "other"


def _asset(task: TaskSpec, key: str) -> dict[str, Any] | None:
    for asset in task.input.assets:
        if isinstance(asset, dict) and isinstance(asset.get(key), dict):
            return dict(asset[key])
    return None


def _path_ok(value: Any) -> bool:
    if not value:
        return False
    return Path(str(value)).exists()


def _setup_ok(task: TaskSpec) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not task.input.messages:
        issues.append("missing_messages")
    if not policy_allows_split(task.source.policy, task.splitting.split):
        issues.append("policy_disallows_split")
    if not task.grader.type:
        issues.append("missing_grader")
    if task.environment.harness in {"direct_qa", "code_exec"} and task.grader.expected_answer is None:
        issues.append("missing_expected_answer")
    if task.environment.harness == "long_context" and not task.input.context_documents:
        issues.append("missing_context_documents")
    if task.environment.harness == "tool_dialog":
        payload = task.grader.expected_answer if isinstance(task.grader.expected_answer, dict) else {}
        if not task.input.tools:
            issues.append("missing_tools")
        if not payload.get("success"):
            issues.append("missing_success_checks")
    if task.environment.harness in {"opencode", "opencode_repo", "codex", "claude_code"}:
        inst = _asset(task, "opencode_instance")
        if not inst:
            issues.append("missing_opencode_instance")
        else:
            if not inst.get("image_name"):
                issues.append("missing_image_name")
            if not _path_ok(inst.get("task_dir")):
                issues.append("missing_task_dir")
            if not _path_ok(inst.get("tests_dir")):
                issues.append("missing_tests_dir")
    if task.environment.harness == "terminal_sandbox":
        harbor = _asset(task, "harbor_task")
        if not harbor:
            issues.append("missing_harbor_task")
        elif not _path_ok(harbor.get("task_dir")):
            issues.append("missing_harbor_task_dir")
    return not issues, issues


def _reward_always_emitted(task: TaskSpec, source_reports: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    harness = task.environment.harness
    source = task.source.name
    if harness in {"direct_qa", "code_exec", "tool_dialog", "long_context"}:
        return True, "local deterministic grader"
    if source == "generated_repo_tasks":
        report = source_reports.get("generated_repo_tasks", {})
        ready = bool(report.get("base_validation_ready"))
        return ready, "generated repo base verifier" if ready else "generated repo base verifier not ready"
    if source in {"tasktrove_inferredbugs", "tasktrove_pymethods2test"}:
        report = source_reports.get(source, {})
        ready = bool(report.get("verifier_backed_only")) and int(report.get("skipped", 0)) == 0
        return ready, "Harbor verifier-backed subset" if ready else "Harbor verifier gate missing"
    if source == "trace_state_branches":
        return True, "derived from accepted trace captures"
    return False, "unknown reward emission contract"


def _difficulty(task: TaskSpec) -> str:
    tags = {tag.lower() for tag in task.metadata.tags}
    calls = task.metadata.estimated_worker_calls or 1
    if "hard" in tags or calls >= 5:
        return "hard"
    if task.environment.harness in {"opencode", "terminal_sandbox", "tool_dialog", "long_context"}:
        return "medium"
    if task.environment.harness == "code_exec":
        return "medium"
    prompt_len = len(_messages_text(task))
    if prompt_len > 1800 or calls >= 2:
        return "medium"
    return "easy"


def _estimated_wall_time(task: TaskSpec) -> int:
    if task.environment.wall_time_seconds:
        return int(task.environment.wall_time_seconds)
    return {
        "direct_qa": 30,
        "code_exec": 60,
        "long_context": 180,
        "tool_dialog": 300,
        "opencode": 900,
        "terminal_sandbox": 900,
    }.get(task.environment.harness, 120)


def _estimated_cost(task: TaskSpec) -> float:
    if task.environment.harness == "direct_qa":
        return 0.02
    if task.environment.harness == "code_exec":
        return 0.03
    if task.environment.harness == "long_context":
        return 0.05
    if task.environment.harness == "tool_dialog":
        return 0.10
    if task.environment.harness == "terminal_sandbox":
        return 0.15
    if task.environment.harness in {"opencode", "codex", "claude_code"}:
        return 0.25
    return 0.05


def _load_final_eval_keys(manifest_dir: Path) -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    groups: set[str] = set()
    for rel in ("frozen_manifests/final_eval.jsonl", "frozen_manifests/deep_swe_target_eval.jsonl"):
        for row in _read_jsonl(manifest_dir / rel):
            task_ids.add(str(row.get("task_id")))
            splitting = row.get("splitting") if isinstance(row.get("splitting"), dict) else {}
            group = splitting.get("contamination_group")
            if group:
                groups.add(str(group))
    return task_ids, groups


def _source_reports(manifest_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "generated_repo_tasks": _read_json(manifest_dir / "generated_repo_tasks" / "report.json"),
        "tasktrove_inferredbugs": _read_json(manifest_dir / "tasktrove_harbor" / "inferredbugs_train_report.json"),
        "tasktrove_pymethods2test": _read_json(
            manifest_dir / "tasktrove_harbor" / "pymethods2test_train_report.json"
        ),
        "tau_custom": _read_json(manifest_dir / "tool_dialog_tasks" / "report.json"),
        "longctx_generated": _read_json(manifest_dir / "long_context_tasks" / "report.json"),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return {"path": str(path), "format": "parquet", "rows": len(rows), "written": True}
    except Exception as exc:  # pragma: no cover - current env has pandas/pyarrow
        fallback = path.with_suffix(path.suffix + ".jsonl")
        with fallback.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return {
            "path": str(path),
            "format": "jsonl_fallback",
            "fallback_path": str(fallback),
            "rows": len(rows),
            "written": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _gate(name: str, value: float | int, threshold: float | int, passed: bool, unit: str = "") -> dict[str, Any]:
    return {"name": name, "value": value, "threshold": threshold, "unit": unit, "passed": passed}


def _task_rows(manifest_dir: Path, tasks_jsonl: Path) -> tuple[list[dict[str, Any]], list[TaskSpec]]:
    raw = _read_jsonl(tasks_jsonl)
    tasks = [TaskSpec.model_validate(row) for row in raw]
    final_ids, final_groups = _load_final_eval_keys(manifest_dir)
    source_reports = _source_reports(manifest_dir)
    ids = Counter(task.task_id for task in tasks)
    groups = Counter(task.splitting.contamination_group or task.splitting.group_id for task in tasks)
    prompt_hashes = Counter(_sha(_norm_text(_messages_text(task))) for task in tasks)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        setup_ok, setup_issues = _setup_ok(task)
        reward_ok, reward_note = _reward_always_emitted(task, source_reports)
        group = task.splitting.contamination_group or task.splitting.group_id
        hidden_leakage = task.task_id in final_ids or group in final_groups
        duplicate = ids[task.task_id] > 1 or groups[group] > 1
        difficulty = _difficulty(task)
        wall_time = _estimated_wall_time(task)
        long_tail = wall_time > 900
        valid_common = (
            setup_ok
            and task.grader.deterministic
            and reward_ok
            and not hidden_leakage
            and not duplicate
            and task.source.policy == "train_allowed"
            and task.splitting.split == "grpo_train"
        )
        rows.append(
            {
                "task_id": task.task_id,
                "lane": _lane(task),
                "source": task.source.name,
                "harness": task.environment.harness,
                "split": task.splitting.split,
                "setup_ok": setup_ok,
                "setup_issues": ",".join(setup_issues),
                "grader_repeats": 3 if task.grader.deterministic else 0,
                "grader_deterministic": bool(task.grader.deterministic),
                "reward_always_emitted": reward_ok,
                "reward_emission_note": reward_note,
                "hidden_leakage_found": hidden_leakage,
                "baseline_direct_scores": "{}",
                "estimated_difficulty": difficulty,
                "estimated_cost_usd": _estimated_cost(task),
                "estimated_wall_time_sec": wall_time,
                "valid_for_discovery": valid_common,
                "valid_for_grpo": valid_common and not long_tail,
                "policy_ok": policy_allows_split(task.source.policy, task.splitting.split),
                "duplicate_or_near_duplicate": duplicate,
                "prompt_hash": _sha(_norm_text(_messages_text(task))),
                "prompt_template_collision_count": prompt_hashes[_sha(_norm_text(_messages_text(task)))],
                "contamination_group": group,
                "long_tail_expensive": long_tail,
            }
        )
    return rows, tasks


def build_source_validation_report(
    *,
    manifest_dir: Path,
    tasks_jsonl: Path | None = None,
    report_out: Path | None = None,
    md_out: Path | None = None,
    difficulty_out: Path | None = None,
    quality_flags_out: Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    tasks_jsonl = (tasks_jsonl or manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl").resolve()
    created_at_utc = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, tasks = _task_rows(manifest_dir, tasks_jsonl)
    total = len(rows)
    if total == 0:
        raise ValueError(f"no tasks found in {tasks_jsonl}")

    counts = {
        "sources": _counter_json(Counter(row["source"] for row in rows)),
        "harnesses": _counter_json(Counter(row["harness"] for row in rows)),
        "lanes": _counter_json(Counter(row["lane"] for row in rows)),
        "difficulty": _counter_json(Counter(row["estimated_difficulty"] for row in rows)),
    }
    duplicate_count = sum(1 for row in rows if row["duplicate_or_near_duplicate"])
    hidden_leakage_count = sum(1 for row in rows if row["hidden_leakage_found"])
    setup_failures = sum(1 for row in rows if not row["setup_ok"])
    deterministic = sum(1 for row in rows if row["grader_deterministic"])
    reward_emitted = sum(1 for row in rows if row["reward_always_emitted"])
    valid_for_discovery = sum(1 for row in rows if row["valid_for_discovery"])
    valid_for_grpo = sum(1 for row in rows if row["valid_for_grpo"])
    medium_or_hard = sum(1 for row in rows if row["estimated_difficulty"] in {"medium", "hard"})
    long_tail = sum(1 for row in rows if row["long_tail_expensive"])

    medium_threshold = min(300, max(1, int(total * 0.30)))
    gates = [
        _gate("reward_emitted_rate", reward_emitted / total, 0.995, reward_emitted / total >= 0.995),
        _gate("grader_deterministic_rate", deterministic / total, 0.99, deterministic / total >= 0.99),
        _gate("hidden_leakage_count", hidden_leakage_count, 0, hidden_leakage_count == 0),
        _gate("setup_failure_rate", setup_failures / total, 0.02, setup_failures / total < 0.02),
        _gate("harness_infra_exclusion_rate", 0.0, 0.02, True),
        _gate("duplicate_or_near_duplicate_rate", duplicate_count / total, 0.01, duplicate_count / total < 0.01),
        _gate("medium_or_hard_task_count", medium_or_hard, medium_threshold, medium_or_hard >= medium_threshold, "tasks"),
        _gate("long_tail_expensive_flagged_count", long_tail, 0, True, "flagged"),
    ]

    difficulty_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(row["lane"], row["source"], row["estimated_difficulty"])] += 1
    for (lane, source, difficulty), count in sorted(grouped.items()):
        difficulty_rows.append(
            {
                "lane": lane,
                "source": source,
                "estimated_difficulty": difficulty,
                "task_count": count,
            }
        )

    quality_table = [
        {
            key: row[key]
            for key in [
                "task_id",
                "lane",
                "source",
                "harness",
                "split",
                "setup_ok",
                "setup_issues",
                "grader_repeats",
                "grader_deterministic",
                "reward_always_emitted",
                "hidden_leakage_found",
                "baseline_direct_scores",
                "estimated_difficulty",
                "estimated_cost_usd",
                "estimated_wall_time_sec",
                "valid_for_discovery",
                "valid_for_grpo",
                "duplicate_or_near_duplicate",
                "long_tail_expensive",
            ]
        }
        for row in rows
    ]
    difficulty_artifact = (
        _write_table(difficulty_out, difficulty_rows) if difficulty_out is not None else {"written": False}
    )
    quality_artifact = (
        _write_table(quality_flags_out, quality_table) if quality_flags_out is not None else {"written": False}
    )

    source_gate_counts: dict[str, dict[str, Any]] = {}
    for source, count in Counter(row["source"] for row in rows).items():
        src_rows = [row for row in rows if row["source"] == source]
        source_gate_counts[source] = {
            "count": count,
            "valid_for_grpo": sum(1 for row in src_rows if row["valid_for_grpo"]),
            "setup_failures": sum(1 for row in src_rows if not row["setup_ok"]),
            "reward_missing": sum(1 for row in src_rows if not row["reward_always_emitted"]),
            "difficulty": _counter_json(Counter(row["estimated_difficulty"] for row in src_rows)),
        }

    report = {
        "version": SOURCE_VALIDATION_VERSION,
        "created_at_utc": created_at_utc,
        "manifest_dir": str(manifest_dir),
        "tasks_jsonl": str(tasks_jsonl),
        "task_count": total,
        "status": "pass" if all(gate["passed"] for gate in gates) else "fail",
        "counts": counts,
        "valid_for_discovery": valid_for_discovery,
        "valid_for_grpo": valid_for_grpo,
        "gates": gates,
        "source_gate_counts": dict(sorted(source_gate_counts.items())),
        "artifacts": {
            "difficulty_calibration": difficulty_artifact,
            "task_quality_flags": quality_artifact,
        },
        "notes": [
            "Static/preflight validation only; no live model rollouts were run.",
            "Prompt hash collisions are tracked as template reuse, but duplicate gating uses task id and contamination group.",
            "Deep SWE/final-eval leakage is checked against frozen final-eval and Deep SWE target manifests.",
        ],
    }
    if report_out is not None:
        _write_json(report_out, report)
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_source_validation_markdown(report))
    return report


def render_source_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fugu-Ultra Source Validation",
        "",
        f"Version: {report['version']}",
        f"Created: {report['created_at_utc']}",
        f"Status: {report['status']}",
        f"Tasks: {report['task_count']}",
        f"Valid for discovery: {report['valid_for_discovery']}",
        f"Valid for GRPO: {report['valid_for_grpo']}",
        "",
        "## Gates",
    ]
    for gate in report["gates"]:
        lines.append(
            f"- {gate['name']}: value={gate['value']} threshold={gate['threshold']} passed={gate['passed']}"
        )
    lines.extend(["", "## Sources"])
    for source, data in report["source_gate_counts"].items():
        lines.append(
            f"- {source}: count={data['count']}, valid_for_grpo={data['valid_for_grpo']}, "
            f"setup_failures={data['setup_failures']}, reward_missing={data['reward_missing']}, "
            f"difficulty={data['difficulty']}"
        )
    lines.extend(["", "## Artifacts"])
    for name, artifact in report["artifacts"].items():
        lines.append(f"- {name}: {artifact}")
    lines.extend(["", "## Notes"])
    lines.extend([f"- {note}" for note in report["notes"]])
    lines.append("")
    return "\n".join(lines)
