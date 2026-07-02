"""Offline gap plan for scaling the GRPO pilot seed.

The GRPO seed contains only tasks with observed workflow disagreement/headroom.
This module compares that seed against a first 300-task pilot target and records
which lanes need more validated, reward-varying tasks before training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .workflow_pool_selection import load_completed_rows

VERSION = "fugu_ultra_grpo_pilot_gap_plan_v1"

DEFAULT_TARGET_LANE_COUNTS = {
    "repo_open_repo_terminal": 80,
    "trace_state_branches": 20,
    "unit_and_scientific_code": 55,
    "math_science_knowledge": 55,
    "tool_dialogue": 60,
    "long_context_memory_planning": 30,
}

SOURCE_LANE_HINTS = {
    "tasktrove_inferredbugs": "repo_open_repo_terminal",
    "tasktrove_pymethods2test": "unit_and_scientific_code",
    "tasktrove_stack_bash_v3": "repo_open_repo_terminal",
    "tau_bench_retail_train": "tool_dialogue",
    "tau_custom": "tool_dialogue",
    "trace_state_branches": "trace_state_branches",
    "longctx_adversarial": "long_context_memory_planning",
    "longctx_counterfactual": "long_context_memory_planning",
    "longctx_stress": "long_context_memory_planning",
    "tasktrove_agent_calendar": "tool_dialogue",
}

TASKTROVE_TASKSPEC_SHARDS = {
    "tasktrove_inferredbugs": "tasktrove_harbor/inferredbugs_train_taskspecs.jsonl",
    "tasktrove_pymethods2test": "tasktrove_harbor/pymethods2test_train_taskspecs.jsonl",
}

TASKTROVE_LOCAL_PARQUETS = {
    "DCAgent/inferredbugs-sandboxes-verifier": (
        "tasktrove_harbor/hf/DCAgent__inferredbugs-sandboxes-verifier/tasks.parquet"
    ),
    "DCAgent/exp_rpt_pymethods2test-v3": (
        "tasktrove_harbor/hf/DCAgent__exp_rpt_pymethods2test-v3/tasks.parquet"
    ),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


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


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _as_reward(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            import pandas as pd

            return int(len(pd.read_parquet(path)))
        except Exception:
            return None


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [
        row
        for row in rows
        if _as_reward(row.get("reward")) is not None and row.get("valid_for_training") is not False
    ]
    if not scored:
        return None
    return max(scored, key=lambda row: (_as_reward(row.get("reward")) or 0.0, str(row.get("arm"))))


def _task_key(row: dict[str, Any]) -> str:
    if row.get("task_jsonl") and row.get("source_task_id"):
        return f"{row['task_jsonl']}::{row['source_task_id']}"
    return str(row.get("tournament_task_id") or row.get("task_id") or row.get("job_id"))


def _group_completed_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_task_key(row)].append(row)
    return groups


def _group_has_success(rows: list[dict[str, Any]]) -> bool:
    return any((_as_reward(row.get("reward")) or 0.0) >= 1.0 for row in rows)


def _summarize_completed(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    lane_stats: dict[str, Counter[str]] = defaultdict(Counter)
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)

    for group in _group_completed_rows(rows).values():
        first = group[0]
        lane = str(first.get("lane") or "unknown")
        source = str(first.get("source") or "unknown")
        rewards = [
            reward
            for row in group
            if row.get("valid_for_training") is not False
            for reward in [_as_reward(row.get("reward"))]
            if reward is not None
        ]
        reward_values = set(rewards)
        best_all = _best(group)
        best_single = _best([row for row in group if row.get("stage") == "single_scaffold"])
        best_role = _best([row for row in group if row.get("stage") == "role_workflow"])
        role_delta = None
        if best_single is not None and best_role is not None:
            role_delta = (_as_reward(best_role.get("reward")) or 0.0) - (_as_reward(best_single.get("reward")) or 0.0)

        for stats in (lane_stats[lane], source_stats[source]):
            stats["task_groups"] += 1
            stats["trainable_rollouts"] += sum(1 for row in group if row.get("valid_for_training") is not False)
            stats["success_rollouts"] += sum(1 for row in group if (_as_reward(row.get("reward")) or 0.0) >= 1.0)
            stats["groups_with_success"] += int(_group_has_success(group))
            stats["reward_variance_groups"] += int(len(reward_values) > 1)
            stats["role_improvement_groups"] += int(role_delta is not None and role_delta > 0)
            stats["role_loss_groups"] += int(role_delta is not None and role_delta < 0)
            stats["workflow_oracle_headroom_groups"] += int(
                best_single is not None
                and best_all is not None
                and (_as_reward(best_all.get("reward")) or 0.0) > (_as_reward(best_single.get("reward")) or 0.0)
            )

    return (
        {lane: dict(stats) for lane, stats in sorted(lane_stats.items())},
        {source: dict(stats) for source, stats in sorted(source_stats.items())},
    )


def _lane_recommendation(
    lane: str,
    deficit: int,
    evidence: dict[str, Any],
    taskcraft: dict[str, Any] | None,
    tasktrove: dict[str, Any],
) -> str:
    if deficit <= 0:
        return "Target already met; keep as validation coverage but do not oversample."
    tasktrove_count = int(tasktrove.get("materialized_train_allowed_task_count") or 0)
    if lane == "repo_open_repo_terminal":
        return (
            f"Use the materialized verifier-backed TaskTrove reservoir first ({tasktrove_count} train-allowed "
            "tasks locally), especially inferredbugs for repo/terminal prefiltering; add generated repo repairs "
            "and scaffolded OpenCode/Codex/Claude trace branches for true repo-harness coverage."
        )
    if lane == "trace_state_branches":
        return "Derive more train-ready branch states from accepted OpenCode/Codex/Claude traces; avoid Deep SWE."
    if lane == "tool_dialogue":
        return "Continue tau-bench retail train shards and role followups; this lane currently has the strongest reward variance."
    if lane == "long_context_memory_planning":
        variance = int(evidence.get("reward_variance_groups", 0))
        if variance > 0:
            return (
                "Continue the counterfactual long-context source family with small OpenRouter-only singles first; "
                "admit only reward-varying rows, then add commercial/role followup on selected groups."
            )
        if taskcraft and taskcraft.get("candidate_count"):
            return (
                "Use TaskCraft only after freezing source docs/pages, auditing chain consistency, and adding deterministic "
                "graders; current generated long-context tasks are saturated."
            )
        return "Import or generate less extractive long-context tasks; current generated sources are saturated."
    if lane == "unit_and_scientific_code":
        return (
            "Use more TaskTrove pymethods2test for fast verifier-backed code signal, then add harder unit/scientific "
            "code tasks where role workflows can beat singles."
        )
    if lane == "math_science_knowledge":
        return (
            "Use existing direct tasks for curriculum, and materialize the recommended TaskTrove math/science/knowledge "
            "diversity shards before inventing new sources."
        )
    if evidence.get("reward_variance_groups", 0):
        return "Expand from the reward-varying source family and run role followup."
    return "Find harder validated tasks; current evidence is too sparse or saturated."


def _priority_lanes(deficits: dict[str, int], evidence_by_lane: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    priorities = []
    for lane, deficit in deficits.items():
        evidence = evidence_by_lane.get(lane, {})
        variance = int(evidence.get("reward_variance_groups", 0))
        role = int(evidence.get("role_improvement_groups", 0))
        score = max(deficit, 0) * 10 - variance * 2 - role * 4
        priorities.append(
            {
                "lane": lane,
                "deficit": deficit,
                "observed_reward_variance_groups": variance,
                "observed_role_improvement_groups": role,
                "priority_score": score,
            }
        )
    return sorted(priorities, key=lambda row: (-row["priority_score"], row["lane"]))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _source_action(source: str, lane: str) -> str:
    if source == "tau_bench_retail_train":
        return "Continue tau-bench train shards; run high-reasoning singles plus role followup on partial/all-fail groups."
    if source == "tasktrove_inferredbugs":
        return "Run small ranked inferredbugs prefilter batches; follow up only on reward-varying, partial-solve, or all-fail groups."
    if source == "tasktrove_pymethods2test":
        return "Use as fast unit-code signal; prefer singles for disagreement, because current role templates do not improve this source."
    if source == "tasktrove_stack_bash_v3":
        return "Use only as a small shell/terminal challenger source until more variance is proven."
    if source == "trace_state_branches":
        return "Create more branch states from accepted train-allowed OpenCode/Codex/Claude traces; keep Deep SWE excluded."
    if source == "longctx_counterfactual":
        return "Continue this source with OpenRouter-only long-context singles; it produced reward variance without Yunwu calls."
    if source == "existing_bank":
        return "Mine remaining direct/code bank rows offline first; spend live calls only on rows with unresolved worker disagreement."
    if lane == "long_context_memory_planning":
        return "Do not expand saturated generated long-context rows; use TaskCraft only after source/grade audit."
    return "Expand cautiously with small high-reasoning prefilters, then admit only reward-varying tasks."


def _source_expansion_queue(
    deficits: dict[str, int],
    evidence_by_source: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for source, evidence in sorted(evidence_by_source.items()):
        lane = SOURCE_LANE_HINTS.get(source)
        if lane is None and source == "existing_bank":
            lane = "math_science_knowledge"
        if lane is None:
            continue
        deficit = int(deficits.get(lane, 0))
        if deficit <= 0:
            continue
        task_groups = int(evidence.get("task_groups") or 0)
        variance = int(evidence.get("reward_variance_groups") or 0)
        role_improvements = int(evidence.get("role_improvement_groups") or 0)
        role_losses = int(evidence.get("role_loss_groups") or 0)
        trainable = int(evidence.get("trainable_rollouts") or 0)
        successes = int(evidence.get("success_rollouts") or 0)
        if task_groups <= 0:
            continue
        priority_score = deficit * 10 + variance * 5 + role_improvements * 12 - role_losses * 4
        queue.append(
            {
                "source": source,
                "lane": lane,
                "lane_deficit": deficit,
                "observed_task_groups": task_groups,
                "observed_reward_variance_groups": variance,
                "observed_role_improvement_groups": role_improvements,
                "observed_success_rate": _rate(successes, trainable),
                "observed_variance_rate": _rate(variance, task_groups),
                "recommended_next_candidate_tasks": min(deficit, 24 if lane == "repo_open_repo_terminal" else 16),
                "priority_score": priority_score,
                "action": _source_action(source, lane),
            }
        )
    return sorted(queue, key=lambda row: (-row["priority_score"], row["source"]))


def _deprioritized_sources(evidence_by_source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, evidence in sorted(evidence_by_source.items()):
        task_groups = int(evidence.get("task_groups") or 0)
        variance = int(evidence.get("reward_variance_groups") or 0)
        groups_with_success = int(evidence.get("groups_with_success") or 0)
        trainable = int(evidence.get("trainable_rollouts") or 0)
        successes = int(evidence.get("success_rollouts") or 0)
        if task_groups < 5 or variance > 0:
            continue
        if groups_with_success == task_groups:
            reason = "saturated_all_solve"
        elif groups_with_success == 0:
            reason = "all_fail_or_no_success_signal"
        else:
            reason = "no_reward_variance"
        rows.append(
            {
                "source": source,
                "observed_task_groups": task_groups,
                "observed_success_rate": _rate(successes, trainable),
                "reason": reason,
                "action": _source_action(source, SOURCE_LANE_HINTS.get(source, "unknown")),
            }
        )
    return rows


def _tasktrove_inventory(manifest_dir: Path, seed_source_counts: Counter[str]) -> dict[str, Any]:
    materialized: dict[str, dict[str, Any]] = {}
    total_materialized = 0
    for source, relpath in TASKTROVE_TASKSPEC_SHARDS.items():
        path = manifest_dir / relpath
        rows = _read_jsonl(path)
        total_materialized += len(rows)
        materialized[source] = {
            "path": str(path.resolve()),
            "task_count": len(rows),
            "seed_count": int(seed_source_counts.get(source, 0)),
            "remaining_unseeded_count": max(0, len(rows) - int(seed_source_counts.get(source, 0))),
            "harnesses": sorted({str(row.get("environment", {}).get("harness")) for row in rows}),
            "split": sorted({str(row.get("splitting", {}).get("split")) for row in rows}),
            "policy": sorted({str(row.get("source", {}).get("policy")) for row in rows}),
        }

    local_parquets: dict[str, dict[str, Any]] = {}
    known_local_parquet_rows = 0
    all_local_parquet_counts_known = True
    for dataset, relpath in TASKTROVE_LOCAL_PARQUETS.items():
        path = manifest_dir / relpath
        rows = _parquet_row_count(path)
        if rows is None:
            all_local_parquet_counts_known = False
        else:
            known_local_parquet_rows += rows
        local_parquets[dataset] = {
            "path": str(path.resolve()),
            "exists": path.exists(),
            "row_count": rows,
        }

    subset_selection = _read_json(manifest_dir / "tasktrove_harbor" / "subset_selection.json")
    recommended_shards = subset_selection.get("recommended_diversity_shards", []) if subset_selection else []
    return {
        "status": "primary_expansion_reservoir",
        "materialized_train_allowed_task_count": total_materialized,
        "materialized_train_allowed": materialized,
        "local_parquet_row_count": known_local_parquet_rows if all_local_parquet_counts_known else None,
        "local_parquet_row_count_known": all_local_parquet_counts_known,
        "local_parquets": local_parquets,
        "recommended_diversity_shards": recommended_shards,
        "recommended_diversity_shard_count": len(recommended_shards),
        "expansion_policy": [
            "Use materialized verifier-backed TaskTrove tasks before inventing new sources for repo/terminal and unit-code deficits.",
            "Run high-reasoning single-prefilter batches first, then role followup only on reward-varying/all-fail/partial-solve groups.",
            "Keep TaskTrove rows train-only; do not let these tasks affect frozen final eval composition.",
        ],
    }


def build_grpo_pilot_gap_plan(
    *,
    manifest_dir: Path,
    report_out: Path | None = None,
    seed_jsonl: Path | None = None,
    target_lane_counts: dict[str, int] | None = None,
    completed_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    target_lane_counts = target_lane_counts or DEFAULT_TARGET_LANE_COUNTS
    seed_jsonl = seed_jsonl or manifest_dir / "grpo_pilot_seed" / "seed_manifest.jsonl"
    seed_rows = _read_jsonl(seed_jsonl)
    rows = completed_rows if completed_rows is not None else load_completed_rows(manifest_dir)
    evidence_by_lane, evidence_by_source = _summarize_completed(rows)
    seed_lane_counts = Counter(str(row.get("lane") or "unknown") for row in seed_rows)
    seed_source_counts = Counter(str(row.get("source") or "unknown") for row in seed_rows)
    deficits = {
        lane: max(0, int(target) - int(seed_lane_counts.get(lane, 0)))
        for lane, target in sorted(target_lane_counts.items())
    }
    taskcraft_report = _read_json(manifest_dir / "taskcraft_source_probe" / "report.json")
    taskcraft_audit = _read_json(manifest_dir / "taskcraft_source_probe" / "readiness_audit.json")
    tasktrove_inventory = _tasktrove_inventory(manifest_dir, seed_source_counts)

    lane_actions = {
        lane: _lane_recommendation(
            lane,
            deficits[lane],
            evidence_by_lane.get(lane, {}),
            taskcraft_report,
            tasktrove_inventory,
        )
        for lane in sorted(target_lane_counts)
    }
    used_files = [
        str(seed_jsonl.resolve()),
        str((manifest_dir / "grpo_pilot_seed" / "report.json").resolve()),
        str((manifest_dir / "workflow_pool_selection_report.json").resolve()),
        str((manifest_dir / "conductor_baseline_report.json").resolve()),
        str((manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl").resolve()),
        str((manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl").resolve()),
        str((manifest_dir / "tasktrove_harbor" / "subset_selection.json").resolve()),
    ]
    if taskcraft_report is not None:
        used_files.append(str((manifest_dir / "taskcraft_source_probe" / "report.json").resolve()))
        used_files.append(str((manifest_dir / "taskcraft_source_probe" / "candidates.jsonl").resolve()))
    if taskcraft_audit is not None:
        used_files.append(str((manifest_dir / "taskcraft_source_probe" / "readiness_audit.json").resolve()))
        used_files.append(str((manifest_dir / "taskcraft_source_probe" / "readiness_evidence.jsonl").resolve()))

    report = {
        "version": VERSION,
        "status": "gap_plan_not_training_manifest",
        "purpose": "Plan expansion from the current disagreement/headroom seed to a first 300-task GRPO pilot.",
        "manifest_dir": str(manifest_dir.resolve()),
        "seed_jsonl": str(seed_jsonl.resolve()),
        "target_task_count": sum(target_lane_counts.values()),
        "target_lane_counts": dict(sorted(target_lane_counts.items())),
        "current_seed_task_count": len(seed_rows),
        "current_seed_lane_counts": _counter_json(seed_lane_counts),
        "current_seed_source_counts": _counter_json(seed_source_counts),
        "lane_deficits": deficits,
        "evidence_by_lane": evidence_by_lane,
        "evidence_by_source": evidence_by_source,
        "priority_lanes": _priority_lanes(deficits, evidence_by_lane),
        "next_expansion_queue": _source_expansion_queue(deficits, evidence_by_source),
        "deprioritized_sources": _deprioritized_sources(evidence_by_source),
        "lane_actions": lane_actions,
        "tasktrove_reservoir": tasktrove_inventory,
        "taskcraft_candidate_status": {
            "available": taskcraft_report is not None,
            "status": taskcraft_report.get("status") if taskcraft_report else None,
            "candidate_count": taskcraft_report.get("candidate_count") if taskcraft_report else 0,
            "candidate_count_before_limit": taskcraft_report.get("candidate_count_before_limit")
            if taskcraft_report
            else 0,
            "raw_dataset_grpo_ready": taskcraft_report.get("raw_dataset_grpo_ready") if taskcraft_report else False,
            "readiness_blockers": taskcraft_report.get("readiness_blockers") if taskcraft_report else [],
            "audit_status": taskcraft_audit.get("status") if taskcraft_audit else None,
            "audit_promote_to_grpo": taskcraft_audit.get("decision", {}).get("promote_to_grpo")
            if taskcraft_audit
            else False,
            "freeze_priority_count": taskcraft_audit.get("freeze_priority_count") if taskcraft_audit else 0,
            "audit_linkage_counts": taskcraft_audit.get("linkage_counts") if taskcraft_audit else {},
            "audit_blocker_counts": taskcraft_audit.get("readiness_blocker_counts") if taskcraft_audit else {},
        },
        "go_no_go": [
            f"Do not start GRPO from the {len(seed_rows)}-task seed alone.",
            "Build the first pilot only from deterministic, reward-varying, budget-feasible tasks.",
            "Keep Deep SWE out of training and discovery expansion.",
            "Use the existing verifier-backed TaskTrove reservoir as the first expansion source for terminal/code deficits.",
            "Do not admit TaskCraft rows until source freeze, answer grading, and chain-consistency audits pass.",
        ],
        "used_files": used_files,
    }
    if report_out is not None:
        _write_json(report_out, report)
    return report
