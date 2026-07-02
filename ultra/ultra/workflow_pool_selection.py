"""Workflow-outcome worker pool selection for Fugu-Ultra.

This report estimates worker/scaffold contribution from completed fixed-workflow
discovery rollouts. It intentionally works on exact worker identities
(``model + scaffold + settings``), not raw model names.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .conductor_baselines import iter_discovery_shards, load_rollout_rows
from .scaffold_tournament import canonical_workers

VERSION = "fugu_ultra_workflow_pool_selection_v1"

REQUIRED_BASELINE_WORKERS = {
    "codex_gpt_coding_agent": "frontier coding scaffold baseline",
    "claude_code_opus_debugger": "frontier coding scaffold baseline",
    "opencode_kimi_builder": "observed coding specialist and required OpenCode baseline",
    "direct_gemini_synth": "frontier direct/long-context baseline",
    "direct_gpt_reasoner": "frontier direct/math/planning baseline",
    "direct_opus_reviewer": "frontier reviewer/debugger baseline",
}

PILOT_LANE_MASKS = {
    "trace_state_branches": [
        "codex_gpt_coding_agent",
        "claude_code_opus_debugger",
        "opencode_kimi_builder",
        "opencode_mimo_repair",
        "opencode_glm_builder",
        "opencode_flash_challenger",
    ],
    "repo_open_repo_terminal": [
        "terminal_gpt_agent",
        "terminal_kimi_agent",
        "terminal_mimo_agent",
        "terminal_glm_agent",
    ],
    "tool_dialogue": [
        "tool_dialog_mimo_agent",
        "tool_dialog_glm_agent",
        "direct_opus_reviewer",
    ],
    "unit_and_scientific_code": [
        "direct_gpt_reasoner",
        "direct_gemini_synth",
        "direct_opus_reviewer",
        "direct_flash_fast",
        "direct_glm_reasoner",
        "direct_mimo_reasoner",
        "direct_minimax_reasoner",
    ],
    "math_science_knowledge": [
        "direct_gpt_reasoner",
        "direct_gemini_synth",
        "direct_opus_reviewer",
        "direct_flash_fast",
        "direct_glm_reasoner",
    ],
    "long_context_memory_planning": [
        "direct_gemini_synth",
        "direct_gpt_reasoner",
        "direct_opus_reviewer",
        "direct_glm_reasoner",
        "direct_mimo_reasoner",
        "direct_minimax_reasoner",
        "direct_flash_fast",
    ],
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("reward") is not None and row.get("valid_for_training") is not False]
    if not scored:
        return None
    return max(scored, key=lambda row: (float(row["reward"]), bool(row.get("success")), str(row.get("arm"))))


def _worker_rows(rows: list[dict[str, Any]], worker: str) -> list[dict[str, Any]]:
    return [row for row in rows if worker in set(row.get("worker_names") or [])]


def _without_worker(rows: list[dict[str, Any]], worker: str) -> list[dict[str, Any]]:
    return [row for row in rows if worker not in set(row.get("worker_names") or [])]


def _normalize_row(row: dict[str, Any], *, shard: str) -> dict[str, Any]:
    reward = row.get("reward")
    success = reward is not None and float(reward) >= 1.0
    return {
        **row,
        "shard": shard,
        "success": success,
        "worker_names": list(row.get("worker_names") or []),
    }


def load_completed_rows(manifest_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in iter_discovery_shards(manifest_dir):
        jobs = manifest_dir / str(spec["jobs"])
        out_dir = manifest_dir / str(spec["out_dir"])
        if not jobs.exists() or not (out_dir / "rollouts").exists():
            continue
        rows.extend(_normalize_row(row, shard=str(spec["name"])) for row in load_rollout_rows(jobs, out_dir))
    return rows


def _task_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("tournament_task_id") or row.get("task_id") or row.get("job_id"))].append(row)
    return groups


def _observed_workers(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({worker for row in rows for worker in (row.get("worker_names") or [])})


def _worker_catalog() -> dict[str, dict[str, Any]]:
    return {worker.name: worker.model_dump() for worker in canonical_workers()}


def estimate_leave_one_out(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = _task_groups(rows)
    workers = _observed_workers(rows)
    worker_stats: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for worker in workers:
        deltas: list[float] = []
        success_drops = 0
        positive_groups = 0
        evaluated = 0
        for task_id, group in sorted(groups.items()):
            if not _worker_rows(group, worker):
                continue
            best_all = _best(group)
            best_without = _best(_without_worker(group, worker))
            if best_all is None or best_without is None:
                continue
            evaluated += 1
            delta = float(best_all["reward"]) - float(best_without["reward"])
            deltas.append(delta)
            if delta > 0:
                positive_groups += 1
                if len(examples[worker]) < 8:
                    examples[worker].append(
                        {
                            "tournament_task_id": task_id,
                            "lane": best_all.get("lane"),
                            "shard": best_all.get("shard"),
                            "best_all_arm": best_all.get("arm"),
                            "best_all_reward": best_all.get("reward"),
                            "best_without_worker_arm": best_without.get("arm"),
                            "best_without_worker_reward": best_without.get("reward"),
                            "reward_delta": delta,
                        }
                    )
            if best_all.get("success") is True and best_without.get("success") is not True:
                success_drops += 1

        worker_rollouts = _worker_rows(rows, worker)
        solo_rollouts = [
            row
            for row in worker_rollouts
            if row.get("stage") == "single_scaffold" and len(row.get("worker_names") or []) == 1
        ]
        outcome_counts = Counter(str(row.get("outcome_class") or "unknown") for row in worker_rollouts)
        lanes = sorted({str(row.get("lane")) for row in worker_rollouts})
        arm_domains = sorted({str(row.get("arm_domain")) for row in worker_rollouts})
        worker_stats[worker] = {
            "evaluated_task_groups": evaluated,
            "mean_leave_one_out_reward_delta": _mean(deltas),
            "total_leave_one_out_reward_delta": sum(deltas),
            "positive_reward_contribution_groups": positive_groups,
            "success_drop_groups": success_drops,
            "rollouts_involving_worker": len(worker_rollouts),
            "successes_involving_worker": sum(1 for row in worker_rollouts if row.get("success") is True),
            "solo_rollouts": len(solo_rollouts),
            "solo_successes": sum(1 for row in solo_rollouts if row.get("success") is True),
            "lanes": lanes,
            "arm_domains": arm_domains,
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "positive_examples": examples[worker],
        }
    return worker_stats


def _recommendations(worker_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    retained: dict[str, list[str]] = {}
    challengers: dict[str, list[str]] = {}
    deferred: dict[str, list[str]] = {}
    mask_workers = {worker for workers in PILOT_LANE_MASKS.values() for worker in workers}
    for worker, stats in sorted(worker_stats.items()):
        reasons: list[str] = []
        if stats["success_drop_groups"] > 0:
            reasons.append(f"leave-one-out success drop on {stats['success_drop_groups']} task group(s)")
        elif stats["positive_reward_contribution_groups"] > 0:
            reasons.append(
                f"positive reward contribution on {stats['positive_reward_contribution_groups']} task group(s)"
            )
        if worker in REQUIRED_BASELINE_WORKERS:
            reasons.append(REQUIRED_BASELINE_WORKERS[worker])
        if reasons:
            retained[worker] = reasons
            continue

        if worker in mask_workers:
            challenge_reasons = ["lane-mask pilot candidate; current discovery is insufficient to prune this role"]
            if stats["solo_successes"] > 0 or stats["successes_involving_worker"] > 0:
                challenge_reasons.append("observed successes but no current leave-one-out contribution")
            challengers[worker] = challenge_reasons
        elif stats["solo_successes"] > 0 or stats["successes_involving_worker"] > 0:
            challengers[worker] = ["observed successes but outside the initial lane masks"]
        else:
            deferred[worker] = ["no observed success or leave-one-out contribution in completed discovery shards"]

    recommended = sorted(set(retained) | {worker for worker in mask_workers if worker in worker_stats})
    return {
        "selection_status": "mvp_grpo_pool_selected_for_pilot_not_final_ultra_claim",
        "retain_workers": retained,
        "challenger_workers": challengers,
        "defer_workers": deferred,
        "recommended_mvp_grpo_workers": recommended,
        "lane_worker_masks": PILOT_LANE_MASKS,
        "selection_rule": [
            "Retain if leave-one-out workflow success or reward drops when the worker is removed.",
            "Retain exact frontier/scaffold baselines needed for credible Ultra comparisons.",
            "Use lane masks early so GRPO does not waste samples on irrelevant scaffolds.",
            "Do not treat saturated long-context/direct shards as evidence to prune frontier workers.",
        ],
    }


def _summarize_tasks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _task_groups(rows)
    reward_variance = 0
    oracle_successes = 0
    best_single_successes = 0
    best_role_successes = 0
    paired_groups = 0
    role_improvements = 0
    role_losses = 0

    for group in groups.values():
        rewards = {row.get("reward") for row in group if row.get("reward") is not None}
        if len(rewards) > 1:
            reward_variance += 1
        best_all = _best(group)
        if best_all and best_all.get("success") is True:
            oracle_successes += 1
        best_single = _best([row for row in group if row.get("stage") == "single_scaffold"])
        best_role = _best([row for row in group if row.get("stage") == "role_workflow"])
        if best_single and best_single.get("success") is True:
            best_single_successes += 1
        if best_role and best_role.get("success") is True:
            best_role_successes += 1
        if best_single and best_role:
            paired_groups += 1
            delta = float(best_role["reward"]) - float(best_single["reward"])
            if delta > 0:
                role_improvements += 1
            elif delta < 0:
                role_losses += 1

    total = len(groups)
    return {
        "rollouts": len(rows),
        "task_groups": total,
        "reward_variance_groups": reward_variance,
        "reward_variance_rate": reward_variance / total if total else None,
        "workflow_oracle_successes": oracle_successes,
        "best_single_successes": best_single_successes,
        "best_role_successes": best_role_successes,
        "paired_single_role_groups": paired_groups,
        "role_improvement_groups": role_improvements,
        "role_loss_groups": role_losses,
    }


def build_workflow_pool_selection_report(
    *,
    manifest_dir: Path,
    report_out: Path | None = None,
    md_out: Path | None = None,
) -> dict[str, Any]:
    rows = load_completed_rows(manifest_dir)
    worker_stats = estimate_leave_one_out(rows)
    catalog = _worker_catalog()
    report = {
        "version": VERSION,
        "purpose": "Select the MVP GRPO worker/scaffold pool from paired workflow outcomes.",
        "manifest_dir": str(manifest_dir.resolve()),
        "scope_note": (
            "This is a candidate pool for the first GRPO pilot. Final Ultra claims still require held-out "
            "online validation and final evaluation against individual model+scaffold baselines."
        ),
        "task_summary": _summarize_tasks(rows),
        "workers": {
            worker: {
                "identity": catalog.get(worker, {"name": worker}),
                **stats,
            }
            for worker, stats in sorted(worker_stats.items())
        },
        "recommendations": _recommendations(worker_stats),
    }
    if report_out is not None:
        _write_json(report_out, report)
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_markdown(report))
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    task = report["task_summary"]
    rec = report["recommendations"]
    lines = [
        "# Fugu-Ultra Workflow Pool Selection",
        "",
        f"Version: `{report['version']}`",
        "",
        report["scope_note"],
        "",
        "## Task Evidence",
        "",
        f"- Rollouts: {task['rollouts']}",
        f"- Task groups: {task['task_groups']}",
        f"- Reward-variance groups: {task['reward_variance_groups']} ({_fmt(task['reward_variance_rate'])})",
        f"- Best single successes: {task['best_single_successes']}",
        f"- Best role successes: {task['best_role_successes']}",
        f"- Workflow-oracle successes: {task['workflow_oracle_successes']}",
        f"- Role improvements/losses: {task['role_improvement_groups']} / {task['role_loss_groups']}",
        "",
        "## Recommended MVP GRPO Workers",
        "",
    ]
    lines.extend([f"- `{worker}`" for worker in rec["recommended_mvp_grpo_workers"]])
    lines.extend(
        [
            "",
            "## Retention Reasons",
            "",
        ]
    )
    for worker, reasons in sorted(rec["retain_workers"].items()):
        lines.append(f"- `{worker}`: {'; '.join(reasons)}")
    lines.extend(
        [
            "",
            "## Worker Contribution",
            "",
            "| worker | LOO groups | positive groups | success drops | solo success | involved success | mean delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for worker, stats in sorted(report["workers"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{worker}`",
                    str(stats["evaluated_task_groups"]),
                    str(stats["positive_reward_contribution_groups"]),
                    str(stats["success_drop_groups"]),
                    f"{stats['solo_successes']}/{stats['solo_rollouts']}",
                    f"{stats['successes_involving_worker']}/{stats['rollouts_involving_worker']}",
                    _fmt(stats["mean_leave_one_out_reward_delta"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Lane Masks",
            "",
        ]
    )
    for lane, workers in rec["lane_worker_masks"].items():
        lines.append(f"- `{lane}`: {', '.join(f'`{worker}`' for worker in workers)}")
    lines.append("")
    return "\n".join(lines)
