"""Pre-RL Conductor baseline reports from fixed-workflow discovery rollouts.

The baseline selectors here are deterministic workflow-template selectors. They
do not train a Conductor and they do not make provider calls; they score already
executed fixed-workflow arms so prompt-only and syntax/topology-SFT baselines are
explicit before GRPO.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

VERSION = "fugu_ultra_conductor_baselines_v1"

DIAGNOSTIC_DISCOVERY_PATH_MARKERS = (
    "bad_gold",
    "capped1024",
    "cap4096",
)

TASKTROVE_UNIT_CODE_SOURCES = {
    "tasktrove_code_contests",
    "tasktrove_pymethods2test",
}

PROMPT_ONLY_ARMS = {
    "repo_coding": "codex_plan__kimi_build__claude_verify",
    "terminal_sandbox": "terminal_gpt_plan__kimi_solve",
    "tool_dialogue": "mimo_tool__opus_review",
    "unit_and_scientific_code": "gpt_code__opus_critic__gpt_revise",
    "math_science_knowledge": "gpt_math__gemini_verify__opus_final",
    "long_context_memory_planning": "long_gpt_extract__gemini_verify__opus_final",
}

SYNTAX_TOPOLOGY_SFT_ARMS = {
    "repo_coding": "kimi_build__claude_debug__kimi_repair",
    "terminal_sandbox": "terminal_kimi_attempt__mimo_repair",
    "tool_dialogue": "glm_tool_attempt__mimo_repair",
    "unit_and_scientific_code": "gpt_algorithm__gemini_check__gpt_final",
    "math_science_knowledge": "opus_answer__gpt_critic__opus_revise",
    "long_context_memory_planning": "long_gemini_answer__opus_review",
}

DEFAULT_SHARDS = (
    {
        "name": "repo_generated_high_reasoning",
        "jobs": "scaffold_tournament_jobs.jsonl",
        "out_dir": "scaffold_discovery_high_reasoning",
    },
    {
        "name": "tasktrove_inferredbugs_harbor",
        "jobs": "scaffold_tournament_jobs.jsonl",
        "out_dir": "scaffold_discovery_high_reasoning_harbor_fixed",
    },
    {
        "name": "tasktrove_pymethods_harbor",
        "jobs": "scaffold_tournament_jobs.jsonl",
        "out_dir": "scaffold_discovery_high_reasoning_pymethods",
    },
    {
        "name": "tasktrove_prefilter_next_singles",
        "jobs": "tasktrove_prefilter_next/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_next/scaffold_discovery_high_reasoning_singles",
    },
    {
        "name": "tasktrove_prefilter_next_role_allfail",
        "jobs": "tasktrove_prefilter_next/role_followup_allfail_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_next/scaffold_discovery_high_reasoning_role_allfail",
    },
    {
        "name": "tasktrove_prefilter_optimized_singles",
        "jobs": "tasktrove_prefilter_optimized/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_optimized/scaffold_discovery_high_reasoning_singles",
    },
    {
        "name": "tasktrove_prefilter_optimized_role_open_allfail",
        "jobs": "tasktrove_prefilter_optimized/role_followup_open_allfail_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_optimized/scaffold_discovery_high_reasoning_role_open_allfail",
    },
    {
        "name": "tasktrove_prefilter_batch_003_singles",
        "jobs": "tasktrove_prefilter_batch_003/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_003/scaffold_discovery_high_reasoning_singles",
    },
    {
        "name": "tasktrove_prefilter_batch_003_role_open_allfail",
        "jobs": "tasktrove_prefilter_batch_003/role_followup_open_allfail_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_003/scaffold_discovery_high_reasoning_role_open_allfail",
    },
    {
        "name": "tasktrove_prefilter_batch_004_open_singles",
        "jobs": "tasktrove_prefilter_batch_004/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_004/scaffold_discovery_high_reasoning_open_singles",
    },
    {
        "name": "tasktrove_prefilter_batch_004_role_open_followup",
        "jobs": "tasktrove_prefilter_batch_004/role_followup_open_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_004/scaffold_discovery_high_reasoning_role_open_followup",
    },
    {
        "name": "tasktrove_prefilter_batch_005_open_singles_v3",
        "jobs": "tasktrove_prefilter_batch_005/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_005/scaffold_discovery_high_reasoning_open_singles_v3",
    },
    {
        "name": "tasktrove_prefilter_batch_006_open_singles",
        "jobs": "tasktrove_prefilter_batch_006/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_006/scaffold_discovery_high_reasoning_open_singles",
    },
    {
        "name": "tasktrove_prefilter_batch_007_open_singles",
        "jobs": "tasktrove_prefilter_batch_007/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_007/scaffold_discovery_high_reasoning_open_singles",
    },
    {
        "name": "tasktrove_prefilter_batch_009_open_singles",
        "jobs": "tasktrove_prefilter_batch_009/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_prefilter_batch_009/scaffold_discovery_high_reasoning_open_singles",
    },
    {
        "name": "tasktrove_stack_bash_v3_open_singles",
        "jobs": "tasktrove_harbor/diversity/stack_bash_v3/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_harbor/diversity/stack_bash_v3/scaffold_discovery_high_reasoning_open_singles",
    },
    {
        "name": "tasktrove_stack_bash_v3_role_probe",
        "jobs": "tasktrove_harbor/diversity/stack_bash_v3/scaffold_tournament_jobs.jsonl",
        "out_dir": "tasktrove_harbor/diversity/stack_bash_v3/scaffold_discovery_high_reasoning_role_probe",
    },
    {
        "name": "direct_reasoning_high",
        "jobs": "scaffold_tournament_jobs.jsonl",
        "out_dir": "scaffold_discovery_high_reasoning_direct",
    },
    {
        "name": "tool_dialog_custom_high",
        "jobs": "scaffold_tournament_jobs.jsonl",
        "out_dir": "scaffold_discovery_high_reasoning_tool_harder",
    },
    {
        "name": "tau_bench_retail_high",
        "jobs": "tau_bench_retail_train/scaffold_tournament_jobs.jsonl",
        "out_dir": "tau_bench_retail_train/scaffold_discovery_high_reasoning",
    },
    {
        "name": "tau_bench_retail_tail_high",
        "jobs": "tau_bench_retail_train_tail/scaffold_tournament_jobs.jsonl",
        "out_dir": "tau_bench_retail_train_tail/scaffold_discovery_high_reasoning",
    },
    {
        "name": "long_context_adversarial_high",
        "jobs": "long_context_adversarial/scaffold_tournament_jobs.jsonl",
        "out_dir": "long_context_adversarial/scaffold_discovery_high_reasoning",
    },
    {
        "name": "long_context_stress_high_v2",
        "jobs": "long_context_stress/scaffold_tournament_jobs.jsonl",
        "out_dir": "long_context_stress/scaffold_discovery_high_reasoning_v2",
    },
)


def is_diagnostic_discovery_path(path: Path) -> bool:
    """Return true for capped probe/retry shards that are not decision-grade."""

    return any(marker in part for part in path.parts for marker in DIAGNOSTIC_DISCOVERY_PATH_MARKERS)


def is_excluded_discovery_shard(path: Path) -> bool:
    """Return true for discovery shards that should not feed aggregate evidence."""

    if is_diagnostic_discovery_path(path):
        return True
    partial_summary = path / "partial_run_summary.json"
    if not partial_summary.exists():
        return False
    try:
        summary = json.loads(partial_summary.read_text())
    except json.JSONDecodeError:
        return True
    decision = str(summary.get("decision") or "")
    status = str(summary.get("status") or "")
    return decision.startswith("do_not_promote") or status.startswith("partial_stopped")


def iter_discovery_shards(manifest_dir: Path) -> list[dict[str, str]]:
    """Return known and auto-discovered completed discovery shards."""

    shards: list[dict[str, str]] = [
        dict(spec) for spec in DEFAULT_SHARDS if not is_excluded_discovery_shard(manifest_dir / str(spec["out_dir"]))
    ]
    seen = {(spec["jobs"], spec["out_dir"]) for spec in shards}

    candidates: list[tuple[Path, Path]] = []
    for root in sorted(manifest_dir.glob("tasktrove_prefilter*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))
        for jobs_path in sorted(root.glob("*jobs.jsonl")):
            if jobs_path.name == "scaffold_tournament_jobs.jsonl":
                continue
            stem = jobs_path.stem.replace("_jobs", "")
            for out_dir in sorted(root.glob(f"scaffold_discovery*{stem}*")):
                if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                    candidates.append((jobs_path, out_dir))

    for root in sorted((manifest_dir / "tasktrove_harbor" / "diversity").glob("*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for root in sorted(manifest_dir.glob("tau_bench*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for root in sorted(manifest_dir.glob("direct_unit_expansion*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for root in sorted(manifest_dir.glob("label_prior*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for root in sorted(manifest_dir.glob("trace_branch*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            jobs = manifest_dir / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for root in sorted(manifest_dir.glob("long_context*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for root in sorted(manifest_dir.glob("expert_disagreement*")):
        jobs = root / "scaffold_tournament_jobs.jsonl"
        if not jobs.exists():
            continue
        for out_dir in sorted(root.glob("scaffold_discovery*")):
            if (out_dir / "rollouts").exists() and not is_excluded_discovery_shard(out_dir):
                candidates.append((jobs, out_dir))

    for jobs, out_dir in candidates:
        rel_jobs = str(jobs.relative_to(manifest_dir))
        rel_out = str(out_dir.relative_to(manifest_dir))
        key = (rel_jobs, rel_out)
        if key in seen:
            continue
        seen.add(key)
        shards.append(
            {
                "name": rel_out.replace("/", "_"),
                "jobs": rel_jobs,
                "out_dir": rel_out,
            }
        )
    return shards


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _as_reward(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _success(rollout: dict[str, Any]) -> bool | None:
    grade = rollout.get("grade")
    if isinstance(grade, dict) and "success" in grade:
        return bool(grade["success"])
    return None


def _normalized_lane(job: dict[str, Any]) -> Any:
    source = str(job.get("source") or "")
    if source in TASKTROVE_UNIT_CODE_SOURCES:
        return "unit_and_scientific_code"
    return job.get("lane")


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("reward") is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: (float(row["reward"]), bool(row.get("success")), str(row.get("arm"))))


def _selection_summary(selected: list[dict[str, Any]], total_task_groups: int) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in selected if row.get("reward") is not None]
    outcome_counts = Counter(str(row.get("outcome_class") or "unknown") for row in selected)
    return {
        "task_groups": total_task_groups,
        "covered_task_groups": len(selected),
        "coverage_rate": len(selected) / total_task_groups if total_task_groups else None,
        "successes": sum(1 for row in selected if row.get("success") is True),
        "grade_successes": sum(1 for row in selected if row.get("grade_success") is True),
        "mean_reward": _mean(rewards),
        "arms": sorted({str(row.get("arm")) for row in selected}),
        "outcome_counts": dict(sorted(outcome_counts.items())),
    }


def _row_from_rollout(job: dict[str, Any], rollout: dict[str, Any], rollout_path: Path) -> dict[str, Any]:
    reward = _as_reward(rollout.get("reward"))
    return {
        "job_id": str(rollout.get("rollout_id") or rollout_path.stem),
        "lane": _normalized_lane(job),
        "arm_domain": job.get("arm_domain"),
        "arm": job.get("arm"),
        "stage": job.get("stage"),
        "source": job.get("source"),
        "source_task_id": job.get("source_task_id"),
        "task_jsonl": job.get("task_jsonl"),
        "task_harness": job.get("task_harness"),
        "tournament_task_id": job.get("tournament_task_id") or rollout.get("task_id"),
        "worker_names": job.get("worker_names") or [],
        "reward": reward,
        "success": reward is not None and reward >= 1.0,
        "grade_success": _success(rollout),
        "valid_for_training": bool(rollout.get("valid_for_training", True)),
        "outcome_class": rollout.get("outcome_class"),
        "rollout_path": str(rollout_path),
    }


def load_rollout_rows(jobs_jsonl: Path, out_dir: Path) -> list[dict[str, Any]]:
    jobs = {str(job["job_id"]): job for job in _read_jsonl(jobs_jsonl)}
    rows: list[dict[str, Any]] = []
    for rollout_path in sorted((out_dir / "rollouts").glob("*.json")):
        rollout = _read_json(rollout_path)
        job_id = str(rollout.get("rollout_id") or rollout_path.stem)
        job = jobs.get(job_id)
        if job is None:
            continue
        rows.append(_row_from_rollout(job, rollout, rollout_path))
    return rows


def summarize_shard(name: str, jobs_jsonl: Path, out_dir: Path) -> dict[str, Any]:
    rows = load_rollout_rows(jobs_jsonl, out_dir)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["tournament_task_id"])].append(row)

    prompt_selected: list[dict[str, Any]] = []
    sft_selected: list[dict[str, Any]] = []
    best_single_selected: list[dict[str, Any]] = []
    best_role_selected: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    reward_variance_groups = 0

    for task_id, group in sorted(by_task.items()):
        rewards = {row["reward"] for row in group if row.get("reward") is not None}
        if len(rewards) > 1:
            reward_variance_groups += 1
        arm_domain = str(group[0].get("arm_domain") or group[0].get("lane") or "")
        prompt_arm = PROMPT_ONLY_ARMS.get(arm_domain)
        sft_arm = SYNTAX_TOPOLOGY_SFT_ARMS.get(arm_domain)
        prompt = _best([row for row in group if row.get("arm") == prompt_arm])
        sft = _best([row for row in group if row.get("arm") == sft_arm])
        best_single = _best([row for row in group if row.get("stage") == "single_scaffold"])
        best_role = _best([row for row in group if row.get("stage") == "role_workflow"])

        if prompt is not None:
            prompt_selected.append(prompt)
        if sft is not None:
            sft_selected.append(sft)
        if best_single is not None:
            best_single_selected.append(best_single)
        if best_role is not None:
            best_role_selected.append(best_role)

        task_rows.append(
            {
                "tournament_task_id": task_id,
                "lane": group[0].get("lane"),
                "arm_domain": arm_domain,
                "prompt_only_arm": prompt_arm,
                "prompt_only_reward": prompt.get("reward") if prompt else None,
                "prompt_only_success": prompt.get("success") if prompt else None,
                "syntax_topology_sft_arm": sft_arm,
                "syntax_topology_sft_reward": sft.get("reward") if sft else None,
                "syntax_topology_sft_success": sft.get("success") if sft else None,
                "best_single_reward": best_single.get("reward") if best_single else None,
                "best_single_success": best_single.get("success") if best_single else None,
                "best_single_arm": best_single.get("arm") if best_single else None,
                "best_role_reward": best_role.get("reward") if best_role else None,
                "best_role_success": best_role.get("success") if best_role else None,
                "best_role_arm": best_role.get("arm") if best_role else None,
                "reward_variance": len(rewards) > 1,
            }
        )

    total_task_groups = len(by_task)
    return {
        "name": name,
        "jobs_jsonl": str(jobs_jsonl.resolve()),
        "out_dir": str(out_dir.resolve()),
        "status": "ok",
        "rollouts": len(rows),
        "task_groups": total_task_groups,
        "task_groups_with_reward_variance": reward_variance_groups,
        "reward_variance_rate": reward_variance_groups / total_task_groups if total_task_groups else None,
        "prompt_only": _selection_summary(prompt_selected, total_task_groups),
        "syntax_topology_sft": _selection_summary(sft_selected, total_task_groups),
        "best_single": _selection_summary(best_single_selected, total_task_groups),
        "best_role": _selection_summary(best_role_selected, total_task_groups),
        "task_examples": task_rows[:25],
    }


def _missing_shard(name: str, jobs_jsonl: Path, out_dir: Path) -> dict[str, Any]:
    return {
        "name": name,
        "jobs_jsonl": str(jobs_jsonl.resolve()),
        "out_dir": str(out_dir.resolve()),
        "status": "missing_rollouts",
        "rollouts": 0,
        "task_groups": 0,
    }


def _aggregate(shards: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [shard for shard in shards if shard.get("status") == "ok"]

    def combine(key: str) -> dict[str, Any]:
        covered = sum(int(shard[key]["covered_task_groups"]) for shard in ok)
        total = sum(int(shard[key]["task_groups"]) for shard in ok)
        successes = sum(int(shard[key]["successes"]) for shard in ok)
        rewards: list[float] = []
        for shard in ok:
            summary = shard[key]
            mean_reward = summary.get("mean_reward")
            if mean_reward is None:
                continue
            rewards.extend([float(mean_reward)] * int(summary["covered_task_groups"]))
        return {
            "task_groups": total,
            "covered_task_groups": covered,
            "coverage_rate": covered / total if total else None,
            "successes": successes,
            "mean_reward": _mean(rewards),
        }

    total_groups = sum(int(shard.get("task_groups", 0)) for shard in ok)
    varying = sum(int(shard.get("task_groups_with_reward_variance", 0)) for shard in ok)
    return {
        "shards_ok": len(ok),
        "task_groups": total_groups,
        "rollouts": sum(int(shard.get("rollouts", 0)) for shard in ok),
        "task_groups_with_reward_variance": varying,
        "reward_variance_rate": varying / total_groups if total_groups else None,
        "prompt_only": combine("prompt_only"),
        "syntax_topology_sft": combine("syntax_topology_sft"),
        "best_single": combine("best_single"),
        "best_role": combine("best_role"),
    }


def build_conductor_baseline_report(
    *,
    manifest_dir: Path,
    report_out: Path | None = None,
    md_out: Path | None = None,
) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    for spec in iter_discovery_shards(manifest_dir):
        jobs = manifest_dir / str(spec["jobs"])
        out_dir = manifest_dir / str(spec["out_dir"])
        if not jobs.exists() or not (out_dir / "rollouts").exists():
            shards.append(_missing_shard(str(spec["name"]), jobs, out_dir))
            continue
        shards.append(summarize_shard(str(spec["name"]), jobs, out_dir))

    report = {
        "version": VERSION,
        "purpose": "Establish prompt-only and syntax/topology-SFT Conductor baselines before GRPO.",
        "manifest_dir": str(manifest_dir.resolve()),
        "baseline_definitions": {
            "prompt_only": {
                "meaning": "Deterministic workflow chosen from written role priors; no reward fitting and no RL.",
                "arms_by_domain": PROMPT_ONLY_ARMS,
            },
            "syntax_topology_sft": {
                "meaning": "Deterministic workflow chosen from syntax/topology SFT priors and known role patterns; no RL.",
                "arms_by_domain": SYNTAX_TOPOLOGY_SFT_ARMS,
            },
        },
        "aggregate": _aggregate(shards),
        "shards": shards,
        "notes": [
            "This report scores only completed fixed-workflow discovery rollouts.",
            "It is a pre-RL baseline artifact, not a trained Conductor checkpoint.",
            "Deep SWE remains excluded from training and discovery; it is final target evaluation only.",
        ],
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
    lines = [
        "# Fugu-Ultra Pre-RL Conductor Baselines",
        "",
        f"Version: `{report['version']}`",
        "",
        "This establishes deterministic prompt-only and syntax/topology-SFT workflow baselines from completed fixed-workflow discovery rollouts. No new provider calls are made.",
        "",
        "## Aggregate",
        "",
        "| baseline | covered/task groups | successes | mean reward |",
        "| --- | ---: | ---: | ---: |",
    ]
    aggregate = report["aggregate"]
    for key, label in [
        ("prompt_only", "prompt-only"),
        ("syntax_topology_sft", "syntax/topology-SFT"),
        ("best_single", "best single worker"),
        ("best_role", "best fixed role workflow"),
    ]:
        row = aggregate[key]
        covered = f"{row['covered_task_groups']}/{row['task_groups']}"
        lines.append(f"| {label} | {covered} | {row['successes']} | {_fmt(row['mean_reward'])} |")
    lines.extend(
        [
            "",
            f"Reward-variance groups: {aggregate['task_groups_with_reward_variance']}/{aggregate['task_groups']} ({_fmt(aggregate['reward_variance_rate'])}).",
            "",
            "## Shards",
            "",
            "| shard | status | groups | variance | prompt-only | syntax/topology-SFT | best single | best role |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for shard in report["shards"]:
        if shard.get("status") != "ok":
            lines.append(f"| {shard['name']} | {shard['status']} | 0 | - | - | - | - | - |")
            continue
        groups = shard["task_groups"]
        variance = f"{shard['task_groups_with_reward_variance']}/{groups}"
        cells = []
        for key in ["prompt_only", "syntax_topology_sft", "best_single", "best_role"]:
            row = shard[key]
            cells.append(f"{row['successes']}/{row['covered_task_groups']} ({_fmt(row['mean_reward'])})")
        lines.append(f"| {shard['name']} | ok | {groups} | {variance} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## Baseline Arms",
            "",
            "Prompt-only:",
        ]
    )
    lines.extend([f"- `{domain}` -> `{arm}`" for domain, arm in sorted(PROMPT_ONLY_ARMS.items())])
    lines.append("")
    lines.append("Syntax/topology-SFT:")
    lines.extend([f"- `{domain}` -> `{arm}`" for domain, arm in sorted(SYNTAX_TOPOLOGY_SFT_ARMS.items())])
    lines.append("")
    return "\n".join(lines)
