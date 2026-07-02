"""Training task distribution plan for Fugu-Ultra.

This module is intentionally lightweight: it does not materialize or execute
tasks. It records the source mix, split policy, and currently available local
artifacts so pool selection and GRPO prep do not drift back into direct-only or
Deep-SWE-as-training shortcuts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .failure_taxonomy import FAILURE_TAXONOMY_ENTRIES

MVP_TASK_MIX = {
    "repo_repair_open_repo_terminal": 250,
    "unit_and_scientific_code": 225,
    "math_science_knowledge": 250,
    "tool_dialogue": 150,
    "long_context_memory_planning": 125,
}

MVP_TASK_MIX_NAME = "MVP GRPO candidate train distribution, pending fixed-workflow discovery"

SOURCE_VALIDATION_GATES = [
    "source policy allows the target split",
    "deterministic verifier or exact direct grader",
    "base/empty patch fails and known-good patch passes for repo/terminal tasks",
    "reward is emitted for success, failure, invalid output, and timeout",
    "one no-model verifier canary for containerized sources",
    "one model-backed canary before pool/tournament inclusion",
    "contamination group assigned before split",
    "hidden-answer leakage check passes",
]

TASKTROVE_VALIDATION_GATES = [
    "reward is always emitted",
    "failure produces reward=0 or documented failure score",
    "timeout is classified explicitly",
    "grader is deterministic over repeated local runs",
    "task setup is reproducible from the bundled environment",
    "invalid output is not silently dropped",
]

MANIFEST_FREEZE_REQUIREMENTS = {
    "status": "required_before_fixed_workflow_discovery",
    "manifests": [
        "online_validation",
        "pool_validation",
        "final_eval",
        "deep_swe_target_eval",
    ],
    "required_fields": [
        "manifest_name",
        "path",
        "row_count",
        "sha256",
        "source_counts",
        "split_counts",
        "created_at_utc",
    ],
    "rule": "fixed-workflow discovery must not influence online-validation, pool-validation, final-eval, or Deep SWE target-eval composition",
}

HARNESS_PARITY_CANARIES = {
    "status": "required_before_fixed_workflow_discovery",
    "harnesses": [
        "opencode",
        "codex_yunwu_gpt_5_5",
        "claude_code_local_bridge_yunwu_opus_4_8",
        "direct_qa",
        "tool_dialogue",
        "terminal_sandbox",
        "long_context",
    ],
    "requirements": [
        "edits files when expected",
        "runs commands when expected",
        "returns patch refs",
        "streams or records complete transcripts",
        "reports usage and externally tracked cost when available",
        "terminates cleanly",
        "grader sees final workspace",
        "tool calls are routed to the correct agent",
        "worker identity logs scaffold, bridge/provider endpoint, model, and settings",
    ],
}

FAILURE_TAXONOMY = FAILURE_TAXONOMY_ENTRIES

SOURCE_VALIDATION_REPORT_SPEC = {
    "required_outputs": [
        "source_validation_report.json",
        "source_validation_report.md",
        "difficulty_calibration.parquet",
        "task_quality_flags.parquet",
    ],
    "minimum_task_fields": [
        "task_id",
        "lane",
        "source",
        "harness",
        "split",
        "setup_ok",
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
    ],
    "gates": [
        "reward emitted >= 99.5%",
        "grader deterministic >= 99%",
        "hidden leakage = 0",
        "setup failures < 1-2%",
        "harness infrastructure exclusions < 2%",
        "duplicate or near-duplicate tasks < 1%",
        "enough medium-difficulty tasks for first pilot",
        "long-tail expensive tasks are flagged, not silently mixed",
        "TaskTrove failed solutions emit explicit failure rewards instead of disappearing",
    ],
}

FIXED_WORKFLOW_DISCOVERY_GATE = {
    "status": "required_before_grpo",
    "sample": "200 candidate tasks plus all accepted trace-branch tasks",
    "lane_mix": {
        "repo_repair_open_repo_terminal": 50,
        "unit_and_scientific_code": 45,
        "math_science_knowledge": 45,
        "tool_dialogue": 35,
        "long_context_memory_planning": 25,
        "trace_state_branch_tasks": "all accepted",
    },
    "templates": [
        "best single worker/scaffold",
        "direct worker selected by lane prior",
        "planner -> solver",
        "solver -> critic -> revise",
        "two independent attempts -> synthesizer",
        "builder -> debugger -> repair",
        "specialist analysis -> finalizer",
        "trace-state current repair / debug / clean-slate reanalysis",
    ],
    "measure": [
        "success rate",
        "best single worker/scaffold",
        "best fixed workflow",
        "paired delta vs best single worker",
        "paired delta vs best fixed baseline",
        "workflow oracle",
        "leave-one-out worker contribution",
        "cost per success",
        "latency",
        "invalid workflow/harness failure rate",
        "within-task reward variance",
        "unique solves",
        "hard-subset behavior",
    ],
    "proceed_if": [
        "best fixed workflow beats best single worker on at least one product-critical lane or workflow-oracle headroom is positive",
        "at least 35-50% of task groups show reward variation among workflows",
    ],
}

WORKER_MASKS = {
    "repo_coding": {
        "candidates": [
            "codex:yunwu-gpt-5.5",
            "claude-code:local-bridge-yunwu-opus-4.8",
            "opencode:kimi-code",
            "opencode:mimo",
            "opencode:glm",
            "opencode:minimax",
            "opencode:deepseek-pro",
        ],
    },
    "math_science_knowledge": {
        "candidates": [
            "direct:gpt-5.5",
            "direct:gemini-3.1-pro",
            "direct:opus-4.8",
            "direct:glm",
            "direct:flash",
        ],
    },
    "tool_dialogue": {
        "candidates": [
            "direct:opus-4.8",
            "direct:gpt-5.5",
            "opencode:mimo",
            "opencode:kimi-code",
            "opencode:glm",
            "opencode:minimax",
        ],
    },
    "long_context": {
        "candidates": [
            "direct:gemini-3.1-pro",
            "direct:gpt-5.5",
            "direct:opus-4.8",
        ],
    },
}

LIKELY_FIRST_PILOT_WORKERS = [
    "direct:gpt-5.5",
    "direct:gemini-3.1-pro",
    "direct:opus-4.8",
    "opencode:kimi-code",
    "opencode:mimo",
    "opencode:glm",
    "codex:yunwu-gpt-5.5",
    "claude-code:local-bridge-yunwu-opus-4.8",
]

WORKER_POOL_SELECTION_RULE = {
    "metric": "MC_m = Acc(S) - Acc(S_without_m)",
    "computed_on": "paired held-out workflow outcomes, not direct worker outcomes",
    "include_worker_if_any": [
        "positive paired workflow contribution",
        "positive hard-subset contribution",
        "moves the cost-quality frontier",
        "enables a role no other worker covers",
    ],
    "initial_quality_first_candidates": LIKELY_FIRST_PILOT_WORKERS,
}

GRPO_PILOT_CONSTRUCTION = {
    "source": "validated candidate tasks with observed workflow disagreement/headroom",
    "size": "300-500 tasks from the 1,000-row candidate pool",
    "max_initial_steps": 3,
    "lane_group_sizes": {
        "math_science_knowledge": "8-16",
        "unit_and_scientific_code": "8",
        "tasktrove_contracts_bugs": "8",
        "repo_coding": "4",
        "tool_dialogue": "4-8",
        "long_context": "4-8",
    },
    "task_filters": [
        "fixed workflows disagree",
        "at least one workflow beats best single worker or creates oracle headroom",
        "different workers win in different roles",
        "deterministic reward",
        "within budget and timeout envelope",
    ],
    "notes": [
        "Do not expose five-step workflows until parse validity, access validity, and fixed-workflow baselines are stable.",
        "The full 1,000-row candidate mix is a pool for sampling, not an automatic first GRPO batch.",
    ],
}

EVALUATION_BASELINES = [
    "best individual direct model",
    "best individual scaffolded worker",
    "best individual model+scaffold selected on dev",
    "best commercial individual worker",
    "best open individual worker",
    "best fixed workflow",
    "best-of-N single-worker sampling",
    "single-worker self-reflection",
    "prompt-only Conductor",
    "syntax/topology-SFT Conductor",
    "GRPO Conductor",
]

CODING_BASELINE_WORKERS = [
    "codex:yunwu-gpt-5.5 alone",
    "claude-code:local-bridge-yunwu-opus-4.8 alone",
    "opencode:kimi-code alone",
    "opencode:mimo alone",
    "direct:gpt-5.5 alone",
    "direct:opus-4.8 alone",
    "direct:gemini-3.1-pro alone",
]

GO_NO_GO_GATES = [
    "workflow parse validity >= 99%",
    "valid access lists >= 99%",
    "reward emitted >= 99.5%",
    "harness/infrastructure exclusions < 2%",
    "within-task reward variance present in >= 35-50% of GRPO groups",
    "fixed workflow beats best single worker on at least one product-critical lane or oracle headroom is positive",
    "GRPO beats prompt-only Conductor on online validation",
    "GRPO trends positive versus best fixed workflow before scale-up",
    "repo branch-state reconstruction is 100% for branch-state RL tasks",
    "hidden-answer leakage is 0",
]


def _count_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "count": 0}
    count = 0
    sources: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    harnesses: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            source = row.get("source")
            if isinstance(source, dict):
                sources[str(source.get("name"))] += 1
                policies[str(source.get("policy"))] += 1
            elif source:
                sources[str(source)] += 1
            elif row.get("source_name"):
                sources[str(row["source_name"])] += 1
            capability = row.get("capability")
            if capability:
                capabilities[str(capability)] += 1
            splitting = row.get("splitting")
            if isinstance(splitting, dict):
                splits[str(splitting.get("split"))] += 1
            environment = row.get("environment")
            if isinstance(environment, dict):
                harnesses[str(environment.get("harness"))] += 1
            metadata = row.get("metadata")
            domain = row.get("domain") or (metadata.get("domain") if isinstance(metadata, dict) else None)
            if domain:
                domains[str(domain)] += 1
    return {
        "path": str(path),
        "exists": True,
        "count": count,
        "sources": dict(sources),
        "capabilities": dict(capabilities),
        "splits": dict(splits),
        "harnesses": dict(harnesses),
        "policies": dict(policies),
        "domains": dict(domains),
    }


def _json_file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "count": 0}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"path": str(path), "exists": True, "count": 1, "valid_json": False}
    return {
        "path": str(path),
        "exists": True,
        "count": 1,
        "valid_json": True,
        "version": payload.get("version"),
        "status": payload.get("overall_status") or payload.get("status"),
        "complete": (
            payload.get("parity_complete")
            or payload.get("freeze_complete")
            or payload.get("frozen")
            or (payload.get("status") == "pass")
            or (payload.get("overall_status") == "pass")
        ),
    }


def _artifact_counts(manifest_dir: Path) -> dict[str, Any]:
    return {
        "existing_router_bank": _count_jsonl(manifest_dir / "manifest.jsonl"),
        "open_direct_probe_evidence": _count_jsonl(manifest_dir / "probe.jsonl"),
        "open_tau_agentic_evidence": _count_jsonl(manifest_dir / "agentic_bank.jsonl"),
        "frontier_direct_matrix_evidence": _count_jsonl(manifest_dir / "pool_matrix_frontier.jsonl"),
        "frontier_tau_live_evidence": _count_jsonl(manifest_dir / "agentic_frontier_tau4.jsonl"),
        "frontier_coding_live_evidence": _count_jsonl(manifest_dir / "agentic_coding_frontier_direct3.jsonl"),
        "deep_swe_eval_taskspecs": _count_jsonl(manifest_dir / "scaffold_repo_taskspecs.jsonl"),
        "tasktrove_harbor_train_taskspecs": _count_jsonl(
            manifest_dir / "tasktrove_harbor" / "inferredbugs_train_taskspecs.jsonl"
        ),
        "tasktrove_pymethods2test_train_taskspecs": _count_jsonl(
            manifest_dir / "tasktrove_harbor" / "pymethods2test_train_taskspecs.jsonl"
        ),
        "training_repo_canary_taskspecs": _count_jsonl(manifest_dir / "training_repo_canaries" / "taskspecs.jsonl"),
        "generated_repo_train_taskspecs": _count_jsonl(manifest_dir / "generated_repo_tasks" / "taskspecs.jsonl"),
        "tool_dialog_train_taskspecs": _count_jsonl(manifest_dir / "tool_dialog_tasks" / "taskspecs.jsonl"),
        "long_context_train_taskspecs": _count_jsonl(manifest_dir / "long_context_tasks" / "taskspecs.jsonl"),
        "mvp_grpo_train_taskspecs": _count_jsonl(
            manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl"
        ),
        "trace_state_branch_candidates": _count_jsonl(
            manifest_dir / "trace_state_branches" / "candidates.jsonl"
        ),
        "trace_capture_branch_candidates": _count_jsonl(
            manifest_dir / "trace_capture" / "branch_candidates.jsonl"
        ),
        "trace_capture_branch_taskspecs": _count_jsonl(
            manifest_dir / "trace_capture" / "branch_taskspecs.jsonl"
        ),
        "trace_capture_jobs": _count_jsonl(manifest_dir / "trace_capture" / "jobs.jsonl"),
        "frozen_online_validation": _count_jsonl(manifest_dir / "frozen_manifests" / "online_validation.jsonl"),
        "frozen_pool_validation": _count_jsonl(manifest_dir / "frozen_manifests" / "pool_validation.jsonl"),
        "frozen_final_eval": _count_jsonl(manifest_dir / "frozen_manifests" / "final_eval.jsonl"),
        "frozen_deep_swe_target_eval": _count_jsonl(
            manifest_dir / "frozen_manifests" / "deep_swe_target_eval.jsonl"
        ),
        "harness_parity_report": _json_file_info(manifest_dir / "harness_parity_report.json"),
        "failure_taxonomy_report": _json_file_info(manifest_dir / "failure_taxonomy_report.json"),
        "source_validation_report": _json_file_info(manifest_dir / "source_validation_report.json"),
        "difficulty_calibration": {
            "path": str(manifest_dir / "difficulty_calibration.parquet"),
            "exists": (manifest_dir / "difficulty_calibration.parquet").exists(),
            "count": 1 if (manifest_dir / "difficulty_calibration.parquet").exists() else 0,
        },
        "task_quality_flags": {
            "path": str(manifest_dir / "task_quality_flags.parquet"),
            "exists": (manifest_dir / "task_quality_flags.parquet").exists(),
            "count": 1 if (manifest_dir / "task_quality_flags.parquet").exists() else 0,
        },
    }


def build_training_distribution_plan(manifest_dir: Path) -> dict[str, Any]:
    """Return the locked training distribution and source plan.

    Percentages are rollout-share targets, not fixed row-count quotas. A source
    may be promoted from pool/diagnostic to GRPO only after its harness canary and
    leakage policy pass.
    """

    manifest_dir = manifest_dir.resolve()
    parity_report_path = manifest_dir / "harness_parity_report.json"
    harness_parity_canaries = dict(HARNESS_PARITY_CANARIES)
    if parity_report_path.exists():
        harness_parity_canaries.update(
            {
                "status": "complete_saved_artifacts",
                "report": str(parity_report_path),
                "scope": (
                    "repo and terminal parity use saved live artifacts; direct/tool/long parity use offline harness tests"
                ),
            }
        )
    failure_taxonomy = FAILURE_TAXONOMY
    failure_taxonomy_report_path = manifest_dir / "failure_taxonomy_report.json"
    failure_taxonomy_status = (
        "complete_frozen_artifact" if failure_taxonomy_report_path.exists() else "required_before_fixed_workflow_discovery"
    )
    source_validation_report_path = manifest_dir / "source_validation_report.json"
    source_validation_status = (
        "complete_passed" if source_validation_report_path.exists() else "required_before_fixed_workflow_discovery"
    )
    return {
        "version": "fugu_ultra_training_distribution_v1",
        "objective": "Train Fugu-Ultra to beat every individual model or model+scaffold worker.",
        "manifest_dir": str(manifest_dir),
        "locked_principles": [
            "Deep SWE is final-evaluation-only; do not use it for routine canaries or GRPO training.",
            "Train orchestration mostly on fast and medium verifiable tasks.",
            "Use hard trace branches for switch/continue/repair decisions instead of replaying full hard tasks.",
            "Promote a source to GRPO only when it has deterministic grading, leakage policy, and at least one passing harness canary.",
            "Keep direct-only tasks as one curriculum lane, not proof of Ultra performance.",
            "The first MVP is text/tool/repo only.",
        ],
        "mvp_task_mix_name": MVP_TASK_MIX_NAME,
        "mvp_task_mix_status": "candidate_pending_fixed_workflow_discovery",
        "mvp_task_mix": MVP_TASK_MIX,
        "mvp_task_mix_notes": [
            "This is a curriculum/proof-of-learning candidate mix, not the final hard-coding Ultra mix.",
            "Task mix totals 1,000 examples across repo, code, reasoning, dialogue, and long-context lanes.",
            "Use a small validated task-source mix before broad expansion; TaskTrove contributes only verifier-backed TaskSpecs.",
            "Prioritize pymethods2test as a fixed RL anchor because the OT-Agent source ablation found it strongest.",
            "Use pymethods2test-style Python contracts as the high-signal fast-verifiable code backbone before broad harder-rollout scaling.",
            "Retain heterogeneous tool/terminal/dialogue sources for OOD generalization even when their ID scores are weaker.",
            "AgentTrove traces are SFT/role-mining material, not reward-bearing TaskSpecs without verifier reconstruction.",
            "Run fixed-workflow discovery before GRPO and sample the pilot from tasks with observed reward variation/headroom.",
            "Raise true repo/harness coverage toward 50-100 validated repo or branch TaskSpecs before making coding-agentic progress claims.",
        ],
        "manifest_freeze_requirements": MANIFEST_FREEZE_REQUIREMENTS,
        "harness_parity_canaries": harness_parity_canaries,
        "failure_taxonomy_status": failure_taxonomy_status,
        "failure_taxonomy_report": str(failure_taxonomy_report_path) if failure_taxonomy_report_path.exists() else None,
        "failure_taxonomy": failure_taxonomy,
        "source_validation_report_spec": SOURCE_VALIDATION_REPORT_SPEC,
        "source_validation_status": source_validation_status,
        "source_validation_report": str(source_validation_report_path) if source_validation_report_path.exists() else None,
        "rollout_mix": [
            {
                "tier": "tier0_workflow_sft",
                "share": 0.0,
                "role": "format/topology warm start; no worker execution",
                "sources": [
                    "synthetic workflow JSON",
                    "fixed role templates",
                    "successful Claude Code/Codex/OpenCode trace summaries",
                ],
                "status": "design_locked_not_materialized",
            },
            {
                "tier": "tier1_fast_verifiable",
                "share": 0.60,
                "role": "cheap GRPO backbone for decomposition, worker choice, aggregation, and verifier behavior",
                "sources": [
                    "existing_bank direct_qa/code_exec train split",
                    "LiveCodeBench old/trainable windows",
                    "BigCodeBench trainable split",
                    "SciCode dev split",
                    "TaskTrove pymethods2test Python-contract tasks",
                    "small generated repo tasks",
                    "short verified terminal/TaskTrove tasks after model canary",
                ],
                "group_size": "8-16",
                "budget": "short",
                "status": "partially_materialized",
            },
            {
                "tier": "tier2_medium_repo_and_tool",
                "share": 0.25,
                "role": "repo editing, execution feedback, tool-dialogue, and repair behavior",
                "sources": [
                    "SWE-smith easy/medium when payloads are recovered or replaced",
                    "custom generated repo tasks",
                    "TaskTrove Harbor verifier-backed tasks",
                    "tau retail/airline/custom tool-dialogue tasks",
                    "small GitHub issue tasks with hidden tests",
                ],
                "group_size": "4-8",
                "budget": "short-medium",
                "status": "harnesses_partially_ready",
            },
            {
                "tier": "tier3_hard_state_branches",
                "share": 0.10,
                "role": "learn when to continue, switch workers, debug, or repair from hard trace states",
                "sources": [
                    "state branches mined from Claude Code traces",
                    "state branches mined from Codex traces",
                    "state branches mined from OpenCode traces",
                    "non-Deep-SWE hard train-allowed repo traces",
                    "trace_state_branches materialized from train-allowed generated repo traces",
                ],
                "group_size": "2-4",
                "budget": "short-medium branch caps",
                "status": "partial_materialized",
            },
            {
                "tier": "tier4_sparse_full_hard_train_allowed",
                "share": 0.05,
                "role": "late-curriculum checkpoint calibration on long tasks without contaminating final target eval",
                "sources": [
                    "hard SWE-smith/custom repo tasks not used in final evaluation",
                    "hard terminal tasks from train-allowed sources",
                    "long tool-dialogue/generated planning tasks",
                ],
                "group_size": "2",
                "budget": "long",
                "status": "late_curriculum_only",
            },
        ],
        "held_out_evaluation": [
            {
                "source": "deep_swe_local",
                "policy": "final_eval_only",
                "use": "target hard repo-coding evaluation; not canary, not GRPO training",
            },
            {"source": "swe_bench_verified/pro_style", "policy": "final_eval_only", "use": "repo-coding eval"},
            {"source": "terminal_bench_official", "policy": "final_eval_only", "use": "terminal-agent eval"},
            {"source": "GPQA/HLE/MATH500/MBPP/HumanEval latest held-out", "policy": "final_eval_only", "use": "hard direct/code eval"},
        ],
        "canary_distribution": {
            "allowed_sources": [
                "training_repo_canary",
                "existing_bank diagnostic sample",
                "TaskTrove Harbor verifier-backed train sample",
                "small generated tool-dialogue sample",
            ],
            "forbidden_sources": ["deep_swe_local", "swe_bench_verified", "terminal_bench_official"],
            "current_passing_canaries": [
                "OpenCode/Kimi on training_repo_canary",
                "Codex/Yunwu GPT-5.5 on training_repo_canary",
                "Claude Code/Yunwu Opus bridge on training_repo_canary",
                "Harbor/Terminus-2 Yunwu GPT-5.5 on TaskTrove inferredbugs",
            ],
        },
        "promotion_gates": [
            *SOURCE_VALIDATION_GATES,
        ],
        "source_validation_gates": SOURCE_VALIDATION_GATES,
        "tasktrove_validation_gates": TASKTROVE_VALIDATION_GATES,
        "fixed_workflow_discovery_gate": FIXED_WORKFLOW_DISCOVERY_GATE,
        "worker_masks": WORKER_MASKS,
        "likely_first_pilot_workers": LIKELY_FIRST_PILOT_WORKERS,
        "worker_pool_selection_rule": WORKER_POOL_SELECTION_RULE,
        "grpo_pilot_construction": GRPO_PILOT_CONSTRUCTION,
        "evaluation_baselines": EVALUATION_BASELINES,
        "coding_baseline_workers": CODING_BASELINE_WORKERS,
        "go_no_go_gates": GO_NO_GO_GATES,
        "immediate_next_actions": [
            "Verify frozen manifest hashes before live discovery.",
            "Use the saved harness parity report as the pre-discovery parity gate.",
            "Use the frozen failure taxonomy and reward mapping report.",
            "Use the source-validation and difficulty-calibration artifacts.",
            "Run live fixed-workflow discovery on the MVP candidate mix plus trace-state branch shard.",
            "Establish prompt-only and syntax/topology-SFT Conductor baselines before RL.",
            "Measure within-task reward variance under sampled workflows before GRPO.",
            "Select the final worker/scaffold pool by paired held-out workflow contribution with lane-specific masks.",
            "Build the first GRPO pilot from validated tasks with observed workflow disagreement/headroom.",
            "Collect richer train-allowed OpenCode/Codex/Claude traces with repo state, patch refs, and execution feedback.",
            "Import additional verified train-allowed repo/code/science/terminal sources only after the MVP path is validated.",
        ],
        "artifacts": _artifact_counts(manifest_dir),
    }


def render_training_distribution_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Fugu-Ultra Training Task Distribution",
        "",
        f"Objective: {plan['objective']}",
        "",
        "## Locked Principles",
        *[f"- {item}" for item in plan["locked_principles"]],
        "",
        "## MVP Task Mix",
        f"- Name: {plan['mvp_task_mix_name']}",
        f"- Status: {plan['mvp_task_mix_status']}",
        *[f"- {name}: {count}" for name, count in plan["mvp_task_mix"].items()],
        *[f"- Note: {item}" for item in plan["mvp_task_mix_notes"]],
        "",
        "## Manifest Freeze Requirements",
        f"- Status: {plan['manifest_freeze_requirements']['status']}",
        f"- Rule: {plan['manifest_freeze_requirements']['rule']}",
        "- Manifests: " + "; ".join(plan["manifest_freeze_requirements"]["manifests"]),
        "- Fields: " + "; ".join(plan["manifest_freeze_requirements"]["required_fields"]),
        "",
        "## Harness Parity Canaries",
        f"- Status: {plan['harness_parity_canaries']['status']}",
        "- Harnesses: " + "; ".join(plan["harness_parity_canaries"]["harnesses"]),
        "- Requirements: " + "; ".join(plan["harness_parity_canaries"]["requirements"]),
        *(
            [f"- Report: {plan['harness_parity_canaries']['report']}"]
            if plan["harness_parity_canaries"].get("report")
            else []
        ),
        *(
            [f"- Scope: {plan['harness_parity_canaries']['scope']}"]
            if plan["harness_parity_canaries"].get("scope")
            else []
        ),
        "",
        "## Failure Taxonomy",
        f"- Status: {plan['failure_taxonomy_status']}",
        *([f"- Report: {plan['failure_taxonomy_report']}"] if plan.get("failure_taxonomy_report") else []),
        *[
            f"- {name}: reward={cfg['reward']}, use={cfg['use']}"
            for name, cfg in plan["failure_taxonomy"].items()
        ],
        "",
        "## Source Validation Report",
        f"- Status: {plan['source_validation_status']}",
        *([f"- Report: {plan['source_validation_report']}"] if plan.get("source_validation_report") else []),
        "- Outputs: " + "; ".join(plan["source_validation_report_spec"]["required_outputs"]),
        "- Fields: " + "; ".join(plan["source_validation_report_spec"]["minimum_task_fields"]),
        "- Gates: " + "; ".join(plan["source_validation_report_spec"]["gates"]),
        "",
        "## Rollout Mix",
    ]
    for tier in plan["rollout_mix"]:
        lines.extend(
            [
                f"### {tier['tier']} ({tier['share']:.0%})",
                f"- Role: {tier['role']}",
                f"- Budget: {tier.get('budget', 'none')}",
                f"- Group size: {tier.get('group_size', 'n/a')}",
                f"- Status: {tier['status']}",
                "- Sources: " + "; ".join(tier["sources"]),
                "",
            ]
        )
    lines.extend(["## Source Validation Gates"])
    lines.extend([f"- {item}" for item in plan["source_validation_gates"]])
    lines.extend(["", "## TaskTrove Validation Gates"])
    lines.extend([f"- {item}" for item in plan["tasktrove_validation_gates"]])
    gate = plan["fixed_workflow_discovery_gate"]
    lines.extend(
        [
            "",
            "## Fixed-Workflow Discovery Gate",
            f"- Status: {gate['status']}",
            f"- Sample: {gate['sample']}",
            "- Lane mix: " + "; ".join(f"{lane}={count}" for lane, count in gate["lane_mix"].items()),
            "- Templates: " + "; ".join(gate["templates"]),
            "- Measure: " + "; ".join(gate["measure"]),
            "- Proceed if: " + "; ".join(gate["proceed_if"]),
            "",
            "## Worker Masks",
        ]
    )
    for lane, mask in plan["worker_masks"].items():
        parts = []
        for key, values in mask.items():
            parts.append(f"{key}=" + ", ".join(values))
        lines.append(f"- {lane}: " + "; ".join(parts))
    lines.extend(
        [
            "",
            "## Worker Pool Selection",
            f"- Metric: {plan['worker_pool_selection_rule']['metric']}",
            f"- Computed on: {plan['worker_pool_selection_rule']['computed_on']}",
            "- Include if: " + "; ".join(plan["worker_pool_selection_rule"]["include_worker_if_any"]),
            "- Initial candidates: " + "; ".join(plan["worker_pool_selection_rule"]["initial_quality_first_candidates"]),
        ]
    )
    pilot = plan["grpo_pilot_construction"]
    lines.extend(
        [
            "",
            "## GRPO Pilot Construction",
            f"- Source: {pilot['source']}",
            f"- Size: {pilot['size']}",
            f"- Max initial steps: {pilot['max_initial_steps']}",
        ]
    )
    lines.extend([f"- Group size {lane}: {size}" for lane, size in pilot["lane_group_sizes"].items()])
    lines.append("- Task filters: " + "; ".join(pilot["task_filters"]))
    lines.extend([f"- Note: {item}" for item in pilot["notes"]])
    lines.extend(["## Held-Out Evaluation"])
    for item in plan["held_out_evaluation"]:
        lines.append(f"- {item['source']} [{item['policy']}]: {item['use']}")
    lines.extend(["", "## Evaluation Baselines"])
    lines.extend([f"- {item}" for item in plan["evaluation_baselines"]])
    lines.extend(["", "## Coding Baseline Workers"])
    lines.extend([f"- {item}" for item in plan["coding_baseline_workers"]])
    lines.extend(["", "## Go/No-Go Gates"])
    lines.extend([f"- {item}" for item in plan["go_no_go_gates"]])
    lines.extend(["", "## Canary Rules"])
    lines.append("- Allowed: " + "; ".join(plan["canary_distribution"]["allowed_sources"]))
    lines.append("- Forbidden: " + "; ".join(plan["canary_distribution"]["forbidden_sources"]))
    lines.extend(["", "## Immediate Next Actions"])
    lines.extend([f"- {item}" for item in plan["immediate_next_actions"]])
    lines.extend(["", "## Current Local Artifacts"])
    for name, artifact in plan["artifacts"].items():
        exists = "yes" if artifact["exists"] else "no"
        lines.append(f"- {name}: exists={exists}, count={artifact['count']}")
    lines.append("")
    return "\n".join(lines)


def write_training_distribution_plan(manifest_dir: Path, out_json: Path, out_md: Path | None = None) -> dict[str, Any]:
    plan = build_training_distribution_plan(manifest_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_training_distribution_markdown(plan))
    return plan
