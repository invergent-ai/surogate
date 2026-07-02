"""Ultra CLI entrypoint (``ultra <subcommand>``)."""

from __future__ import annotations

import argparse
import asyncio
import json


def _add_stepzero(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stepzero", help="run the step-zero workflow-headroom test (live)")
    p.add_argument(
        "--workers",
        required=True,
        help="comma-separated id=model_slug; index 0 is the anchor. "
        "e.g. flash=deepseek/deepseek-v4-flash,glm=z-ai/glm-5.2,mimo=...",
    )
    p.add_argument("--n-tasks", type=int, default=100, dest="n_tasks")
    p.add_argument("--split", default="grpo_train", help="bank split to sample (grpo_train|online_validation)")
    p.add_argument("--harness", default="direct_qa", help="task harness filter (direct_qa|code_exec)")
    p.add_argument(
        "--all-difficulties",
        action="store_true",
        dest="all_difficulties",
        help="don't restrict to medium-difficulty (router 'discriminative') tasks",
    )
    p.add_argument(
        "--worker-assignment",
        default=None,
        dest="worker_assignment",
        help="override scaffold worker indices; 'name=i,j;...' e.g. "
        "D_debate_synthesize=0,1,2;F_execute_critic_revise=0,2",
    )
    p.add_argument("--reps", type=int, default=3, help="draws per (arm, task) — denoises cells")
    p.add_argument("--folds", type=int, default=5, help="cross-fit folds for Δ_fixed")
    p.add_argument("--max-tokens", type=int, default=4096, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--reasoning", default="high", help="worker reasoning effort")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--budget", type=float, default=None, help="USD spend cap (default: unlimited)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bank-path", default=None, dest="bank_path")
    p.add_argument("--out", default=None, help="write the report JSON to this path")


def _add_pool_select(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "pool-select",
        help="render the offline Ultra pool-selection report from existing evidence",
    )
    p.add_argument("--manifest-dir", default=None, help="fugu_clean_v1 manifest directory")
    p.add_argument("--budget", type=float, default=200.0, help="paid follow-up budget cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write Markdown report to this path")


def _add_pool_tournament(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "pool-tournament",
        help="run or dry-run the bounded paid Ultra pool-selection tournament",
    )
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--split", default="online_validation")
    p.add_argument("--tasks-per-domain", type=int, default=10)
    p.add_argument("--all-difficulties", action="store_true")
    p.add_argument("--open-success-min", type=int, default=None)
    p.add_argument("--open-success-max", type=int, default=None)
    p.add_argument("--stages", default="single,same_worker,mixed,challenger")
    p.add_argument("--arms", default=None)
    p.add_argument("--budget", type=float, default=200.0)
    p.add_argument("--stop-ratio", type=float, default=0.8)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--reasoning", default="high")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="./.ultra_cache/completions")
    p.add_argument("--out-dir", default="./pool_tournament")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def _add_pool_tournament_analyze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "pool-tournament-analyze",
        help="summarize a pool tournament rollouts JSONL file",
    )
    p.add_argument("rollouts_jsonl")
    p.add_argument("--out", default=None, help="write summary JSON to this path")


def _add_trace_convert_opencode(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trace-convert-opencode",
        help="convert saved OpenCode rollout JSONL into canonical AgentTrace records",
    )
    p.add_argument("input_jsonl")
    p.add_argument("--out-dir", required=True)


def _add_scaffold_tournament_plan(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-tournament-plan",
        help="render the preregistered scaffold-aware Ultra role-tournament plan",
    )
    p.add_argument("--repo-tasks", type=int, default=50)
    p.add_argument("--unit-code-tasks", type=int, default=45)
    p.add_argument("--direct-tasks", type=int, default=45)
    p.add_argument("--tool-dialog-tasks", type=int, default=35)
    p.add_argument("--long-context-tasks", type=int, default=25)
    p.add_argument("--coding-tasks", type=int, default=None, help="legacy alias for --repo-tasks")
    p.add_argument("--out", default=None)


def _add_scaffold_tournament_manifest(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-tournament-manifest",
        help="select concrete tasks and jobs for the scaffold-aware Ultra tournament",
    )
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--repo-tasks", type=int, default=50)
    p.add_argument("--unit-code-tasks", type=int, default=45)
    p.add_argument("--direct-tasks", type=int, default=45)
    p.add_argument("--tool-dialog-tasks", type=int, default=35)
    p.add_argument("--long-context-tasks", type=int, default=25)
    p.add_argument("--coding-tasks", type=int, default=None, help="legacy alias for --repo-tasks")
    p.add_argument("--tasks-jsonl", default=None)
    p.add_argument("--branch-tasks-jsonl", default=None)
    p.add_argument(
        "--include-pool-validation",
        action="store_true",
        help="allow explicit pool_only/pool_validation TaskSpecs from --tasks-jsonl; never used for GRPO manifests",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs-out", default=None)


def _add_scaffold_tournament_readiness(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-tournament-readiness",
        help="analyze which scaffold-tournament jobs are runnable with current harness adapters",
    )
    p.add_argument("manifest_json")
    p.add_argument("--out", default=None)


def _add_scaffold_materialize_repo(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-materialize-repo",
        help="materialize scaffold-tournament repo-coding tasks into canonical TaskSpecs",
    )
    p.add_argument("manifest_json")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)


def _add_scaffold_canary(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-canary",
        help="run one scaffold-aware arm on one TaskSpec through the normal executor",
    )
    p.add_argument("--tasks-jsonl", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--task-id", default=None)
    p.add_argument("--rollout-id", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--reasoning", default="high")
    p.add_argument("--budget", choices=["short", "medium", "long", "max"], default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--artifact-dir", default=None)
    p.add_argument("--agent-trace-out", default=None)


def _add_scaffold_discovery_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-discovery-run",
        help="run or dry-run selected fixed-workflow discovery jobs",
    )
    p.add_argument("--jobs-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--job-id", action="append", default=[])
    p.add_argument("--lanes", default=None, help="comma-separated lane filter")
    p.add_argument("--arms", default=None, help="comma-separated arm filter")
    p.add_argument("--stages", default=None, help="comma-separated stage filter")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--reasoning", default="high")
    p.add_argument("--budget", choices=["short", "medium", "long", "max"], default=None)
    p.add_argument("--provider", default=None, help="force one provider; default routes commercial via Yunwu and open/specialists via OpenRouter")
    p.add_argument("--dotenv", default=None, help="dotenv file to load before live calls; default is repo .env")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--requests-per-minute", type=float, default=None)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--job-timeout-s", type=float, default=None)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument(
        "--docker-network-janitor",
        action="store_true",
        help="preflight stale Harbor/TaskTrove Docker Compose networks before terminal_sandbox jobs",
    )
    p.add_argument(
        "--docker-network-janitor-dry-run",
        action="store_true",
        help="report stale Docker networks but do not remove them during the preflight",
    )
    p.add_argument("--live", action="store_true", help="make live provider/CLI calls")
    p.add_argument("--fake", action="store_true", help="execute with FakeProvider for local tests")
    p.add_argument("--dry-run", action="store_true", help="only select jobs and write a report")


def _add_docker_network_janitor(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "docker-network-janitor",
        help="dry-run or remove stale Harbor/TaskTrove Docker Compose networks",
    )
    p.add_argument("--delete", action="store_true", help="actually remove stale detached networks")
    p.add_argument("--max-remove", type=int, default=200)
    p.add_argument("--name-prefix", action="append", default=[], help="repeatable Compose project/name prefix allowlist")
    p.add_argument("--all-compose", action="store_true", help="allow every detached Docker Compose network")
    p.add_argument("--docker-bin", default="docker")
    p.add_argument("--report-out", default=None)


def _add_scaffold_discovery_analyze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-discovery-analyze",
        help="summarize fixed-workflow discovery rollouts",
    )
    p.add_argument("--jobs-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report-out", required=True)


def _add_scaffold_discovery_followup(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-discovery-followup",
        help="plan targeted follow-up jobs from completed discovery evidence",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--jobs-jsonl", default=None)
    p.add_argument("--out-json", required=True)
    p.add_argument("--jobs-out", default=None)
    p.add_argument(
        "--mode",
        choices=["targeted", "complete-variance", "role-followup", "single-prefilter", "all-missing"],
        default="targeted",
    )
    p.add_argument("--sources", default=None, help="comma-separated source filter")
    p.add_argument("--lanes", default=None, help="comma-separated lane filter")
    p.add_argument("--arm-domains", default=None, help="comma-separated arm-domain filter")
    p.add_argument("--stages", default=None, help="comma-separated stage filter")
    p.add_argument("--max-jobs", type=int, default=32)
    p.add_argument("--max-task-groups", type=int, default=12)


def _add_training_repo_canaries(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "training-repo-canaries",
        help="materialize small train-allowed repo tasks for harness canaries",
    )
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--image-tag", default="fugu-ultra/training-repo-canary:slugkit-v1")
    p.add_argument("--no-build", action="store_true", help="write TaskSpecs without building the Docker image")


def _add_generated_repo_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "generated-repo-tasks",
        help="materialize train-allowed generated repo-repair tasks",
    )
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--image-prefix", default="fugu-ultra/generated-repo")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-build", action="store_true", help="write TaskSpecs without building Docker images")


def _add_tool_dialog_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tool-dialog-tasks",
        help="materialize train-allowed custom tau-style tool-dialogue tasks",
    )
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--limit", type=int, default=None)


def _add_tau_bench_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tau-bench-tasks",
        help="materialize train-allowed tau-bench retail tasks",
    )
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--selection", choices=["default", "high_action"], default="default")
    p.add_argument("--tasks-train-path", default=None)


def _add_long_context_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "long-context-tasks",
        help="materialize train-allowed generated long-context document-pack tasks",
    )
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--limit", type=int, default=None)


def _add_long_context_adversarial_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "long-context-adversarial-tasks",
        help="materialize train-allowed adversarial long-context tasks",
    )
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--limit", type=int, default=None)


def _add_long_context_stress_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "long-context-stress-tasks",
        help="materialize train-allowed stress long-context tasks",
    )
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--limit", type=int, default=None)


def _add_harbor_materialize(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "harbor-materialize",
        help="materialize Harbor task bundles, including TaskTrove subsets, into canonical TaskSpecs",
    )
    p.add_argument("root")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--source-name", default="tasktrove_harbor")
    p.add_argument("--source-version", default="v3")
    p.add_argument("--policy", default="pool_only")
    p.add_argument("--split", default="pool_validation")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--include-no-verifier",
        action="store_true",
        help="include Harbor task bundles without verifier/tests payloads",
    )


def _add_tasktrove_parquet_materialize(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tasktrove-parquet-materialize",
        help="extract TaskTrove Harbor bundles from a parquet shard and emit canonical TaskSpecs",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet", default=None, help="local TaskTrove tasks.parquet path")
    source.add_argument("--hf-file", default=None, help="dataset-relative Hugging Face parquet path")
    p.add_argument("--hf-repo", default="open-thoughts/TaskTrove")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--extract-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--source-name", default="tasktrove_harbor")
    p.add_argument("--source-version", default="v3")
    p.add_argument("--policy", default="train_allowed")
    p.add_argument("--split", default="grpo_train")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--include-path", action="append", default=[], help="repeatable TaskTrove path to materialize")
    p.add_argument(
        "--include-paths-jsonl",
        default=None,
        help="JSONL rows with tasktrove_path, path, or task_id fields to materialize from the parquet",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--include-no-verifier",
        action="store_true",
        help="include Harbor task bundles without verifier/tests payloads",
    )


def _add_training_distribution_plan(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "training-distribution-plan",
        help="write the locked Fugu-Ultra training task distribution plan",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", default=None)


def _add_manifest_freeze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "manifest-freeze",
        help="freeze validation and final-eval manifests with content hashes",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--md-out", default=None)
    p.add_argument("--created-at-utc", default=None)


def _add_harness_parity_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "harness-parity-report",
        help="summarize already-run Ultra harness parity canaries",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--created-at-utc", default=None)


def _add_failure_taxonomy_freeze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "failure-taxonomy-freeze",
        help="write the frozen rollout failure taxonomy and reward mapping",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--created-at-utc", default=None)


def _add_source_validate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "source-validate",
        help="run source-level validation and difficulty calibration for a task mix",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--tasks-jsonl", default=None)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", required=True)
    p.add_argument("--difficulty-out", required=True)
    p.add_argument("--quality-flags-out", required=True)
    p.add_argument("--created-at-utc", default=None)


def _add_mvp_data_mix(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "mvp-data-mix",
        help="write the 1,000-row MVP candidate train distribution",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--seed", type=int, default=0)


def _add_data_recipe_build(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "data-recipe-build",
        help="write data-recipe source manifests and canonical local TaskSpec shards",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--skip-existing-bank",
        action="store_true",
        help="only write source manifest/registry artifacts",
    )


def _add_trace_state_branches(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trace-state-branches",
        help="audit saved traces and rollout records for state-level branch candidates",
    )
    p.add_argument("--trace-jsonl", action="append", default=[])
    p.add_argument("--trace-glob", action="append", default=[])
    p.add_argument("--rollout-json", action="append", default=[])
    p.add_argument("--rollout-glob", action="append", default=[])
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)


def _add_trace_capture_plan(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trace-capture-plan",
        help="write the first train-allowed repo trace-capture job manifest",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--tasks-jsonl", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--jobs-out", required=True)
    p.add_argument("--task-limit", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--arms",
        default=None,
        help="comma-separated scaffold arms; default is OpenCode/Kimi, Codex/GPT, Claude/Opus solo",
    )


def _add_trace_capture_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trace-capture-run",
        help="execute trace-capture jobs and write rollout/AgentTrace artifacts",
    )
    p.add_argument("--jobs-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--job-id", action="append", default=[])
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--reasoning", default="high")
    p.add_argument("--dotenv", default=None, help="dotenv file to load before live trace-capture calls; default is repo .env")


def _add_trace_branch_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "trace-branch-tasks",
        help="materialize train-ready trace checkpoints into branch-repair TaskSpecs",
    )
    p.add_argument("--branch-candidates-jsonl", required=True)
    p.add_argument("--base-tasks-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--limit", type=int, default=None)


def _add_conductor_baseline_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "conductor-baseline-report",
        help="score pre-RL prompt-only and syntax/topology-SFT baselines from completed discovery rollouts",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)


def _add_workflow_pool_selection_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "workflow-pool-selection-report",
        help="estimate worker/scaffold contribution from completed workflow discovery rollouts",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)


def _add_grpo_pilot_seed(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "grpo-pilot-seed",
        help="build a seed set of GRPO pilot tasks from observed workflow disagreement/headroom",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--task-jsonl-out", default=None)


def _add_grpo_pilot_gap_plan(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "grpo-pilot-gap-plan",
        help="plan expansion from the current GRPO seed to a 300-task pilot",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--seed-jsonl", default=None)


def _add_grpo_pilot_freeze(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "grpo-pilot-freeze",
        help="freeze the GRPO pilot seed into a hash-locked training manifest",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--seed-jsonl", required=True)
    p.add_argument("--tasks-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--gap-plan-json", default=None)
    p.add_argument("--target-task-count", type=int, default=300)
    p.add_argument("--created-at-utc", default=None)


def _add_grpo_pilot_config(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "grpo-pilot-config",
        help="build the first GRPO pilot config from frozen tasks and selected workers",
    )
    p.add_argument("--freeze-report-json", required=True)
    p.add_argument("--pool-report-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--max-workflow-steps", type=int, default=3)


def _add_commercial_replay(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "commercial-replay",
        help="build replay and SFT artifacts from completed commercial-inclusive rollouts",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)


def _add_workflow_sft_warmstart(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "workflow-sft-warmstart",
        help="build the offline workflow-SFT warm-start dataset and Qwen3-8B config",
    )
    p.add_argument("--commercial-sft-jsonl", required=True)
    p.add_argument("--tasks-jsonl", required=True)
    p.add_argument("--pilot-config-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--examples-per-arm", type=int, default=2)


def _add_tasktrove_prefilter_batch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tasktrove-prefilter-batch",
        help="select a fresh verifier-backed TaskTrove prefilter shard and discovery jobs",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--inferredbugs-count", type=int, default=6)
    p.add_argument("--pymethods-count", type=int, default=6)
    p.add_argument(
        "--source-count",
        action="append",
        default=[],
        help="repeatable source=count override, e.g. tasktrove_r2egym=16; replaces inferredbugs/pymethods counts",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selection", choices=["ranked", "sequential"], default="ranked")


def _add_agenttrove_exact_prefilter_batch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "agenttrove-exact-prefilter-batch",
        help="select and materialize exact local TaskTrove tasks from AgentTrove disagreement priors",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--tasktrove-root", required=True)
    p.add_argument("--exact-matches-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--source-count",
        action="append",
        default=[],
        help="repeatable source=count override, e.g. tasktrove_r2egym=20",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-attempts", type=int, default=4)
    p.add_argument("--min-teacher-count", type=int, default=3)
    p.add_argument("--min-model-count", type=int, default=2)
    p.add_argument("--min-success-rate", type=float, default=0.25)
    p.add_argument("--max-success-rate", type=float, default=0.75)
    p.add_argument("--overwrite", action="store_true")


def _add_tasktrove_reservoir_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tasktrove-reservoir-report",
        help="summarize locally materialized TaskTrove sources before live discovery",
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--report-out", required=True)


def _add_agenttrove_disagreement_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "agenttrove-disagreement-report",
        help="rank AgentTrove tasks with mixed historical outcomes as TaskTrove prefilter priors",
    )
    p.add_argument("--parquet", action="append", default=[], help="local AgentTrove parquet shard")
    p.add_argument("--hf-file", action="append", default=[], help="dataset-relative AgentTrove parquet shard")
    p.add_argument("--hf-repo", default="open-thoughts/AgentTrove")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--manifest-dir", default=None, help="optional local manifest dir for exact TaskTrove match flags")
    p.add_argument("--candidates-out", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--source-filter", default=None, help="comma-separated AgentTrove source labels to keep")
    p.add_argument("--min-attempts", type=int, default=2)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--limit-rows-per-file", type=int, default=None)
    p.add_argument(
        "--allow-self-reported-completion",
        action="store_true",
        help="use assistant task_complete true/false as a weak prior when verifier outcome metadata is absent",
    )


def _add_expert_disagreement_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "expert-disagreement-tasks",
        help="materialize expert-designed verifier-backed tasks intended to induce workflow disagreement",
    )
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--version", choices=["v1", "v2"], default="v1")


def _add_taskcraft_source_probe(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "taskcraft-source-probe",
        help="filter TaskCraft text-only PDF/HTML rows into a controlled candidate-source report",
    )
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--candidates-out", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--limit", type=int, default=200)


def _add_taskcraft_readiness_audit(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "taskcraft-readiness-audit",
        help="audit TaskCraft candidates for source-freeze and deterministic-grader readiness",
    )
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--candidates-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--evidence-out", default=None)


def _add_acrouter_disagreement_candidates(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "acrouter-disagreement-candidates",
        help="extract CodeRouterBench OOD176 partial-solve tasks as reconstruction candidates",
    )
    p.add_argument("--coderouterbench-dir", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--index-out", default=None)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)


def _add_acrouter_reconstruction_queue(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "acrouter-reconstruction-queue",
        help="rank ACRouter OOD176 disagreement candidates by reconstruction readiness",
    )
    p.add_argument("--candidates-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--md-out", default=None)
    p.add_argument("--ready-swebench-out", default=None)
    p.add_argument("--ready-swebench-report-out", default=None)
    p.add_argument("--load-swebench-verified", action="store_true")
    p.add_argument("--detect-docker-images", action="store_true")
    p.add_argument("--image-prefix", default="swebench/sweb.eval.x86_64")


def _add_acrouter_swebench_smoke(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "acrouter-swebench-smoke",
        help="smoke the held-out SWE-bench Verified grader on one ACRouter ready candidate",
    )
    p.add_argument("--ready-jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--instance-id", default=None)
    p.add_argument("--patch-source", choices=["gold", "empty"], default="gold")
    p.add_argument("--image-prefix", default="swebench/sweb.eval.x86_64")
    p.add_argument("--eval-timeout", type=int, default=1200)
    p.add_argument("--network", default="none")


def _add_acrouter_swebench_tasks(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "acrouter-swebench-tasks",
        help="materialize held-out ACRouter SWE-bench ready candidates as pool-validation TaskSpecs",
    )
    p.add_argument("--ready-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--limit", type=int, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ultra", description="Fugu-Ultra Conductor toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_stepzero(sub)
    _add_pool_select(sub)
    _add_pool_tournament(sub)
    _add_pool_tournament_analyze(sub)
    _add_trace_convert_opencode(sub)
    _add_scaffold_tournament_plan(sub)
    _add_scaffold_tournament_manifest(sub)
    _add_scaffold_tournament_readiness(sub)
    _add_scaffold_materialize_repo(sub)
    _add_scaffold_canary(sub)
    _add_scaffold_discovery_run(sub)
    _add_docker_network_janitor(sub)
    _add_scaffold_discovery_analyze(sub)
    _add_scaffold_discovery_followup(sub)
    _add_training_repo_canaries(sub)
    _add_generated_repo_tasks(sub)
    _add_tool_dialog_tasks(sub)
    _add_tau_bench_tasks(sub)
    _add_long_context_tasks(sub)
    _add_long_context_adversarial_tasks(sub)
    _add_long_context_stress_tasks(sub)
    _add_harbor_materialize(sub)
    _add_tasktrove_parquet_materialize(sub)
    _add_training_distribution_plan(sub)
    _add_manifest_freeze(sub)
    _add_harness_parity_report(sub)
    _add_failure_taxonomy_freeze(sub)
    _add_source_validate(sub)
    _add_mvp_data_mix(sub)
    _add_data_recipe_build(sub)
    _add_trace_state_branches(sub)
    _add_trace_capture_plan(sub)
    _add_trace_capture_run(sub)
    _add_trace_branch_tasks(sub)
    _add_conductor_baseline_report(sub)
    _add_workflow_pool_selection_report(sub)
    _add_grpo_pilot_seed(sub)
    _add_grpo_pilot_gap_plan(sub)
    _add_grpo_pilot_freeze(sub)
    _add_grpo_pilot_config(sub)
    _add_commercial_replay(sub)
    _add_workflow_sft_warmstart(sub)
    _add_tasktrove_prefilter_batch(sub)
    _add_agenttrove_exact_prefilter_batch(sub)
    _add_tasktrove_reservoir_report(sub)
    _add_agenttrove_disagreement_report(sub)
    _add_expert_disagreement_tasks(sub)
    _add_taskcraft_source_probe(sub)
    _add_taskcraft_readiness_audit(sub)
    _add_acrouter_disagreement_candidates(sub)
    _add_acrouter_reconstruction_queue(sub)
    _add_acrouter_swebench_smoke(sub)
    _add_acrouter_swebench_tasks(sub)
    args = parser.parse_args()

    if args.cmd == "stepzero":
        from .stepzero_run import run_cli

        asyncio.run(run_cli(args))
    elif args.cmd == "pool-select":
        from pathlib import Path

        from .pool_selection import default_manifest_dir, render_report

        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else default_manifest_dir()
        report = render_report(manifest_dir, budget_usd=args.budget, seed=args.seed)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report)
        print(report)
    elif args.cmd == "pool-tournament":
        from pathlib import Path

        from .pool_tournament import default_manifest_path, run_tournament

        if args.manifest_path is None:
            args.manifest_path = str(default_manifest_path())
        print(json.dumps(asyncio.run(run_tournament(args)), indent=2))
    elif args.cmd == "pool-tournament-analyze":
        from pathlib import Path

        from .pool_tournament import analyze_rollout_file

        summary = analyze_rollout_file(Path(args.rollouts_jsonl))
        text = json.dumps(summary, indent=2)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
        print(text)
    elif args.cmd == "trace-convert-opencode":
        from pathlib import Path

        from .traces.opencode_rollouts import convert_rollouts

        report = convert_rollouts(Path(args.input_jsonl), Path(args.out_dir))
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "scaffold-tournament-plan":
        from pathlib import Path

        from .scaffold_tournament import build_plan

        task_mix = {
            "repo_open_repo_terminal": args.coding_tasks if args.coding_tasks is not None else args.repo_tasks,
            "unit_and_scientific_code": args.unit_code_tasks,
            "math_science_knowledge": args.direct_tasks,
            "tool_dialog": args.tool_dialog_tasks,
            "long_context_memory_planning": args.long_context_tasks,
        }
        plan = build_plan(task_mix)
        text = json.dumps(plan, indent=2, sort_keys=True)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n")
        print(text)
    elif args.cmd == "scaffold-tournament-manifest":
        from pathlib import Path

        from .scaffold_tournament import default_manifest_dir, write_concrete_manifest

        task_mix = {
            "repo_open_repo_terminal": args.coding_tasks if args.coding_tasks is not None else args.repo_tasks,
            "unit_and_scientific_code": args.unit_code_tasks,
            "math_science_knowledge": args.direct_tasks,
            "tool_dialog": args.tool_dialog_tasks,
            "long_context_memory_planning": args.long_context_tasks,
        }
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else default_manifest_dir()
        manifest = write_concrete_manifest(
            manifest_dir,
            Path(args.out),
            Path(args.jobs_out) if args.jobs_out else None,
            task_mix=task_mix,
            seed=args.seed,
            tasks_jsonl=Path(args.tasks_jsonl) if args.tasks_jsonl else None,
            branch_tasks_jsonl=Path(args.branch_tasks_jsonl) if args.branch_tasks_jsonl else None,
            include_pool_validation=args.include_pool_validation,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.cmd == "scaffold-tournament-readiness":
        from pathlib import Path

        from .scaffold_tournament import write_readiness

        report = write_readiness(Path(args.manifest_json), Path(args.out) if args.out else None)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "scaffold-materialize-repo":
        from pathlib import Path

        from .scaffold_materialize import materialize_repo_tasks

        report = materialize_repo_tasks(
            Path(args.manifest_json),
            Path(args.out_jsonl),
            Path(args.report_out) if args.report_out else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "scaffold-canary":
        from .scaffold_canary import run_cli

        asyncio.run(run_cli(args))
    elif args.cmd == "scaffold-discovery-run":
        from pathlib import Path

        from .scaffold_discovery_run import _split_csv, run_scaffold_discovery_jobs

        dry_run = args.dry_run or not (args.live or args.fake)
        report = asyncio.run(
            run_scaffold_discovery_jobs(
                jobs_jsonl=Path(args.jobs_jsonl),
                out_dir=Path(args.out_dir),
                report_out=Path(args.report_out),
                dry_run=dry_run,
                live=args.live,
                fake=args.fake,
                limit=args.limit,
                job_ids=set(args.job_id) if args.job_id else None,
                lanes=_split_csv(args.lanes),
                arms=_split_csv(args.arms),
                stages=_split_csv(args.stages),
                resume=args.resume,
                parallel=args.parallel,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
                budget=args.budget,
                provider_name=args.provider,
                dotenv=Path(args.dotenv) if args.dotenv else None,
                max_concurrency=args.max_concurrency,
                requests_per_minute=args.requests_per_minute,
                timeout_s=args.timeout_s,
                job_timeout_s=args.job_timeout_s,
                max_retries=args.max_retries,
                docker_network_janitor=args.docker_network_janitor,
                docker_network_janitor_dry_run=args.docker_network_janitor_dry_run,
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "docker-network-janitor":
        from .docker_janitor import run_cli

        report = run_cli(args)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "scaffold-discovery-analyze":
        from pathlib import Path

        from .scaffold_discovery_run import analyze_scaffold_discovery

        report = analyze_scaffold_discovery(
            jobs_jsonl=Path(args.jobs_jsonl),
            out_dir=Path(args.out_dir),
            report_out=Path(args.report_out),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "scaffold-discovery-followup":
        from pathlib import Path

        from .discovery_followup import build_discovery_followup_plan, split_csv

        report = build_discovery_followup_plan(
            manifest_dir=Path(args.manifest_dir),
            jobs_jsonl=Path(args.jobs_jsonl) if args.jobs_jsonl else None,
            out_json=Path(args.out_json),
            jobs_out=Path(args.jobs_out) if args.jobs_out else None,
            sources=split_csv(args.sources),
            lanes=split_csv(args.lanes),
            arm_domains=split_csv(args.arm_domains),
            stages=split_csv(args.stages),
            mode=args.mode,
            max_jobs=args.max_jobs,
            max_task_groups=args.max_task_groups,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "training-repo-canaries":
        from pathlib import Path

        from .training_repo_canary import materialize_training_repo_canaries

        report = materialize_training_repo_canaries(
            work_dir=Path(args.work_dir),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            image_tag=args.image_tag,
            build=not args.no_build,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "generated-repo-tasks":
        from pathlib import Path

        from .generated_repo_tasks import materialize_generated_repo_tasks

        report = materialize_generated_repo_tasks(
            work_dir=Path(args.work_dir),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            image_prefix=args.image_prefix,
            build=not args.no_build,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "tool-dialog-tasks":
        from pathlib import Path

        from .tool_dialog_tasks import materialize_tool_dialog_tasks

        report = materialize_tool_dialog_tasks(
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "tau-bench-tasks":
        from pathlib import Path

        from .tau_bench_tasks import materialize_tau_bench_tasks

        report = materialize_tau_bench_tasks(
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            limit=args.limit,
            offset=args.offset,
            selection=args.selection,
            tasks_train_path=Path(args.tasks_train_path) if args.tasks_train_path else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "long-context-tasks":
        from pathlib import Path

        from .long_context_tasks import materialize_long_context_tasks

        report = materialize_long_context_tasks(
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "long-context-adversarial-tasks":
        from pathlib import Path

        from .long_context_adversarial_tasks import materialize_long_context_adversarial_tasks

        report = materialize_long_context_adversarial_tasks(
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "long-context-stress-tasks":
        from pathlib import Path

        from .long_context_stress_tasks import materialize_long_context_stress_tasks

        report = materialize_long_context_stress_tasks(
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "harbor-materialize":
        from pathlib import Path

        from .sources.harbor import materialize_harbor_tasks

        report = materialize_harbor_tasks(
            Path(args.root),
            Path(args.out_jsonl),
            Path(args.report_out) if args.report_out else None,
            source_name=args.source_name,
            source_version=args.source_version,
            policy=args.policy,
            split=args.split,
            limit=args.limit,
            verifier_backed_only=not args.include_no_verifier,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "tasktrove-parquet-materialize":
        from pathlib import Path

        from .sources.harbor import download_tasktrove_parquet, materialize_tasktrove_parquet

        parquet_path = (
            Path(args.parquet)
            if args.parquet
            else download_tasktrove_parquet(
                hf_file=args.hf_file,
                repo_id=args.hf_repo,
                cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
            )
        )
        include_paths = set(args.include_path or [])
        if args.include_paths_jsonl:
            with Path(args.include_paths_jsonl).open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row.get("tasktrove_path") or row.get("path") or row.get("task_id")
                    if value not in (None, ""):
                        include_paths.add(str(value))
        report = materialize_tasktrove_parquet(
            parquet_path=parquet_path,
            extract_dir=Path(args.extract_dir),
            out_jsonl=Path(args.out_jsonl),
            report_path=Path(args.report_out),
            source_name=args.source_name,
            source_version=args.source_version,
            policy=args.policy,
            split=args.split,
            include_paths=include_paths or None,
            limit=args.limit,
            offset=args.offset,
            seed=args.seed,
            shuffle=args.shuffle,
            overwrite=args.overwrite,
            verifier_backed_only=not args.include_no_verifier,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "training-distribution-plan":
        from pathlib import Path

        from .training_distribution import write_training_distribution_plan

        plan = write_training_distribution_plan(
            manifest_dir=Path(args.manifest_dir),
            out_json=Path(args.out_json),
            out_md=Path(args.out_md) if args.out_md else None,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif args.cmd == "manifest-freeze":
        from pathlib import Path

        from .manifest_freeze import build_manifest_freeze

        report = build_manifest_freeze(
            manifest_dir=Path(args.manifest_dir),
            out_dir=Path(args.out_dir),
            report_out=Path(args.report_out) if args.report_out else None,
            md_out=Path(args.md_out) if args.md_out else None,
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "harness-parity-report":
        from pathlib import Path

        from .harness_parity import build_harness_parity_report

        report = build_harness_parity_report(
            manifest_dir=Path(args.manifest_dir),
            repo_root=Path(args.repo_root),
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "failure-taxonomy-freeze":
        from pathlib import Path

        from .failure_taxonomy import build_failure_taxonomy_report

        report = build_failure_taxonomy_report(
            manifest_dir=Path(args.manifest_dir),
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "source-validate":
        from pathlib import Path

        from .source_validation import build_source_validation_report

        report = build_source_validation_report(
            manifest_dir=Path(args.manifest_dir),
            tasks_jsonl=Path(args.tasks_jsonl) if args.tasks_jsonl else None,
            report_out=Path(args.report_out),
            md_out=Path(args.md_out),
            difficulty_out=Path(args.difficulty_out),
            quality_flags_out=Path(args.quality_flags_out),
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "mvp-data-mix":
        from pathlib import Path

        from .mvp_data_mix import build_mvp_grpo_mix

        report = build_mvp_grpo_mix(
            manifest_dir=Path(args.manifest_dir),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out) if args.report_out else None,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "data-recipe-build":
        from pathlib import Path

        from .data_recipe import write_data_recipe_artifacts

        report = write_data_recipe_artifacts(
            manifest_dir=Path(args.manifest_dir),
            out_dir=Path(args.out_dir),
            include_existing_bank=not args.skip_existing_bank,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "trace-state-branches":
        from pathlib import Path

        from .trace_state_branches import build_trace_state_branch_report, expand_globs

        report = build_trace_state_branch_report(
            trace_jsonls=[*[Path(path) for path in args.trace_jsonl], *expand_globs(args.trace_glob)],
            rollout_jsons=[*[Path(path) for path in args.rollout_json], *expand_globs(args.rollout_glob)],
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "trace-capture-plan":
        from pathlib import Path

        from .trace_capture_plan import DEFAULT_TRACE_ARMS, build_trace_capture_plan

        arms = tuple(args.arms.split(",")) if args.arms else DEFAULT_TRACE_ARMS
        plan = build_trace_capture_plan(
            manifest_dir=Path(args.manifest_dir),
            tasks_jsonl=Path(args.tasks_jsonl),
            out_json=Path(args.out_json),
            jobs_out=Path(args.jobs_out),
            task_limit=args.task_limit,
            seed=args.seed,
            arms=arms,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif args.cmd == "trace-capture-run":
        from pathlib import Path

        from .trace_capture_run import run_trace_capture_jobs

        report = asyncio.run(
            run_trace_capture_jobs(
                jobs_jsonl=Path(args.jobs_jsonl),
                report_out=Path(args.report_out),
                limit=args.limit,
                job_ids=set(args.job_id) if args.job_id else None,
                resume=not args.no_resume,
                parallel=args.parallel,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning=args.reasoning,
                dotenv=Path(args.dotenv) if args.dotenv else None,
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "trace-branch-tasks":
        from pathlib import Path

        from .trace_branch_tasks import materialize_trace_branch_tasks

        report = materialize_trace_branch_tasks(
            branch_candidates_jsonl=Path(args.branch_candidates_jsonl),
            base_tasks_jsonl=Path(args.base_tasks_jsonl),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out),
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "conductor-baseline-report":
        from pathlib import Path

        from .conductor_baselines import build_conductor_baseline_report

        report = build_conductor_baseline_report(
            manifest_dir=Path(args.manifest_dir),
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "workflow-pool-selection-report":
        from pathlib import Path

        from .workflow_pool_selection import build_workflow_pool_selection_report

        report = build_workflow_pool_selection_report(
            manifest_dir=Path(args.manifest_dir),
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "grpo-pilot-seed":
        from pathlib import Path

        from .grpo_pilot_seed import build_grpo_pilot_seed

        report = build_grpo_pilot_seed(
            manifest_dir=Path(args.manifest_dir),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out),
            task_jsonl_out=Path(args.task_jsonl_out) if args.task_jsonl_out else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "grpo-pilot-gap-plan":
        from pathlib import Path

        from .grpo_pilot_gap_plan import build_grpo_pilot_gap_plan

        report = build_grpo_pilot_gap_plan(
            manifest_dir=Path(args.manifest_dir),
            seed_jsonl=Path(args.seed_jsonl) if args.seed_jsonl else None,
            report_out=Path(args.report_out),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "grpo-pilot-freeze":
        from pathlib import Path

        from .grpo_pilot_freeze import build_grpo_pilot_freeze

        report = build_grpo_pilot_freeze(
            manifest_dir=Path(args.manifest_dir),
            seed_jsonl=Path(args.seed_jsonl),
            tasks_jsonl=Path(args.tasks_jsonl),
            out_dir=Path(args.out_dir),
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
            gap_plan_json=Path(args.gap_plan_json) if args.gap_plan_json else None,
            target_task_count=args.target_task_count,
            created_at_utc=args.created_at_utc,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "grpo-pilot-config":
        from pathlib import Path

        from .grpo_pilot_config import build_grpo_pilot_config

        report = build_grpo_pilot_config(
            freeze_report_json=Path(args.freeze_report_json),
            pool_report_json=Path(args.pool_report_json),
            out_json=Path(args.out_json),
            md_out=Path(args.md_out) if args.md_out else None,
            max_workflow_steps=args.max_workflow_steps,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "commercial-replay":
        from pathlib import Path

        from .commercial_replay import build_commercial_replay

        report = build_commercial_replay(
            manifest_dir=Path(args.manifest_dir),
            out_dir=Path(args.out_dir),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "workflow-sft-warmstart":
        from pathlib import Path

        from .workflow_sft_warmstart import build_workflow_sft_warmstart

        report = build_workflow_sft_warmstart(
            commercial_sft_jsonl=Path(args.commercial_sft_jsonl),
            tasks_jsonl=Path(args.tasks_jsonl),
            pilot_config_json=Path(args.pilot_config_json),
            out_dir=Path(args.out_dir),
            examples_per_arm=args.examples_per_arm,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "tasktrove-prefilter-batch":
        from pathlib import Path

        from .tasktrove_prefilter import build_tasktrove_prefilter_batch

        source_counts = None
        if args.source_count:
            source_counts = {}
            for item in args.source_count:
                if "=" not in item:
                    raise SystemExit(f"--source-count must be source=count: {item}")
                source, raw_count = item.split("=", 1)
                source = source.strip()
                try:
                    count = int(raw_count)
                except ValueError as exc:
                    raise SystemExit(f"--source-count count must be an integer: {item}") from exc
                source_counts[source] = count

        report = build_tasktrove_prefilter_batch(
            manifest_dir=Path(args.manifest_dir),
            out_dir=Path(args.out_dir),
            inferredbugs_count=args.inferredbugs_count,
            pymethods_count=args.pymethods_count,
            source_counts=source_counts,
            seed=args.seed,
            selection=args.selection,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "agenttrove-exact-prefilter-batch":
        from pathlib import Path

        from .tasktrove_prefilter import build_agenttrove_exact_prefilter_batch

        source_counts = None
        if args.source_count:
            source_counts = {}
            for item in args.source_count:
                if "=" not in item:
                    raise SystemExit(f"--source-count must be source=count: {item}")
                source, raw_count = item.split("=", 1)
                source = source.strip()
                try:
                    count = int(raw_count)
                except ValueError as exc:
                    raise SystemExit(f"--source-count count must be an integer: {item}") from exc
                source_counts[source] = count

        report = build_agenttrove_exact_prefilter_batch(
            exact_matches_jsonl=Path(args.exact_matches_jsonl),
            tasktrove_root=Path(args.tasktrove_root),
            manifest_dir=Path(args.manifest_dir),
            out_dir=Path(args.out_dir),
            source_counts=source_counts,
            seed=args.seed,
            min_attempts=args.min_attempts,
            min_teacher_count=args.min_teacher_count,
            min_model_count=args.min_model_count,
            min_success_rate=args.min_success_rate,
            max_success_rate=args.max_success_rate,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "tasktrove-reservoir-report":
        from pathlib import Path

        from .tasktrove_prefilter import build_tasktrove_reservoir_report

        report = build_tasktrove_reservoir_report(
            manifest_dir=Path(args.manifest_dir),
            report_out=Path(args.report_out),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "agenttrove-disagreement-report":
        from pathlib import Path

        from .agenttrove_disagreement import download_agenttrove_parquet, scan_agenttrove_disagreement
        from .scaffold_discovery_run import _split_csv

        parquet_paths = [Path(path) for path in args.parquet]
        parquet_paths.extend(
            download_agenttrove_parquet(
                hf_file=hf_file,
                repo_id=args.hf_repo,
                cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
            )
            for hf_file in args.hf_file
        )
        if not parquet_paths:
            raise SystemExit("provide at least one --parquet or --hf-file")
        report = scan_agenttrove_disagreement(
            parquet_paths=parquet_paths,
            candidates_out=Path(args.candidates_out),
            report_out=Path(args.report_out),
            manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
            source_filter=set(_split_csv(args.source_filter) or []),
            min_attempts=args.min_attempts,
            top_k=args.top_k,
            batch_size=args.batch_size,
            limit_rows_per_file=args.limit_rows_per_file,
            allow_self_reported_completion=args.allow_self_reported_completion,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "expert-disagreement-tasks":
        from pathlib import Path

        from .expert_disagreement_tasks import materialize_expert_disagreement_tasks

        report = materialize_expert_disagreement_tasks(
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out),
            limit=args.limit,
            version=args.version,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "taskcraft-source-probe":
        from pathlib import Path

        from .taskcraft_source import build_taskcraft_source_probe

        report = build_taskcraft_source_probe(
            dataset_dir=Path(args.dataset_dir),
            candidates_out=Path(args.candidates_out),
            report_out=Path(args.report_out),
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "taskcraft-readiness-audit":
        from pathlib import Path

        from .taskcraft_source import build_taskcraft_readiness_audit

        report = build_taskcraft_readiness_audit(
            dataset_dir=Path(args.dataset_dir),
            candidates_jsonl=Path(args.candidates_jsonl),
            report_out=Path(args.report_out),
            evidence_out=Path(args.evidence_out) if args.evidence_out else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "acrouter-disagreement-candidates":
        from pathlib import Path

        from .acrouter_candidates import extract_ood176_disagreement_candidates

        report = extract_ood176_disagreement_candidates(
            coderouterbench_dir=Path(args.coderouterbench_dir),
            out_jsonl=Path(args.out_jsonl),
            index_out=Path(args.index_out) if args.index_out else None,
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "acrouter-reconstruction-queue":
        from pathlib import Path

        from .acrouter_candidates import build_ood176_reconstruction_queue

        report = build_ood176_reconstruction_queue(
            candidates_jsonl=Path(args.candidates_jsonl),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out),
            md_out=Path(args.md_out) if args.md_out else None,
            ready_swebench_out=Path(args.ready_swebench_out) if args.ready_swebench_out else None,
            ready_swebench_report_out=(
                Path(args.ready_swebench_report_out) if args.ready_swebench_report_out else None
            ),
            load_swebench_verified=args.load_swebench_verified,
            detect_docker_images=args.detect_docker_images,
            image_prefix=args.image_prefix,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "acrouter-swebench-smoke":
        from pathlib import Path

        from .acrouter_swebench import run_swebench_ready_smoke

        report = run_swebench_ready_smoke(
            ready_jsonl=Path(args.ready_jsonl),
            out=Path(args.out),
            log_dir=Path(args.log_dir),
            instance_id=args.instance_id,
            patch_source=args.patch_source,
            image_prefix=args.image_prefix,
            eval_timeout=args.eval_timeout,
            network=args.network,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.cmd == "acrouter-swebench-tasks":
        from pathlib import Path

        from .acrouter_swebench import materialize_swebench_ready_tasks

        report = materialize_swebench_ready_tasks(
            ready_jsonl=Path(args.ready_jsonl),
            out_jsonl=Path(args.out_jsonl),
            report_out=Path(args.report_out),
            limit=args.limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
