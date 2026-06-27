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
    p.add_argument("--coding-tasks", type=int, default=15)
    p.add_argument("--tool-dialog-tasks", type=int, default=10)
    p.add_argument("--direct-tasks", type=int, default=12)
    p.add_argument("--out", default=None)


def _add_scaffold_tournament_manifest(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scaffold-tournament-manifest",
        help="select concrete tasks and jobs for the scaffold-aware Ultra tournament",
    )
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--coding-tasks", type=int, default=15)
    p.add_argument("--tool-dialog-tasks", type=int, default=10)
    p.add_argument("--direct-tasks", type=int, default=12)
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
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--include-no-verifier",
        action="store_true",
        help="include Harbor task bundles without verifier/tests payloads",
    )


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
    _add_harbor_materialize(sub)
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
            "coding_repo": args.coding_tasks,
            "tool_dialog": args.tool_dialog_tasks,
            "direct_reasoning": args.direct_tasks,
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
            "coding_repo": args.coding_tasks,
            "tool_dialog": args.tool_dialog_tasks,
            "direct_reasoning": args.direct_tasks,
        }
        manifest_dir = Path(args.manifest_dir) if args.manifest_dir else default_manifest_dir()
        manifest = write_concrete_manifest(
            manifest_dir,
            Path(args.out),
            Path(args.jobs_out) if args.jobs_out else None,
            task_mix=task_mix,
            seed=args.seed,
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
    elif args.cmd == "harbor-materialize":
        from pathlib import Path

        from .sources.harbor import materialize_harbor_tasks

        report = materialize_harbor_tasks(
            Path(args.root),
            Path(args.out_jsonl),
            Path(args.report_out) if args.report_out else None,
            source_name=args.source_name,
            source_version=args.source_version,
            limit=args.limit,
            verifier_backed_only=not args.include_no_verifier,
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
